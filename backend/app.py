import os
from datetime import datetime
from flask import Flask, request, jsonify
import requests
import json
from dotenv import load_dotenv
import os
import torch

from weights_loader import load_fttransformer_from_config_json_and_pth
from flask_cors import CORS

load_dotenv()  # reads .env into environment variables

app = Flask(__name__)

pth_path = "aeolus_model_weights-2.pth"
json_pth_path = "ft_transformer_weights-2.json"

import numpy as np
import torch

CAT_FEATURES = [
    "OP_CARRIER",
    "OP_CARRIER_FL_NUM",
    "ORIGIN_INDEX",
    "DEST_INDEX",
    "CRS_DEP_HOURS",
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "MONTH",
]

NUM_FEATURES = [
    "FLIGHTS",
    "O_TEMP","O_PRCP","O_WSPD",
    "D_TEMP","D_PRCP","D_WSPD",
    "O_LATITUDE","O_LONGITUDE",
    "D_LATITUDE","D_LONGITUDE",
    "CRS_DEP_MINS",
]

def load_label_encoders(path="label_encoders.json"):
    with open(path, "r") as f:
        raw = json.load(f)

    encoders = {}
    for col, classes in raw.items():
        # build value -> id map
        encoders[col] = {str(v): i for i, v in enumerate(classes)}
    return encoders


def encode_cat(col: str, value, card: int) -> int:
    """
    Training-compatible encoding:
      - LabelEncoder on str(value) gives 0..n-1
      - shift by +1 so 0 is reserved for unknown/pad
      - clamp to [0..card]
    """
    s = str(value)

    mapping = LABEL_ENCODERS.get(col)
    if mapping is None:
        # fallback: unknown
        return 0

    if s in mapping:
        idx = mapping[s] + 1  # << IMPORTANT +1 shift
    else:
        idx = 0  # unknown

    # clamp just in case
    if idx < 0:
        return 0
    if idx > card:
        return card
    return idx

CORS(app)


def load_airport_map(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)
    
LABEL_ENCODERS = load_label_encoders("label_encoders.json")
CAT_CARDINALITIES = [15, 6854, 271, 300, 31, 7, 11, 24]

app.config["AIRPORT_MAP"] = load_airport_map("airport_map.json")

AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY", "")
AVIATIONSTACK_BASE_URL = "http://api.aviationstack.com/v1/flights"

airport_map = app.config.get("AIRPORT_MAP", {})  # recommended: set at startup
n_airports = len(airport_map)

device = "cuda" if torch.cuda.is_available() else "cpu"

model, meta = load_fttransformer_from_config_json_and_pth(
    config_json_path=json_pth_path,
    weights_pth_path=pth_path,
    device=device,
    strict=True,
)

print("Loaded. Missing:", len(meta["missing_keys"]), "Unexpected:", len(meta["unexpected_keys"]))

def encode_label(encoders, col, value):
    """
    Exact match to LabelEncoder.fit_transform(data[col].astype(str))
    with safe handling of unseen values.
    """
    s = str(value)
    enc = encoders[col]

    if s in enc:
        return enc[s]

    # unseen category → map to most frequent / fallback
    # safest choice: map to 0 (first learned class)
    return 0

def predict_one(model, params: dict, device: str):
    x_cat_vals = [
        encode_cat(CAT_FEATURES[i], params[CAT_FEATURES[i]], CAT_CARDINALITIES[i])
        for i in range(len(CAT_FEATURES))
    ]
    x_num_vals = [float(params[k]) for k in NUM_FEATURES]

    x_cat = torch.tensor([x_cat_vals], dtype=torch.long, device=device)
    x_num = torch.tensor([x_num_vals], dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        y = model(x_num, x_cat)
    return float(y.squeeze().cpu().item())


def get_airport_features(iata_code: str) -> dict:
    rec = airport_map.get(iata_code)
    if not rec:
        raise KeyError(f"Unknown airport code in mapping: {iata_code}")
    return rec

def fetch_weather(lat: float, lon: float, flight_date: str, crs_dep_time: int) -> dict:
    """
    Fetch weather near departure time using Open-Meteo.

    Returns:
    {
        "temp": float,   # temperature (°C)
        "prcp": float,   # precipitation (mm)
        "wspd": float    # wind speed (km/h)
    }
    """

    # Convert CRS_DEP_TIME (e.g. 1430) → hour/minute
    hour = crs_dep_time // 100
    minute = crs_dep_time % 100

    # Build datetime
    dep_dt = datetime.strptime(flight_date, "%Y-%m-%d").replace(
        hour=hour, minute=minute
    )

    # Open-Meteo works on hourly granularity → round down to hour
    dep_hour = dep_dt.replace(minute=0, second=0)

    # Decide whether to use forecast or historical endpoint
    today = datetime.utcnow().date()
    is_historical = dep_hour.date() < today

    if is_historical:
        base_url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        base_url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,windspeed_10m",
        "start_date": dep_hour.date().isoformat(),
        "end_date": dep_hour.date().isoformat(),
        "timezone": "UTC"
    }

    r = requests.get(base_url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    # Find matching hour
    dep_hour_str = dep_hour.strftime("%Y-%m-%dT%H:00")
    try:
        idx = times.index(dep_hour_str)
    except ValueError:
        raise ValueError(f"No weather data found for {dep_hour_str}")

    return {
        "temp": float(hourly["temperature_2m"][idx]),
        "prcp": float(hourly["precipitation"][idx]),
        "wspd": float(hourly["windspeed_10m"][idx]),
    }

def hhmm_from_iso(dt_str: str) -> int:
    """
    Convert an ISO datetime string like '2026-02-01T14:30:00+00:00'
    into HHMM int like 1430 (local to whatever the API returns).
    """
    # datetime.fromisoformat handles offsets like +00:00
    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    return dt.hour * 100 + dt.minute

def split_hhmm(crs_dep_time: int) -> tuple[int, int]:
    t = int(crs_dep_time)
    hours = max(0, min(23, t // 100))
    mins = max(0, min(59, t % 100))
    return hours, mins

def fetch_flight_from_aviationstack(marketing_carrier: str,
                                   marketing_flight_number: str,
                                   flight_date: str,
                                   origin: str,
                                   destination: str) -> dict:
    """
    Query AviationStack for a specific flight/date/route.
    Returns inferred OP_CARRIER, OP_CARRIER_FL_NUM, CRS_DEP_TIME.
    """
    if not AVIATIONSTACK_API_KEY:
        raise RuntimeError("Missing AVIATIONSTACK_API_KEY env var")

    params = {
        "access_key": AVIATIONSTACK_API_KEY,

        # Filters (AviationStack supports these; exact behavior can vary by plan)
        # "flight_date": flight_date,                  # YYYY-MM-DD
        "flight_iata": f"{marketing_carrier}{marketing_flight_number}",  # e.g. AA123
        "dep_iata": origin,
        "arr_iata": destination,
        # Optional: limit results
        "limit": 10
    }

    r = requests.get(AVIATIONSTACK_BASE_URL, params=params, timeout=15)
    if not r.ok:
        # Surface aviationstack's own error payload
        try:
            err = r.json()
        except Exception:
            err = {"raw": r.text}
        raise RuntimeError(f"Aviationstack error {r.status_code}: {err}")

    payload = r.json()

    data = payload.get("data") or []
    if not data:
        return {}

    # Choose the "best" match (first result). You may want smarter selection.
    flight = data[0]

    # Operating carrier / number might live under different keys depending on API response
    # These are common fields in AviationStack:
    # flight["airline"]["iata"], flight["flight"]["number"], flight["flight"]["iata"]
    op_carrier = (flight.get("airline") or {}).get("iata") or marketing_carrier
    op_fl_num = (flight.get("flight") or {}).get("number") or marketing_flight_number

    # Scheduled departure time often in:
    # flight["departure"]["scheduled"] like "2026-02-01T14:30:00+00:00"
    dep_scheduled = (flight.get("departure") or {}).get("scheduled")
    if not dep_scheduled:
        # fallback: sometimes there are other fields, but keep it simple
        raise ValueError("Scheduled departure time not found in API response")

    crs_dep_time = hhmm_from_iso(dep_scheduled)

    return {
        "OP_CARRIER": op_carrier,
        "OP_CARRIER_FL_NUM": int(op_fl_num) if str(op_fl_num).isdigit() else op_fl_num,
        "CRS_DEP_TIME": crs_dep_time
    }

@app.route("/api/flight", methods=["POST"])
def flight_lookup():
    """
    Input JSON:
    {
      "marketing_carrier": "AA",
      "marketing_flight_number": "123",
      "flight_date": "2026-02-01",
      "origin": "JFK",
      "destination": "LAX"
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing or invalid JSON payload"}), 400

    required = ["marketing_carrier", "marketing_flight_number", "flight_date", "origin", "destination"]
    missing = [k for k in required if k not in data or data.get(k) in (None, "")]
    if missing:
        return jsonify({"error": "Missing required fields", "missing_fields": missing}), 400

    marketing_carrier = str(data["marketing_carrier"]).strip().upper()
    marketing_flight_number = str(data["marketing_flight_number"]).strip()
    flight_date = str(data["flight_date"]).strip()
    origin = str(data["origin"]).strip().upper()
    destination = str(data["destination"]).strip().upper()

    # Basic date format check
    try:
        dt = datetime.strptime(flight_date, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "flight_date must be YYYY-MM-DD"}), 400

    try:
        inferred = fetch_flight_from_aviationstack(
            marketing_carrier=marketing_carrier,
            marketing_flight_number=marketing_flight_number,
            flight_date=flight_date,
            origin=origin,
            destination=destination
        )
    except requests.HTTPError as e:
        return jsonify({"error": "Upstream API error", "details": str(e)}), 502
    except Exception as e:
        return jsonify({"error": "Failed to infer flight details", "details": str(e)}), 500

    if not inferred:
        return jsonify({"error": "No matching flight found"}), 404
    
    # get origin & destination index + coordinates from mapping
    try:
        o = get_airport_features(origin)
        d = get_airport_features(destination)
    except KeyError as e:
        return jsonify({"error": str(e)}), 400

    origin_index = int(o["index"])
    dest_index = int(d["index"])

    o_lat = float(o["lat"])
    o_lon = float(o["lon"])
    d_lat = float(d["lat"])
    d_lon = float(d["lon"])

    # get origin & destination weather (temp, prcp, wspd)
    o_weather = fetch_weather(o_lat, o_lon, flight_date, inferred["CRS_DEP_TIME"])
    d_weather = fetch_weather(d_lat, d_lon, flight_date, inferred["CRS_DEP_TIME"])

    crs_dep_time = int(inferred["CRS_DEP_TIME"])
    crs_dep_hours, crs_dep_mins = split_hhmm(crs_dep_time)

    # set up params for model inference (match your training column names)
    params = {
        "OP_CARRIER": inferred["OP_CARRIER"],
        "OP_CARRIER_FL_NUM": inferred["OP_CARRIER_FL_NUM"],
        "ORIGIN_INDEX": origin_index,
        "DEST_INDEX": dest_index,
        "CRS_DEP_HOURS": crs_dep_hours,
        "CRS_DEP_MINS": crs_dep_mins,
        "DAY_OF_MONTH": int(dt.day),
        "DAY_OF_WEEK": int(dt.isoweekday()), 
        "MONTH": int(dt.month),
        "FLIGHTS": 1,  # change if better source

        "O_TEMP": float(o_weather["temp"]),
        "O_PRCP": float(o_weather["prcp"]),
        "O_WSPD": float(o_weather["wspd"]),
        "D_TEMP": float(d_weather["temp"]),
        "D_PRCP": float(d_weather["prcp"]),
        "D_WSPD": float(d_weather["wspd"]),

        "O_LATITUDE": o_lat,
        "O_LONGITUDE": o_lon,
        "D_LATITUDE": d_lat,
        "D_LONGITUDE": d_lon,
    }

    # If your model expects columns in a strict order, you can enforce it:
    FEATURE_ORDER = [
        "OP_CARRIER",
        "OP_CARRIER_FL_NUM",
        "ORIGIN_INDEX",
        "DEST_INDEX",
        "CRS_DEP_HOURS",
        "DAY_OF_MONTH",
        "DAY_OF_WEEK",
        "MONTH",
        "FLIGHTS",
        "O_TEMP",
        "O_PRCP",
        "O_WSPD",
        "D_TEMP",
        "D_PRCP",
        "D_WSPD",
        "O_LATITUDE",
        "O_LONGITUDE",
        "D_LATITUDE",
        "D_LONGITUDE",
        "CRS_DEP_MINS",
    ]
    params = {k: params[k] for k in FEATURE_ORDER}

    # return jsonify(params), 200

    
    # infer delay prediction from model
    prediction = predict_one(model, params, device)

    return jsonify({
        "message": "Flight feature payload received",
        "prediction": prediction
    }), 200
    


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
