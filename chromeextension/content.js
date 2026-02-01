// content.js — Aeolus predictor (Google Flights)
// Key improvements:
// - calls your API ONLY ONCE per unique flight (per page session)
// - carrier+flight extracted from DOM token (e.g. "DL 2344"), not price
// - negative delays are ALWAYS green "Accurate"
// - injects status below "Departure time" in DOM
// - supports multiple cards with caching + strong de-dupe
// - optional debug panel + verbose logs

const STORAGE_DEFAULTS = {
  backendUrl: "http://127.0.0.1:5000",
  dateOverride: "",
  enabled: true,
  debug: true,
  debugPanel: false
};

const state = {
  settings: { ...STORAGE_DEFAULTS },
  // cacheKey -> { status: "ok"|"pending"|"error", html, colorClass, predictionMinutes, ts }
  cache: new Map(),
  // depEl -> injected line element
  injectedByDepEl: new WeakMap(),
  panelEl: null,
  stylesInjected: false
};

// HARD de-dupe: only query once per unique key during the lifetime of the page.
const seenKeys = new Set();

const LOG_PREFIX = "[AEOLUS]";
function log(...args) {
  if (state.settings.debug) console.log(LOG_PREFIX, ...args);
}
function warn(...args) {
  if (state.settings.debug) console.warn(LOG_PREFIX, ...args);
}
function error(...args) {
  if (state.settings.debug) console.error(LOG_PREFIX, ...args);
}

function normalizeSpaces(s) {
  return (s || "")
    .replace(/\u00A0/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function ensureStylesInjected() {
  if (state.stylesInjected) return;
  const style = document.createElement("style");
  style.textContent = `
    .aeolus-prediction {
      margin-top: 4px;
      font-size: 14px; /* bigger */
      line-height: 1.25;
      font-weight: 600;
    }
    .aeolus-prediction--loading { opacity: 0.75; font-weight: 500; }
    .aeolus-prediction--error { opacity: 0.9; font-weight: 600; }
    .aeolus-green { color: #1a7f37; }
    .aeolus-yellow { color: #b58100; }
    .aeolus-red { color: #c62828; }
  `;
  document.documentElement.appendChild(style);
  state.stylesInjected = true;
}

function panelEnsure() {
  if (!state.settings.debugPanel) return null;
  if (state.panelEl && state.panelEl.isConnected) return state.panelEl;

  const el = document.createElement("div");
  el.style.position = "fixed";
  el.style.right = "12px";
  el.style.bottom = "12px";
  el.style.zIndex = "999999";
  el.style.maxWidth = "520px";
  el.style.maxHeight = "42vh";
  el.style.overflow = "auto";
  el.style.fontSize = "12px";
  el.style.fontFamily = "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
  el.style.background = "rgba(0,0,0,0.78)";
  el.style.color = "#fff";
  el.style.padding = "10px";
  el.style.borderRadius = "10px";
  el.style.boxShadow = "0 6px 20px rgba(0,0,0,0.25)";
  el.style.whiteSpace = "pre-wrap";

  const header = document.createElement("div");
  header.style.display = "flex";
  header.style.justifyContent = "space-between";
  header.style.alignItems = "center";
  header.style.marginBottom = "8px";

  const title = document.createElement("div");
  title.textContent = "AEOLUS debug";
  title.style.fontWeight = "700";

  const btn = document.createElement("button");
  btn.textContent = "Clear";
  btn.style.cursor = "pointer";
  btn.style.border = "0";
  btn.style.borderRadius = "8px";
  btn.style.padding = "6px 10px";
  btn.onclick = () => {
    const body = el.querySelector("#aeolus-debug-body");
    if (body) body.textContent = "";
  };

  const body = document.createElement("div");
  body.id = "aeolus-debug-body";

  header.appendChild(title);
  header.appendChild(btn);
  el.appendChild(header);
  el.appendChild(body);

  document.documentElement.appendChild(el);
  state.panelEl = el;
  return el;
}

function panelWrite(line) {
  if (!state.settings.debugPanel) return;
  panelEnsure();
  const body = state.panelEl?.querySelector("#aeolus-debug-body");
  if (!body) return;
  const t = new Date().toISOString().slice(11, 19);
  body.textContent += `[${t}] ${line}\n`;
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function debounce(fn, wait = 1200) {
  let t = null;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), wait);
  };
}

function pad2(n) {
  return String(n).padStart(2, "0");
}

function toYMD(date) {
  return `${date.getFullYear()}-${pad2(date.getMonth() + 1)}-${pad2(date.getDate())}`;
}

function normalizeBaseUrl(url) {
  return (url || "").replace(/\/+$/, "");
}

// ===== Extractors =====

function extractIataCodesFromText(text) {
  const re = /\(([A-Z]{3})\)/g;
  const codes = [];
  let m;
  while ((m = re.exec(text)) !== null) codes.push(m[1]);
  return [...new Set(codes)];
}

function looksLikeCarrierFlight(s) {
  // Must look like "DL 2344" or "DL2344"
  // Carrier must include at least one letter (avoid "29 0" / prices / times)
  // Flight number 2–5 digits (avoid "0")
  const t = normalizeSpaces(s).toUpperCase();
  const m = t.match(/^([A-Z0-9]{2,3})\s*(\d{2,5})$/);
  if (!m) return null;

  const carrier = m[1];
  const flightNo = m[2];

  if (!/[A-Z]/.test(carrier)) return null;
  return { marketing_carrier: carrier, marketing_flight_number: flightNo };
}

function extractCarrierAndFlightNumberFromDom(root) {
  // Google Flights often has the token in a <span>, e.g. "DL 2344"
  const nodes = root.querySelectorAll("span, div");
  for (const n of nodes) {
    const txt = normalizeSpaces(n.textContent);
    if (!txt) continue;
    if (txt.length > 12) continue; // keep tiny tokens only
    const parsed = looksLikeCarrierFlight(txt);
    if (parsed) return parsed;
  }
  return null;
}

function findDepartureTimeEls(scope = document) {
  // aria-label="Departure time: 19:15."
  return Array.from(
    scope.querySelectorAll('[aria-label^="Departure time:"] , [aria-label*="Departure time"]')
  );
}

function getTimeText(el) {
  return normalizeSpaces(el?.textContent || "");
}

function parseTimeToMinutesSinceMidnight(timeText) {
  let m = timeText.match(/^(\d{1,2}):(\d{2})$/);
  if (m) return parseInt(m[1], 10) * 60 + parseInt(m[2], 10);

  m = timeText.match(/^(\d{1,2}):(\d{2})\s*([AP]M)$/i);
  if (m) {
    let h = parseInt(m[1], 10);
    const mins = parseInt(m[2], 10);
    const ap = m[3].toUpperCase();
    if (ap === "PM" && h !== 12) h += 12;
    if (ap === "AM" && h === 12) h = 0;
    return h * 60 + mins;
  }

  return null;
}

function minutesToHHMM(mins) {
  mins = ((mins % (24 * 60)) + (24 * 60)) % (24 * 60);
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  return `${pad2(h)}:${pad2(m)}`;
}

function detectFlightDateFromPage() {
  // Best-effort. Use settings.dateOverride if this fails.
  const labels = Array.from(document.querySelectorAll("[aria-label]"))
    .map((el) => el.getAttribute("aria-label"))
    .filter(Boolean);

  const interesting = labels.filter((s) =>
    /departure date|depart|outbound|return date|return/i.test(s)
  );

  for (const label of interesting) {
    const part = label.includes(":") ? label.split(":").slice(1).join(":").trim() : label.trim();
    const hasYear = /\b20\d{2}\b/.test(part);
    const assumed = hasYear ? part : `${part} ${new Date().getFullYear()}`;
    const dt = new Date(assumed);
    if (!isNaN(dt.getTime())) {
      const ymd = toYMD(dt);
      log("Detected flight date:", ymd, "from:", label);
      panelWrite(`Detected date ${ymd} from "${label}"`);
      return ymd;
    }
  }

  warn("Could not detect flight date from page. Consider dateOverride.");
  panelWrite("Could not detect flight date; consider dateOverride");
  return null;
}

function isLikelyFlightCardRoot(root) {
  const text = normalizeSpaces(root.innerText || "");
  if (text.length < 40) return false;

  const depEls = findDepartureTimeEls(root);
  if (depEls.length === 0) return false;

  const cf = extractCarrierAndFlightNumberFromDom(root);
  if (!cf) return false;

  const iatas = extractIataCodesFromText(text);
  if (iatas.length < 2) return false;

  return true;
}

function findCardRootFromDepartureEl(depEl) {
  let node = depEl;
  for (let i = 0; i < 12 && node; i++) {
    const el = node instanceof HTMLElement ? node : null;
    if (el && isLikelyFlightCardRoot(el)) return el;
    node = node.parentElement;
  }
  return null;
}

function findOpenCardRoots() {
  const depEls = findDepartureTimeEls(document);
  log("Departure time nodes found:", depEls.length);
  panelWrite(`Departure time nodes found: ${depEls.length}`);

  const roots = [];
  for (const depEl of depEls) {
    const root = findCardRootFromDepartureEl(depEl);
    if (root) roots.push(root);
  }

  const uniq = [...new Set(roots)];
  log("Likely open card roots:", uniq.length);
  panelWrite(`Likely open card roots: ${uniq.length}`);
  return uniq;
}

function buildFlightArgsFromCard(root, flightDate) {
  const text = normalizeSpaces(root.innerText || "");

  const cf = extractCarrierAndFlightNumberFromDom(root);
  log("Carrier/flight parsed:", cf);
  panelWrite(`Carrier/flight parsed: ${cf ? cf.marketing_carrier + " " + cf.marketing_flight_number : "null"}`);

  if (!cf) return { error: "Could not parse carrier/flight number", debugText: text };

  const iatas = extractIataCodesFromText(text);
  if (iatas.length < 2) return { error: "Could not parse origin/destination IATA", debugText: text };

  const origin = iatas[0];
  const destination = iatas[iatas.length - 1];

  if (!flightDate) return { error: "Could not detect flight date" };

  return {
    marketing_carrier: cf.marketing_carrier,
    marketing_flight_number: cf.marketing_flight_number,
    origin,
    destination,
    flight_date: flightDate
  };
}

function makeCacheKey(args) {
  return [
    args.flight_date,
    args.origin,
    args.destination,
    args.marketing_carrier,
    args.marketing_flight_number
  ].join("|");
}

// ===== DOM injection =====

function ensureInjectedLine(depEl) {
  let existing = state.injectedByDepEl.get(depEl);
  if (existing && existing.isConnected) return existing;

  const line = document.createElement("div");
  line.className = "aeolus-prediction aeolus-prediction--loading";
  line.textContent = "Checking…";
  depEl.insertAdjacentElement("afterend", line);

  state.injectedByDepEl.set(depEl, line);
  return line;
}

function setLoading(line) {
  line.classList.add("aeolus-prediction--loading");
  line.classList.remove("aeolus-prediction--error");
  line.classList.remove("aeolus-green", "aeolus-yellow", "aeolus-red");
  line.textContent = "Checking…";
}

function setError(line, message) {
  line.classList.remove("aeolus-prediction--loading");
  line.classList.add("aeolus-prediction--error");
  line.classList.remove("aeolus-green", "aeolus-yellow", "aeolus-red");
  line.textContent = `Unavailable (${message})`;
}

function setResult(line, html, colorClass) {
  line.classList.remove("aeolus-prediction--loading");
  line.classList.remove("aeolus-prediction--error");
  line.classList.remove("aeolus-green", "aeolus-yellow", "aeolus-red");
  if (colorClass) line.classList.add(colorClass);
  line.innerHTML = html;
}

// ===== Display logic =====
// Rules:
// - if delay < 0 => green Accurate
// - else if delay < 5 => green Accurate
// - else if 5..15 => yellow "Slight delay: ~time"
// - else => red "Major delay: ~time"

function computeDisplay(depTimeText, predictionMinutes) {
  const depMins = parseTimeToMinutesSinceMidnight(depTimeText);
  if (depMins == null) return null;

  const delay = Math.round(Number(predictionMinutes));
  if (!isFinite(delay)) return null;

  const predictedTimeStr = minutesToHHMM(depMins + delay);

  if (delay < 0) {
    return { html: "Accurate", colorClass: "aeolus-green" };
  }
  if (delay < 5) {
    return { html: "Accurate", colorClass: "aeolus-green" };
  }
  if (delay <= 15) {
    return {
      html: `Slight delay: <span style="font-weight:700;">~${predictedTimeStr}</span>`,
      colorClass: "aeolus-yellow"
    };
  }
  return {
    html: `Major delay: <span style="font-weight:700;">~${predictedTimeStr}</span>`,
    colorClass: "aeolus-red"
  };
}

// ===== Backend call =====

// Preferred: via background service worker (avoids CORS if host_permissions is set)
function callBackgroundPredict(args) {
  return new Promise((resolve, reject) => {
    if (!chrome?.runtime?.sendMessage) {
      reject(new Error("chrome.runtime.sendMessage not available"));
      return;
    }

    chrome.runtime.sendMessage({ type: "AEOLUS_PREDICT", args }, (resp) => {
      const lastErr = chrome.runtime.lastError;
      if (lastErr) {
        reject(new Error(lastErr.message || "runtime message error"));
        return;
      }
      if (!resp) {
        reject(new Error("No response from background"));
        return;
      }
      if (!resp.ok) {
        reject(new Error(resp.error || "Background fetch failed"));
        return;
      }
      resolve(resp.data);
    });
  });
}

// Fallback: direct fetch (will CORS fail if backend doesn't allow it)
async function directFetchPredict(args) {
  const base = normalizeBaseUrl(state.settings.backendUrl);
  const url = `${base}/api/flight`;

  log("DIRECT POST (may CORS fail):", url, args);
  panelWrite(`DIRECT POST ${url}`);

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args)
  });

  const text = await res.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch {}
  if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`);
  return data;
}

function getHandledKey(depEl) {
  return depEl?.dataset?.aeolusKey || null;
}
function markHandled(depEl, key) {
  if (!depEl || !key) return;
  depEl.dataset.aeolusKey = key;
}

async function processCard(root, flightDate) {
  const depEl = findDepartureTimeEls(root)[0];
  const depTimeText = getTimeText(depEl);

  if (!depEl || !depTimeText) return;

  const line = ensureInjectedLine(depEl);

  const args = buildFlightArgsFromCard(root, flightDate);
  if (args.error) {
    setError(line, args.error);
    return;
  }

  const key = makeCacheKey(args);

  // If this exact element was already associated with this key, just re-render from cache.
  const existingKey = getHandledKey(depEl);
  if (existingKey === key) {
    const cached = state.cache.get(key);
    if (cached?.status === "ok") setResult(line, cached.html, cached.colorClass);
    else if (cached?.status === "pending") setLoading(line);
    return;
  }

  // Mark element immediately so repeated scans don't requeue it.
  markHandled(depEl, key);

  // HARD DE-DUPE: if we've already attempted this key once, do NOT request again.
  if (seenKeys.has(key)) {
    const cached = state.cache.get(key);
    if (cached?.status === "ok") setResult(line, cached.html, cached.colorClass);
    else if (cached?.status === "pending") setLoading(line);
    else setError(line, "already checked");
    return;
  }

  // If we already have cached OK, render it and mark as seen (optional but consistent).
  const cached = state.cache.get(key);
  if (cached?.status === "ok") {
    seenKeys.add(key);
    setResult(line, cached.html, cached.colorClass);
    return;
  }
  if (cached?.status === "pending") {
    setLoading(line);
    return;
  }

  // First time for this key => allow exactly one request.
  seenKeys.add(key);
  state.cache.set(key, { status: "pending", ts: Date.now() });
  setLoading(line);

  try {
    let resp;
    try {
      resp = await callBackgroundPredict(args);
    } catch (bgErr) {
      warn("Background predict failed, trying direct fetch:", bgErr.message);
      resp = await directFetchPredict(args); // may CORS fail
    }

    const predictionMinutes = resp?.prediction;
    const computed = computeDisplay(depTimeText, predictionMinutes);

    if (!computed) {
      state.cache.set(key, { status: "error", ts: Date.now() });
      setError(line, "bad time/prediction");
      return;
    }

    state.cache.set(key, {
      status: "ok",
      predictionMinutes,
      html: computed.html,
      colorClass: computed.colorClass,
      ts: Date.now()
    });

    setResult(line, computed.html, computed.colorClass);
  } catch (e) {
    // IMPORTANT: keep seenKeys so we don't retry/spam on failure.
    state.cache.set(key, { status: "error", ts: Date.now() });
    setError(line, e.message || "request failed");
  }
}

// ===== Main scan loop =====

async function loadSettings() {
  const stored = await chrome.storage.sync.get(STORAGE_DEFAULTS);
  state.settings = { ...STORAGE_DEFAULTS, ...stored };
  ensureStylesInjected();
  panelEnsure();
  log("Settings loaded:", state.settings);
  panelWrite(`Settings: ${JSON.stringify(state.settings)}`);
}

const scan = debounce(async () => {
  if (!state.settings.enabled) return;

  const pageDate = state.settings.dateOverride?.trim() || detectFlightDateFromPage();
  const roots = findOpenCardRoots();

  for (const root of roots) {
    processCard(root, pageDate);
    await sleep(25);
  }
}, 1200); // slower scan to reduce churn

async function init() {
  await loadSettings();

  chrome.storage.onChanged.addListener(async (changes, area) => {
    if (area !== "sync") return;
    await loadSettings();
    scan();
  });

  const observer = new MutationObserver(() => scan());
  observer.observe(document.documentElement, { childList: true, subtree: true });

  scan();
}

init().catch((e) => {
  error("Init failed:", e);
  panelWrite(`Init failed: ${e.message || e}`);
});
