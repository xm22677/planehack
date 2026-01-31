# Flight Delay Predictor Chrome Extension

A Chrome extension that predicts flight delays on Google Flights by scraping flight details and querying a prediction API.

## Features

- Automatically detects when you expand a flight card on Google Flights
- Extracts: carrier code, flight number, date, origin, destination
- Calls your prediction API with these parameters
- Displays predicted departure time in green below the scheduled time

## Installation

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select the `chrome-extension` folder from this project

## Configuration

Edit `content.js` and update the `API_ENDPOINT` constant:

```javascript
const API_ENDPOINT = 'https://your-api-endpoint.com/predict-delay';
```

### Expected API Format

**Request:**
```
GET /predict-delay?carrier=B6&flightNumber=1723&flightDate=2026-02-15&origin=JFK&destination=LAX
```

**Response:**
```json
{
  "delayMinutes": 30
}
```

- Positive values = delay
- Zero = on time
- Negative values = early departure

## How It Works

1. Extension monitors Google Flights pages
2. When a flight card is expanded (shows detailed info), it:
   - Scrapes carrier code (e.g., "B6")
   - Scrapes flight number (e.g., "1723")
   - Extracts airport codes (origin/destination)
   - Parses the flight date
3. Calls your API with these parameters
4. Injects predicted time below the departure time

## Files

- `manifest.json` - Extension configuration
- `content.js` - Main logic (scraping + API calls + DOM injection)
- `styles.css` - Styling for injected elements
- `icons/` - Extension icons (you'll need to add these)

## Adding Icons

Create a folder called `icons` and add:
- `icon16.png` (16x16)
- `icon48.png` (48x48)
- `icon128.png` (128x128)

## Troubleshooting

Open Chrome DevTools on a Google Flights page and check the console for messages starting with `[Flight Delay Predictor]`.
