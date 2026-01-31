// Flight Delay Predictor - Content Script
// Runs on Google Flights pages to predict delays

(function() {
  'use strict';

  // Configuration
  const API_ENDPOINT = 'https://your-api-endpoint.com/predict-delay'; // Replace with your actual API
  const POLL_INTERVAL = 2000; // Check for new flight cards every 2 seconds
  const PROCESSED_ATTR = 'data-delay-processed';

  /**
   * Parse time string (e.g., "20:29") to minutes since midnight
   */
  function parseTime(timeStr) {
    const match = timeStr.match(/(\d{1,2}):(\d{2})/);
    if (!match) return null;
    return parseInt(match[1]) * 60 + parseInt(match[2]);
  }

  /**
   * Format minutes since midnight back to time string
   */
  function formatTime(minutes) {
    const hours = Math.floor(minutes / 60) % 24;
    const mins = minutes % 60;
    return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}`;
  }

  /**
   * Extract flight data from an expanded flight card
   */
  function extractFlightData(flightCard) {
    function normalizeWhitespace(str) {
      return (str || '')
        // Google UIs frequently use NBSP / narrow NBSP in separate spans
        .replace(/[\u00A0\u2007\u202F]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
    }

    const data = {
      carrier: null,
      flightNumber: null,
      flightDate: null,
      origin: null,
      destination: null,
      departureTime: null,
      departureTimeElement: null
    };

    try {
      const cardText = normalizeWhitespace(flightCard.textContent || '');

      // Strategy 1: Look for individual elements containing just "B6 224" or "B6&nbsp;224"
      // This is the most reliable - Google often puts carrier+number in its own span
      const allElements = flightCard.querySelectorAll('span, div');
      for (const el of allElements) {
        const text = normalizeWhitespace(el.textContent || '');
        // Match elements that contain primarily just a carrier code + flight number
        // Pattern: starts with 2-char code, followed by 1-4 digit number
        const directMatch = text.match(/^([A-Z][A-Z0-9])\s*(\d{1,4})$/);
        if (directMatch) {
          data.carrier = directMatch[1];
          data.flightNumber = directMatch[2];
          break;
        }
      }

      // Strategy 2: Look for "· B6 1723" pattern in longer text
      if (!data.carrier) {
        for (const el of allElements) {
          const text = normalizeWhitespace(el.textContent || '');
          const flightMatch = text.match(/·\s*([A-Z][A-Z0-9])\s*(\d{1,4})\b/);
          if (flightMatch) {
            data.carrier = flightMatch[1];
            data.flightNumber = flightMatch[2];
            break;
          }
        }
      }

      // Strategy 3: Look for "AirlineName XX 123" pattern (e.g., "JetBlue B6 224")
      if (!data.carrier) {
        const airlineMatch = cardText.match(
          /\b(?:JetBlue|American|United|Delta|Southwest|Spirit|Alaska|Frontier|Hawaiian|Sun Country|Allegiant|Breeze|Avelo)\s+([A-Z][A-Z0-9])\s*(\d{1,4})\b/i
        );
        if (airlineMatch) {
          data.carrier = airlineMatch[1].toUpperCase();
          data.flightNumber = airlineMatch[2];
        }
      }

      // Strategy 4: Fallback - find any "XX 1234" pattern in the card
      if (!data.carrier) {
        const fallbackMatch = cardText.match(/\b([A-Z][A-Z0-9])\s+(\d{1,4})\b/);
        if (fallbackMatch) {
          data.carrier = fallbackMatch[1];
          data.flightNumber = fallbackMatch[2];
        }
      }

      // Extract origin and destination from airport codes
      // Look for patterns like "John F. Kennedy International Airport (JFK)"
      const airportMatches = [];
      const airportRegex = /\(([A-Z]{3})\)/g;
      let match;
      while ((match = airportRegex.exec(cardText)) !== null) {
        airportMatches.push(match[1]);
      }
      if (airportMatches.length >= 2) {
        data.origin = airportMatches[0];
        data.destination = airportMatches[1];
      }

      // Extract departure time - look for time pattern near the top
      const timeElements = flightCard.querySelectorAll('div, span');
      for (const el of timeElements) {
        const text = (el.textContent || '').trim();
        // Match time format like "20:29" or "8:30"
        if (/^\d{1,2}:\d{2}$/.test(text)) {
          const parent = el.parentElement;
          // Make sure this isn't already processed and looks like a departure time
          if (!el.querySelector('.flight-delay-prediction') && 
              !el.getAttribute(PROCESSED_ATTR)) {
            data.departureTime = text;
            data.departureTimeElement = el;
            break;
          }
        }
      }

      // Extract flight date from the card header (e.g., "Return · Sun 15 Feb")
      const datePatterns = [
        /(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/i,
        /(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})/i
      ];
      
      for (const pattern of datePatterns) {
        const dateMatch = cardText.match(pattern);
        if (dateMatch) {
          const months = {
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
            'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
            'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
          };
          
          let day, monthStr;
          if (/^\d/.test(dateMatch[1])) {
            day = dateMatch[1].padStart(2, '0');
            monthStr = dateMatch[2].toLowerCase();
          } else {
            monthStr = dateMatch[1].toLowerCase();
            day = dateMatch[2].padStart(2, '0');
          }
          
          const month = months[monthStr];
          // Assume current or next year
          const year = new Date().getFullYear();
          data.flightDate = `${year}-${month}-${day}`;
          break;
        }
      }

      // Also try to get date from URL
      if (!data.flightDate) {
        const urlMatch = window.location.href.match(/(\d{4}-\d{2}-\d{2})/);
        if (urlMatch) {
          data.flightDate = urlMatch[1];
        }
      }

    } catch (error) {
      console.error('[Flight Delay Predictor] Error extracting flight data:', error);
    }

    return data;
  }

  /**
   * Call the prediction API
   */
  async function predictDelay(flightData) {
    try {
      const params = new URLSearchParams({
        carrier: flightData.carrier,
        flightNumber: flightData.flightNumber,
        flightDate: flightData.flightDate,
        origin: flightData.origin,
        destination: flightData.destination
      });

      const response = await fetch(`${API_ENDPOINT}?${params}`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`API returned ${response.status}`);
      }

      const data = await response.json();
      // Expected response: { delayMinutes: number }
      return data.delayMinutes || 0;
    } catch (error) {
      console.error('[Flight Delay Predictor] API error:', error);
      throw error;
    }
  }

  /**
   * Inject the predicted delay into the DOM
   */
  function injectPrediction(element, departureTime, delayMinutes) {
    // Remove any existing prediction
    const existing = element.parentElement?.querySelector('.flight-delay-prediction');
    if (existing) {
      existing.remove();
    }

    // Create prediction element
    const prediction = document.createElement('div');
    prediction.className = 'flight-delay-prediction';

    if (delayMinutes === 0) {
      prediction.textContent = 'Predicted: On time';
    } else if (delayMinutes > 0) {
      const depMinutes = parseTime(departureTime);
      if (depMinutes !== null) {
        const predictedTime = formatTime(depMinutes + delayMinutes);
        prediction.textContent = `Predicted: ${predictedTime} (+${delayMinutes} min)`;
      } else {
        prediction.textContent = `Predicted delay: +${delayMinutes} min`;
      }
    } else {
      // Early arrival
      const depMinutes = parseTime(departureTime);
      if (depMinutes !== null) {
        const predictedTime = formatTime(depMinutes + delayMinutes);
        prediction.textContent = `Predicted: ${predictedTime} (${delayMinutes} min early)`;
      }
    }

    // Insert after the departure time element
    element.parentElement?.insertBefore(prediction, element.nextSibling);
  }

  /**
   * Show loading state
   */
  function showLoading(element) {
    const existing = element.parentElement?.querySelector('.flight-delay-prediction');
    if (existing) {
      existing.remove();
    }

    const loading = document.createElement('div');
    loading.className = 'flight-delay-prediction loading';
    loading.textContent = 'Predicting delay...';
    element.parentElement?.insertBefore(loading, element.nextSibling);
  }

  /**
   * Show error state
   */
  function showError(element) {
    const existing = element.parentElement?.querySelector('.flight-delay-prediction');
    if (existing) {
      existing.remove();
    }

    const error = document.createElement('div');
    error.className = 'flight-delay-prediction error';
    // Just show the red dot (from ::before pseudo-element), no text
    element.parentElement?.insertBefore(error, element.nextSibling);
  }

  /**
   * Process a flight card
   */
  async function processFlightCard(card) {
    if (card.getAttribute(PROCESSED_ATTR)) {
      return;
    }

    const flightData = extractFlightData(card);

    // Validate we have all required data
    if (!flightData.carrier || !flightData.flightNumber || 
        !flightData.origin || !flightData.destination ||
        !flightData.departureTimeElement) {
      console.log('[Flight Delay Predictor] Incomplete flight data:', flightData);
      return;
    }

    card.setAttribute(PROCESSED_ATTR, 'true');

    console.log('[Flight Delay Predictor] Processing flight:', flightData);

    // Show loading state
    showLoading(flightData.departureTimeElement);

    try {
      const delayMinutes = await predictDelay(flightData);
      injectPrediction(flightData.departureTimeElement, flightData.departureTime, delayMinutes);
    } catch (error) {
      showError(flightData.departureTimeElement);
    }
  }

  /**
   * Find and process all expanded flight cards
   */
  function scanForFlightCards() {
    console.log('[Flight Delay Predictor] Scanning for flight cards...');
    
    // Strategy 1: Look for sections containing "Departing flight" or "Return flight"
    const allElements = document.querySelectorAll('*');
    const flightSections = [];
    
    allElements.forEach(el => {
      const text = el.textContent || '';
      if ((text.includes('Departing flight') || text.includes('Return flight') || text.includes('Selected flights')) &&
          /\([A-Z]{3}\)/.test(text) &&
          /\d{1,2}:\d{2}/.test(text)) {
        // Find the closest container that wraps just this flight
        let container = el;
        while (container.parentElement && 
               container.parentElement.textContent === container.textContent) {
          container = container.parentElement;
        }
        if (!flightSections.includes(container)) {
          flightSections.push(container);
        }
      }
    });

    // Strategy 2: Look for cards with flight info pattern (carrier code + number)
    document.querySelectorAll('div, section').forEach(card => {
      const text = card.textContent || '';
      // Must have airport codes, times, and flight number pattern
      const hasAirportCode = /\([A-Z]{3}\)/.test(text);
      const hasFlightNumber = /·\s*[A-Z0-9]{2}\s+\d{1,4}/.test(text);
      const hasTime = /\d{1,2}:\d{2}/.test(text);
      const hasCarrier = /JetBlue|American|United|Delta|Southwest|Spirit|Alaska|Frontier/.test(text);
      
      if (hasAirportCode && hasTime && (hasFlightNumber || hasCarrier)) {
        // Check this isn't too large (avoid processing the whole page)
        if (text.length < 2000 && !flightSections.includes(card)) {
          flightSections.push(card);
        }
      }
    });

    console.log('[Flight Delay Predictor] Found potential cards:', flightSections.length);

    flightSections.forEach(card => {
      if (!card.getAttribute(PROCESSED_ATTR)) {
        processFlightCard(card);
      }
    });
  }

  /**
   * Initialize the extension
   */
  function init() {
    console.log('[Flight Delay Predictor] Initializing...');

    // Initial scan
    scanForFlightCards();

    // Watch for DOM changes (new flights being expanded)
    const observer = new MutationObserver((mutations) => {
      let shouldScan = false;
      for (const mutation of mutations) {
        if (mutation.addedNodes.length > 0 || 
            mutation.type === 'attributes') {
          shouldScan = true;
          break;
        }
      }
      if (shouldScan) {
        // Debounce the scan
        clearTimeout(window._flightDelayDebounce);
        window._flightDelayDebounce = setTimeout(scanForFlightCards, 500);
      }
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['aria-expanded', 'class']
    });

    // Also poll periodically as a fallback
    setInterval(scanForFlightCards, POLL_INTERVAL);
  }

  // Start when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
