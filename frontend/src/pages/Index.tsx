import { useState } from "react";
import { Plane } from "lucide-react";
import FlyingPlane from "@/components/FlyingPlane";
import FlightForm, { FlightData } from "@/components/FlightForm";
import FlightResult from "@/components/FlightResult";
import LoadingState from "@/components/LoadingState";

interface FlightResultData {
  delayMinutes: number;
  flightDate: string;
  scheduledDepTime: number;
  arrDelayMinutes: number;
  scheduledArrTime: number;
}

const Index = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<FlightResultData | null>(null);
  const [flightInfo, setFlightInfo] = useState<{
    carrier: string;
    flightNumber: string;
    origin: string;
    destination: string;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (data: FlightData) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    setFlightInfo({
      carrier: data.marketingCarrier.toUpperCase(),
      flightNumber: data.flightNumber,
      origin: data.origin.toUpperCase(),
      destination: data.destination.toUpperCase(),
    });

    try {
      const response = await fetch('http://127.0.0.1:5000/api/flight', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          marketing_carrier: data.marketingCarrier.toUpperCase(),
          marketing_flight_number: data.flightNumber,
          flight_date: data.flightDate,
          origin: data.origin.toUpperCase(),
          destination: data.destination.toUpperCase(),
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to fetch flight data');
      }

      const resultData = await response.json();
      
      if (resultData.prediction === undefined) {
        throw new Error('No delay prediction returned from server');
      }
      
      setResult({
        delayMinutes: Math.round(resultData.prediction),
        flightDate: data.flightDate,
        scheduledDepTime: resultData.CRS_DEP_TIME,
        arrDelayMinutes: Math.round(resultData['arr prediction']),
        scheduledArrTime: resultData.CRS_ARR_TIME,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch flight data');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setFlightInfo(null);
    setError(null);
  };

  return (
    <div className="min-h-screen sky-background relative overflow-hidden">
      <FlyingPlane />
      
      <div className="relative z-10 min-h-screen flex flex-col items-center justify-center px-4 py-12">
        {/* Header */}
        <div className="text-center mb-10 animate-fade-up">
          <div className="inline-flex items-center justify-center gap-3 mb-4">
            <Plane className="w-10 h-10 text-primary animate-float" />
            <h1 className="text-4xl md:text-5xl font-bold title-gradient pb-1">
              Real Flight Time
            </h1>
          </div>
          <p className="text-muted-foreground text-lg max-w-md mx-auto">
            Get real flight delay predictions and actual departure times
          </p>
        </div>

        {/* Main Card */}
        <div className="w-full max-w-lg glass-card rounded-3xl p-8 animate-scale-in" style={{ animationDelay: '0.2s' }}>
          <FlightForm onSubmit={handleSubmit} isLoading={isLoading} />

          {isLoading && (
            <div className="mt-8 pt-8 border-t border-border/50">
              <LoadingState />
            </div>
          )}

          {result && flightInfo && !isLoading && (
            <div className="mt-8 pt-8 border-t border-border/50">
              <FlightResult
                delayMinutes={result.delayMinutes}
                flightDate={result.flightDate}
                flightInfo={flightInfo}
                scheduledDepTime={result.scheduledDepTime}
                arrDelayMinutes={result.arrDelayMinutes}
                scheduledArrTime={result.scheduledArrTime}
              />
            </div>
          )}

          {error && (
            <div className="text-center py-8">
              <p className="text-destructive mb-4">{error}</p>
              <button
                onClick={handleReset}
                className="text-primary hover:text-primary/80 font-medium transition-colors"
              >
                Try Again
              </button>
            </div>
          )}
        </div>

      </div>
    </div>
  );
};

export default Index;
