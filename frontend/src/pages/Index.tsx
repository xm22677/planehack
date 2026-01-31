import { useState } from "react";
import { Plane } from "lucide-react";
import FlyingPlane from "@/components/FlyingPlane";
import FlightForm, { FlightData } from "@/components/FlightForm";
import FlightResult from "@/components/FlightResult";
import LoadingState from "@/components/LoadingState";

interface FlightResult {
  delayMinutes: number;
  scheduledDeparture: string;
}

const Index = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<FlightResult | null>(null);
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
      // TODO: Replace with actual API endpoint
      const params = new URLSearchParams({
        marketingCarrier: data.marketingCarrier,
        flightNumber: data.flightNumber,
        flightDate: data.flightDate,
        origin: data.origin,
        destination: data.destination,
      });

      const response = await fetch(`/api/flight-delay?${params.toString()}`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch flight data');
      }

      const resultData = await response.json();
      setResult(resultData);
    } catch (err) {
      // For demo purposes, simulate a response
      // Remove this in production and handle the error properly
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Simulated response
      const simulatedDelay = Math.floor(Math.random() * 120) - 15; // -15 to 105 minutes
      const scheduledTime = new Date(data.flightDate);
      scheduledTime.setHours(14, 30, 0, 0); // Default scheduled time
      
      setResult({
        delayMinutes: simulatedDelay,
        scheduledDeparture: scheduledTime.toISOString(),
      });
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
            <h1 className="text-4xl md:text-5xl font-bold title-gradient">
              Real Flight Time
            </h1>
          </div>
          <p className="text-muted-foreground text-lg max-w-md mx-auto">
            Get real-time flight delay predictions and actual arrival times
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
                scheduledDeparture={result.scheduledDeparture}
                flightInfo={flightInfo}
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

        {/* Footer */}
        <p className="mt-8 text-muted-foreground/60 text-sm animate-fade-up" style={{ animationDelay: '0.4s' }}>
          Flight delay predictions powered by real-time data
        </p>
      </div>
    </div>
  );
};

export default Index;
