import { Clock, AlertTriangle, CheckCircle, Plane } from "lucide-react";

interface FlightResultProps {
  delayMinutes: number;
  scheduledDeparture: string;
  flightInfo: {
    carrier: string;
    flightNumber: string;
    origin: string;
    destination: string;
  };
}

const FlightResult = ({ delayMinutes, scheduledDeparture, flightInfo }: FlightResultProps) => {
  const scheduledTime = new Date(scheduledDeparture);
  const actualTime = new Date(scheduledTime.getTime() + delayMinutes * 60 * 1000);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: true,
    });
  };

  const formatDate = (date: Date) => {
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
    });
  };

  const isDelayed = delayMinutes > 0;
  const isEarly = delayMinutes < 0;

  return (
    <div className="animate-fade-up space-y-6">
      {/* Flight Route Header */}
      <div className="flex items-center justify-center gap-4 py-4">
        <div className="text-center">
          <p className="text-3xl font-bold text-foreground">{flightInfo.origin}</p>
          <p className="text-sm text-muted-foreground">Origin</p>
        </div>
        <div className="flex-1 flex items-center justify-center">
          <div className="w-full h-px bg-border relative">
            <Plane className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-6 h-6 text-primary rotate-90" />
          </div>
        </div>
        <div className="text-center">
          <p className="text-3xl font-bold text-foreground">{flightInfo.destination}</p>
          <p className="text-sm text-muted-foreground">Destination</p>
        </div>
      </div>

      <div className="text-center mb-4">
        <p className="text-muted-foreground text-sm">
          {flightInfo.carrier} {flightInfo.flightNumber}
        </p>
      </div>

      {/* Delay Status Card */}
      <div className={`rounded-2xl p-6 text-center ${
        isDelayed 
          ? 'bg-destructive/10 border border-destructive/20' 
          : isEarly 
            ? 'bg-green-500/10 border border-green-500/20'
            : 'bg-primary/10 border border-primary/20'
      }`}>
        <div className="flex items-center justify-center gap-2 mb-3">
          {isDelayed ? (
            <AlertTriangle className="w-6 h-6 text-destructive" />
          ) : (
            <CheckCircle className="w-6 h-6 text-green-500" />
          )}
          <span className={`text-lg font-medium ${
            isDelayed ? 'text-destructive' : 'text-green-600'
          }`}>
            {isDelayed ? 'Flight Delayed' : isEarly ? 'Arriving Early' : 'On Time'}
          </span>
        </div>
        
        <div className={`text-5xl font-bold mb-2 ${
          isDelayed ? 'text-destructive' : 'text-green-600'
        }`}>
          {isDelayed ? '+' : ''}{delayMinutes} min
        </div>
        
        <p className="text-muted-foreground text-sm">
          {isDelayed 
            ? 'Expected delay from scheduled time' 
            : isEarly 
              ? 'Expected to arrive early'
              : 'Flight is on schedule'}
        </p>
      </div>

      {/* Time Breakdown */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-secondary/50 rounded-xl p-4 text-center">
          <Clock className="w-5 h-5 text-muted-foreground mx-auto mb-2" />
          <p className="text-xs text-muted-foreground uppercase tracking-wide mb-1">Scheduled</p>
          <p className="text-xl font-semibold text-foreground">{formatTime(scheduledTime)}</p>
          <p className="text-xs text-muted-foreground">{formatDate(scheduledTime)}</p>
        </div>
        
        <div className={`rounded-xl p-4 text-center ${
          isDelayed 
            ? 'bg-destructive/10' 
            : 'bg-green-500/10'
        }`}>
          <Plane className={`w-5 h-5 mx-auto mb-2 ${
            isDelayed ? 'text-destructive' : 'text-green-600'
          }`} />
          <p className="text-xs text-muted-foreground uppercase tracking-wide mb-1">Actual</p>
          <p className={`text-xl font-semibold ${
            isDelayed ? 'text-destructive' : 'text-green-600'
          }`}>
            {formatTime(actualTime)}
          </p>
          <p className="text-xs text-muted-foreground">{formatDate(actualTime)}</p>
        </div>
      </div>
    </div>
  );
};

export default FlightResult;
