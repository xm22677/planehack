import { AlertTriangle, CheckCircle, Plane, Calendar, Clock } from "lucide-react";

interface FlightResultProps {
  delayMinutes: number;
  flightDate: string;
  flightInfo: {
    carrier: string;
    flightNumber: string;
    origin: string;
    destination: string;
  };
  scheduledDepTime: number;
}

const FlightResult = ({ delayMinutes, flightDate, flightInfo, scheduledDepTime }: FlightResultProps) => {
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  const formatTime = (time: number) => {
    const hours = Math.floor(time / 100);
    const minutes = time % 100;
    const period = hours >= 12 ? 'PM' : 'AM';
    const displayHours = hours % 12 || 12;
    return `${displayHours}:${minutes.toString().padStart(2, '0')} ${period}`;
  };

  const calculateActualTime = (scheduled: number, delayMins: number) => {
    const hours = Math.floor(scheduled / 100);
    const minutes = scheduled % 100;
    const totalMinutes = hours * 60 + minutes + delayMins;
    const newHours = Math.floor(totalMinutes / 60) % 24;
    const newMinutes = totalMinutes % 60;
    return newHours * 100 + newMinutes;
  };

  const getDelayCategory = (minutes: number) => {
    if (minutes < 10) {
      return { label: "Departing On Time", color: "text-green-600", bgColor: "bg-green-500/10", borderColor: "border-green-500/20" };
    } else if (minutes >= 10 && minutes < 20) {
      return { label: "Departing A Bit Late", color: "text-yellow-600", bgColor: "bg-yellow-500/10", borderColor: "border-yellow-500/20" };
    } else if (minutes >= 20 && minutes < 60) {
      return { label: "Departing Late", color: "text-red-500", bgColor: "bg-red-500/10", borderColor: "border-red-500/20" };
    } else {
      return { label: "Departing Very Late", color: "text-red-700", bgColor: "bg-red-600/20", borderColor: "border-red-600/30" };
    }
  };

  const isEarly = delayMinutes < 0;
  const category = getDelayCategory(delayMinutes);
  const actualTime = calculateActualTime(scheduledDepTime, delayMinutes);

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
      <div className={`rounded-2xl p-8 text-center ${
        isEarly 
          ? 'bg-green-500/10 border border-green-500/20'
          : `${category.bgColor} border ${category.borderColor}`
      }`}>
        <div className="flex items-center justify-center gap-3 mb-4">
          {isEarly || delayMinutes < 10 ? (
            <CheckCircle className="w-8 h-8 text-green-500" />
          ) : (
            <AlertTriangle className={`w-8 h-8 ${category.color}`} />
          )}
        </div>
        
        <div className={`text-3xl font-bold ${
          isEarly ? 'text-green-600' : category.color
        }`}>
          {isEarly ? 'Departing On Time' : category.label}
        </div>
      </div>

      {/* Time Info */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-secondary/50 rounded-xl p-4 text-center">
          <Clock className="w-5 h-5 text-muted-foreground mx-auto mb-2" />
          <p className="text-xs text-muted-foreground uppercase tracking-wide mb-1">Scheduled</p>
          <p className="text-lg font-semibold text-foreground">{formatTime(scheduledDepTime)}</p>
        </div>
        <div className={`rounded-xl p-4 text-center ${
          isEarly 
            ? 'bg-green-500/10 border border-green-500/20'
            : `${category.bgColor} border ${category.borderColor}`
        }`}>
          <Clock className={`w-5 h-5 mx-auto mb-2 ${isEarly ? 'text-green-600' : category.color}`} />
          <p className="text-xs text-muted-foreground uppercase tracking-wide mb-1">Actual</p>
          <p className={`text-lg font-semibold ${isEarly ? 'text-green-600' : category.color}`}>~{formatTime(actualTime)}</p>
        </div>
      </div>

      {/* Flight Date */}
      <div className="bg-secondary/50 rounded-xl p-4 text-center">
        <Calendar className="w-5 h-5 text-muted-foreground mx-auto mb-2" />
        <p className="text-xs text-muted-foreground uppercase tracking-wide mb-1">Flight Date</p>
        <p className="text-lg font-semibold text-foreground">{formatDate(flightDate)}</p>
      </div>
    </div>
  );
};

export default FlightResult;
