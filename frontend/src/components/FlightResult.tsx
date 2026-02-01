import { AlertTriangle, CheckCircle, Plane, Calendar, Clock, PlaneLanding, PlaneTakeoff } from "lucide-react";

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
  arrDelayMinutes: number;
  scheduledArrTime: number;
}

const FlightResult = ({ 
  delayMinutes, 
  flightDate, 
  flightInfo, 
  scheduledDepTime,
  arrDelayMinutes,
  scheduledArrTime 
}: FlightResultProps) => {
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

  const getDepDelayCategory = (minutes: number) => {
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

  const getArrDelayCategory = (minutes: number) => {
    if (minutes < 20) {
      return { label: "Arriving On Time", color: "text-green-600", bgColor: "bg-green-500/10", borderColor: "border-green-500/20" };
    } else if (minutes >= 20 && minutes < 40) {
      return { label: "Arriving A Bit Late", color: "text-yellow-600", bgColor: "bg-yellow-500/10", borderColor: "border-yellow-500/20" };
    } else if (minutes >= 40 && minutes < 120) {
      return { label: "Arriving Late", color: "text-red-500", bgColor: "bg-red-500/10", borderColor: "border-red-500/20" };
    } else {
      return { label: "Arriving Very Late", color: "text-red-700", bgColor: "bg-red-600/20", borderColor: "border-red-600/30" };
    }
  };

  const depIsEarly = delayMinutes < 0;
  const arrIsEarly = arrDelayMinutes < 0;
  const depCategory = getDepDelayCategory(delayMinutes);
  const arrCategory = getArrDelayCategory(arrDelayMinutes);
  const actualDepTime = calculateActualTime(scheduledDepTime, delayMinutes);
  const actualArrTime = calculateActualTime(scheduledArrTime, arrDelayMinutes);

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

      {/* Two boxes side by side: Departure and Arrival */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-stretch">
        {/* Departure Box */}
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-center gap-2 text-muted-foreground h-6">
            <PlaneTakeoff className="w-5 h-5" />
            <span className="text-sm font-medium uppercase tracking-wide">Departure</span>
          </div>
          
          {/* Departure Status Card */}
          <div className={`rounded-2xl p-6 text-center flex-1 flex flex-col items-center justify-center ${
            depIsEarly 
              ? 'bg-green-500/10 border border-green-500/20'
              : `${depCategory.bgColor} border ${depCategory.borderColor}`
          }`}>
            <div className="flex items-center justify-center gap-2 mb-3">
              {depIsEarly || delayMinutes < 10 ? (
                <CheckCircle className="w-6 h-6 text-green-500" />
              ) : (
                <AlertTriangle className={`w-6 h-6 ${depCategory.color}`} />
              )}
            </div>
            
            <div className={`text-xl font-bold ${
              depIsEarly ? 'text-green-600' : depCategory.color
            }`}>
              {depIsEarly ? 'Departing On Time' : depCategory.label}
            </div>
          </div>

          {/* Departure Scheduled Time */}
          <div className="bg-white rounded-xl p-4 text-center border border-border/30 h-[76px] flex flex-col items-center justify-center">
            <Clock className="w-4 h-4 text-muted-foreground mb-1" />
            <p className="text-xs text-muted-foreground uppercase tracking-wide mb-1">Scheduled</p>
            <p className="text-base font-semibold text-foreground">{formatTime(scheduledDepTime)}</p>
          </div>

          {/* Departure Actual Time */}
          <div className="bg-white rounded-xl p-4 text-center border border-border/30 h-[76px] flex flex-col items-center justify-center">
            <Clock className={`w-4 h-4 mb-1 ${depIsEarly ? 'text-green-600' : depCategory.color}`} />
            <p className="text-xs text-muted-foreground uppercase tracking-wide mb-1">Actual</p>
            <p className={`text-base font-semibold ${depIsEarly ? 'text-green-600' : depCategory.color}`}>~{formatTime(actualDepTime)}</p>
          </div>
        </div>

        {/* Arrival Box */}
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-center gap-2 text-muted-foreground h-6">
            <PlaneLanding className="w-5 h-5" />
            <span className="text-sm font-medium uppercase tracking-wide">Arrival</span>
          </div>
          
          {/* Arrival Status Card */}
          <div className={`rounded-2xl p-6 text-center flex-1 flex flex-col items-center justify-center ${
            arrIsEarly 
              ? 'bg-green-500/10 border border-green-500/20'
              : `${arrCategory.bgColor} border ${arrCategory.borderColor}`
          }`}>
            <div className="flex items-center justify-center gap-2 mb-3">
              {arrIsEarly || arrDelayMinutes < 20 ? (
                <CheckCircle className="w-6 h-6 text-green-500" />
              ) : (
                <AlertTriangle className={`w-6 h-6 ${arrCategory.color}`} />
              )}
            </div>
            
            <div className={`text-xl font-bold ${
              arrIsEarly ? 'text-green-600' : arrCategory.color
            }`}>
              {arrIsEarly ? 'Arriving On Time' : arrCategory.label}
            </div>
          </div>

          {/* Arrival Scheduled Time */}
          <div className="bg-white rounded-xl p-4 text-center border border-border/30 h-[76px] flex flex-col items-center justify-center">
            <Clock className="w-4 h-4 text-muted-foreground mb-1" />
            <p className="text-xs text-muted-foreground uppercase tracking-wide mb-1">Scheduled</p>
            <p className="text-base font-semibold text-foreground">{formatTime(scheduledArrTime)}</p>
          </div>

          {/* Arrival Actual Time */}
          <div className="bg-white rounded-xl p-4 text-center border border-border/30 h-[76px] flex flex-col items-center justify-center">
            <Clock className={`w-4 h-4 mb-1 ${arrIsEarly ? 'text-green-600' : arrCategory.color}`} />
            <p className="text-xs text-muted-foreground uppercase tracking-wide mb-1">Actual</p>
            <p className={`text-base font-semibold ${arrIsEarly ? 'text-green-600' : arrCategory.color}`}>~{formatTime(actualArrTime)}</p>
          </div>
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
