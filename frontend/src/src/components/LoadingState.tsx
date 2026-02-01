import { Plane } from "lucide-react";

const LoadingState = () => {
  return (
    <div className="animate-scale-in flex flex-col items-center justify-center py-12 space-y-6">
      <div className="relative">
        <div className="w-24 h-24 rounded-full border-4 border-primary/20 border-t-primary animate-spin" />
        <Plane className="absolute inset-0 m-auto w-10 h-10 text-primary animate-pulse-slow" />
      </div>
      
      <div className="text-center space-y-2">
        <p className="text-lg font-medium text-foreground">Checking flight status...</p>
        <p className="text-sm text-muted-foreground">Fetching real-time delay information</p>
      </div>

      <div className="flex gap-1.5">
        <div className="w-2.5 h-2.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: '0ms' }} />
        <div className="w-2.5 h-2.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: '150ms' }} />
        <div className="w-2.5 h-2.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: '300ms' }} />
      </div>
    </div>
  );
};

export default LoadingState;
