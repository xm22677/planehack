import { Plane } from "lucide-react";

const FlyingPlane = () => {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
      {/* Clouds */}
      <div className="absolute top-20 left-[10%] w-32 h-16 bg-white/40 rounded-full blur-xl animate-float" style={{ animationDelay: '0s' }} />
      <div className="absolute top-40 left-[30%] w-48 h-20 bg-white/30 rounded-full blur-2xl animate-float" style={{ animationDelay: '2s' }} />
      <div className="absolute top-16 right-[20%] w-40 h-16 bg-white/35 rounded-full blur-xl animate-float" style={{ animationDelay: '1s' }} />
      <div className="absolute top-60 right-[10%] w-56 h-24 bg-white/25 rounded-full blur-2xl animate-float" style={{ animationDelay: '3s' }} />
      <div className="absolute bottom-40 left-[5%] w-44 h-18 bg-white/30 rounded-full blur-xl animate-float" style={{ animationDelay: '4s' }} />
      <div className="absolute bottom-60 right-[25%] w-36 h-14 bg-white/35 rounded-full blur-xl animate-float" style={{ animationDelay: '2.5s' }} />
      
      {/* Flying planes */}
      <div className="absolute top-[15%] animate-fly" style={{ animationDelay: '0s' }}>
        <Plane className="w-12 h-12 text-primary/60 rotate-45" />
      </div>
      <div className="absolute top-[45%] animate-fly" style={{ animationDelay: '7s', animationDuration: '25s' }}>
        <Plane className="w-8 h-8 text-primary/40 rotate-45" />
      </div>
      <div className="absolute top-[70%] animate-fly" style={{ animationDelay: '14s', animationDuration: '22s' }}>
        <Plane className="w-10 h-10 text-primary/50 rotate-45" />
      </div>
    </div>
  );
};

export default FlyingPlane;
