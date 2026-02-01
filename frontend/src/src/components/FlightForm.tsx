import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Plane, Calendar, MapPin, Hash, Building2 } from "lucide-react";

interface FlightFormProps {
  onSubmit: (data: FlightData) => void;
  isLoading: boolean;
}

export interface FlightData {
  marketingCarrier: string;
  flightNumber: string;
  flightDate: string;
  origin: string;
  destination: string;
}

const FlightForm = ({ onSubmit, isLoading }: FlightFormProps) => {
  const [formData, setFormData] = useState<FlightData>({
    marketingCarrier: "",
    flightNumber: "",
    flightDate: "",
    origin: "",
    destination: "",
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="space-y-2">
          <Label htmlFor="marketingCarrier" className="flex items-center gap-2 text-foreground/80">
            <Building2 className="w-4 h-4 text-primary" />
            Marketing Carrier
          </Label>
          <Input
            id="marketingCarrier"
            name="marketingCarrier"
            placeholder="e.g., AA, UA, DL"
            value={formData.marketingCarrier}
            onChange={handleChange}
            required
            className="h-12 bg-secondary/50 border-border/50 focus:border-primary focus:ring-primary/20 transition-all duration-300 uppercase"
            maxLength={3}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="flightNumber" className="flex items-center gap-2 text-foreground/80">
            <Hash className="w-4 h-4 text-primary" />
            Flight Number
          </Label>
          <Input
            id="flightNumber"
            name="flightNumber"
            placeholder="e.g., 1234"
            value={formData.flightNumber}
            onChange={handleChange}
            required
            className="h-12 bg-secondary/50 border-border/50 focus:border-primary focus:ring-primary/20 transition-all duration-300"
            maxLength={5}
          />
        </div>
      </div>

      <div className="space-y-2">
        <Label htmlFor="flightDate" className="flex items-center gap-2 text-foreground/80">
          <Calendar className="w-4 h-4 text-primary" />
          Flight Date
        </Label>
        <Input
          id="flightDate"
          name="flightDate"
          type="date"
          value={formData.flightDate}
          onChange={handleChange}
          required
          className="h-12 bg-secondary/50 border-border/50 focus:border-primary focus:ring-primary/20 transition-all duration-300"
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="space-y-2">
          <Label htmlFor="origin" className="flex items-center gap-2 text-foreground/80">
            <MapPin className="w-4 h-4 text-primary" />
            Origin
          </Label>
          <Input
            id="origin"
            name="origin"
            placeholder="e.g., JFK, LAX"
            value={formData.origin}
            onChange={handleChange}
            required
            className="h-12 bg-secondary/50 border-border/50 focus:border-primary focus:ring-primary/20 transition-all duration-300 uppercase"
            maxLength={3}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="destination" className="flex items-center gap-2 text-foreground/80">
            <MapPin className="w-4 h-4 text-accent" />
            Destination
          </Label>
          <Input
            id="destination"
            name="destination"
            placeholder="e.g., SFO, ORD"
            value={formData.destination}
            onChange={handleChange}
            required
            className="h-12 bg-secondary/50 border-border/50 focus:border-primary focus:ring-primary/20 transition-all duration-300 uppercase"
            maxLength={3}
          />
        </div>
      </div>

      <Button
        type="submit"
        disabled={isLoading}
        className="w-full h-14 text-lg font-semibold bg-primary hover:bg-primary/90 text-primary-foreground shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-0.5"
      >
        {isLoading ? (
          <div className="flex items-center gap-3">
            <Plane className="w-5 h-5 animate-spin-slow" />
            <span>Checking Flight...</span>
          </div>
        ) : (
          <div className="flex items-center gap-3">
            <Plane className="w-5 h-5" />
            <span>Check Flight Delay</span>
          </div>
        )}
      </Button>
    </form>
  );
};

export default FlightForm;
