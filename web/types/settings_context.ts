export interface SettingsContextType {
  snowMode: boolean;
  savePrediction: boolean;
  language: "vi" | "en";
  setSnowMode: (val: boolean) => void;
  setSavePrediction: (val: boolean) => void;
  setLanguage: (val: "vi" | "en") => void;
  isReady: boolean;
}
