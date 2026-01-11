export interface SettingsContextType {
    snowMode: boolean;
    savePrediction: boolean;
    setSnowMode: (val: boolean) => void;
    setSavePrediction: (val: boolean) => void;
    isReady: boolean;
}
