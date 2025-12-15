export interface PredictionResult {
  prediction: number;
  probability: number;
  shap_waterfall: string;
  shap_bar: string;
  lime: string;
}
