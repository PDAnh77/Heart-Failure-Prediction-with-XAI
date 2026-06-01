export interface PredictionResult {
  prediction: number;
  probability: number;
  shap_waterfall: string;
  shap_bar: string;
  lime: string;
  prediction_history: PredictionHistoryBase;
}

export interface PredictionHistoryBase {
  id: string;
  created_at: string;
}

export interface PredictionHistoryDetail {
  input_data: {
    age: number;
    sex: string;
    max_hr: number;
    oldpeak: number;
    st_slope: string;
    fasting_bs: number;
    resting_bp: number;
    cholesterol: number;
    resting_ecg: string;
    chest_pain_type: string;
    exercise_angina: string;
  };
  predicted_label: number;
  predicted_probability: number;
  prediction_xai: {
    lime: string;
    shap_bar: string;
    shap_waterfall: string;
  };
  created_at: string;
}

export interface UnifiedHistoryItem {
  id: string;
  type: "single" | "batch";
  created_at: string;
}
