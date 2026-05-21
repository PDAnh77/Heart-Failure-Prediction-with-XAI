export interface FSResult {
  fs_dataset_id: string;
  baseline_accuracy: number;
  original_feature_count: number;
  best_ga_accuracy: number;
  selected_features: string[];
  feature_count: number;
  found_at_generation: number;
  balancing?: {
    method: "smote" | "adasyn" | "none" | string;
    before?: Record<string, number>;
    after?: Record<string, number>;
    skipped?: string;
  } | null;
}

export interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
}

export interface EvalResult {
  status: string;
  model_evaluated: string;
  metrics_before: Metrics;
  metrics_after: Metrics;
  confusion_matrix_chart_url: string;
  roc_chart_url: string | null;
}
