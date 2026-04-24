export interface FSResult {
    fs_dataset_id: string;
    baseline_accuracy: number;
    original_feature_count: number;
    best_ga_accuracy: number;
    selected_features: string[];
    feature_count: number;
    found_at_generation: number;
}
