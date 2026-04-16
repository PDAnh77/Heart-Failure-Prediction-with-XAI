export interface DatasetSummary {
    rows: number;
    columns: number;
    target_column: string;
    categorical_features: string[];
    numerical_features: string[];
    column_types: Record<string, string>;
    missing_values: Record<string, number>;
}