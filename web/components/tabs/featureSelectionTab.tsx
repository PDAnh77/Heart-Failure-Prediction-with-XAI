"use client";
import { useEffect, useState, useRef, useCallback } from "react";
import Image from "next/image";
import { FiTrendingUp, FiInfo, FiDownload, FiPlay, FiBarChart2 } from "react-icons/fi";
import { FaStar, FaCheckCircle, FaBalanceScale } from "react-icons/fa";
import { FaFilter } from "react-icons/fa6";
import { IoSettingsSharp } from "react-icons/io5";
import { api } from "@/lib/api";
import toast from "react-hot-toast";
import { EvalResult, FSResult } from "@/types/feature_selection";
import ImageModal from "../modals/imageModal";

interface FeatureSelectionTabProps {
    targetColumn: string;
    processedId: string | null;
    size: number;
    mutationRate: number;
    testSize: number;
    nParents?: number;
    balancing: string;
}

const LOADING_MESSAGES = [
    "Preparing and validating dataset...",
    "Analyzing feature relevance...",
    "Exploring potential feature combinations...",
    "Evaluating model performance across selections...",
    "Finalizing the most impactful feature set..."
];

export default function FeatureSelectionTab({
    targetColumn,
    processedId,
    size,
    mutationRate,
    testSize,
    nParents,
    balancing
}: FeatureSelectionTabProps) {
    // UI States
    const [isLoading, setIsLoading] = useState(true);
    const [loadingMessageIdx, setLoadingMessageIdx] = useState(0);
    const [result, setResult] = useState<FSResult | null>(null);
    const [isDownloading, setIsDownloading] = useState(false);
    const [selectedImage, setSelectedImage] = useState<string | null>(null); // State cho Modal
    const hasFetched = useRef(false);

    // Local states for parameter configuration
    const [localSize, setLocalSize] = useState<number>(size);
    const [localMutationRate, setLocalMutationRate] = useState<number>(mutationRate);
    const [localTestSize, setLocalTestSize] = useState<number>(testSize);
    const [localNParents, setLocalNParents] = useState<number | string>(nParents ?? "");

    // --- State evaluation ---
    const [evalResult, setEvalResult] = useState<EvalResult | null>(null);
    const [isEvalLoading, setIsEvalLoading] = useState(false);

    // Thêm State để dễ dàng cấu hình phương pháp Balancing (ADASYN, SMOTE, none)
    const [localBalancing, setLocalBalancing] = useState<string>(
        balancing.toLowerCase() === "yes" ? "adasyn" : (balancing.toLowerCase() || "none")
    );

    // Rotate loading message every 5 seconds
    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (isLoading) {
            interval = setInterval(() => {
                setLoadingMessageIdx((prev) => (prev + 1) % LOADING_MESSAGES.length);
            }, 5000);
        }
        return () => clearInterval(interval);
    }, [isLoading]);

    const fetchEvaluation = async (fsDatasetId: string) => {
        setIsEvalLoading(true);
        try {
            const res = await api.get(`/datasets/${processedId}/feature-selection-evaluation`, {
                params: {
                    fs_dataset_id: fsDatasetId,
                    target_column: targetColumn,
                    test_size: localTestSize
                }
            });
            setEvalResult(res.data);
        } catch (error) {
            toast.error("Could not load deep evaluation metrics.");
        } finally {
            setIsEvalLoading(false);
        }
    };

    // Extracted API call function to allow re-running
    const runFS = useCallback(async (isManualRerun: boolean = false) => {
        if (!processedId || !targetColumn) return;

        setIsLoading(true);
        setLoadingMessageIdx(0);

        try {
            const requestParams: Record<string, any> = {
                target_column: targetColumn,
                size: localSize,
                mutation_rate: localMutationRate,
                test_size: localTestSize,
                balancing_method: localBalancing,
            };

            if (localNParents !== "" && localNParents !== undefined) {
                requestParams.n_parents = Number(localNParents);
            }

            const res = await api.get(`/datasets/${processedId}/feature-selection`, {
                params: requestParams
            });

            setResult(res.data);

            if (isManualRerun) {
                toast.success("Feature selection rerun completed successfully!");
            } else {
                toast.success("Feature selection completed successfully!");
            }

            if (res.data.fs_dataset_id) {
                fetchEvaluation(res.data.fs_dataset_id);
            }
        } catch (error) {
            toast.error("Feature selection failed. Please check your parameters.");
        } finally {
            setIsLoading(false);
            setEvalResult(null);
        }
    }, [processedId, targetColumn, localSize, localMutationRate, localTestSize, localNParents, localBalancing]);

    // Initial API call when component mounts
    useEffect(() => {
        if (!hasFetched.current && processedId && targetColumn) {
            hasFetched.current = true;
            runFS(false);
        }
    }, [processedId, targetColumn, runFS]);

    // --- Validation Logic before Rerun ---
    const handleRerun = () => {
        if (localSize < 10 || localSize > 200) {
            toast.error("Population size must be between 10 and 200.");
            return;
        }

        if (localMutationRate < 0.01 || localMutationRate > 0.5) {
            toast.error("Mutation rate must be between 0.01 and 0.5.");
            return;
        }

        if (localTestSize < 0.1 || localTestSize > 0.5) {
            toast.error("Test size must be between 0.1 and 0.5.");
            return;
        }

        if (localNParents !== "") {
            const parsedParents = Number(localNParents);
            if (parsedParents <= 0) {
                toast.error("Number of parents must be greater than 0.");
                return;
            }
            if (parsedParents >= localSize) {
                toast.error("Number of parents must be less than Population size.");
                return;
            }
        }

        runFS(true);
    };

    const handleDownload = async () => {
        if (!result?.fs_dataset_id) {
            toast.error("Dataset ID not available");
            return;
        }

        setIsDownloading(true);
        try {
            const response = await api.get(
                `/datasets/${result.fs_dataset_id}/download`,
                {
                    responseType: 'blob',
                    params: { target_column: targetColumn }
                }
            );

            const blob = new Blob([response.data], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `feature_selected_dataset.csv`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
        } catch (error) {
            toast.error("Failed to download dataset");
        } finally {
            setIsDownloading(false);
        }
    };

    const inputClass = "w-full p-2 mt-1 border border-gray-300 dark:border-gray-600 rounded-lg bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white focus:ring-2 focus:ring-[#4361EE] outline-none transition-shadow";

    // Extract balancing info returned from backend (shape: { method, before?, after?, skipped? })
    // Use only when the returned `method` matches the current `localBalancing` selection
    const balancingStats = result?.balancing && result.balancing.method === localBalancing ? result.balancing : null;

    const MetricCompare = ({ label, before, after }: { label: string, before: number, after: number }) => {
        const isBetter = after >= before;
        return (
            <div className="flex flex-col items-center p-3 bg-gray-50 dark:bg-gray-900/50 rounded-xl border border-gray-100 dark:border-gray-800">
                <span className="text-xs font-bold text-gray-500 uppercase">{label}</span>
                <div className="mt-2 flex items-center gap-3">
                    <span className="text-lg font-medium text-gray-400">{(before * 100).toFixed(1)}%</span>
                    <span className="text-gray-300 dark:text-gray-600">→</span>
                    <span className={`text-xl font-bold ${isBetter ? 'text-[#4361EE] dark:text-indigo-300' : 'text-red-500'}`}>
                        {(after * 100).toFixed(1)}%
                    </span>
                </div>
            </div>
        );
    };

    return (
        <div>
            <div className="flex-1 mt-6">
                {isLoading ? (
                    <div className="flex flex-col items-center justify-center min-h-[60vh]">
                        <div className="flex items-center gap-3 mb-6">
                            <div className="w-8 h-8 border-2 border-[#4361EE] border-t-transparent rounded-full animate-spin"></div>
                            <h3 className="text-2xl font-bold text-gray-800 dark:text-gray-200">
                                Running feature selection
                            </h3>
                        </div>

                        <p className="text-[#4361EE] font-medium text-lg animate-pulse transition-all duration-500 text-center px-4">
                            {LOADING_MESSAGES[loadingMessageIdx]}
                        </p>
                        <p className="text-sm text-gray-400 mt-6 bg-gray-50 dark:bg-gray-800 px-4 py-2 rounded-lg">
                            This process may take 20-30 seconds. Please do not close the browser.
                        </p>
                    </div>
                ) : (
                    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-500">

                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
                            <div>
                                <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                                    Feature selection results (Genetic algorithm)
                                </h2>
                                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1.5 max-w-2xl">
                                    The algorithm has identified the most predictive subset of features. This helps in reducing dimensionality, preventing overfitting, and improving model training efficiency.
                                </p>
                            </div>

                            <button
                                onClick={handleDownload}
                                disabled={isDownloading}
                                className="flex items-center hover:cursor-pointer justify-center shrink-0 min-w-[200px] gap-2 px-5 py-2.5 bg-linear-to-r from-[#4361EE] to-[#3a52d5] text-white font-semibold rounded-xl shadow-md hover:shadow-lg hover:from-[#3a52d5] hover:to-[#2e41b0] transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                <FiDownload className="text-lg" />
                                {isDownloading ? "Downloading..." : "Download dataset"}
                            </button>
                        </div>

                        {/* Stats Cards */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {/* Card 1: Best Accuracy */}
                            <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 hover:border-[#1A535C] transition-colors relative flex flex-col">
                                <div className="flex justify-between items-start">
                                    <p className="text-sm font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                                        Best accuracy
                                    </p>
                                    <div className="w-10 h-10 bg-[#1A535C]/10 rounded-full flex items-center justify-center shrink-0">
                                        <FaCheckCircle className="text-xl text-[#1A535C] dark:text-teal-400" />
                                    </div>
                                </div>

                                <div className="mt-2 flex items-baseline gap-1">
                                    <p className="text-5xl font-black text-gray-900 dark:text-white">
                                        {((result?.best_ga_accuracy || 0) * 100).toFixed(2)}
                                    </p>
                                    <span className="text-3xl font-bold text-gray-600 dark:text-gray-400">%</span>
                                </div>

                                <div className="mt-4 flex items-center gap-1.5 text-sm font-bold text-[#1A535C] dark:text-teal-400">
                                    <FiTrendingUp className="text-base" />
                                    <span>
                                        +{(((result?.best_ga_accuracy || 0) - (result?.baseline_accuracy || 0)) * 100).toFixed(2)}% vs Baseline
                                    </span>
                                </div>
                            </div>

                            {/* Card 2: Feature Count */}
                            <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 hover:border-[#4361EE] transition-colors relative flex flex-col">
                                <div className="flex justify-between items-start">
                                    <p className="text-sm font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                                        Feature count
                                    </p>
                                    <div className="w-10 h-10 bg-[#4361EE]/10 rounded-full flex items-center justify-center shrink-0">
                                        <FaFilter className="text-xl text-[#4361EE]" />
                                    </div>
                                </div>

                                <div className="mt-2 flex items-baseline gap-2">
                                    <p className="text-5xl font-black text-gray-900 dark:text-white">
                                        {result?.feature_count}
                                    </p>
                                    <span className="text-lg font-medium text-gray-600 dark:text-gray-400">
                                        Selected
                                    </span>
                                </div>

                                <div className="mt-4 flex items-center gap-1.5 text-sm font-medium text-gray-500 dark:text-gray-400">
                                    <FiInfo className="text-base" />
                                    <span>
                                        Reduced from {result?.original_feature_count} initial features
                                    </span>
                                </div>
                            </div>

                        </div>

                        {/* Data Balancing Info */}
                        <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 hover:border-purple-500 transition-colors relative flex flex-col">

                            <div className="flex justify-between items-center mb-2">
                                <p className="text-sm font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                                    Data Balancing: <span className="text-black dark:text-white ml-1">{localBalancing === "none" ? "None" : localBalancing}</span>
                                </p>
                                <div className="w-10 h-10 bg-purple-500/10 rounded-full flex items-center justify-center shrink-0">
                                    <FaBalanceScale className="text-xl text-purple-600 dark:text-purple-400" />
                                </div>
                            </div>

                            <div className="flex-1 flex flex-col justify-center">
                                {localBalancing === "none" ? (
                                    <p className="text-gray-500 font-medium">Dataset used as-is. No balancing applied.</p>
                                ) : balancingStats?.skipped ? (
                                    <div className="bg-orange-50 dark:bg-orange-900/10 p-3 rounded-xl border border-orange-100 dark:border-orange-800/30">
                                        <p className="text-sm font-bold text-orange-600 dark:text-orange-400 uppercase mb-1">Skipped</p>
                                        <p className="text-sm text-orange-700 dark:text-orange-300">{balancingStats.skipped}</p>
                                    </div>
                                ) : balancingStats?.before && balancingStats?.after ? (
                                    <div className="grid grid-cols-2 gap-4 text-sm">
                                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded-xl border border-gray-100 dark:border-gray-800">
                                            <p className="font-bold text-gray-500 uppercase mb-2 border-b border-gray-200 dark:border-gray-700 pb-1.5">Original</p>
                                            <div className="space-y-2">
                                                {Object.entries(balancingStats.before).map(([cls, count]) => (
                                                    <div key={cls} className="flex justify-between text-gray-700 dark:text-gray-300">
                                                        <span>Class {cls}:</span>
                                                        <span className="font-mono font-bold">{count as number}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                        <div className="bg-purple-50 dark:bg-purple-900/10 p-3 rounded-xl border border-purple-100 dark:border-purple-800/30">
                                            <p className="font-bold text-purple-500 uppercase mb-2 border-b border-purple-200 dark:border-purple-800/30 pb-1.5">Balanced</p>
                                            <div className="space-y-2">
                                                {Object.entries(balancingStats.after).map(([cls, count]) => (
                                                    <div key={cls} className="flex justify-between text-purple-700 dark:text-purple-300">
                                                        <span>Class {cls}:</span>
                                                        <span className="font-mono font-bold">{count as number}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                ) : (
                                    <p className="text-base text-gray-500 font-medium italic">Processing results...</p>
                                )}
                            </div>
                        </div>

                        {/* Feature List */}
                        <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-100 dark:border-gray-700 overflow-hidden shadow-sm mt-2">
                            <div className="py-4 px-6 border-b border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 flex items-center gap-2">
                                <FaStar className="text-[#4361EE] text-xl" />
                                <h3 className="font-bold text-gray-800 dark:text-gray-200">
                                    Optimal feature subset
                                </h3>
                            </div>
                            <div className="p-6">
                                <div className="flex flex-wrap gap-3">
                                    {result?.selected_features.map((feat) => (
                                        <span
                                            key={feat}
                                            className="px-4 py-2 bg-indigo-50 dark:bg-indigo-900/20 text-[#4361EE] dark:text-indigo-300 rounded-xl border border-indigo-100 dark:border-indigo-800/30 font-semibold shadow-sm"
                                        >
                                            {feat}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* --- DEEP EVALUATION --- */}
                        <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-100 dark:border-gray-700 overflow-hidden shadow-sm mt-6">
                            <div className="py-4 px-6 border-b border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <FiBarChart2 className="text-[#4361EE] text-xl" />
                                    <h3 className="font-bold text-gray-800 dark:text-gray-200">Impact evaluation</h3>
                                </div>
                                {isEvalLoading && (
                                    <div className="flex items-center gap-2 text-sm text-[#4361EE] animate-pulse">
                                        <div className="w-5 h-5 border-2 border-gray-500 border-t-transparent rounded-full animate-spin"></div>
                                    </div>
                                )}
                            </div>

                            <div className="p-6">
                                {isEvalLoading ? (
                                    <div className="h-64 flex items-center justify-center gap-2 bg-gray-50 dark:bg-gray-900/20 rounded-xl border border-dashed border-gray-200 dark:border-gray-700">
                                        <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                                        <p className="text-gray-500">Please wait while we evaluate the selected features...</p>
                                    </div>
                                ) : evalResult ? (
                                    <div className="space-y-6">
                                        {/* Metrics Table */}
                                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                            <MetricCompare label="Accuracy" before={evalResult.metrics_before.accuracy} after={evalResult.metrics_after.accuracy} />
                                            <MetricCompare label="Recall (Sensitivity)" before={evalResult.metrics_before.recall} after={evalResult.metrics_after.recall} />
                                            <MetricCompare label="Precision" before={evalResult.metrics_before.precision} after={evalResult.metrics_after.precision} />
                                            <MetricCompare label="F1-Score" before={evalResult.metrics_before.f1_score} after={evalResult.metrics_after.f1_score} />
                                        </div>

                                        {/* Charts - Hiển thị thành 2 cột cho gọn */}
                                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                            {evalResult.confusion_matrix_chart_url && (
                                                <div
                                                    className="relative group cursor-pointer rounded-xl border border-gray-100 dark:border-gray-700 overflow-hidden bg-white"
                                                    onClick={() => setSelectedImage(evalResult.confusion_matrix_chart_url)}
                                                >
                                                    <Image
                                                        src={evalResult.confusion_matrix_chart_url}
                                                        alt="Confusion Matrix Comparison"
                                                        width={600}
                                                        height={400}
                                                        className="w-full h-auto object-contain"
                                                    />
                                                    <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 bg-black/10 transition-opacity">
                                                        <span className="bg-black/70 text-white text-xs px-3 py-1.5 rounded-md font-medium shadow-sm backdrop-blur-sm">Click to expand</span>
                                                    </div>
                                                </div>
                                            )}

                                            {evalResult.roc_chart_url && (
                                                <div
                                                    className="relative group cursor-pointer rounded-xl border border-gray-100 dark:border-gray-700 overflow-hidden bg-white"
                                                    onClick={() => setSelectedImage(evalResult.roc_chart_url)}
                                                >
                                                    <Image
                                                        src={evalResult.roc_chart_url}
                                                        alt="ROC Curve Comparison"
                                                        width={600}
                                                        height={400}
                                                        className="w-full h-auto object-contain"
                                                    />
                                                    <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 bg-black/10 transition-opacity">
                                                        <span className="bg-black/70 text-white text-xs px-3 py-1.5 rounded-md font-medium shadow-sm backdrop-blur-sm">Click to expand</span>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                ) : (
                                    <div className="text-center text-gray-500 py-8">Evaluation data is not available.</div>
                                )}
                            </div>
                        </div>

                        {/* --- CONFIGURATION PANEL --- */}
                        <div className="bg-white dark:bg-gray-800 mb-6 mt-8 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700">
                            <div className="py-4 px-6 border-b border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 flex items-center gap-2">
                                <IoSettingsSharp className="text-[#4361EE] text-xl" />
                                <h3 className="font-bold text-gray-900 dark:text-white">
                                    Algorithm parameters
                                </h3>
                            </div>
                            {/* Layout Grid được nới rộng ra để chứa thêm ô Balancing */}
                            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6 p-6">

                                {/* Data Balancing Method Dropdown */}
                                <div>
                                    <label className="block text-sm font-semibold text-gray-800 dark:text-gray-200">
                                        Data balancing
                                    </label>
                                    <select
                                        value={localBalancing}
                                        onChange={(e) => setLocalBalancing(e.target.value)}
                                        className={inputClass}
                                    >
                                        <option value="none">None (No balancing)</option>
                                        <option value="adasyn">ADASYN</option>
                                        <option value="smote">SMOTE</option>
                                    </select>
                                    <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">Choose method to handle imbalanced classes.</p>
                                </div>

                                {/* Population Size */}
                                <div>
                                    <label className="block text-sm font-semibold text-gray-800 dark:text-gray-200">
                                        Population size
                                    </label>
                                    <input
                                        type="number"
                                        value={localSize}
                                        min={10} max={200}
                                        onChange={(e) => setLocalSize(Number(e.target.value))}
                                        className={inputClass}
                                    />
                                    <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">Individuals in each generation (10 - 200).</p>
                                </div>

                                {/* Mutation Rate */}
                                <div>
                                    <label className="block text-sm font-semibold text-gray-800 dark:text-gray-200">
                                        Mutation rate
                                    </label>
                                    <input
                                        type="number"
                                        step="0.01"
                                        min={0.01} max={0.5}
                                        value={localMutationRate}
                                        onChange={(e) => setLocalMutationRate(Number(e.target.value))}
                                        className={inputClass}
                                    />
                                    <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">Probability of feature flipping (0.01 - 0.5).</p>
                                </div>

                                {/* Number of Parents */}
                                <div>
                                    <label className="block text-sm font-semibold text-gray-800 dark:text-gray-200">
                                        Number of parents
                                    </label>
                                    <input
                                        type="number"
                                        value={localNParents}
                                        onChange={(e) => setLocalNParents(e.target.value ? Number(e.target.value) : "")}
                                        placeholder="Default"
                                        className={inputClass}
                                    />
                                    <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">Best individuals kept. (Less than Population).</p>
                                </div>

                                {/* Test Size */}
                                <div>
                                    <label className="block text-sm font-semibold text-gray-800 dark:text-gray-200">
                                        Test size
                                    </label>
                                    <input
                                        type="number"
                                        step="0.05"
                                        min={0.1} max={0.5}
                                        value={localTestSize}
                                        onChange={(e) => setLocalTestSize(Number(e.target.value))}
                                        className={inputClass}
                                    />
                                    <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">Proportion of data for evaluation (0.1 - 0.5).</p>
                                </div>
                            </div>

                            {/* Rerun Button */}
                            <div className="flex justify-end py-4 px-6 border-t border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/20 rounded-b-2xl">
                                <button
                                    onClick={handleRerun}
                                    className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-[#4361EE] text-white font-semibold shadow-md cursor-pointer hover:bg-[#3a52d5] hover:shadow-lg focus:ring-4 focus:ring-indigo-500/30 transition-all disabled:opacity-70 disabled:cursor-not-allowed"
                                >
                                    <FiPlay className="text-sm" />
                                    Rerun selection
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            <ImageModal
                imageUrl={selectedImage}
                onClose={() => setSelectedImage(null)}
            />
        </div>
    );
}