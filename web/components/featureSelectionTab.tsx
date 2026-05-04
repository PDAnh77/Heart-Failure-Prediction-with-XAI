"use client";
import { useEffect, useState, useRef, useCallback } from "react";
import { FiTrendingUp, FiInfo, FiDownload, FiPlay, FiLoader } from "react-icons/fi";
import { FaStar, FaCheckCircle } from "react-icons/fa";
import { FaFilter } from "react-icons/fa6";
import { IoSettingsSharp } from "react-icons/io5";
import { api } from "@/lib/api";
import toast from "react-hot-toast";
import { FSResult } from "@/types/feature_selection";

interface FeatureSelectionTabProps {
    targetColumn: string;
    processedId: string | null;
    size: number;
    mutationRate: number;
    testSize: number;
    nParents?: number;
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
    nParents
}: FeatureSelectionTabProps) {
    // UI States
    const [isLoading, setIsLoading] = useState(true);
    const [loadingMessageIdx, setLoadingMessageIdx] = useState(0);
    const [result, setResult] = useState<FSResult | null>(null);
    const [isDownloading, setIsDownloading] = useState(false);
    const hasFetched = useRef(false);

    // Local states for parameter configuration
    const [localSize, setLocalSize] = useState<number>(size);
    const [localMutationRate, setLocalMutationRate] = useState<number>(mutationRate);
    const [localTestSize, setLocalTestSize] = useState<number>(testSize);
    const [localNParents, setLocalNParents] = useState<number | string>(nParents ?? "");

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
            }
        } catch (error) {
            console.error("Feature Selection Error:", error);
            toast.error("Feature selection failed. Please check your parameters.");
        } finally {
            setIsLoading(false);
        }
    }, [processedId, targetColumn, localSize, localMutationRate, localTestSize, localNParents]);

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
            toast.error("Population Size must be between 10 and 200.");
            return;
        }

        if (localMutationRate < 0.01 || localMutationRate > 0.5) {
            toast.error("Mutation Rate must be between 0.01 and 0.5.");
            return;
        }

        if (localTestSize < 0.1 || localTestSize > 0.5) {
            toast.error("Test Size must be between 0.1 and 0.5.");
            return;
        }

        if (localNParents !== "") {
            const parsedParents = Number(localNParents);
            if (parsedParents <= 0) {
                toast.error("Number of Parents must be greater than 0.");
                return;
            }
            if (parsedParents >= localSize) {
                toast.error("Number of Parents must be less than Population Size.");
                return;
            }
        }

        // All validations passed, execute run
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
            console.error("Download Error:", error);
            toast.error("Failed to download dataset");
        } finally {
            setIsDownloading(false);
        }
    };

    const inputClass = "w-full p-2 mt-1 border border-gray-300 dark:border-gray-600 rounded-lg bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white focus:ring-2 focus:ring-[#4361EE] outline-none transition-shadow";

    return (
        <div>
            <div className="flex-1 mt-6">
                {isLoading ? (
                    <div className="flex flex-col items-center justify-center min-h-[60vh]">
                        <div className="flex items-center gap-3 mb-6">
                            <FiLoader className="w-8 h-8 text-[#4361EE] animate-spin" />
                            <h3 className="text-2xl font-bold text-gray-800 dark:text-gray-200">
                                Running Feature Selection
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
                                    Feature Selection Results
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
                                {isDownloading ? "Downloading..." : "Download Dataset"}
                            </button>
                        </div>

                        {/* Stats Cards */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {/* Card 1: Best Accuracy */}
                            <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 hover:border-[#1A535C] transition-colors relative flex flex-col">
                                <div className="flex justify-between items-start">
                                    <p className="text-sm font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                                        Best Accuracy
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
                                        +{(((result?.best_ga_accuracy || 0) - (result?.baseline_accuracy || 0)) * 100).toFixed(2)}% vs baseline
                                    </span>
                                </div>
                            </div>

                            {/* Card 2: Feature Count */}
                            <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 hover:border-[#4361EE] transition-colors relative flex flex-col">
                                <div className="flex justify-between items-start">
                                    <p className="text-sm font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                                        Feature Count
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
                                        Selected Columns
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

                        {/* Feature List */}
                        <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-100 dark:border-gray-700 overflow-hidden shadow-sm">
                            <div className="py-4 px-6 border-b border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 flex items-center gap-2">
                                <FaStar className="text-[#4361EE] text-xl" />
                                <h3 className="font-bold text-gray-800 dark:text-gray-200">
                                    Optimal Feature Subset
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

                        {/* --- CONFIGURATION PANEL --- */}
                        <div className="bg-white dark:bg-gray-800 mb-6 mt-8 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700">
                            <div className="py-4 px-6 border-b border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 flex items-center gap-2">
                                <IoSettingsSharp className="text-[#4361EE] text-xl" />
                                <h3 className="font-bold text-gray-900 dark:text-white">
                                    Algorithm Parameters
                                </h3>
                            </div>
                            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-6 p-6">
                                {/* Population Size */}
                                <div>
                                    <label className="block text-sm font-semibold text-gray-800 dark:text-gray-200">
                                        Population Size
                                    </label>
                                    <input
                                        type="number"
                                        value={localSize}
                                        min={10} max={200}
                                        onChange={(e) => setLocalSize(Number(e.target.value))}
                                        className={inputClass}
                                    />
                                    <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">Number of individuals in each generation. (Range: 10 - 200).</p>
                                </div>

                                {/* Mutation Rate */}
                                <div>
                                    <label className="block text-sm font-semibold text-gray-800 dark:text-gray-200">
                                        Mutation Rate
                                    </label>
                                    <input
                                        type="number"
                                        step="0.01"
                                        min={0.01} max={0.5}
                                        value={localMutationRate}
                                        onChange={(e) => setLocalMutationRate(Number(e.target.value))}
                                        className={inputClass}
                                    />
                                    <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">Probability of a feature flipping its state. (Range: 0.01 - 0.5).</p>
                                </div>

                                {/* Number of Parents */}
                                <div>
                                    <label className="block text-sm font-semibold text-gray-800 dark:text-gray-200">
                                        Number of Parents
                                    </label>
                                    <input
                                        type="number"
                                        value={localNParents}
                                        onChange={(e) => setLocalNParents(e.target.value ? Number(e.target.value) : "")}
                                        placeholder="Leave empty for default"
                                        className={inputClass}
                                    />
                                    <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">Best individuals kept for breeding. Must be less than Population Size.</p>
                                </div>

                                {/* Test Size */}
                                <div>
                                    <label className="block text-sm font-semibold text-gray-800 dark:text-gray-200">
                                        Test Size
                                    </label>
                                    <input
                                        type="number"
                                        step="0.05"
                                        min={0.1} max={0.5}
                                        value={localTestSize}
                                        onChange={(e) => setLocalTestSize(Number(e.target.value))}
                                        className={inputClass}
                                    />
                                    <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">Proportion of data used for evaluation. (Range: 0.1 - 0.5).</p>
                                </div>
                            </div>

                            {/* Rerun Button */}
                            <div className="flex justify-end py-4 px-6 border-t border-gray-100 dark:border-gray-700">
                                <button
                                    onClick={handleRerun}
                                    className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-indigo-600 text-white font-semibold shadow-sm cursor-pointer hover:bg-indigo-700 focus:ring-4 focus:ring-indigo-500/30 transition-all disabled:opacity-70 disabled:cursor-not-allowed"
                                >
                                    <FiPlay className="text-sm" />
                                    Rerun Selection
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}