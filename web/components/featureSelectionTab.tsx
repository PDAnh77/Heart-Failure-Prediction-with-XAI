"use client";
import { useEffect, useState, useRef } from "react";
import { FiCheckCircle, FiCpu, FiTrendingUp, FiInfo, FiFilter } from "react-icons/fi";
import { FaStar, FaCheckCircle } from "react-icons/fa";
import { FaFilter } from "react-icons/fa6";
import { api } from "@/lib/api";
import toast from "react-hot-toast";
import { FSResult } from "@/types/feature_selection";

interface FeatureSelectionTabProps {
    targetColumn: string;
    processedId: string | null;
}

const LOADING_MESSAGES = [
    "Preparing and validating dataset...",
    "Analyzing feature relevance...",
    "Exploring potential feature combinations...",
    "Evaluating model performance across selections...",
    "Finalizing the most impactful feature set..."
];

export default function FeatureSelectionTab({ targetColumn, processedId }: FeatureSelectionTabProps) {
    const [isLoading, setIsLoading] = useState(true);
    const [loadingMessageIdx, setLoadingMessageIdx] = useState(0);
    const [result, setResult] = useState<FSResult | null>(null);
    const hasFetched = useRef(false);

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

    // Call API when processedId is available from parent
    useEffect(() => {
        if (!processedId || !targetColumn) {
            return;
        }

        if (hasFetched.current) return;

        const runFS = async () => {
            hasFetched.current = true;
            setIsLoading(true);
            try {
                const res = await api.get(`/datasets/${processedId}/feature-selection`, {
                    params: { target_column: targetColumn }
                });
                setResult(res.data);
                toast.success("Feature selection completed successfully!");
            } catch (error) {
                console.error("Feature Selection Error:", error);
                toast.error("Feature selection failed.");
                hasFetched.current = false;
            } finally {
                setIsLoading(false);
            }
        };

        runFS();
    }, [processedId, targetColumn]);

    return (
        <div>
            <div className="flex-1 mt-6">
                {isLoading ? (
                    <div className="flex flex-col items-center justify-center min-h-[60vh]">
                        <div className="relative mb-8">
                            <div className="w-20 h-20 border-4 border-gray-200 dark:border-gray-700 border-t-[#4361EE] rounded-full animate-spin"></div>
                            <div className="absolute inset-0 flex items-center justify-center">
                                <FiCpu className="w-6 h-6 text-[#4361EE] animate-pulse" />
                            </div>
                        </div>
                        <h3 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-4">
                            Running Feature Selection
                        </h3>
                        <p className="text-[#4361EE] font-medium text-lg animate-pulse transition-all duration-500 text-center px-4">
                            {LOADING_MESSAGES[loadingMessageIdx]}
                        </p>
                        <p className="text-sm text-gray-400 mt-6 bg-gray-50 dark:bg-gray-800 px-4 py-2 rounded-lg">
                            This process may take 20-30 seconds. Please do not close the browser.
                        </p>
                    </div>
                ) : (
                    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-500">

                        <h2 className="text-xl font-bold mb-4 text-gray-900 dark:text-white">
                            Feature Selection Results
                        </h2>


                        {/* Stats Cards */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {/* Card 1: Best Accuracy */}
                            <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 hover:border-[#1A535C] transition-colors relative flex flex-col">
                                {/* Tiêu đề & Icon góc phải */}
                                <div className="flex justify-between items-start">
                                    <p className="text-sm font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                                        Best Accuracy
                                    </p>
                                    <div className="w-10 h-10 bg-[#1A535C]/10 rounded-full flex items-center justify-center shrink-0">
                                        <FaCheckCircle className="text-xl text-[#1A535C] dark:text-teal-400" />
                                    </div>
                                </div>

                                {/* Số lớn */}
                                <div className="mt-2 flex items-baseline gap-1">
                                    <p className="text-5xl font-black text-gray-900 dark:text-white">
                                        {((result?.best_ga_accuracy || 0) * 100).toFixed(2)}
                                    </p>
                                    <span className="text-3xl font-bold text-gray-600 dark:text-gray-400">%</span>
                                </div>

                                {/* Dòng thông tin phụ ở dưới */}
                                <div className="mt-4 flex items-center gap-1.5 text-sm font-bold text-[#1A535C] dark:text-teal-400">
                                    <FiTrendingUp className="text-base" />
                                    <span>
                                        +{(((result?.best_ga_accuracy || 0) - (result?.baseline_accuracy || 0)) * 100).toFixed(2)}% vs baseline
                                    </span>
                                </div>
                            </div>

                            {/* Card 2: Feature Count */}
                            <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 hover:border-[#4361EE] transition-colors relative flex flex-col">
                                {/* Tiêu đề & Icon góc phải */}
                                <div className="flex justify-between items-start">
                                    <p className="text-sm font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                                        Feature Count
                                    </p>
                                    <div className="w-10 h-10 bg-[#4361EE]/10 rounded-full flex items-center justify-center shrink-0">
                                        <FaFilter className="text-xl text-[#4361EE]" />
                                    </div>
                                </div>

                                {/* Số lớn */}
                                <div className="mt-2 flex items-baseline gap-2">
                                    <p className="text-5xl font-black text-gray-900 dark:text-white">
                                        {result?.feature_count}
                                    </p>
                                    <span className="text-lg font-medium text-gray-600 dark:text-gray-400">
                                        Selected Columns
                                    </span>
                                </div>

                                {/* Dòng thông tin phụ ở dưới */}
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
                            <div className="p-4 border-b border-gray-100 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 flex items-center gap-2">
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
                    </div>
                )}
            </div>
        </div>
    );
}