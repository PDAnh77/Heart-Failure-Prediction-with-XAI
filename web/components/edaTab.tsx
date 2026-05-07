"use client";
import { useEffect, useState, useRef, UIEvent } from "react";
import { FiLoader, FiPieChart, FiAlertCircle, FiCheck, FiDatabase, FiDownload, FiGrid, FiLayers } from "react-icons/fi";
import Image from "next/image";
import toast from "react-hot-toast";
import { api } from "@/lib/api";
import ImageModal from "@/components/imageModal";
import { DatasetSummary } from "@/types/dataset_summary";

interface EdaTabProps {
    datasetId: string;
    targetColumn: string;
    imputation: string;
    balancing: string;
    onProcessed: (processedId: string) => void;
}

export default function EDATab({ datasetId, targetColumn, imputation, balancing, onProcessed }: EdaTabProps) {
    const [summary, setSummary] = useState<DatasetSummary | null>(null);
    const [charts, setCharts] = useState<Record<string, string>>({});
    const [selectedImage, setSelectedImage] = useState<string | null>(null);
    const [duplicatesRemoved, setDuplicatesRemoved] = useState<number>(0);
    const [processedRowCount, setProcessedRowCount] = useState<number | null>(null);

    const fetchedDatasetId = useRef<string | null>(null);

    // --- STATES CHO DATA PREVIEW ---
    const [processedId, setProcessedId] = useState<string | null>(null);
    const [originalRows, setOriginalRows] = useState<any[]>([]);
    const [processedRows, setProcessedRows] = useState<any[]>([]);
    const [originalOffset, setOriginalOffset] = useState(0);
    const [processedOffset, setProcessedOffset] = useState(0);
    const [hasMoreOriginal, setHasMoreOriginal] = useState(true);
    const [hasMoreProcessed, setHasMoreProcessed] = useState(true);
    const [loadingMoreOrg, setLoadingMoreOrg] = useState(false);
    const [loadingMoreProc, setLoadingMoreProc] = useState(false);

    const [isDownloading, setIsDownloading] = useState(false);

    const [isLoadingSummary, setIsLoadingSummary] = useState(true);
    const [isLoadingPreprocess, setIsLoadingPreprocess] = useState(false);
    const [isLoadingEda, setIsLoadingEda] = useState(false);

    const extractRows = (data: any) => {
        if (Array.isArray(data)) return data;
        return data?.data || data?.rows || data?.items || [];
    };

    useEffect(() => {
        // Chỉ chạy khi đã có đủ thông tin
        if (!datasetId || !targetColumn) return;

        // Tránh gọi API 2 lần
        if (fetchedDatasetId.current === datasetId) return;

        const fetchData = async () => {
            fetchedDatasetId.current = datasetId;

            try {
                // --- 1. TẢI VÀ HIỂN THỊ SUMMARY ---
                setIsLoadingSummary(true);
                const summaryRes = await api.get(`/datasets/${datasetId}/summary`, {
                    params: { target_column: targetColumn }
                });
                setSummary(summaryRes.data);
                setIsLoadingSummary(false);

                // --- 2. PREPROCESS VÀ LẤY DATA PREVIEW ---
                setIsLoadingPreprocess(true);
                const preprocessRes = await api.post(`/datasets/${datasetId}/preprocess`, null, {
                    params: {
                        target_column: targetColumn,
                        imputation_method: imputation
                    }
                });

                const procId = preprocessRes.data.processed_dataset_id;
                setProcessedId(procId);

                // Truyền ID dataset đã xử lý lên trang cha
                onProcessed(procId);

                setDuplicatesRemoved(preprocessRes.data.duplicates_removed || 0);
                setProcessedRowCount(preprocessRes.data.rows || null);

                // Lấy dữ liệu bảng ngay sau khi có procId
                const [orgRowsRes, procRowsRes] = await Promise.all([
                    api.get(`/datasets/${datasetId}/rows`, { params: { limit: 10, offset: 0 } }),
                    api.get(`/datasets/${procId}/rows`, { params: { limit: 10, offset: 0 } })
                ]);

                setOriginalRows(extractRows(orgRowsRes.data));
                setProcessedRows(extractRows(procRowsRes.data));
                setIsLoadingPreprocess(false);

                // --- 3. TẢI BIỂU ĐỒ EDA ---
                setIsLoadingEda(true);
                const edaRes = await api.get(`/datasets/${procId}/eda`, { params: { target_column: targetColumn } });

                setCharts(edaRes.data.charts || {});
                setIsLoadingEda(false);

                toast.success("Analysis completed successfully!");
            } catch (error) {
                console.error("EDA Error:", error);
                toast.error("Failed to load analysis data.");
                fetchedDatasetId.current = null;
            } finally {
                setIsLoadingSummary(false);
                setIsLoadingPreprocess(false);
                setIsLoadingEda(false);
            }
        };

        fetchData();
    }, [datasetId, targetColumn, onProcessed]);

    const loadMoreRows = async (type: 'original' | 'processed') => {
        if (type === 'original') {
            if (loadingMoreOrg || !hasMoreOriginal || !datasetId) return;
            setLoadingMoreOrg(true);
            try {
                const nextOffset = originalOffset + 10;
                const res = await api.get(`/datasets/${datasetId}/rows`, { params: { limit: 10, offset: nextOffset } });
                const newRows = extractRows(res.data);

                if (newRows.length === 0) setHasMoreOriginal(false);
                else {
                    setOriginalRows(prev => [...prev, ...newRows]);
                    setOriginalOffset(nextOffset);
                }
            } catch (error) {
                console.error("Failed to load more original rows", error);
            } finally {
                setLoadingMoreOrg(false);
            }
        } else {
            if (loadingMoreProc || !hasMoreProcessed || !processedId) return;
            setLoadingMoreProc(true);
            try {
                const nextOffset = processedOffset + 10;
                const res = await api.get(`/datasets/${processedId}/rows`, { params: { limit: 10, offset: nextOffset } });
                const newRows = extractRows(res.data);

                if (newRows.length === 0) setHasMoreProcessed(false);
                else {
                    setProcessedRows(prev => [...prev, ...newRows]);
                    setProcessedOffset(nextOffset);
                }
            } catch (error) {
                console.error("Failed to load more processed rows", error);
            } finally {
                setLoadingMoreProc(false);
            }
        }
    };

    const handleTableScroll = (e: UIEvent<HTMLDivElement>, type: 'original' | 'processed') => {
        const { scrollTop, scrollHeight, clientHeight } = e.currentTarget;

        if (scrollHeight - scrollTop - clientHeight < 50) {
            loadMoreRows(type);
        }
    };

    // --- HÀM XỬ LÝ TẢI DATASET ---
    const handleDownloadProcessed = async () => {
        if (!processedId) return;
        try {
            setIsDownloading(true);

            // Sử dụng responseType 'blob' để nhận file
            const response = await api.get(`/datasets/${processedId}/download`, {
                responseType: 'blob'
            });

            // Tạo link tải file tạm thời và trigger click
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.download = `preprocessed_dataset.csv`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
        } catch (error) {
            console.error("Download Error:", error);
            toast.error("Failed to download the dataset.");
        } finally {
            setIsDownloading(false);
        }
    };

    const chartConfigurations = [
        {
            key: "target_distribution",
            title: "Target Distribution",
            description: "Illustrates the distribution of the target variable. This helps identify if the dataset is balanced or imbalanced, which is crucial for model evaluation."
        },
        {
            key: "target_correlation",
            title: "Target Correlation",
            description: "Shows how strongly each individual numerical feature correlates directly with the target variable, highlighting the most predictive factors."
        },
        {
            key: "full_correlation",
            title: "Feature Correlation Matrix",
            description: "Displays the correlation matrix across all features. Distinct colored regions indicate strong relationships, useful for spotting multicollinearity."
        },
        {
            key: "anova_score",
            title: "ANOVA F-Scores (Numerical)",
            description: "Measures the linear dependency between numerical features and the target. Features with higher F-scores are highly informative for the model."
        },
        {
            key: "chi_square_score",
            title: "Chi-Square Scores (Categorical)",
            description: "Evaluates the statistical significance of categorical features against the target variable. Higher scores indicate stronger relevance."
        }
    ];

    const renderChartImage = (imageUrl: string, title: string, description: string, index: number) => {
        if (!imageUrl) return null;

        return (
            <div className="bg-white p-5 rounded-2xl shadow-sm border border-gray-200 dark:bg-gray-800 dark:border-gray-700 w-full flex flex-col md:flex-row items-center gap-6 overflow-hidden">
                <div
                    className="relative w-full md:w-[50%] h-[300px] rounded-xl bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-700/50 p-2 shrink-0 cursor-pointer group"
                    onClick={() => setSelectedImage(imageUrl)}
                >
                    <Image
                        src={imageUrl}
                        alt={title}
                        fill
                        sizes="(max-width: 768px) 100vw, 50vw"
                        className="object-contain p-2"
                        loading="lazy"
                    />
                    <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 bg-black/10 transition-opacity">
                        <span className="bg-black/50 text-white text-xs px-2 py-1 rounded-md">Click to expand</span>
                    </div>
                </div>

                <div className="w-full md:w-[55%] flex flex-col justify-center py-2">
                    <div className="inline-flex items-center justify-center px-3 py-1 rounded-full bg-indigo-50 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 text-xs font-bold uppercase tracking-wider mb-3 w-fit">
                        Plot {index + 1}
                    </div>
                    <h4 className="font-bold text-xl mb-3 text-gray-800 dark:text-gray-200">{title}</h4>
                    <p className="text-gray-600 dark:text-gray-400 text-sm md:text-base leading-relaxed">
                        {description}
                    </p>
                </div>
            </div>
        );
    };

    const renderDataTable = (data: any[], type: 'original' | 'processed', isLoadingMore: boolean, hasMore: boolean) => {
        if (data.length === 0) return <div className="p-4 text-center text-gray-500">No data available</div>;
        const columns = Object.keys(data[0]);

        return (
            <div
                className="overflow-auto h-90 rounded-xl overscroll-none border border-gray-200 dark:border-gray-700 custom-scrollbar relative"
                onScroll={(e) => handleTableScroll(e, type)}
            >
                <table className="min-w-full w-max text-sm text-left text-gray-500 dark:text-gray-300">
                    <thead className="text-gray-700 bg-gray-50 dark:bg-gray-900/80 dark:text-gray-100 sticky top-0 z-10 shadow-sm backdrop-blur-sm">
                        <tr>
                            {columns.map(col => (
                                <th key={col} className="px-4 py-3 border-b border-r border-gray-200 dark:border-gray-700 w-28 min-w-28 max-w-28 wrap-break-word">{col}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {data.map((row, index) => (
                            <tr
                                key={index}
                                className="bg-white border-b border-gray-400/50 dark:bg-gray-800 dark:border-gray-700"
                            >
                                {columns.map((col) => {
                                    const rawValue = row[col];
                                    const fullValue = String(rawValue ?? "");

                                    const displayValue =
                                        typeof rawValue === "number" &&
                                            !Number.isInteger(rawValue) &&
                                            type === "processed"
                                            ? Number(rawValue)
                                            : fullValue;

                                    return (
                                        <td
                                            key={col}
                                            className="px-4 py-2 border-r border-gray-400/50 dark:border-gray-700"
                                        >
                                            <div className="relative group max-w-[100px]">
                                                <div className="truncate">
                                                    {displayValue}
                                                </div>

                                                {/* tooltip khi hover */}
                                                {fullValue.length > 6 && (
                                                    <div
                                                        className="absolute left-0 bottom-full mb-1 hidden group-hover:block bg-black text-white text-xs px-2 py-1 rounded whitespace-pre-wrap z-20 max-w-xs wrap-break-word shadow-lg"
                                                    >
                                                        {fullValue}
                                                    </div>
                                                )}
                                            </div>
                                        </td>
                                    );
                                })}
                            </tr>
                        ))}
                    </tbody>
                </table>
                {isLoadingMore && (
                    <div className="py-3 flex justify-center items-center bg-white dark:bg-gray-800 border-t border-gray-100 dark:border-gray-700">
                        <FiLoader className="w-5 h-5 animate-spin text-indigo-600 mr-2" />
                        <span className="text-xs text-gray-500">Loading more rows...</span>
                    </div>
                )}
            </div>
        );
    };

    const getImputationInfo = (method: string) => {
        switch (method) {
            case "knn":
                return {
                    label: "KNN Imputer",
                    detail: "Estimates missing values using nearest neighbors in feature space."
                };
            case "mice":
                return {
                    label: "MICE Imputer",
                    detail: "Iteratively imputes each feature using the others as predictors."
                };
            case "mean":
                return {
                    label: "Mean / Mode",
                    detail: "Uses mean for numerical features and mode for categorical features."
                };
            case "default":
            default:
                return {
                    label: "Median / Mode",
                    detail: "Uses median for numerical features and mode for categorical features."
                };
        }
    };

    return (
        <>
            <div className="mt-6 space-y-8 animate-in fade-in slide-in-from-bottom-2 duration-700">

                {/* --- Dataset Scale & Missing Values --- */}
                {isLoadingSummary ? (
                    <div className="flex gap-2 items-center justify-center py-10 rounded-2xl border border-gray-100 dark:border-gray-700">
                        <FiLoader className="w-8 h-8 animate-spin text-[#4361EE]" />
                        <p className="text-gray-500 font-medium animate-pulse">Analyzing data structure...</p>
                    </div>
                ) : summary && (
                    <div className="flex flex-col">

                        <div className="flex gap-3 mb-4">
                            <FiDatabase className="w-7 h-7 text-[#2EC4B6]" />
                            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Dataset summary</h2>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

                            {/* CARD 1: Dataset Scale (1/3 chiều rộng) */}
                            <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 flex flex-col justify-center h-fit">
                                <div className="flex items-center gap-2 mb-5 font-bold text-gray-800 dark:text-gray-200">
                                    <FiLayers className="w-5 h-5 text-[#4361EE]" />
                                    <h3 className="text-lg font-bold text-gray-900 dark:text-white">Dataset scale</h3>
                                </div>

                                <div className="flex flex-col gap-4">
                                    <div className="flex gap-4 flex-wrap">
                                        <div className="flex-1 bg-gray-50 dark:bg-gray-900/50 rounded-xl p-4 flex flex-col justify-center border border-gray-100/50 dark:border-gray-700/50">
                                            <span className="text-[10px] font-bold text-gray-400 dark:text-gray-400 uppercase tracking-widest mb-1">Total Rows</span>
                                            <span className="text-3xl font-black text-[#4361EE]">{summary.rows.toLocaleString()}</span>
                                        </div>
                                        <div className="flex-1 bg-gray-50 dark:bg-gray-900/50 rounded-xl p-4 flex flex-col justify-center border border-gray-100/50 dark:border-gray-700/50">
                                            <span className="text-[10px] font-bold text-gray-400 dark:text-gray-400 uppercase tracking-widest mb-1">Total Columns</span>
                                            <span className="text-3xl font-black text-[#9B5DE5]">{summary.columns}</span>
                                        </div>
                                    </div>

                                    {processedRowCount !== null && processedRowCount !== summary.rows && (
                                        <div className="w-full bg-gray-50 dark:bg-gray-900/50 rounded-xl p-4 flex items-center justify-between border border-gray-100/50 dark:border-gray-700/50">
                                            <div className="flex flex-col">
                                                <span className="text-[10px] font-bold text-gray-400 dark:text-gray-400 uppercase tracking-widest mb-1">
                                                    Cleaned rows
                                                </span>
                                                <div className="flex items-baseline gap-2">
                                                    <span className="text-3xl font-black text-[#2EC4B6]">
                                                        {processedRowCount.toLocaleString()}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* CARD 2: Missing Values / Data Quality */}
                            <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 md:col-span-2 flex flex-col relative overflow-hidden">

                                {Object.values(summary.missing_values).some(v => v > 0) || duplicatesRemoved > 0 ? (
                                    // --- TRẠNG THÁI CÓ LỖI ---
                                    <div className="flex flex-col gap-4">
                                        {/* Header: Tiêu đề & Badge Attention */}
                                        <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
                                            <div className="flex items-center gap-2 font-bold text-gray-800 dark:text-gray-200">
                                                <FiAlertCircle className="w-5 h-5 text-red-600" />
                                                <h3 className="text-lg font-bold text-gray-900 dark:text-white">Data quality</h3>
                                            </div>
                                        </div>

                                        {/* Khối Duplicates */}
                                        {duplicatesRemoved > 0 && (
                                            <div className="flex items-center justify-between py-3 px-5 bg-orange-50/70 dark:bg-orange-900/10 rounded-xl border border-orange-100 dark:border-orange-800/30">
                                                <div className="flex items-center gap-3 text-orange-900 dark:text-orange-300 font-semibold">
                                                    <span>Duplicate rows:</span>
                                                </div>
                                                <span className="text-orange-600 dark:text-orange-500 font-bold">
                                                    {duplicatesRemoved} removed
                                                </span>
                                            </div>
                                        )}

                                        {/* Lưới Missing Values */}
                                        {Object.values(summary.missing_values).some(v => v > 0) && (
                                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-h-[140px] overflow-y-auto pr-2 custom-scrollbar">
                                                {Object.entries(summary.missing_values)
                                                    .filter(([_, v]) => v > 0)
                                                    .map(([key, val]) => (
                                                        <div key={key} className="flex flex-wrap items-center justify-between py-3 px-5 bg-red-50/50 dark:bg-red-900/10 rounded-xl border border-red-100 dark:border-red-800/30">
                                                            <div className="flex items-center gap-2.5 text-gray-800 dark:text-gray-200">
                                                                <div className="w-1.5 h-1.5 rounded-full bg-red-400"></div>
                                                                <span>{key}:</span>
                                                            </div>
                                                            <span className="text-red-500">
                                                                {val} rows missing
                                                            </span>
                                                        </div>
                                                    ))}
                                            </div>
                                        )}

                                        <div className="text-sm text-gray-500 dark:text-gray-400">
                                            * Data integrity scan identified inconsistencies. Automated cleanup applied where possible.
                                        </div>
                                    </div>
                                ) : (
                                    // --- TRẠNG THÁI HOÀN HẢO (Không có lỗi) ---
                                    <>
                                        <div className="flex items-center gap-2 mb-4 font-semibold text-lg">
                                            <FiAlertCircle className="w-5 h-5" />
                                            <span>Data quality</span>
                                        </div>
                                        <div className="flex items-center gap-5 mt-2">
                                            <div className="w-12 h-12 rounded-full bg-[#2EC4B6]/15 flex items-center justify-center shrink-0">
                                                <FiCheck className="w-6 h-6 text-[#2EC4B6]" strokeWidth={3} />
                                            </div>
                                            <div className="z-10">
                                                <h4 className="text-xl font-bold text-[#1f8c82] dark:text-[#2EC4B6] mb-1">Perfect data integrity</h4>
                                                <p className="text-sm text-gray-500 dark:text-gray-400">No missing values or duplicates were detected in the dataset.</p>
                                            </div>
                                        </div>
                                    </>
                                )}
                            </div>
                        </div>

                        {/* --- Features Details (Chia 2 cột) --- */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">

                            {/* CARD 3: Numerical Features */}
                            <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700">
                                <div className="flex items-center justify-between mb-6">
                                    <div className="flex items-center">
                                        <div className="w-1.5 h-6 bg-[#4361EE] rounded-full mr-3"></div>
                                        <h3 className="text-lg font-bold text-gray-900 dark:text-white">Numerical features</h3>
                                    </div>
                                </div>
                                <div className="flex flex-wrap gap-3">
                                    {summary.numerical_features.map(feat => (
                                        <div key={feat} className="flex items-center gap-2 px-3 py-2 bg-gray-50 dark:bg-gray-900/60 rounded-lg border border-gray-100 dark:border-gray-800 hover:border-gray-200 dark:hover:border-gray-700 transition-colors">
                                            <span className="text-sm font-semibold text-gray-700 dark:text-gray-300">{feat}</span>
                                            <span className="text-[10px] text-gray-600 dark:text-gray-300">({summary.column_types?.[feat] || 'N/A'})</span>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* CARD 4: Categorical Features*/}
                            <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700">
                                <div className="flex items-center justify-between mb-6">
                                    <div className="flex items-center">
                                        <div className="w-1.5 h-6 bg-[#9B5DE5] rounded-full mr-3"></div>
                                        <h3 className="text-lg font-bold text-gray-900 dark:text-white">Categorical features</h3>
                                    </div>
                                </div>
                                <div className="flex flex-wrap gap-3">
                                    {summary.categorical_features.map(feat => (
                                        <div key={feat} className="flex items-center gap-2 px-3 py-2 bg-gray-50 dark:bg-gray-900/60 rounded-lg border border-gray-100 dark:border-gray-800 hover:border-gray-200 dark:hover:border-gray-700 transition-colors">
                                            <span className="text-sm font-semibold text-gray-700 dark:text-gray-300">{feat}</span>
                                            <span className="text-[10px] text-gray-600 dark:text-gray-300">({summary.column_types?.[feat] || 'N/A'})</span>
                                        </div>
                                    ))}
                                </div>
                            </div>

                        </div>
                    </div>
                )}

                {/* --- DATA PREVIEW (TRƯỚC & SAU) --- */}
                {isLoadingPreprocess ? (
                    <div className="flex gap-2 items-center justify-center py-12 rounded-2xl border border-gray-100 dark:border-gray-700">
                        <FiLoader className="w-8 h-8 animate-spin text-[#2EC4B6]" />
                        <p className="text-gray-500 font-medium animate-pulse">Optimizing and preparing dataset...</p>
                    </div>
                ) : processedId && (
                    <div className="mb-8">
                        <div className="flex gap-3 mb-4">
                            <FiGrid className="w-7 h-7 text-[#2EC4B6]" />
                            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Dataset comparison</h2>
                        </div>

                        <div className="px-0 md:px-6 space-y-10">

                            {/* Bảng Dữ Liệu Gốc */}
                            <div className="space-y-4">
                                <div className="flex items-center justify-between">
                                    <h3 className="text-lg font-bold text-gray-800 dark:text-gray-200 flex items-center gap-2">
                                        <span className="w-3 h-3 rounded-full bg-gray-400"></span> Original data
                                    </h3>
                                </div>

                                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                                    {/* Vùng Bảng (Chiếm 2/3) */}
                                    <div className="lg:col-span-2">
                                        {renderDataTable(originalRows, 'original', loadingMoreOrg, hasMoreOriginal)}
                                    </div>

                                    {/* Vùng Thông tin (Chiếm 1/3) */}
                                    <div className="bg-gray-50 dark:bg-gray-800/50 p-5 rounded-xl border border-gray-200 dark:border-gray-700 h-fit">
                                        <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3 text-base border-b border-gray-200 dark:border-gray-700 pb-2">
                                            About original dataset
                                        </h4>
                                        <p className="text-sm text-gray-600 dark:text-gray-300 mb-4 leading-relaxed">
                                            This is your raw, untouched dataset exactly as it was uploaded. It serves as the baseline to verify the integrity of the automated preprocessing steps.
                                        </p>
                                        <ul className="text-sm text-gray-600 dark:text-gray-300 space-y-2">
                                            <li className="flex items-start gap-2">
                                                <span className="text-gray-400">•</span>
                                                <span><strong>Raw format:</strong> Features retain their original scales, string text, and categorical labels.</span>
                                            </li>
                                            <li className="flex items-start gap-2">
                                                <span className="text-gray-400">•</span>
                                                <span><strong>Potential issues:</strong> May contain duplicate rows, missing values (NaN/Null), or unencoded variables.</span>
                                            </li>
                                            <li className="flex items-start gap-2">
                                                <span className="text-gray-400">•</span>
                                                <span><strong>Algorithm readiness:</strong> Not yet optimized for training machine learning models.</span>
                                            </li>
                                        </ul>
                                    </div>
                                </div>
                            </div>

                            {/* Bảng Dữ Liệu Đã Xử Lý */}
                            <div className="space-y-4">
                                <div className="flex items-center justify-between">
                                    <h3 className="text-lg font-bold text-gray-800 dark:text-gray-200 flex items-center gap-2">
                                        <span className="w-3 h-3 rounded-full bg-[#2EC4B6]"></span> Preprocessed data
                                    </h3>

                                    <button
                                        onClick={handleDownloadProcessed}
                                        disabled={isDownloading || !processedId}
                                        className="flex items-center hover:cursor-pointer justify-center min-w-[180px] gap-2 px-4 py-2 bg-linear-to-r from-[#2EC4B6] to-[#25a095] text-white text-sm font-medium rounded-lg shadow-sm hover:shadow-md hover:from-[#25a095] hover:to-[#1e8278] transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-sm disabled:hover:from-[#2EC4B6] disabled:hover:to-[#25a095]"
                                    >
                                        <FiDownload className="w-4 h-4" />
                                        {isDownloading ? "Downloading..." : "Download dataset"}
                                    </button>
                                </div>

                                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                                    {/* Vùng Bảng (Chiếm 2/3) */}
                                    <div className="lg:col-span-2">
                                        {renderDataTable(processedRows, 'processed', loadingMoreProc, hasMoreProcessed)}
                                    </div>

                                    {/* Vùng Thông tin (Chiếm 1/3) */}
                                    <div className="bg-[#2EC4B6]/5 dark:bg-[#2EC4B6]/10 p-5 rounded-xl border border-[#2EC4B6]/20 dark:border-[#2EC4B6]/30 h-fit">
                                        <h4 className="font-semibold text-[#1f8c82] dark:text-[#2EC4B6] mb-3 text-base border-b border-[#2EC4B6]/20 pb-2">
                                            About preprocessed dataset
                                        </h4>
                                        <p className="text-sm text-gray-600 dark:text-gray-300 mb-4 leading-relaxed">
                                            The dataset has been automatically cleaned and standardized through our pipeline to ensure optimal model performance:
                                        </p>
                                        <ul className="text-sm text-gray-600 dark:text-gray-300 space-y-2">
                                            <li className="flex items-start gap-2">
                                                <span className="text-[#2EC4B6] font-bold">✓</span>
                                                <span><strong>Data cleaning:</strong> Dropped rows with missing target values and removed exact duplicates to prevent bias.</span>
                                            </li>
                                            <li className="flex items-start gap-2">
                                                <span className="text-[#2EC4B6] font-bold">✓</span>
                                                <span><strong>Imputation:</strong> <em>{getImputationInfo(imputation).label}</em> - {getImputationInfo(imputation).detail}</span>
                                            </li>
                                            <li className="flex items-start gap-2">
                                                <span className="text-[#2EC4B6] font-bold">✓</span>
                                                <span><strong>Encoding:</strong> Applied <em>Label Encoding</em> to convert categorical strings into machine-readable numbers.</span>
                                            </li>
                                            <li className="flex items-start gap-2">
                                                <span className="text-[#2EC4B6] font-bold">✓</span>
                                                <span><strong>Scaling:</strong> Used <em>Standard Scaler</em> on numerical features to ensure equal contribution across variables (Mean=0, Variance=1).</span>
                                            </li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* SECTION EDA PLOTS */}
                {isLoadingEda ? (
                    <div className="flex gap-2 items-center justify-center py-16 rounded-2xl border border-gray-100 dark:border-gray-700">
                        <FiLoader className="w-8 h-8 animate-spin text-[#9B5DE5]" />
                        <p className="text-gray-500 font-medium animate-pulse">Generating analytical charts...</p>
                    </div>
                ) : Object.keys(charts).length > 0 && (
                    <div className="mx-auto w-full mb-6">
                        <div className="flex items-center gap-3 my-4">
                            <FiPieChart className="w-7 h-7 text-[#4361EE]" />
                            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Exploratory data analysis</h2>
                        </div>

                        <div className="grid grid-cols-1 gap-6 px-6">
                            {chartConfigurations.map((config, idx) => {
                                const imgUrl = charts[config.key];
                                if (!imgUrl) return null;

                                return (
                                    <div key={config.key} className="w-full">
                                        {renderChartImage(imgUrl, config.title, config.description, idx)}
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}
            </div>

            <ImageModal
                imageUrl={selectedImage}
                onClose={() => setSelectedImage(null)}
            />
        </>
    );
}