"use client";
import { useEffect, useState, useRef, UIEvent } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { useAuth } from "@/context/authcontext"
import { FiLoader, FiPieChart, FiAlertCircle, FiCheck, FiDatabase } from "react-icons/fi";
import { HiOutlineDatabase } from "react-icons/hi";
import Image from "next/image";
import toast from "react-hot-toast";
import { api } from "@/lib/api";
import ImageModal from "@/components/imageModal";

interface DatasetSummary {
    rows: number;
    columns: number;
    target_column: string;
    categorical_features: string[];
    numerical_features: string[];
    column_types: Record<string, string>;
    missing_values: Record<string, number>;
}

export default function EDA() {
    const searchParams = useSearchParams();
    const router = useRouter();

    const datasetId = searchParams.get("id");
    const targetColumn = searchParams.get("target");

    const [isLoading, setIsLoading] = useState(true);
    const [summary, setSummary] = useState<DatasetSummary | null>(null);
    const [charts, setCharts] = useState<Record<string, string>>({});
    const [loadingStep, setLoadingStep] = useState("");
    const [selectedImage, setSelectedImage] = useState<string | null>(null);
    const [duplicatesRemoved, setDuplicatesRemoved] = useState<number>(0);
    const { user, loading } = useAuth();
    const fetchedDatasetId = useRef<string | null>(null);
    const [processedRowCount, setProcessedRowCount] = useState<number | null>(null);

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

    // ref theo dõi lần đầu tiên vào trang
    const isFirstMount = useRef(true);

    const extractRows = (data: any) => {
        if (Array.isArray(data)) return data;
        return data?.data || data?.rows || data?.items || [];
    };

    useEffect(() => {
        if (loading) return; 
        
        if (!user) {
            // Chỉ hiện thông báo lỗi nếu đây là lúc vừa mới load vào trang
            if (isFirstMount.current) {
                toast.error("Please login first.");
                isFirstMount.current = false;
            }
            
            router.push("/login");
            return;
        }

        // Nếu code chạy được đến đây (tức là có user), tắt cờ first mount
        isFirstMount.current = false;
    }, [user, loading, router]);

    useEffect(() => {
        if (loading || !user) return;

        if (!datasetId || !targetColumn) {
            toast.error("Missing analysis information.");
            router.push("/");
            return;
        }

        if (fetchedDatasetId.current === datasetId) {
            return;
        }

        const fetchData = async () => {
            fetchedDatasetId.current = datasetId;

            try {
                setIsLoading(true);

                setLoadingStep("Analyzing data structure...");
                const summaryRes = await api.get(`/datasets/${datasetId}/summary`, {
                    params: { target_column: targetColumn }
                });
                setSummary(summaryRes.data);

                setLoadingStep("Optimizing dataset...");
                const preprocessRes = await api.post(`/datasets/${datasetId}/preprocess`, null, {
                    params: { target_column: targetColumn }
                });

                const procId = preprocessRes.data.processed_dataset_id;
                setProcessedId(procId);
                setDuplicatesRemoved(preprocessRes.data.duplicates_removed || 0);
                setProcessedRowCount(preprocessRes.data.rows || null);

                setLoadingStep("Fetching charts from the system...");
                const [edaRes, orgRowsRes, procRowsRes] = await Promise.all([
                    api.get(`/datasets/${procId}/eda`, { params: { target_column: targetColumn } }),
                    api.get(`/datasets/${datasetId}/rows`, { params: { limit: 10, offset: 0 } }),
                    api.get(`/datasets/${procId}/rows`, { params: { limit: 10, offset: 0 } })
                ]);

                setCharts(edaRes.data.charts || {});
                setOriginalRows(extractRows(orgRowsRes.data));
                setProcessedRows(extractRows(procRowsRes.data));

                toast.success("Analysis completed successfully!");
            } catch (error) {
                console.error("EDA Error:", error);
                toast.error("Failed to load analysis data.");
                fetchedDatasetId.current = null;
            } finally {
                setIsLoading(false);
            }
        };

        fetchData();
    }, [user, loading, datasetId, targetColumn]);

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

    // --- HÀM BẮT SỰ KIỆN CUỘN BẢNG ---
    const handleTableScroll = (e: UIEvent<HTMLDivElement>, type: 'original' | 'processed') => {
        const { scrollTop, scrollHeight, clientHeight } = e.currentTarget;

        if (scrollHeight - scrollTop - clientHeight < 50) {
            loadMoreRows(type);
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
                        className="object-contain transition-transform duration-300 p-2 group-hover:scale-105"
                        loading="lazy"
                    />
                    <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-black/5 dark:bg-white/5 rounded-xl z-10">
                        <span className="bg-gray-900/70 text-white px-3 py-1.5 rounded-full text-xs font-medium backdrop-blur-sm shadow-sm">
                            Click to zoom
                        </span>
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

    // Hàm render bảng dữ liệu chung
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
                            <tr key={index} className="bg-white border-b border-gray-400/50 dark:bg-gray-800 dark:border-gray-700">
                                {columns.map(col => (
                                    <td key={col} className="px-4 py-2 border-r border-gray-400/50 dark:border-gray-700">
                                        {/* Format lại số thập phân cho gọn nếu quá dài */}
                                        {typeof row[col] === 'number' && !Number.isInteger(row[col])
                                            ? Number(row[col]).toFixed(4)
                                            : String(row[col])}
                                    </td>
                                ))}
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
                {!hasMore && data.length > 0 && (
                    <div className="py-3 text-center bg-gray-50 dark:bg-gray-900/50 border-t border-gray-100 dark:border-gray-700">
                        <span className="text-xs text-gray-400">End of dataset</span>
                    </div>
                )}
            </div>
        );
    };

    return (
        <>
            <div className="p-4 flex flex-col h-full relative">
                <div className="flex items-center gap-4 mb-8">
                    <div>
                        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Detailed Dataset Analysis</h1>
                        <p className="text-gray-500 dark:text-gray-400 mt-1">Target Feature: <span className="font-bold text-[#4361EE] uppercase bg-[#4361EE]/10 px-2 py-0.5 rounded ml-1">{targetColumn}</span></p>
                    </div>
                </div>

                {isLoading ? (
                    <div className="flex-1 flex flex-col items-center justify-center min-h-[400px]">
                        <FiLoader className="w-10 h-10 animate-spin text-[#4361EE] mb-4" />
                        <p className="text-lg font-medium animate-pulse">{loadingStep}</p>
                    </div>
                ) : (
                    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-2 duration-700">

                        {summary && (
                            <div className="flex flex-col gap-6">

                                {/* --- Dataset Scale & Missing Values --- */}
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

                                    {/* CARD 1: Dataset Scale (1/3 chiều rộng) */}
                                    <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 flex flex-col justify-center h-fit">
                                        <div className="flex items-center gap-2 mb-5 font-bold text-gray-800 dark:text-gray-200">
                                            <HiOutlineDatabase className="w-5 h-5 text-[#4361EE]" />
                                            <span>Dataset Scale</span>
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
                                                            Cleaned Rows
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
                                                        <span>Data Quality</span>
                                                    </div>
                                                </div>

                                                {/* Khối Duplicates */}
                                                {duplicatesRemoved > 0 && (
                                                    <div className="flex items-center justify-between py-3 px-5 bg-orange-50/70 dark:bg-orange-900/10 rounded-xl border border-orange-100 dark:border-orange-800/30">
                                                        <div className="flex items-center gap-3 text-orange-900 dark:text-orange-300 font-semibold">
                                                            <span>Duplicate Rows:</span>
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
                                                                <div key={key} className="flex items-center justify-between py-3 px-5 bg-red-50/50 dark:bg-red-900/10 rounded-xl border border-red-100 dark:border-red-800/30">
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
                                                    <span>Data Quality</span>
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
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

                                    {/* CARD 3: Numerical Features */}
                                    <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700">
                                        <div className="flex items-center justify-between mb-6">
                                            <div className="flex items-center">
                                                <div className="w-1.5 h-6 bg-[#4361EE] rounded-full mr-3"></div>
                                                <h3 className="text-lg font-bold text-gray-900 dark:text-white">Numerical Features</h3>
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
                                                <h3 className="text-lg font-bold text-gray-900 dark:text-white">Categorical Features</h3>
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
                        <div>
                            <div className="flex gap-3 mb-4">
                                <FiDatabase className="w-7 h-7 text-[#2EC4B6]" />
                                <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Dataset Comparation</h2>
                            </div>

                            <div className="px-6">
                                {/* Bảng Dữ Liệu Gốc */}
                                <div className="space-y-4 mb-4">
                                    <div className="flex items-center justify-between">
                                        <h3 className="text-lg font-bold text-gray-800 dark:text-gray-200 flex items-center gap-2">
                                            <span className="w-3 h-3 rounded-full bg-gray-400"></span> Original Data
                                        </h3>
                                    </div>
                                    {renderDataTable(originalRows, 'original', loadingMoreOrg, hasMoreOriginal)}
                                </div>

                                {/* Bảng Dữ Liệu Đã Xử Lý */}
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between">
                                        <h3 className="text-lg font-bold text-gray-800 dark:text-gray-200 flex items-center gap-2">
                                            <span className="w-3 h-3 rounded-full bg-[#2EC4B6]"></span> Preprocessed Data
                                        </h3>
                                    </div>
                                    {renderDataTable(processedRows, 'processed', loadingMoreProc, hasMoreProcessed)}
                                </div>
                            </div>
                        </div>

                        {/* SECTION EDA PLOTS */}
                        <div className="mx-auto w-full mb-6">
                            <div className="flex items-center gap-3 mb-4 mt-4">
                                <FiPieChart className="w-7 h-7 text-[#4361EE]" />
                                <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Exploratory Data Analysis</h2>
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
