"use client";
import { useEffect, useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import { FiDownload, FiUsers, FiAlertTriangle, FiCheckCircle, FiActivity, FiX } from "react-icons/fi";
import { api } from "@/lib/api";
import toast from "react-hot-toast";
import { BatchResult } from "@/types/batch";
import Image from "next/image";
import ImageModal from "@/components/imageModal";

interface PatientRow {
    prediction_result?: number;
    prediction_probability?: number;
    [key: string]: any;
}

export default function BatchResultPage() {
    const router = useRouter();
    const [resultData, setResultData] = useState<BatchResult | null>(null);
    const [originalFileType, setOriginalFileType] = useState<string>("csv");
    const [isDownloading, setIsDownloading] = useState(false);
    const [targetColumn, setTargetColumn] = useState<string>("");
    const hasInitializedRef = useRef(false);
    const loadedPagesRef = useRef<Set<number>>(new Set());

    // State quản lý Modal ảnh (toàn cục)
    const [modalImageUrl, setModalImageUrl] = useState<string | null>(null);

    // --- STATES CHO INFINITE SCROLL ---
    const [patients, setPatients] = useState<PatientRow[]>([]);
    const [page, setPage] = useState(0);
    const [loadingMore, setLoadingMore] = useState(false);
    const [hasMore, setHasMore] = useState(true);
    const limit = 12;

    // --- STATES CHO BỆNH NHÂN DETAIL MODAL & XAI ---
    const [selectedPatient, setSelectedPatient] = useState<PatientRow | null>(null);
    const [patientXAI, setPatientXAI] = useState<any>(null);
    const [loadingXAI, setLoadingXAI] = useState(false);

    // Xử lý Intersection Observer để kích hoạt infinite scroll
    const observer = useRef<IntersectionObserver | null>(null);
    const lastElementRef = useCallback((node: HTMLDivElement) => {
        if (loadingMore) return;
        if (observer.current) observer.current.disconnect();
        observer.current = new IntersectionObserver(entries => {
            if (entries[0].isIntersecting && hasMore) {
                setPage(prev => prev + 1);
            }
        });
        if (node) observer.current.observe(node);
    }, [loadingMore, hasMore]);

    // Khởi tạo data từ Session Storage
    useEffect(() => {
        if (hasInitializedRef.current) return;
        hasInitializedRef.current = true;
        const storedData = sessionStorage.getItem('batchPredictionResult');
        const storedFileType = sessionStorage.getItem('uploadFileType');
        const storedTargetColumn = sessionStorage.getItem('batchPredictionTargetColumn');

        if (storedData) {
            const parsedData = JSON.parse(storedData);
            setResultData(parsedData);
            setPatients([]);
            setPage(0);
            setHasMore(true);
            loadedPagesRef.current = new Set();
            if (storedFileType) {
                setOriginalFileType(storedFileType);
            }
            if (storedTargetColumn) {
                setTargetColumn(storedTargetColumn);
            }
            // Gọi fetch trang đầu tiên
            fetchPatients(parsedData.file_id, 0);
        } else {
            toast.error("No prediction data found.");
            router.push("/predict/batch");
        }
    }, [router]);

    // Lắng nghe thay đổi của page để load thêm data
    useEffect(() => {
        if (page > 0 && resultData?.file_id) {
            fetchPatients(resultData.file_id, page);
        }
    }, [page]);

    // Hàm gọi API lấy dữ liệu dòng
    const fetchPatients = async (fileId: string, currentPage: number) => {
        if (loadedPagesRef.current.has(currentPage)) return;
        loadedPagesRef.current.add(currentPage);
        try {
            setLoadingMore(true);
            const offset = currentPage * limit;
            const res = await api.get(`/datasets/${fileId}/rows`, {
                params: { limit, offset }
            });

            const newData = res.data.data;
            setPatients(prev => {
                const merged = [...prev, ...newData];
                if (newData.length < limit || merged.length >= res.data.total_rows) {
                    setHasMore(false);
                }
                return merged;
            });
        } catch (error) {
            loadedPagesRef.current.delete(currentPage);
            toast.error("Failed to load patient list");
        } finally {
            setLoadingMore(false);
        }
    };

    // Hàm gọi API tải dataset
    const handleDownloadProcessed = async () => {
        const processedId = resultData?.file_id;
        if (!processedId) return;

        try {
            setIsDownloading(true);
            const response = await api.get(`/datasets/${processedId}/download`, {
                responseType: 'blob',
                params: { file_type: originalFileType }
            });

            // Lấy đuôi file động dựa vào originalFileType (mặc định fallback về csv)
            const fileExtension = originalFileType.includes('xls') ? 'xlsx' : 'csv';

            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.download = `prediction_data.${fileExtension}`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
        } catch (error) {
            toast.error("Failed to download the dataset.");
        } finally {
            setIsDownloading(false);
        }
    };

    // Hàm xử lý khi click vào Card bệnh nhân
    const handleViewPatientDetail = async (patient: PatientRow) => {
        setSelectedPatient(patient);
        setPatientXAI(null); // Clear data XAI cũ

        try {
            setLoadingXAI(true);
            const res = await api.post(`/predictions/xai/on-demand`, patient);
            setPatientXAI(res.data);
        } catch (error) {
            toast.error("Failed to generate Explainable AI for this patient.");
        } finally {
            setLoadingXAI(false);
        }
    };

    if (!resultData) {
        return <div className="p-8 text-center text-gray-500">Loading results...</div>;
    }

    const { summary, batch_shap_bar, batch_shap_beeswarm, file_id } = resultData;
    const resultKey = targetColumn || "prediction_result";
    const probabilityKey = targetColumn ? `${targetColumn}_prediction_probability` : "prediction_probability";
    const hiddenKeys = new Set([resultKey, probabilityKey, "prediction_result", "prediction_probability"]);

    return (
        <div className="p-4 min-h-screen flex flex-col">
            {/* Header & Controls */}
            <div className="flex flex-col sm:flex-row sm:justify-between sm:items-start mb-8 pb-4 gap-4">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Prediction Results</h1>
                    <p className="text-gray-500 dark:text-gray-400 mt-1">
                        Overview of AI analysis results based on the uploaded patient data.
                    </p>
                </div>

                <button
                    onClick={handleDownloadProcessed}
                    disabled={isDownloading || !file_id}
                    className="flex items-center hover:cursor-pointer justify-center shrink-0 min-w-[200px] gap-2 px-5 py-2.5 bg-linear-to-r from-[#4361EE] to-[#3a52d5] text-white font-semibold rounded-xl shadow-md hover:shadow-lg hover:from-[#3a52d5] hover:to-[#2e41b0] transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    <FiDownload className="text-lg" />
                    {isDownloading ? "Downloading..." : "Download data"}
                </button>
            </div>

            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                {/* Total Patients */}
                <div className="bg-white dark:bg-gray-800 p-5 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm flex flex-col justify-between relative overflow-hidden">
                    <div className="flex justify-between items-center text-[#8B7E74] dark:text-gray-400">
                        <p className="text-xs font-bold tracking-wider uppercase">Total Patients</p>
                        <FiUsers className="w-5 h-5 text-[#8B7E74]/70 dark:text-gray-500" />
                    </div>
                    <p className="text-4xl font-bold text-[#111827] dark:text-white mt-2">{summary.total}</p>
                </div>

                {/* High Risk */}
                <div className="bg-white dark:bg-gray-800 p-5 rounded-xl border border-red-200 dark:border-red-900/50 shadow-sm flex flex-col justify-between relative overflow-hidden">
                    {/* Left Border */}
                    <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-[#B91C1C] dark:text-red-500"></div>

                    <div className="flex justify-between items-center text-[#8B7E74] dark:text-gray-400 pl-2">
                        <p className="text-xs font-bold tracking-wider uppercase">High Risk (Disease)</p>
                        <FiAlertTriangle className="w-5 h-5 text-[#B91C1C] dark:text-red-500" />
                    </div>
                    <p className="text-4xl font-bold text-[#B91C1C] dark:text-red-500 mt-2 pl-2 flex items-baseline gap-2">
                        {summary.disease}
                        <span className="text-2xl font-bold">({(summary.disease_ratio * 100).toFixed(1)}%)</span>
                    </p>
                </div>

                {/* Low Risk */}
                <div className="bg-white dark:bg-gray-800 p-5 rounded-xl border border-green-200 dark:border-green-900/50 shadow-sm flex flex-col justify-between relative overflow-hidden">
                    {/* Left Border */}
                    <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-[#047857] dark:text-green-500"></div>

                    <div className="flex justify-between items-center text-[#8B7E74] dark:text-gray-400 pl-2">
                        <p className="text-xs font-bold tracking-wider uppercase">Low Risk (Normal)</p>
                        <FiCheckCircle className="w-5 h-5 text-[#047857] dark:text-green-500" />
                    </div>
                    <p className="text-4xl font-bold text-[#047857] dark:text-green-500 mt-2 pl-2 flex items-baseline gap-2">
                        {summary.normal}
                        <span className="text-2xl font-bold">({(summary.normal_ratio * 100).toFixed(1)}%)</span>
                    </p>
                </div>
            </div>

            {/* Global XAI Charts */}
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 mt-4">Batch explainability analysis</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-10">
                {/* SHAP Bar Chart */}
                <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm flex flex-col">
                    <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-4">Feature Importance</h3>
                    <div
                        className="relative w-full aspect-video flex-1 bg-gray-50 dark:bg-gray-900 rounded-lg overflow-hidden cursor-pointer hover:opacity-90 transition-opacity"
                        onClick={() => setModalImageUrl(batch_shap_bar)}
                    >
                        <Image src={batch_shap_bar} alt="SHAP Bar Chart" fill sizes="(max-width: 1024px) 100vw, 50vw" className="object-contain p-2 priority" />
                        <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 bg-black/10 transition-opacity">
                            <span className="bg-black/50 text-white text-xs px-2 py-1 rounded-md">Click to expand</span>
                        </div>
                    </div>
                </div>

                {/* SHAP Beeswarm Chart */}
                <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm flex flex-col">
                    <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-4">Feature Impact Distribution</h3>
                    <div
                        className="relative w-full aspect-video flex-1 bg-gray-50 dark:bg-gray-900 rounded-lg overflow-hidden cursor-pointer hover:opacity-90 transition-opacity"
                        onClick={() => setModalImageUrl(batch_shap_beeswarm)}
                    >
                        <Image src={batch_shap_beeswarm} alt="SHAP Beeswarm Chart" fill sizes="(max-width: 1024px) 100vw, 50vw" className="object-contain p-2" priority />
                        <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 bg-black/10 transition-opacity">
                            <span className="bg-black/50 text-white text-xs px-2 py-1 rounded-md">Click to expand</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* --- DANH SÁCH BỆNH NHÂN --- */}
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">Patient detail list</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 xl:grid-cols-2 gap-4">
                {patients.map((patient, index) => {
                    const resultValue = patient[resultKey] ?? patient.prediction_result;
                    const isHighRisk = Number(resultValue) === 1;

                    const displayAge = patient.Age || patient.age || patient.age_years || 'N/A';
                    const displaySex = patient.Sex || patient.sex || patient.gender || 'N/A';
                    const prob = patient[probabilityKey] ?? patient.prediction_probability ?? 0;

                    return (
                        <div
                            key={index}
                            ref={index === patients.length - 1 ? lastElementRef : null}
                            onClick={() => handleViewPatientDetail(patient)}
                            className={`p-4 rounded-xl border cursor-pointer transition-all duration-300 hover:shadow-md dark:hover:shadow-lg ${isHighRisk
                                ? "border-red-200 dark:border-red-800 dark:hover:bg-red-950/20 dark:hover:border-red-700 dark:hover:shadow-red-900/30"
                                : "border-green-200 dark:border-green-800 dark:hover:bg-green-950/20 dark:hover:border-green-700 dark:hover:shadow-green-900/30"
                                }`}
                        >
                            <div className="flex justify-between items-start mb-3">
                                <span className="font-semibold text-gray-700 dark:text-gray-200">
                                    Patient #{index + 1}
                                </span>
                                <span className={`px-2.5 py-1 text-[10px] font-bold rounded-full uppercase tracking-wider ${isHighRisk ? "bg-red-200 text-red-800" : "bg-green-200 text-green-800"
                                    }`}>
                                    {isHighRisk ? "High Risk" : "Normal"}
                                </span>
                            </div>
                            <div className="text-sm text-gray-600 dark:text-gray-400 space-y-1.5">
                                <p className="flex justify-between"><span>Age:</span> <span className="font-medium text-gray-800 dark:text-gray-300">{displayAge}</span></p>
                                <p className="flex justify-between"><span>Sex:</span> <span className="font-medium text-gray-800 dark:text-gray-300">{displaySex}</span></p>
                                <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-700/50 flex justify-between">
                                    <span>Confidence:</span>
                                    <span className="font-bold text-gray-900 dark:text-white">{(prob * 100).toFixed(1)}%</span>
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Trạng thái Loading */}
            {loadingMore && (
                <div className="flex justify-center py-4">
                    <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                </div>
            )}

            {/* --- MODAL CHI TIẾT BỆNH NHÂN & XAI --- */}
            {selectedPatient && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
                    <div className="bg-white dark:bg-gray-800 rounded-2xl w-full max-w-5xl max-h-[90vh] flex flex-col shadow-2xl relative">
                        {(() => {
                            const selectedResultValue = selectedPatient[resultKey] ?? selectedPatient.prediction_result;
                            const selectedProbabilityValue = selectedPatient[probabilityKey] ?? selectedPatient.prediction_probability;
                            const isSelectedHighRisk = Number(selectedResultValue) === 1;

                            return (
                                <>

                                    {/* Header Modal */}
                                    <div className="p-6 border-b border-gray-100 dark:border-gray-700 flex justify-between items-center">
                                        <h2 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                                            <FiActivity className="text-blue-500" />
                                            Patient analysis
                                        </h2>
                                        <button
                                            onClick={() => setSelectedPatient(null)}
                                            className="p-2 text-gray-400 hover:cursor-pointer hover:text-gray-700 hover:bg-gray-100 dark:hover:text-white dark:hover:bg-gray-700 rounded-full transition-colors"
                                        >
                                            <FiX className="w-6 h-6" />
                                        </button>
                                    </div>

                                    {/* Body Modal */}
                                    <div className="p-6 overflow-y-auto">
                                        {/* Thanh tóm tắt kết quả */}
                                        <div className={`p-5 rounded-xl mb-6 flex justify-between items-center border ${isSelectedHighRisk
                                            ? 'bg-red-50 border-red-200 dark:bg-red-900/10 dark:border-red-800/50'
                                            : 'bg-green-50 border-green-200 dark:bg-green-900/10 dark:border-green-800/50'
                                            }`}>
                                            <div>
                                                <p className="text-sm font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">AI Prediction</p>
                                                <p className={`text-2xl font-bold ${isSelectedHighRisk ? 'text-red-600 dark:text-red-500' : 'text-green-600 dark:text-green-500'}`}>
                                                    {isSelectedHighRisk ? 'Heart Disease Detected' : 'Normal / Low risk'}
                                                </p>
                                            </div>
                                            <div className="text-right">
                                                <p className="text-sm font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">Confidence Score</p>
                                                <p className="text-3xl font-bold text-gray-900 dark:text-white">
                                                    {((selectedProbabilityValue || 0) * 100).toFixed(1)}%
                                                </p>
                                            </div>
                                        </div>

                                        {/* Thông tin bệnh nhân */}
                                        <div className="mb-6 p-5 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
                                            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">Patient information</h3>
                                            <div className="grid grid-cols-2 md:grid-cols-4 gap-y-4 gap-x-4">
                                                {Object.entries(selectedPatient)
                                                    .filter(([key]) => !hiddenKeys.has(key))
                                                    .map(([key, value]) => (
                                                        <div key={key}>
                                                            <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase wrap-break-word">
                                                                {key.replace(/_/g, ' ')}
                                                            </p>
                                                            <p className="text-base font-semibold text-gray-900 dark:text-white truncate">
                                                                {value !== null && value !== undefined && value !== '' ? String(value) : 'N/A'}
                                                            </p>
                                                        </div>
                                                    ))}
                                            </div>
                                        </div>

                                        {/* XAI */}
                                        {loadingXAI ? (
                                            <div className="flex flex-col items-center justify-center py-16 bg-gray-50 dark:bg-gray-900/50 rounded-xl border border-dashed border-gray-200 dark:border-gray-700">
                                                <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
                                                <p className="text-gray-600 dark:text-gray-400 font-medium">Generating AI Explainability...</p>
                                                <p className="text-sm text-gray-400 mt-2">This may take a few seconds as we analyze the features.</p>
                                            </div>
                                        ) : patientXAI ? (
                                            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

                                                {/* 1. SHAP Waterfall */}
                                                {patientXAI.shap_waterfall && (
                                                    <div className="border border-gray-200 dark:border-gray-700 rounded-xl p-3 bg-white dark:bg-gray-800">
                                                        <div className="mb-2 text-center">
                                                            <p className="text-sm font-bold text-gray-800 dark:text-gray-200">Risk factor breakdown</p>
                                                        </div>
                                                        <div
                                                            className="relative w-full aspect-square bg-gray-50 dark:bg-gray-900 rounded-lg overflow-hidden cursor-pointer hover:opacity-90"
                                                            onClick={() => setModalImageUrl(patientXAI.shap_waterfall)}
                                                        >
                                                            <Image src={patientXAI.shap_waterfall} loading="lazy" alt="SHAP Waterfall" fill sizes="(max-width: 1024px) 100vw, 33vw" className="object-contain p-2" />
                                                            <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 bg-black/10 transition-opacity">
                                                                <span className="bg-black/50 text-white text-xs px-2 py-1 rounded-md">Click to expand</span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                )}

                                                {/* 2. SHAP Bar */}
                                                {patientXAI.shap_bar && (
                                                    <div className="border border-gray-200 dark:border-gray-700 rounded-xl p-3 bg-white dark:bg-gray-800">
                                                        <div className="mb-2 text-center">
                                                            <p className="text-sm font-bold text-gray-800 dark:text-gray-200">Top influencing factors</p>
                                                        </div>
                                                        <div
                                                            className="relative w-full aspect-square bg-gray-50 dark:bg-gray-900 rounded-lg overflow-hidden cursor-pointer hover:opacity-90"
                                                            onClick={() => setModalImageUrl(patientXAI.shap_bar)}
                                                        >
                                                            <Image src={patientXAI.shap_bar} loading="lazy" alt="SHAP Bar" fill sizes="(max-width: 1024px) 100vw, 33vw" className="object-contain p-2" />
                                                            <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 bg-black/10 transition-opacity">
                                                                <span className="bg-black/50 text-white text-xs px-2 py-1 rounded-md">Click to expand</span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                )}

                                                {/* 3. LIME Analysis */}
                                                {patientXAI.lime && (
                                                    <div className="border border-gray-200 dark:border-gray-700 rounded-xl p-3 bg-white dark:bg-gray-800">
                                                        <div className="mb-2 text-center">
                                                            <p className="text-sm font-bold text-gray-800 dark:text-gray-200">Local feature impact</p>
                                                        </div>
                                                        <div
                                                            className="relative w-full aspect-square bg-gray-50 dark:bg-gray-900 rounded-lg overflow-hidden cursor-pointer hover:opacity-90"
                                                            onClick={() => setModalImageUrl(patientXAI.lime)}
                                                        >
                                                            <Image src={patientXAI.lime} loading="lazy" alt="LIME" fill sizes="(max-width: 1024px) 100vw, 33vw" className="object-contain p-2" />
                                                            <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 bg-black/10 transition-opacity">
                                                                <span className="bg-black/50 text-white text-xs px-2 py-1 rounded-md">Click to expand</span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        ) : (
                                            <div className="text-center py-12 text-gray-500">
                                                Could not generate explainability charts for this patient.
                                            </div>
                                        )}
                                    </div>
                                </>
                            );
                        })()}
                    </div>
                </div>
            )}

            <ImageModal
                imageUrl={modalImageUrl}
                onClose={() => setModalImageUrl(null)}
            />
        </div>
    );
}