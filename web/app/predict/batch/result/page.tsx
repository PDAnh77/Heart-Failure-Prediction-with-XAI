"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { FiDownload, FiUsers, FiAlertTriangle, FiCheckCircle } from "react-icons/fi";
import { api } from "@/lib/api";
import toast from "react-hot-toast";
import { BatchResult } from "@/types/batch";
import Image from "next/image";
import ImageModal from "@/components/imageModal";

export default function BatchResultPage() {
    const router = useRouter();
    const [resultData, setResultData] = useState<BatchResult | null>(null);
    const [originalFileType, setOriginalFileType] = useState<string>("csv");
    const [isDownloading, setIsDownloading] = useState(false);

    // State quản lý việc mở/đóng Modal ảnh
    const [modalImageUrl, setModalImageUrl] = useState<string | null>(null);

    useEffect(() => {
        const storedData = sessionStorage.getItem('batchPredictionResult');
        const storedFileType = sessionStorage.getItem('uploadFileType');

        if (storedData) {
            setResultData(JSON.parse(storedData));
            if (storedFileType) {
                setOriginalFileType(storedFileType);
            }
        } else {
            toast.error("No prediction data found.");
            router.push("/predict/batch");
        }
    }, [router]);

    const handleDownloadProcessed = async () => {
        const processedId = resultData?.file_id;
        if (!processedId) return;

        try {
            setIsDownloading(true);
            const response = await api.get(`/datasets/${processedId}/download`, {
                responseType: 'blob',
                params: {
                    file_type: originalFileType
                }
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

    if (!resultData) {
        return <div className="p-8 text-center text-gray-500">Loading results...</div>;
    }

    const { summary, batch_shap_bar, batch_shap_beeswarm, file_id } = resultData;

    return (
        <div className="p-4 h-full flex flex-col">
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

            {/* Explainable AI Visualizations */}
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">AI explainability analysis</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-10">
                {/* SHAP Bar Chart */}
                <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm flex flex-col">
                    <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-4">Feature Importance</h3>
                    <div
                        className="relative w-full aspect-video flex-1 bg-gray-50 dark:bg-gray-900 rounded-lg overflow-hidden cursor-pointer hover:opacity-90 transition-opacity"
                        onClick={() => setModalImageUrl(batch_shap_bar)}
                    >
                        <Image
                            src={batch_shap_bar}
                            alt="SHAP Bar Chart"
                            fill
                            sizes="(max-width: 1024px) 100vw, 50vw"
                            className="object-contain p-2"
                        />
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
                        <Image
                            src={batch_shap_beeswarm}
                            alt="SHAP Beeswarm Chart"
                            fill
                            sizes="(max-width: 1024px) 100vw, 50vw"
                            className="object-contain p-2"
                        />
                        <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 bg-black/10 transition-opacity">
                            <span className="bg-black/50 text-white text-xs px-2 py-1 rounded-md">Click to expand</span>
                        </div>
                    </div>
                </div>
            </div>

            <ImageModal
                imageUrl={modalImageUrl}
                onClose={() => setModalImageUrl(null)}
            />
        </div>
    );
}