"use client";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import Image from "next/image";
import { FaUser, FaNotesMedical } from "react-icons/fa6";
import { FaHeartbeat } from "react-icons/fa";
import { PredictionHistoryDetail } from "@/types/prediction";
import { useAuth } from "@/context/authcontext";
import toast from "react-hot-toast";

export default function PredictionDetailPage() {
    const params = useParams();
    const predictionId = params.predictionId;
    const [result, setResult] = useState<PredictionHistoryDetail | null>(null);
    const [loadingPrediction, setLoadingPrediction] = useState(true);
    const { loading, user } = useAuth();

    const GLOSSARY_DATA = [
        { term: "Age", definition: "Patient's age." },
        { term: "Sex", definition: "M: Male, F: Female." },
        { term: "Chest Pain Type", definition: "TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic." },
        { term: "Resting BP", definition: "Resting blood pressure (in mmHg on admission to the hospital)." },
        { term: "Cholesterol", definition: "Serum cholesterol in mm/dl." },
        { term: "Fasting BS", definition: "Fasting blood sugar. 1 if > 120 mg/dl, 0 otherwise." },
        { term: "Resting ECG", definition: "Resting electrocardiogram results: Normal, ST: ST-T wave abnormality, LVH: Left ventricular hypertrophy." },
        { term: "Max HR", definition: "Maximum heart rate achieved." },
        { term: "Exercise Angina", definition: "Exercise-induced angina. Y: Yes, N: No." },
        { term: "Oldpeak", definition: "ST depression induced by exercise relative to rest." },
        { term: "ST Slope", definition: "The slope of the peak exercise ST segment: Up, Flat, Down." },
    ];

    useEffect(() => {
        if (!loading && user) {
            const fetchDetail = async () => {
                try {
                    const res = await api.get(`/prediction-history/${predictionId}`);
                    setResult(res.data);
                } catch (error) {
                    toast.error("Prediction data not found");
                } finally {
                    setLoadingPrediction(false);
                }
            };
            if (predictionId) {
                fetchDetail();
            }
        }
    }, [predictionId, loading, user]);

    const formatDateTime = (dateString: string) => {
        const date = new Date(dateString);

        const parts = new Intl.DateTimeFormat('vi-VN', {
            hour: '2-digit',
            minute: '2-digit',
            day: '2-digit',
            month: '2-digit',
            year: 'numeric',
        }).formatToParts(date);

        const hour = parts.find(p => p.type === 'hour')?.value;
        const minute = parts.find(p => p.type === 'minute')?.value;
        const day = parts.find(p => p.type === 'day')?.value;
        const month = parts.find(p => p.type === 'month')?.value;
        const year = parts.find(p => p.type === 'year')?.value;

        return `${hour}:${minute} - ${day}/${month}/${year}`;
    };

    // Truyền URL img Supabase vào src
    const renderChartImage = (imageUrl: string, title: string) => {
        return (
            <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200 dark:bg-gray-800 dark:border-gray-700">
                <h4 className="text-center font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</h4>
                <div className="relative w-full max-w-2xl mx-auto overflow-hidden rounded-lg bg-gray-100 dark:bg-gray-900 min-h-[200px] flex items-center justify-center">
                    <Image
                        src={imageUrl}
                        alt={title}
                        width={0}
                        height={0}
                        sizes="100vw"
                        className="w-full h-auto object-contain hover:scale-102 transition-transform duration-300"
                        priority
                    />
                </div>
            </div>
        );
    };

    if (loadingPrediction) return (
        <div className="flex h-screen items-center justify-center">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500"></div>
        </div>
    );

    if (!result) return (
        <div className="flex flex-col h-screen items-center justify-center text-gray-500">
            <p>Patient data not found.</p>
        </div>
    );

    return (
        <div className="min-h-screen p-4">
            <h1 className="text-2xl mb-12 font-bold text-gray-900 dark:text-white">Analysis Report ({formatDateTime(result.created_at.toString())})</h1>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* LEFT COLUMN: Patient Info */}
                <div className="lg:col-span-1 space-y-6">
                    {/* 1. Patient Data Card */}
                    <div className="bg-white dark:bg-[#18181B] p-6 rounded-2xl shadow-sm border border-gray-200 dark:border-[#FFFFFF1A]">
                        <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                            <FaUser className="text-indigo-500" /> Patient info
                        </h3>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                            <InfoItem label="Age" value={result.input_data.age} />
                            <InfoItem label="Sex" value={result.input_data.sex} />
                            <InfoItem label="Chest Pain" value={result.input_data.chest_pain_type} />
                            <InfoItem label="Resting BP" value={`${result.input_data.resting_bp} mmHg`} />
                            <InfoItem label="Cholesterol" value={`${result.input_data.cholesterol} mm/dl`} />
                            <InfoItem label="Fasting BS" value={result.input_data.fasting_bs ? "> 120 mg/dl" : "< 120 mg/dl"} />
                            <InfoItem label="Resting ECG" value={result.input_data.resting_ecg} />
                            <InfoItem label="Max HR" value={result.input_data.max_hr} />
                            <InfoItem label="Exercise Angina" value={result.input_data.exercise_angina} />
                            <InfoItem label="Oldpeak" value={result.input_data.oldpeak} />
                            <InfoItem label="ST Slope" value={result.input_data.st_slope} />
                        </div>
                    </div>

                    {/* 2. Medical Glossary Card */}
                    <div className="bg-indigo-50/50 dark:bg-[#18181B] p-6 rounded-2xl shadow-sm border border-indigo-100 dark:border-[#FFFFFF1A]">
                        <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                            <FaNotesMedical className="text-indigo-500" /> Medical terms
                        </h3>
                        <div className="space-y-3 pr-2 custom-scrollbar">
                            {GLOSSARY_DATA.map((item, index) => (
                                <div key={index} className="text-sm border-b border-indigo-100 dark:border-gray-700 pb-2 last:border-0 last:pb-0">
                                    <p className="font-semibold text-indigo-700 dark:text-indigo-400 text-xs uppercase mb-1">{item.term}</p>
                                    <p>{item.definition}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* RIGHT COLUMN: Results & Charts */}
                <div className="lg:col-span-2">
                    <div className={`p-6 rounded-2xl border-l-8 shadow-lg mb-8 transition-all bg-white dark:bg-[#18181B] 
                                ${result.predicted_label === 1
                            ? 'border-red-500 shadow-red-500/10'
                            : 'border-green-500 shadow-green-500/10'
                        }`}>
                        <div className="flex flex-col md:flex-row justify-between items-center gap-6">
                            <div>
                                <h3 className="text-lg font-medium text-gray-500 dark:text-gray-400 flex items-center gap-2">
                                    <FaHeartbeat /> Diagnosis Prediction:
                                </h3>
                                <p className={`text-4xl font-extrabold mt-2 tracking-tight 
                                            ${result.predicted_label === 1
                                        ? 'text-red-600 dark:text-red-500'
                                        : 'text-green-600 dark:text-green-500'
                                    }`}>
                                    {result.predicted_label === 1 ? "HEART FAILURE RISK" : "NORMAL"}
                                </p>
                                <p className="text-sm mt-2 text-gray-500 dark:text-gray-400">
                                    Based on the provided clinical indicators.
                                </p>
                            </div>

                            <div className="flex flex-col items-center justify-center bg-gray-50 dark:bg-black/20 p-4 rounded-xl">
                                <span className="text-xs uppercase font-bold text-gray-400 mb-1">Confidence</span>
                                <div className="relative flex items-center justify-center">
                                    <span className={`text-3xl font-black`}>
                                        {(result.predicted_probability * 100).toFixed(1)}%
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* 2. Biểu đồ phân tích (XAI Charts) */}
                    <div className="mb-6">
                        <h3 className="text-xl font-bold mb-6 text-gray-900 dark:text-white flex items-center">
                            <span className="mr-2 bg-indigo-100 text-indigo-700 text-xs font-bold px-2.5 py-1 rounded-md dark:bg-indigo-500/20 dark:text-indigo-300">XAI MODEL</span>
                            AI Logic Explanation
                        </h3>

                        <div className="grid grid-cols-1 gap-8">
                            <div>
                                {renderChartImage(result.prediction_xai.shap_waterfall, "Feature Impact Analysis (SHAP Waterfall)")}
                                <p className="text-sm text-gray-500 mt-2 text-center italic dark:text-gray-200">
                                    Visualizes how individual factors shift the prediction from the baseline.
                                    <span className="font-bold text-red-500"> Red bars</span> indicate factors increasing heart failure risk,
                                    while <span className="font-bold text-blue-500"> Blue bars</span> indicate factors decreasing the risk.
                                </p>
                            </div>
                            <div>
                                {renderChartImage(result.prediction_xai.shap_bar, "Global Feature Importance (SHAP Bar)")}
                                <p className="text-sm text-gray-500 mt-2 text-center italic dark:text-gray-200">
                                    Ranks the health indicators by their absolute impact on this prediction. Longer bars mean the AI considered these factors most critical for this patient.
                                </p>
                            </div>
                            <div>
                                {renderChartImage(result.prediction_xai.lime, "Local Interpretation (LIME)")}
                                <p className="text-sm text-gray-500 mt-2 text-center italic dark:text-gray-200">
                                    Independent verification: Analyzing which specific features support a "High Risk" diagnosis versus those supporting a "Normal" diagnosis.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

const InfoItem = ({ label, value }: { label: string, value: string | number }) => (
    <div className="flex flex-col border-b border-gray-100 dark:border-gray-700/50 pb-2 last:border-0 last:pb-0">
        <span className="text-xs text-gray-400 uppercase font-semibold">{label}</span>
        <span className="text-gray-800 dark:text-gray-200 font-medium truncate" title={String(value)}>
            {value}
        </span>
    </div>
);
