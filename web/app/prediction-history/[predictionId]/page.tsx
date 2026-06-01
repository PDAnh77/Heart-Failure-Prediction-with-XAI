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
import ImageModal from "@/components/modals/imageModal";
import { useLocale, useTranslations } from "next-intl";

export default function PredictionDetailPage() {
    const t = useTranslations("predictionHistoryDetail");
    const locale = useLocale();
    const params = useParams();
    const predictionId = params.predictionId;
    const [result, setResult] = useState<PredictionHistoryDetail | null>(null);
    const [loadingPrediction, setLoadingPrediction] = useState(true);
    const { loading, user } = useAuth();

    const [modalImageUrl, setModalImageUrl] = useState<string | null>(null);

    const GLOSSARY_DATA = [
        { term: t("patientFields.age"), definition: t("glossary.age") },
        { term: t("patientFields.sex"), definition: t("glossary.sex") },
        { term: t("patientFields.chestPain"), definition: t("glossary.chestPainType") },
        { term: t("patientFields.restingBp"), definition: t("glossary.restingBp") },
        { term: t("patientFields.cholesterol"), definition: t("glossary.cholesterol") },
        { term: t("patientFields.fastingBs"), definition: t("glossary.fastingBs") },
        { term: t("patientFields.restingEcg"), definition: t("glossary.restingEcg") },
        { term: t("patientFields.maxHr"), definition: t("glossary.maxHr") },
        { term: t("patientFields.exerciseAngina"), definition: t("glossary.exerciseAngina") },
        { term: t("patientFields.oldpeak"), definition: t("glossary.oldpeak") },
        { term: t("patientFields.stSlope"), definition: t("glossary.stSlope") },
    ];

    useEffect(() => {
        if (!loading && user) {
            const fetchDetail = async () => {
                try {
                    const res = await api.get(`/predictions/${predictionId}`);
                    setResult(res.data);
                } catch (error) {
                    toast.error(t("toast.notFound"));
                } finally {
                    setLoadingPrediction(false);
                }
            };
            if (predictionId) {
                fetchDetail();
            }
        }
    }, [predictionId, loading, user, t]);

    const formatDateTime = (dateString: string) => {
        const date = new Date(dateString);

        const localeTag = locale === "vi" ? "vi-VN" : "en-US";
        const parts = new Intl.DateTimeFormat(localeTag, {
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

    const renderChartImage = (imageUrl: string, title: string) => {
        if (!imageUrl) return null;
        return (
            <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200 dark:bg-gray-800 dark:border-gray-700">
                <h4 className="text-center font-semibold mb-3 text-gray-700 dark:text-gray-300">{title}</h4>
                <div
                    className="relative w-full max-w-2xl mx-auto overflow-hidden rounded-lg bg-gray-100 dark:bg-gray-900 min-h-[200px] flex items-center justify-center cursor-pointer hover:opacity-90 transition-opacity"
                    onClick={() => setModalImageUrl(imageUrl)}
                >
                    <Image
                        src={imageUrl}
                        alt={title}
                        width={0}
                        height={0}
                        sizes="100vw"
                        className="w-full h-auto object-contain"
                        priority
                    />
                    <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 bg-black/10 transition-opacity z-10">
                        <span className="bg-black/50 text-white text-xs px-2 py-1 rounded-md">{t("clickToExpand")}</span>
                    </div>
                </div>
            </div>
        );
    };

    if (loadingPrediction) return (
        <div className="relative min-h-full">
            <div className="absolute inset-0 flex items-center justify-center bg-white/50 dark:bg-gray-900/50 backdrop-blur-sm z-10">
                <div className="py-3 flex justify-center gap-2 items-center">
                    <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                    <p className="text-gray-500">{t("loading")}</p>
                </div>
            </div>
        </div>
    );

    if (!result) {
        return (
            <div className="relative min-h-full">
                <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-500 bg-white/50 dark:bg-gray-900/50 backdrop-blur-sm z-10">
                    <p>{t("patientNotFound")}</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen p-4">
            <h1 className="text-2xl mb-8 font-bold text-gray-900 dark:text-white">{t("title")} ({formatDateTime(result.created_at.toString())})</h1>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* LEFT COLUMN: Patient Info */}
                <div className="lg:col-span-1 space-y-6">
                    {/* 1. Patient Data Card */}
                    <div className="bg-white dark:bg-[#18181B] p-6 rounded-2xl shadow-sm border border-gray-200 dark:border-[#FFFFFF1A]">
                        <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                            <FaUser className="text-indigo-500" /> {t("patientInfo")}
                        </h3>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                            <InfoItem label={t("patientFields.age")} value={result.input_data.age} />
                            <InfoItem label={t("patientFields.sex")} value={result.input_data.sex} />
                            <InfoItem label={t("patientFields.chestPain")} value={result.input_data.chest_pain_type} />
                            <InfoItem label={t("patientFields.restingBp")} value={`${result.input_data.resting_bp} mmHg`} />
                            <InfoItem label={t("patientFields.cholesterol")} value={`${result.input_data.cholesterol} mm/dl`} />
                            <InfoItem label={t("patientFields.fastingBs")} value={result.input_data.fasting_bs ? "> 120 mg/dl" : "< 120 mg/dl"} />
                            <InfoItem label={t("patientFields.restingEcg")} value={result.input_data.resting_ecg} />
                            <InfoItem label={t("patientFields.maxHr")} value={result.input_data.max_hr} />
                            <InfoItem label={t("patientFields.exerciseAngina")} value={result.input_data.exercise_angina} />
                            <InfoItem label={t("patientFields.oldpeak")} value={result.input_data.oldpeak} />
                            <InfoItem label={t("patientFields.stSlope")} value={result.input_data.st_slope} />
                        </div>
                    </div>

                    {/* 2. Medical Glossary Card */}
                    <div className="bg-indigo-50/50 dark:bg-[#18181B] p-6 rounded-2xl shadow-sm border border-indigo-100 dark:border-[#FFFFFF1A]">
                        <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                            <FaNotesMedical className="text-indigo-500" /> {t("medicalTerms")}
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
                                    <FaHeartbeat /> {t("diagnosisPrediction")}
                                </h3>
                                <p className={`text-4xl font-extrabold mt-2 tracking-tight 
                                            ${result.predicted_label === 1
                                        ? 'text-red-600 dark:text-red-500'
                                        : 'text-green-600 dark:text-green-500'
                                    }`}>
                                    {result.predicted_label === 1 ? t("heartFailureRisk") : t("normal")}
                                </p>
                                <p className="text-sm mt-2 text-gray-500 dark:text-gray-400">
                                    {t("clinicalIndicators")}
                                </p>
                            </div>

                            <div className="flex flex-col items-center justify-center bg-gray-50 dark:bg-black/20 p-4 rounded-xl">
                                <span className="text-xs uppercase font-bold text-gray-400 mb-1">{t("confidence")}</span>
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
                            <span className="mr-2 bg-indigo-100 text-indigo-700 text-xs font-bold px-2.5 py-1 rounded-md dark:bg-indigo-500/20 dark:text-indigo-300">{t("xaiModel")}</span>
                            {t("aiLogicExplanation")}
                        </h3>

                        <div className="grid grid-cols-1 gap-8">
                            <div>
                                {renderChartImage(result.prediction_xai.shap_waterfall, t("charts.shapWaterfallTitle"))}
                                <p className="text-sm text-gray-500 mt-2 text-center italic dark:text-gray-200">
                                    {t("charts.shapWaterfallDescription")}
                                    <span className="font-bold text-red-500"> {t("charts.shapWaterfallRed")}</span> {t("charts.shapWaterfallMiddle")}
                                    <span className="font-bold text-blue-500"> {t("charts.shapWaterfallBlue")}</span> {t("charts.shapWaterfallEnd")}
                                </p>
                            </div>
                            <div>
                                {renderChartImage(result.prediction_xai.shap_bar, t("charts.shapBarTitle"))}
                                <p className="text-sm text-gray-500 mt-2 text-center italic dark:text-gray-200">
                                    {t("charts.shapBarDescription")}
                                </p>
                            </div>
                            <div>
                                {renderChartImage(result.prediction_xai.lime, t("charts.limeTitle"))}
                                <p className="text-sm text-gray-500 mt-2 text-center italic dark:text-gray-200">
                                    {t("charts.limeDescription")}
                                </p>
                            </div>
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

const InfoItem = ({ label, value }: { label: string, value: string | number }) => (
    <div className="flex flex-col border-b border-gray-100 dark:border-gray-700/50 pb-2 last:border-0 last:pb-0">
        <span className="text-xs text-gray-400 uppercase font-semibold">{label}</span>
        <span className="text-gray-800 dark:text-gray-200 font-medium truncate" title={String(value)}>
            {value}
        </span>
    </div>
);