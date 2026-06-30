import { FiActivity, FiX } from "react-icons/fi";
import Image from "next/image";
import { useTranslations } from "next-intl";

export interface PatientRow {
    prediction_result?: number;
    prediction_probability?: number;
    [key: string]: any;
}

interface PatientDetailModalProps {
    patient: PatientRow | null;
    onClose: () => void;
    resultKey: string;
    probabilityKey: string;
    hiddenKeys: Set<string>;
    loadingXAI: boolean;
    patientXAI: any;
    onImageClick: (url: string) => void;
}

export default function PatientDetailModal({ patient, onClose, resultKey, probabilityKey, hiddenKeys, loadingXAI, patientXAI, onImageClick }: PatientDetailModalProps) {
    const t = useTranslations("predictionBatchHistoryDetail");

    if (!patient) return null;

    const selectedResultValue = patient[resultKey] ?? patient.prediction_result;
    const selectedProbabilityValue = patient[probabilityKey] ?? patient.prediction_probability;
    const isSelectedHighRisk = Number(selectedResultValue) === 1;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
            <div className="bg-white dark:bg-gray-800 rounded-2xl w-full max-w-5xl max-h-[90vh] flex flex-col shadow-2xl relative">
                {/* Header Modal */}
                <div className="p-6 border-b border-gray-100 dark:border-gray-700 flex justify-between items-center">
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                        <FiActivity className="text-blue-500" />
                        {t("modal.title")}
                    </h2>
                    <button
                        onClick={onClose}
                        className="p-2 text-gray-400 hover:cursor-pointer hover:text-gray-700 hover:bg-gray-100 dark:hover:text-white dark:hover:bg-gray-700 rounded-full transition-colors"
                    >
                        <FiX className="w-6 h-6" />
                    </button>
                </div>

                {/* Body Modal */}
                <div className="p-6 overflow-y-auto">
                    {/* Kết quả tóm tắt */}
                    <div className={`p-5 rounded-xl mb-6 flex justify-between items-center border ${isSelectedHighRisk
                        ? 'bg-red-50 border-red-200 dark:bg-red-900/10 dark:border-red-800/50'
                        : 'bg-green-50 border-green-200 dark:bg-green-900/10 dark:border-green-800/50'
                        }`}>
                        <div>
                            <p className="text-sm font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">{t("modal.aiPrediction")}</p>
                            <p className={`text-2xl font-bold ${isSelectedHighRisk ? 'text-red-600 dark:text-red-500' : 'text-green-600 dark:text-green-500'}`}>
                                {isSelectedHighRisk ? t("modal.highRisk") : t("modal.lowRisk")}
                            </p>
                        </div>
                        <div className="text-right">
                            <p className="text-sm font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">{t("modal.confidenceScore")}</p>
                            <p className="text-3xl font-bold text-gray-900 dark:text-white">
                                {((selectedProbabilityValue || 0) * 100).toFixed(1)}%
                            </p>
                        </div>
                    </div>

                    {/* Thông tin bệnh nhân */}
                    <div className="mb-6 p-5 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
                        <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">{t("modal.patientInfo")}</h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-y-4 gap-x-4">
                            {Object.entries(patient)
                                .filter(([key]) => !hiddenKeys.has(key))
                                .map(([key, value]) => (
                                    <div key={key}>
                                        <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase wrap-break-word">
                                            {key.replace(/_/g, ' ')}
                                        </p>
                                        <p className="text-base font-semibold text-gray-900 dark:text-white truncate">
                                            {value !== null && value !== undefined && value !== '' ? String(value) : t("patientList.na")}
                                        </p>
                                    </div>
                                ))}
                        </div>
                    </div>

                    {/* XAI */}
                    {loadingXAI ? (
                        <div className="flex flex-col items-center justify-center py-16 bg-gray-50 dark:bg-gray-900/50 rounded-xl border border-dashed border-gray-200 dark:border-gray-700">
                            <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
                            <p className="text-gray-600 dark:text-gray-400 font-medium">{t("state.generatingXai")}</p>
                            <p className="text-sm text-gray-400 mt-2">{t("state.generatingXaiDesc")}</p>
                        </div>
                    ) : patientXAI ? (
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                            {/* 1. SHAP Waterfall */}
                            {patientXAI.shap_waterfall && (
                                <div className="border border-gray-200 dark:border-gray-700 rounded-xl p-3 bg-white dark:bg-gray-800">
                                    <div className="mb-2 text-center">
                                        <p className="text-sm font-bold text-gray-800 dark:text-gray-200">{t("modal.charts.shapWaterfallTitle")}</p>
                                    </div>
                                    <div
                                        className="relative w-full aspect-square bg-gray-50 dark:bg-gray-900 rounded-lg overflow-hidden cursor-pointer hover:opacity-90"
                                        onClick={() => onImageClick(patientXAI.shap_waterfall)}
                                    >
                                        <Image src={patientXAI.shap_waterfall} loading="lazy" alt="SHAP Waterfall" fill sizes="(max-width: 1024px) 100vw, 33vw" className="object-contain p-2" />
                                        <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 bg-black/10 transition-opacity">
                                            <span className="bg-black/50 text-white text-xs px-2 py-1 rounded-md">{t("xai.clickToExpand")}</span>
                                        </div>
                                    </div>
                                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                                        {t("modal.charts.shapWaterfallDesc")}
                                    </p>
                                </div>
                            )}

                            {/* 2. SHAP Bar */}
                            {patientXAI.shap_bar && (
                                <div className="border border-gray-200 dark:border-gray-700 rounded-xl p-3 bg-white dark:bg-gray-800">
                                    <div className="mb-2 text-center">
                                        <p className="text-sm font-bold text-gray-800 dark:text-gray-200">{t("modal.charts.shapBarTitle")}</p>
                                    </div>
                                    <div
                                        className="relative w-full aspect-square bg-gray-50 dark:bg-gray-900 rounded-lg overflow-hidden cursor-pointer hover:opacity-90"
                                        onClick={() => onImageClick(patientXAI.shap_bar)}
                                    >
                                        <Image src={patientXAI.shap_bar} loading="lazy" alt="SHAP Bar" fill sizes="(max-width: 1024px) 100vw, 33vw" className="object-contain p-2" />
                                        <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 bg-black/10 transition-opacity">
                                            <span className="bg-black/50 text-white text-xs px-2 py-1 rounded-md">{t("xai.clickToExpand")}</span>
                                        </div>
                                    </div>
                                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                                        {t("modal.charts.shapBarDesc")}
                                    </p>
                                </div>
                            )}

                            {/* 3. LIME Analysis */}
                            {patientXAI.lime && (
                                <div className="border border-gray-200 dark:border-gray-700 rounded-xl p-3 bg-white dark:bg-gray-800">
                                    <div className="mb-2 text-center">
                                        <p className="text-sm font-bold text-gray-800 dark:text-gray-200">{t("modal.charts.limeTitle")}</p>
                                    </div>
                                    <div
                                        className="relative w-full aspect-square bg-gray-50 dark:bg-gray-900 rounded-lg overflow-hidden cursor-pointer hover:opacity-90"
                                        onClick={() => onImageClick(patientXAI.lime)}
                                    >
                                        <Image src={patientXAI.lime} loading="lazy" alt="LIME" fill sizes="(max-width: 1024px) 100vw, 33vw" className="object-contain p-2" />
                                        <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 bg-black/10 transition-opacity">
                                            <span className="bg-black/50 text-white text-xs px-2 py-1 rounded-md">{t("xai.clickToExpand")}</span>
                                        </div>
                                    </div>
                                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                                        {t("modal.charts.limeDesc")}
                                    </p>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="text-center py-12 text-gray-500">
                            {t("state.xaiError")}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}