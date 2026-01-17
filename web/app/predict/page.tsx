"use client"
import { useRouter } from "next/navigation";
import { FormEvent, useState, useRef, useEffect } from "react";
import { PredictionResult } from "@/types/prediction";
import { Patient } from "@/types/patient"
import { useSettings } from "@/context/settingscontext";
import { useAuth } from "@/context/authcontext";
import Image from "next/image";
import { api } from "@/lib/api";
import toast from "react-hot-toast";
import { FaHeartbeat } from "react-icons/fa";

export default function Predict() {
    const router = useRouter();
    const [submitting, setSubmitting] = useState(false);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const [invalidFields, setInvalidFields] = useState<string[]>([]);
    const formRef = useRef<HTMLFormElement>(null);
    const autoFillBtnRef = useRef<HTMLButtonElement>(null);
    const topRef = useRef<HTMLDivElement>(null);
    const resultRef = useRef<HTMLDivElement>(null);
    const { user, loading: authLoading, logout, triggerRefreshHistory } = useAuth();
    const { savePrediction } = useSettings();

    const RequiredMark = () => <span className="text-red-500">*</span>;

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const fieldName = e.target.name;
        if (invalidFields.includes(fieldName)) {
            setInvalidFields(prev => prev.filter(item => item !== fieldName));
        }
    };

    const validateForm = (formData: FormData) => {
        const errors: string[] = [];
        const checkRange = (name: string, min: number, max: number) => {
            const value = formData.get(name);
            if (!value) {
                errors.push(name);
            } else {
                const num = Number(value);
                if (isNaN(num) || num < min || num > max) {
                    errors.push(name);
                }
            }
        };

        // Check required (select/text)
        const checkRequired = (name: string) => {
            const value = formData.get(name);
            if (!value || value === "") errors.push(name);
        };

        checkRequired("gender");
        checkRange("age", 1, 120);
        checkRequired("chest-pain-type");
        checkRange("resting-bp", 50, 250);
        checkRange("cholesterol", 0, 600);
        checkRequired("fasting-bs");
        checkRequired("resting-ecg");
        checkRange("max-hr", 60, 220);
        checkRequired("exercise-angina");
        checkRange("oldpeak", 0, 6.2);
        checkRequired("st-slope");

        setInvalidFields(errors);
        return errors.length === 0;
    };

    // Nếu có tên trong danh sách lỗi -> trả về class error
    const getInputClass = (fieldName: string) => {
        const baseStructure = "block w-full rounded-lg shadow-sm transition h-10 px-3 text-base outline-1 -outline-offset-1 md:text-base/6 focus:outline-2 focus:-outline-offset-2";
        if (invalidFields.includes(fieldName)) {
            return `${baseStructure} bg-red-50 text-red-800 placeholder:text-red-800/50 border-red-500 outline-red-500 focus:outline-red-600 dark:bg-red-200/10 dark:text-red-100 dark:placeholder:text-red-100/50 dark:border-red-500/50 dark:outline-red-500/50 dark:focus:outline-red-500/50`;
        }
        return `${baseStructure} bg-white text-gray-900 outline-gray-300 placeholder:text-gray-400 focus:outline-indigo-600 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500`;
    };

    const handleAutoFillForm = async () => {
        toast.dismiss();
        setInvalidFields([]); // Reset lỗi khi auto fill
        if (!user) {
            toast.error("Please sign in to continue.");
            router.push("/login");
            return;
        }
        if (!formRef.current) return;
        const form = formRef.current;

        try {
            if (autoFillBtnRef.current) {
                autoFillBtnRef.current.innerText = "Loading...";
                autoFillBtnRef.current.disabled = true;
            }

            const res = await api.get("/patients/rand");
            const patient: Patient = res.data.data[0];

            if (!patient) return;

            const setValue = (selector: string, value: any) => {
                const element = form.querySelector(selector) as HTMLInputElement | HTMLSelectElement;
                if (element) element.value = String(value);
            }

            setValue('#age', patient.age);
            setValue('#gender', patient.sex);
            setValue('#chest-pain-type', patient.chest_pain_type);
            setValue('#resting-bp', patient.resting_bp);
            setValue('#cholesterol', patient.cholesterol);
            setValue('#fasting-bs', patient.fasting_bs);
            setValue('#resting-ecg', patient.resting_ecg);
            setValue('#max-hr', patient.max_hr);
            setValue('#exercise-angina', patient.exercise_angina);
            setValue('#oldpeak', patient.oldpeak);
            setValue('#st-slope', patient.st_slope);

        } catch (error: any) {
            if (error.response?.status === 401) {
                logout();
                toast.error("Session expired. Please sign in again.");
                router.push("/login");
                return;
            }
            console.error(error.message);
            toast.error("Failed to auto-fill form.");
        } finally {
            if (autoFillBtnRef.current) {
                autoFillBtnRef.current.innerText = "Auto-fill sample";
                autoFillBtnRef.current.disabled = false;
            }
        }
    };

    const handleReset = () => {
        formRef.current?.reset();
        setResult(null);
        setInvalidFields([]);
        setSubmitting(false);
        toast.dismiss();

        topRef.current?.scrollIntoView({
            behavior: "smooth",
            block: "start",
        });
    };

    useEffect(() => {
        if (result && resultRef.current) {
            resultRef.current.scrollIntoView({
                behavior: "smooth",
                block: "start",
            });
        }
    }, [result]);

    const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        toast.dismiss();

        if (authLoading) {
            toast.error("Checking authentication, try again in a moment.");
            return;
        }

        if (!user) {
            toast.error("Please sign in to continue.");
            router.push("/login");
            return;
        }

        const formData = new FormData(e.currentTarget);

        if (!validateForm(formData)) {
            topRef.current?.scrollIntoView({ behavior: "smooth" });
            return;
        }

        setSubmitting(true);
        setResult(null);

        const payload = {
            age: parseInt(formData.get("age") as string),
            sex: formData.get("gender"),
            chest_pain_type: formData.get("chest-pain-type"),
            resting_bp: parseInt(formData.get("resting-bp") as string),
            cholesterol: parseInt(formData.get("cholesterol") as string),
            fasting_bs: parseInt(formData.get("fasting-bs") as string),
            resting_ecg: formData.get("resting-ecg"),
            max_hr: parseInt(formData.get("max-hr") as string),
            exercise_angina: formData.get("exercise-angina"),
            oldpeak: parseFloat(formData.get("oldpeak") as string),
            st_slope: formData.get("st-slope"),
            save_prediction: savePrediction
        };
        // console.log("Payload sending to API:", JSON.stringify(payload, null, 2));

        try {
            const res = await api.post("/predict", payload);
            const data: PredictionResult = res.data;
            setResult(data);
            // console.log("API result:", data);
            triggerRefreshHistory();
        } catch (error: any) {
            if (error.response?.status === 401) {
                toast.error("Session expired. Please sign in again.");
                logout();
                router.push("/login");
                return;
            }
            console.error(error);
            toast.error("An error occurred during prediction.");
        } finally {
            setSubmitting(false);
        }
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
                        loading="lazy"
                    />
                </div>
            </div>
        );
    };

    return (
        <div className="p-4">
            <h1 ref={topRef} className="text-2xl font-bold">Patient Information</h1>
            <p className="mt-1 text-base/6 text-gray-600 dark:text-gray-400">Please provide accurate patient information for better prediction results.</p>
            <div className="mt-4">
                <button
                    type="button"
                    onClick={handleAutoFillForm}
                    ref={autoFillBtnRef}
                    className="rounded-md px-4 py-2 bg-gray-200 text-sm font-medium dark:text-black hover:bg-gray-300 cursor-pointer"
                >
                    Auto-fill sample
                </button>
            </div>

            {/* Thêm noValidate */}
            <form ref={formRef} onSubmit={handleSubmit} noValidate className="mt-12 border-b border-gray-900/10 pb-8 px-2 md:px-8 dark:border-white/10">
                <div className="mt-10 grid grid-cols-1 gap-x-6 gap-y-8 md:grid-cols-6">
                    {/* --- Gender --- */}
                    <div className="md:col-span-3">
                        <label htmlFor="gender" className="block text-base/6 font-medium text-gray-900 dark:text-white">Gender <RequiredMark /></label>
                        <div className="mt-2">
                            <select
                                id="gender"
                                name="gender"
                                defaultValue="M"
                                onChange={handleInputChange}
                                className={getInputClass("gender")}
                            >
                                <option value="M">Male</option>
                                <option value="F">Female</option>
                            </select>
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Patient's gender.</p>
                    </div>

                    {/* --- Age --- */}
                    <div className="md:col-span-3">
                        <label htmlFor="age" className="block text-base/6 font-medium text-gray-900 dark:text-white">Age <RequiredMark /></label>
                        <div className="mt-2">
                            <input
                                id="age"
                                type="number"
                                name="age"
                                placeholder="e.g. 45"
                                onChange={handleInputChange}
                                className={getInputClass("age")}
                            />
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Allowed range: <strong>1 - 120</strong> years.</p>
                    </div>

                    {/* --- Chest Pain Type --- */}
                    <div className="md:col-span-3">
                        <label htmlFor="chest-pain-type" className="block text-base/6 font-medium text-gray-900 dark:text-white">Chest pain type <RequiredMark /></label>
                        <div className="mt-2">
                            <select
                                id="chest-pain-type"
                                defaultValue=""
                                name="chest-pain-type"
                                onChange={handleInputChange}
                                className={getInputClass("chest-pain-type")}
                            >
                                <option value="" disabled>Select the type of chest pain...</option>
                                <option value="TA">Typical Angina</option>
                                <option value="ATA">Atypical Angina</option>
                                <option value="NAP">Non-Anginal Pain</option>
                                <option value="ASY">Asymptomatic</option>
                            </select>
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Subjective description of pain reported by patient.</p>
                    </div>

                    {/* --- Resting BP --- */}
                    <div className="md:col-span-3">
                        <label htmlFor="resting-bp" className="block text-base/6 font-medium text-gray-900 dark:text-white">Resting blood pressure <RequiredMark /></label>
                        <div className="mt-2">
                            <input
                                id="resting-bp"
                                type="number"
                                name="resting-bp"
                                placeholder="e.g. 120"
                                onChange={handleInputChange}
                                className={getInputClass("resting-bp")}
                            />
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Systolic BP in <strong>mmHg</strong> (Range: 50 - 250).</p>
                    </div>

                    {/* --- Cholesterol --- */}
                    <div className="md:col-span-3">
                        <label htmlFor="cholesterol" className="block text-base/6 font-medium text-gray-900 dark:text-white">Cholesterol <RequiredMark /></label>
                        <div className="mt-2">
                            <input
                                id="cholesterol"
                                type="number"
                                name="cholesterol"
                                placeholder="e.g. 210"
                                onChange={handleInputChange}
                                className={getInputClass("cholesterol")}
                            />
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Serum cholesterol in <strong>mg/dl</strong> (Range: 0 - 600).</p>
                    </div>

                    {/* --- Fasting BS --- */}
                    <div className="md:col-span-3">
                        <label htmlFor="fasting-bs" className="block text-base/6 font-medium text-gray-900 dark:text-white">Fasting blood sugar <RequiredMark /></label>
                        <div className="mt-2">
                            <select
                                id="fasting-bs"
                                name="fasting-bs"
                                defaultValue=""
                                onChange={handleInputChange}
                                className={getInputClass("fasting-bs")}
                            >
                                <option value="" disabled> Is fasting blood sugar {'>'} 120 mg/dl?
                                </option>
                                <option value="1"> Yes ({'>'} 120 mg/dl)
                                </option>
                                <option value="0"> No (≤ 120 mg/dl)
                                </option>
                            </select>
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Measured in <strong>mg/dl</strong> after fasting.</p>
                    </div>

                    {/* --- Resting ECG --- */}
                    <div className="md:col-span-3">
                        <label htmlFor="resting-ecg" className="block text-base/6 font-medium text-gray-900 dark:text-white">Resting ECG <RequiredMark /></label>
                        <div className="mt-2">
                            <select
                                id="resting-ecg"
                                defaultValue=""
                                name="resting-ecg"
                                onChange={handleInputChange}
                                className={getInputClass("resting-ecg")}
                            >
                                <option value="" disabled>Select result...</option>
                                <option value="Normal">Normal</option>
                                <option value="ST">ST-T wave abnormality</option>
                                <option value="LVH">Left ventricular hypertrophy</option>
                            </select>
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Result based on ST-T wave or Estes criteria.</p>
                    </div>

                    {/* --- Max Heart Rate --- */}
                    <div className="md:col-span-3">
                        <label htmlFor="max-hr" className="block text-base/6 font-medium text-gray-900 dark:text-white">Max heart rate <RequiredMark /></label>
                        <div className="mt-2">
                            <input
                                id="max-hr"
                                type="number"
                                name="max-hr"
                                placeholder="e.g. 150"
                                onChange={handleInputChange}
                                className={getInputClass("max-hr")}
                            />
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Max HR achieved in <strong>bpm</strong> (Range: 60 - 220).</p>
                    </div>

                    {/* --- Exercise Angina --- */}
                    <div className="md:col-span-3">
                        <label htmlFor="exercise-angina" className="block text-base/6 font-medium text-gray-900 dark:text-white">Exercise-induced angina <RequiredMark /></label>
                        <div className="mt-2">
                            <select
                                id="exercise-angina"
                                defaultValue=""
                                name="exercise-angina"
                                onChange={handleInputChange}
                                className={getInputClass("exercise-angina")}
                            >
                                <option value="" disabled>Did patient have angina?</option>
                                <option value="Y">Yes</option>
                                <option value="N">No</option>
                            </select>
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Pain specifically caused by exercise.</p>
                    </div>

                    {/* --- Oldpeak --- */}
                    <div className="md:col-span-3">
                        <label htmlFor="oldpeak" className="block text-base/6 font-medium text-gray-900 dark:text-white">Oldpeak <RequiredMark /></label>
                        <div className="mt-2">
                            <input
                                id="oldpeak"
                                type="number"
                                name="oldpeak"
                                step="0.1"
                                placeholder="e.g. 1.5"
                                onChange={handleInputChange}
                                className={getInputClass("oldpeak")}
                            />
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">ST depression induced by exercise (Range: 0 - 6.2).</p>
                    </div>

                    {/* --- ST Slope --- */}
                    <div className="md:col-span-3">
                        <label htmlFor="st-slope" className="block text-base/6 font-medium text-gray-900 dark:text-white">ST Slope <RequiredMark /></label>
                        <div className="mt-2">
                            <select
                                id="st-slope"
                                defaultValue=""
                                name="st-slope"
                                onChange={handleInputChange}
                                className={getInputClass("st-slope")}
                            >
                                <option value="" disabled>Select the slope curve...</option>
                                <option value="Up">Upsloping</option>
                                <option value="Flat">Flat</option>
                                <option value="Down">Downsloping</option>
                            </select>
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Slope of the peak exercise ST segment.</p>
                    </div>
                </div>

                <div className="mt-6 flex items-center justify-end gap-x-2">
                    <button type="button" onClick={handleReset}
                        className="rounded-md px-8 py-2 text-sm font-semibold text-gray-900 cursor-pointer hover:bg-gray-100 hover:text-indigo-600 dark:text-white dark:hover:bg-white/10 dark:hover:text-indigo-400 transition-colors">
                        Reset
                    </button>
                    <button type="submit" disabled={submitting}
                        className="rounded-md bg-indigo-600 px-8 py-2 text-sm font-semibold text-white shadow-sm cursor-pointer hover:bg-indigo-500 focus-visible:outline focus-visible:outline-offset-2 focus-visible:outline-indigo-600">
                        {submitting ? "Analyzing..." : "Predict"}
                    </button>
                </div>
            </form>

            {/* --- HIỂN THỊ KẾT QUẢ --- */}
            {result && (
                <div ref={resultRef} className="mt-12 mb-12 animate-fade-in px-2 md:px-8">
                    <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white pb-2 border-b border-gray-200 dark:border-gray-700">
                        Analysis Results
                    </h2>

                    {/* 1. Kết quả chẩn đoán */}
                    <div className={`p-4 rounded-xl border-l-8 shadow-md mb-8 transition-all ${result.prediction === 1 ? 'bg-red-50 border-red-500 dark:bg-red-900/20' : 'bg-green-50 border-green-500 dark:bg-green-900/20'}`}>
                        <div className="flex flex-col md:flex-row justify-between items-center gap-4">
                            <div>
                                <h3 className="text-lg font-semibold flex items-center gap-2 text-gray-700 dark:text-gray-300">
                                    <FaHeartbeat /> Diagnosis Prediction:
                                </h3>
                                <p className={`text-3xl font-bold mt-1 ${result.prediction === 1 ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'}`}>
                                    {result.prediction === 1 ? "RISK DETECTED" : "NORMAL"}
                                </p>
                            </div>
                            <div className="text-center bg-white/60 dark:bg-black/20 p-4 rounded-lg">
                                <span className="block text-sm text-gray-500 dark:text-gray-400">Confidence Score</span>
                                <span className="text-3xl font-bold text-gray-800 dark:text-white">{(result.probability * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>

                    {/* 2. Biểu đồ phân tích */}
                    <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white flex items-center gap-2">
                        <span className="bg-indigo-100 text-indigo-800 text-xs font-medium px-2.5 py-0.5 rounded dark:bg-indigo-900 dark:text-indigo-300">AI Logic</span>
                        Detailed Explanation
                    </h3>

                    <div className="grid grid-cols-1 gap-8">
                        <div>
                            {renderChartImage(result.shap_waterfall, "Feature Impact Analysis (SHAP Waterfall)")}
                            <p className="text-sm text-gray-500 mt-2 text-center italic dark:text-gray-200">
                                Visualizes how individual factors shift the prediction from the baseline.
                                <span className="font-bold text-red-500"> Red bars</span> indicate factors increasing heart failure risk,
                                while <span className="font-bold text-blue-500"> Blue bars</span> indicate factors decreasing the risk.
                            </p>
                        </div>
                        <div>
                            {renderChartImage(result.shap_bar, "Global Feature Importance (SHAP Bar)")}
                            <p className="text-sm text-gray-500 mt-2 text-center italic dark:text-gray-200">
                                Ranks the health indicators by their absolute impact on this prediction. Longer bars mean the AI considered these factors most critical for this patient.
                            </p>
                        </div>
                        <div>
                            {renderChartImage(result.lime, "Local Interpretation (LIME)")}
                            <p className="text-sm text-gray-500 mt-2 text-center italic dark:text-gray-200">
                                Independent verification: Analyzing which specific features support a "High Risk" diagnosis versus those supporting a "Normal" diagnosis.
                            </p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}