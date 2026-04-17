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
import { FaHeartbeat, FaUserAlt, FaNotesMedical } from "react-icons/fa";

export default function Predict() {
    const router = useRouter();
    const [submitting, setSubmitting] = useState(false);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const [invalidFields, setInvalidFields] = useState<string[]>([]);
    const formRef = useRef<HTMLFormElement>(null);
    const autoFillBtnRef = useRef<HTMLButtonElement>(null);
    const topRef = useRef<HTMLDivElement>(null);
    const resultRef = useRef<HTMLDivElement>(null);
    const { user, loading: authLoading, logout, pushNewHistoryItem } = useAuth();
    const { savePrediction } = useSettings();

    const RequiredMark = () => <span className="text-red-500 font-bold ml-1">*</span>;

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
        checkRange("oldpeak", -5, 10);
        checkRequired("st-slope");

        setInvalidFields(errors);
        return errors.length === 0;
    };

    // Style ô input bo tròn, nền xám như ảnh minh họa
    const getInputClass = (fieldName: string, hasSuffix: boolean = false) => {
        const baseStructure = `block w-full rounded-xl transition h-12 px-4 text-base outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white dark:focus:bg-gray-900 ${hasSuffix ? 'pr-16' : ''}`;
        if (invalidFields.includes(fieldName)) {
            return `${baseStructure} bg-red-50 text-red-800 placeholder:text-red-300 border border-red-500 dark:bg-red-900/20 dark:text-red-200`;
        }
        return `${baseStructure} bg-gray-100 text-gray-900 border border-transparent placeholder:text-gray-400 dark:bg-gray-800 dark:text-white`;
    };

    const handleAutoFillForm = async () => {
        toast.dismiss();
        setInvalidFields([]);
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
            const patient: Patient = res.data;

            if (!patient) return;

            // Hàm set value được nâng cấp để hỗ trợ radio buttons
            const setValue = (name: string, value: any) => {
                if (name === "exercise-angina") {
                    const radio = form.querySelector(`input[name="exercise-angina"][value="${value}"]`) as HTMLInputElement;
                    if (radio) radio.checked = true;
                    return;
                }
                const element = form.querySelector(`[name="${name}"]`) as HTMLInputElement | HTMLSelectElement;
                if (element) element.value = String(value);
            }

            setValue('age', patient.age);
            setValue('gender', patient.sex);
            setValue('chest-pain-type', patient.chest_pain_type);
            setValue('resting-bp', patient.resting_bp);
            setValue('cholesterol', patient.cholesterol);
            setValue('fasting-bs', patient.fasting_bs);
            setValue('resting-ecg', patient.resting_ecg);
            setValue('max-hr', patient.max_hr);
            setValue('exercise-angina', patient.exercise_angina);
            setValue('oldpeak', patient.oldpeak);
            setValue('st-slope', patient.st_slope);

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

        try {
            const res = await api.post("/predictions", payload);
            const data: PredictionResult = res.data;
            setResult(data);
            pushNewHistoryItem(data.prediction_history);
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

    const renderChartImage = (imageUrl: string, title: string) => {
        if (!imageUrl) return null
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
        <div className="p-4 mx-auto">
            <h1 ref={topRef} className="text-2xl font-bold text-gray-900 dark:text-white">Patient Information</h1>
            <p className="mt-1 text-gray-600 dark:text-gray-400">Please provide accurate patient information for better prediction results</p>
            <div className="mt-5">
                <button
                    type="button"
                    onClick={handleAutoFillForm}
                    ref={autoFillBtnRef}
                    className="rounded-lg px-5 py-2.5 bg-gray-100 border border-gray-200 text-base font-semibold text-gray-700 hover:bg-gray-200 cursor-pointer transition-colors dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-700"
                >
                    Auto-fill sample
                </button>
            </div>

            <form ref={formRef} onSubmit={handleSubmit} noValidate className="mt-10">
                {/* CARD 1: Demographic Data */}
                <div className="bg-white dark:bg-gray-800/60 p-6 md:p-8 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 mb-6">
                    <h3 className="text-xl font-bold text-gray-800 dark:text-gray-200 flex items-center gap-2 mb-6 pb-4 border-b border-gray-200 dark:border-gray-700">
                        <FaUserAlt className="text-indigo-600 dark:text-indigo-400" /> Demographic Data
                    </h3>
                    <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                        {/* Gender */}
                        <div>
                            <label htmlFor="gender" className="block text-base font-semibold text-gray-800 dark:text-gray-200 mb-2">Gender <RequiredMark /></label>
                            <select id="gender" name="gender" defaultValue="M" onChange={handleInputChange} className={getInputClass("gender")}>
                                <option value="M">Male</option>
                                <option value="F">Female</option>
                            </select>
                            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Patient's gender.</p>
                        </div>
                        {/* Age */}
                        <div>
                            <label htmlFor="age" className="block text-base font-semibold text-gray-800 dark:text-gray-200 mb-2">Age <RequiredMark /></label>
                            <input id="age" type="number" name="age" placeholder="e.g. 45" onChange={handleInputChange} className={getInputClass("age")} />
                            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Allowed range: <strong>1 - 120</strong> years.</p>
                        </div>
                    </div>
                </div>

                {/* CARD 2: Cardiovascular Metrics */}
                <div className="bg-white dark:bg-gray-800/60 p-6 md:p-8 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 mb-6">
                    <h3 className="text-xl font-bold text-gray-800 dark:text-gray-200 flex items-center gap-2 mb-6 pb-4 border-b border-gray-200 dark:border-gray-700">
                        <FaHeartbeat className="text-indigo-600 dark:text-indigo-400" /> Cardiovascular Metrics
                    </h3>
                    <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                        {/* Resting BP */}
                        <div>
                            <label htmlFor="resting-bp" className="block text-base font-semibold text-gray-800 dark:text-gray-200 mb-2">Resting Blood Pressure <RequiredMark /></label>
                            <div className="relative">
                                <input id="resting-bp" type="number" name="resting-bp" placeholder="e.g. 120" onChange={handleInputChange} className={getInputClass("resting-bp", true)} />
                                <div className="absolute inset-y-0 right-0 flex items-center pr-4 pointer-events-none">
                                    <span className="text-[10px] font-bold text-gray-400 dark:text-gray-500 uppercase tracking-widest">MMHG</span>
                                </div>
                            </div>
                            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Systolic BP in <strong>mmHg</strong> (Range: 50 - 250).</p>
                        </div>
                        {/* Cholesterol */}
                        <div>
                            <label htmlFor="cholesterol" className="block text-base font-semibold text-gray-800 dark:text-gray-200 mb-2">Serum Cholesterol <RequiredMark /></label>
                            <div className="relative">
                                <input id="cholesterol" type="number" name="cholesterol" placeholder="e.g. 210" onChange={handleInputChange} className={getInputClass("cholesterol", true)} />
                                <div className="absolute inset-y-0 right-0 flex items-center pr-4 pointer-events-none">
                                    <span className="text-[10px] font-bold text-gray-400 dark:text-gray-500 uppercase tracking-widest">MG/DL</span>
                                </div>
                            </div>
                            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Serum cholesterol in <strong>mg/dl</strong> (Range: 0 - 600).</p>
                        </div>
                        {/* Max HR */}
                        <div>
                            <label htmlFor="max-hr" className="block text-base font-semibold text-gray-800 dark:text-gray-200 mb-2">Max Heart Rate Achieved <RequiredMark /></label>
                            <div className="relative">
                                <input id="max-hr" type="number" name="max-hr" placeholder="e.g. 150" onChange={handleInputChange} className={getInputClass("max-hr", true)} />
                                <div className="absolute inset-y-0 right-0 flex items-center pr-4 pointer-events-none">
                                    <span className="text-[10px] font-bold text-gray-400 dark:text-gray-500 uppercase tracking-widest">BPM</span>
                                </div>
                            </div>
                            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Max HR achieved in <strong>bpm</strong> (Range: 60 - 220).</p>
                        </div>
                    </div>
                </div>

                {/* CARD 3: Symptoms & Clinical History */}
                <div className="bg-white dark:bg-gray-800/60 p-6 md:p-8 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 mb-6">
                    <h3 className="text-xl font-bold text-gray-800 dark:text-gray-200 flex items-center gap-2 mb-6 pb-4 border-b border-gray-200 dark:border-gray-700">
                        <FaNotesMedical className="text-indigo-600 dark:text-indigo-400" /> Symptoms & Clinical History
                    </h3>
                    <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                        {/* Chest Pain Type */}
                        <div>
                            <label htmlFor="chest-pain-type" className="block text-base font-semibold text-gray-800 dark:text-gray-200 mb-2">Chest Pain Type <RequiredMark /></label>
                            <select id="chest-pain-type" defaultValue="" name="chest-pain-type" onChange={handleInputChange} className={getInputClass("chest-pain-type")}>
                                <option value="" disabled>Select the type of chest pain...</option>
                                <option value="TA">Typical Angina</option>
                                <option value="ATA">Atypical Angina</option>
                                <option value="NAP">Non-Anginal Pain</option>
                                <option value="ASY">Asymptomatic</option>
                            </select>
                            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Subjective description of pain reported by patient.</p>
                        </div>
                        {/* Fasting BS */}
                        <div>
                            <label htmlFor="fasting-bs" className="block text-base font-semibold text-gray-800 dark:text-gray-200 mb-2">Fasting Blood Sugar <RequiredMark /></label>
                            <select id="fasting-bs" name="fasting-bs" defaultValue="" onChange={handleInputChange} className={getInputClass("fasting-bs")}>
                                <option value="" disabled>Is fasting blood sugar &gt; 120 mg/dl?</option>
                                <option value="1">Yes (&gt; 120 mg/dl)</option>
                                <option value="0">No (≤ 120 mg/dl)</option>
                            </select>
                            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Measured in <strong>mg/dl</strong> after fasting.</p>
                        </div>
                        {/* Resting ECG */}
                        <div>
                            <label htmlFor="resting-ecg" className="block text-base font-semibold text-gray-800 dark:text-gray-200 mb-2">Resting ECG <RequiredMark /></label>
                            <select id="resting-ecg" defaultValue="" name="resting-ecg" onChange={handleInputChange} className={getInputClass("resting-ecg")}>
                                <option value="" disabled>Select result...</option>
                                <option value="Normal">Normal</option>
                                <option value="ST">ST-T wave abnormality</option>
                                <option value="LVH">Left ventricular hypertrophy</option>
                            </select>
                            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Result based on ST-T wave or Estes criteria.</p>
                        </div>
                        {/* Exercise Angina */}
                        <div>
                            <label className="block text-base font-semibold text-gray-800 dark:text-gray-200 mb-3">Exercise Induced Angina <RequiredMark /></label>
                            <div className="flex items-center gap-8 h-10">
                                <label className="flex items-center gap-2 cursor-pointer group">
                                    <input type="radio" name="exercise-angina" value="Y" className="w-4 h-4 text-indigo-600 bg-gray-100 border-gray-300 focus:ring-indigo-500 dark:ring-offset-gray-800 dark:bg-gray-700 dark:border-gray-600" onChange={handleInputChange} />
                                    <span className="text-base font-medium text-gray-700 group-hover:text-indigo-600 dark:text-gray-300">Yes</span>
                                </label>
                                <label className="flex items-center gap-2 cursor-pointer group">
                                    <input type="radio" name="exercise-angina" value="N" className="w-4 h-4 text-indigo-600 bg-gray-100 border-gray-300 focus:ring-indigo-500 dark:ring-offset-gray-800 dark:bg-gray-700 dark:border-gray-600" onChange={handleInputChange} />
                                    <span className="text-base font-medium text-gray-700 group-hover:text-indigo-600 dark:text-gray-300">No</span>
                                </label>
                            </div>
                            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Pain specifically caused by exercise.</p>
                            {invalidFields.includes("exercise-angina") && (
                                <p className="mt-1 text-sm text-red-500">Please select an option.</p>
                            )}
                        </div>
                        {/* Oldpeak */}
                        <div>
                            <label htmlFor="oldpeak" className="block text-base font-semibold text-gray-800 dark:text-gray-200 mb-2">Oldpeak <RequiredMark /></label>
                            <input id="oldpeak" type="number" name="oldpeak" step="0.1" min="-5" max="10" placeholder="e.g. 1.5 or -0.5" onChange={handleInputChange} className={getInputClass("oldpeak")} />
                            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">ST deviation induced by exercise (typically -5 to 10; negative values indicate ST elevation).</p>
                        </div>
                        {/* ST Slope */}
                        <div>
                            <label htmlFor="st-slope" className="block text-base font-semibold text-gray-800 dark:text-gray-200 mb-2">ST Slope <RequiredMark /></label>
                            <select id="st-slope" defaultValue="" name="st-slope" onChange={handleInputChange} className={getInputClass("st-slope")}>
                                <option value="" disabled>Select the slope curve...</option>
                                <option value="Up">Upsloping</option>
                                <option value="Flat">Flat</option>
                                <option value="Down">Downsloping</option>
                            </select>
                            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Slope of the peak exercise ST segment.</p>
                        </div>
                    </div>
                </div>

                {/* Các Nút Hành Động */}
                <div className="mt-8 flex items-center justify-end gap-x-4 border-t border-gray-100 dark:border-gray-800 pt-6">
                    <button type="button" onClick={handleReset}
                        className="rounded-xl px-8 py-3 text-base font-semibold text-gray-700 bg-white border border-gray-200 cursor-pointer hover:bg-gray-50 hover:text-indigo-600 transition-colors shadow-sm dark:bg-gray-800 dark:border-gray-700 dark:text-gray-200 dark:hover:bg-gray-700">
                        Reset Form
                    </button>
                    <button type="submit" disabled={submitting}
                        className="rounded-xl bg-indigo-600 px-8 py-3 text-base font-semibold text-white shadow-sm cursor-pointer hover:bg-indigo-700 focus:ring-4 focus:ring-indigo-500/30 transition-all">
                        {submitting ? "Analyzing..." : "Predict"}
                    </button>
                </div>
            </form>

            {/* --- HIỂN THỊ KẾT QUẢ --- */}
            {result && (
                <div ref={resultRef} className="mt-8 animate-fade-in">
                    <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white pb-2 border-b border-gray-200 dark:border-gray-700">
                        Analysis Results
                    </h2>

                    <div className={`p-5 rounded-2xl border border-l-8 shadow-sm mb-8 transition-all ${result.prediction === 1 ? 'bg-red-50 border-red-500 dark:bg-red-900/20' : 'bg-green-50 border-green-500 dark:bg-green-900/20'}`}>
                        <div className="flex flex-col md:flex-row justify-between items-center gap-4">
                            <div>
                                <h3 className="text-lg font-bold flex items-center gap-2 text-gray-800 dark:text-gray-200">
                                    <FaHeartbeat /> Diagnosis Prediction:
                                </h3>
                                <p className={`text-3xl font-black mt-2 ${result.prediction === 1 ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'}`}>
                                    {result.prediction === 1 ? "RISK DETECTED" : "NORMAL"}
                                </p>
                            </div>
                            <div className="text-center bg-white dark:bg-gray-800 p-4 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700">
                                <span className="block text-xs uppercase tracking-wider font-bold text-gray-400 dark:text-gray-500 mb-1">Confidence Score</span>
                                <span className="text-3xl font-black text-gray-800 dark:text-white">{(result.probability * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>

                    <h3 className="text-xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-3">
                        <span className="bg-indigo-100 text-indigo-700 text-xs font-bold px-3 py-1 rounded-full dark:bg-indigo-900 dark:text-indigo-300">AI Logic</span>
                        Detailed Explanation
                    </h3>

                    <div className="grid grid-cols-1 gap-8">
                        <div>
                            {renderChartImage(result.shap_waterfall, "Feature Impact Analysis (SHAP Waterfall)")}
                            <p className="text-sm text-gray-500 mt-3 text-center italic dark:text-gray-400 max-w-3xl mx-auto">
                                Visualizes how individual factors shift the prediction from the baseline.
                                <span className="font-bold text-red-500"> Red bars</span> indicate factors increasing heart failure risk,
                                while <span className="font-bold text-blue-500"> Blue bars</span> indicate factors decreasing the risk.
                            </p>
                        </div>
                        <div>
                            {renderChartImage(result.shap_bar, "Global Feature Importance (SHAP Bar)")}
                            <p className="text-sm text-gray-500 mt-3 text-center italic dark:text-gray-400 max-w-3xl mx-auto">
                                Ranks the health indicators by their absolute impact on this prediction. Longer bars mean the AI considered these factors most critical for this patient.
                            </p>
                        </div>
                        <div>
                            {renderChartImage(result.lime, "Local Interpretation (LIME)")}
                            <p className="text-sm text-gray-500 mt-3 text-center italic dark:text-gray-400 max-w-3xl mx-auto">
                                Independent verification: Analyzing which specific features support a "High Risk" diagnosis versus those supporting a "Normal" diagnosis.
                            </p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}