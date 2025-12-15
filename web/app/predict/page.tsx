"use client"
import { useRouter } from "next/navigation";
import { FormEvent, useState, useRef } from "react";
import { toast } from "react-toastify";
import { PredictionResult } from "@/types/prediction";
import Image from "next/image";

export default function Predict() {
    const router = useRouter();
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const formRef = useRef<HTMLFormElement>(null);
    const topRef = useRef<HTMLDivElement>(null);

    const handleReset = () => {
        formRef.current?.reset();
        setResult(null);
        setLoading(false);
        toast.dismiss();

        topRef.current?.scrollIntoView({
            behavior: "smooth",
            block: "start",
        });

        toast.info("Form cleared.");
    };

    const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        toast.dismiss();
        const token = localStorage.getItem("access_token");
        if (!token) {
            toast.warning("You must be logged in to submit a prediction.");
            router.push("/login");
            return;
        }

        setLoading(true);
        setResult(null);

        const formData = new FormData(e.currentTarget);

        const rawFastingBS = parseInt(formData.get("fasting-bs") as string)
        const fastingBS = rawFastingBS > 120 ? 1 : 0;

        const payload = {
            age: parseInt(formData.get("age") as string),
            sex: formData.get("gender"),
            chest_pain_type: formData.get("chest-pain-type"),
            resting_bp: parseInt(formData.get("resting-bp") as string),
            cholesterol: parseInt(formData.get("cholesterol") as string),
            fasting_bs: fastingBS,
            resting_ecg: formData.get("resting-ecg"),
            max_hr: parseInt(formData.get("max-hr") as string),
            exercise_angina: formData.get("exercise-angina"),
            oldpeak: parseFloat(formData.get("oldpeak") as string),
            st_slope: formData.get("st-slope"),
        };

        console.log("Payload sending to API:", JSON.stringify(payload, null, 2));

        try {
            const res = await fetch("https://heart-failure-api-uwqj.onrender.com/api/predict/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${token}`
                },
                body: JSON.stringify(payload),
            });

            if (!res.ok) {
                if (res.status === 401) {
                    toast.error("Session expired. Please sign in again.");
                    router.push("/login");
                    return;
                }
                throw new Error("API Error");
            }

            const data: PredictionResult = await res.json();
            console.log("API result:", data);
            setResult(data);
            toast.success("Success!");
        } catch (error) {
            console.error(error);
            toast.error("An error occurred during prediction.");
        } finally {
            setLoading(false);
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

            <form ref={formRef} onSubmit={handleSubmit} className="mt-16 border-b border-gray-900/10 pb-12 px-12 dark:border-white/10">
                <div className="mt-10 grid grid-cols-1 gap-x-6 gap-y-8 sm:grid-cols-6">

                    {/* --- First Name --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="first-name" className="block text-base/6 font-medium text-gray-900 dark:text-white">First name</label>
                        <div className="mt-2">
                            <input
                                id="first-name" type="text" name="first-name" autoComplete="given-name"
                                placeholder="e.g. John"
                                className="block w-full rounded-lg bg-white px-3 py-1.5 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500"
                            />
                        </div>
                    </div>

                    {/* --- Last Name --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="last-name" className="block text-base/6 font-medium text-gray-900 dark:text-white">Last name</label>
                        <div className="mt-2">
                            <input
                                id="last-name" type="text" name="last-name" autoComplete="family-name"
                                placeholder="e.g. Doe"
                                className="block w-full rounded-lg bg-white px-3 py-1.5 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500"
                            />
                        </div>
                    </div>

                    {/* --- Gender --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="gender" className="block text-base/6 font-medium text-gray-900 dark:text-white">Gender</label>
                        <div className="mt-2">
                            <select id="gender" required defaultValue="M" name="gender" className="block w-full rounded-lg bg-white px-3 py-2 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500">
                                <option value="M">Male</option>
                                <option value="F">Female</option>
                            </select>
                        </div>
                    </div>

                    {/* --- Age --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="age" className="block text-base/6 font-medium text-gray-900 dark:text-white">Age</label>
                        <div className="mt-2">
                            <input
                                id="age" required type="number" name="age" min={1} max={120}
                                placeholder="e.g. 45"
                                className="block w-full rounded-lg bg-white px-3 py-1.5 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500"
                            />
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Patient's age in years.</p>
                    </div>

                    {/* --- Chest Pain Type --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="chest-pain-type" className="block text-base/6 font-medium text-gray-900 dark:text-white">Chest pain type</label>
                        <div className="mt-2">
                            <select id="chest-pain-type" required defaultValue="" name="chest-pain-type" className="block w-full rounded-lg bg-white px-3 py-2 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500">
                                <option value="" disabled>Select the type of chest pain...</option>
                                <option value="TA">Typical Angina</option>
                                <option value="ATA">Atypical Angina</option>
                                <option value="NAP">Non-Anginal Pain</option>
                                <option value="ASY">Asymptomatic</option>
                            </select>
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Subjective description of pain reported by the patient.</p>
                    </div>

                    {/* --- Resting BP --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="resting-bp" className="block text-base/6 font-medium text-gray-900 dark:text-white">Resting blood pressure</label>
                        <div className="mt-2">
                            <input
                                id="resting-bp" required type="number" name="resting-bp" min={1}
                                placeholder="e.g. 120"
                                className="block w-full rounded-lg bg-white px-3 py-1.5 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500"
                            />
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Systolic blood pressure in <strong>mmHg</strong>.</p>
                    </div>

                    {/* --- Cholesterol --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="cholesterol" className="block text-base/6 font-medium text-gray-900 dark:text-white">Cholesterol</label>
                        <div className="mt-2">
                            <input
                                id="cholesterol" required type="number" name="cholesterol" min={1}
                                placeholder="e.g. 210"
                                className="block w-full rounded-lg bg-white px-3 py-1.5 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500"
                            />
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Serum cholesterol in <strong>mg/dl</strong>.</p>
                    </div>

                    {/* --- Fasting BS --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="fasting-bs" className="block text-base/6 font-medium text-gray-900 dark:text-white">Fasting blood sugar</label>
                        <div className="mt-2">
                            <input
                                id="fasting-bs" required type="number" name="fasting-bs" min={1}
                                placeholder="e.g. 95"
                                className="block w-full rounded-lg bg-white px-3 py-1.5 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500"
                            />
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Measured in <strong>mg/dl</strong> after fasting.</p>
                    </div>

                    {/* --- Resting ECG --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="resting-ecg" className="block text-base/6 font-medium text-gray-900 dark:text-white">Resting ECG</label>
                        <div className="mt-2">
                            <select id="resting-ecg" required defaultValue="" name="resting-ecg" className="block w-full rounded-lg bg-white px-3 py-2 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500">
                                <option value="" disabled>Select result...</option>
                                <option value="Normal">Normal</option>
                                <option value="ST">ST-T wave abnormality</option>
                                <option value="LVH">Left ventricular hypertrophy</option>
                            </select>
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Result based on ST-T wave or Estes' criteria.</p>
                    </div>

                    {/* --- Max Heart Rate --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="max-hr" className="block text-base/6 font-medium text-gray-900 dark:text-white">Max heart rate</label>
                        <div className="mt-2">
                            <input
                                id="max-hr" required type="number" name="max-hr" min={60} max={220}
                                placeholder="e.g. 150"
                                className="block w-full rounded-lg bg-white px-3 py-1.5 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500"
                            />
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Maximum heart rate achieved (bpm).</p>
                    </div>

                    {/* --- Exercise Angina --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="exercise-angina" className="block text-base/6 font-medium text-gray-900 dark:text-white">Exercise-induced angina</label>
                        <div className="mt-2">
                            <select id="exercise-angina" required defaultValue="" name="exercise-angina" className="block w-full rounded-lg bg-white px-3 py-2 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500">
                                <option value="" disabled>Did patient have angina?</option>
                                <option value="Y">Yes</option>
                                <option value="N">No</option>
                            </select>
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Pain specifically caused by exercise.</p>
                    </div>

                    {/* --- Oldpeak --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="oldpeak" className="block text-base/6 font-medium text-gray-900 dark:text-white">Oldpeak</label>
                        <div className="mt-2">
                            <input
                                id="oldpeak" required type="number" name="oldpeak" step="0.1" min="0"
                                placeholder="e.g. 1.5"
                                className="block w-full rounded-lg bg-white px-3 py-1.5 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500"
                            />
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">ST depression induced by exercise vs rest.</p>
                    </div>

                    {/* --- ST Slope --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="st-slope" className="block text-base/6 font-medium text-gray-900 dark:text-white">ST Slope</label>
                        <div className="mt-2">
                            <select id="st-slope" required defaultValue="" name="st-slope" className="block w-full rounded-lg bg-white px-3 py-2 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500">
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
                    <button
                        type="button"
                        onClick={handleReset}
                        className="rounded-md px-8 py-2 text-sm font-semibold text-gray-900 cursor-pointer hover:bg-gray-100 hover:text-indigo-600 dark:text-white dark:hover:bg-white/10 dark:hover:text-indigo-400 transition-colors"
                    >
                        Reset
                    </button>
                    <button
                        type="submit"
                        disabled={loading}
                        className="rounded-md bg-indigo-600 px-8 py-2 text-sm font-semibold text-white shadow-sm cursor-pointer hover:bg-indigo-500 focus-visible:outline focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
                    >
                        {loading ? "Analyzing..." : "Predict"}
                    </button>
                </div>
            </form>

            {/* --- HIỂN THỊ KẾT QUẢ --- */}
            {result && (
                <div className="mt-12 mb-12 animate-fade-in px-2 sm:px-8">
                    <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white pb-2 border-b border-gray-200 dark:border-gray-700">
                        Analysis Results
                    </h2>

                    {/* 1. Kết quả chẩn đoán */}
                    <div className={`p-6 rounded-xl border-l-8 shadow-md mb-8 transition-all ${result.prediction === 1 ? 'bg-red-50 border-red-500 dark:bg-red-900/20' : 'bg-green-50 border-green-500 dark:bg-green-900/20'}`}>
                        <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
                            <div>
                                <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-300">Diagnosis Prediction:</h3>
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

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        <div className="lg:col-span-2">
                            {renderChartImage(result.shap_waterfall, "Feature Impact Analysis (SHAP Waterfall)")}
                            <p className="text-sm text-gray-500 mt-2 text-center italic">
                                Visualizes how individual factors shift the prediction from the baseline.
                                <span className="font-bold text-red-500"> Red bars</span> indicate factors increasing heart failure risk,
                                while <span className="font-bold text-blue-500"> Blue bars</span> indicate factors decreasing the risk.
                            </p>
                        </div>
                        <div className="lg:col-span-2">
                            {renderChartImage(result.shap_bar, "Global Feature Importance (SHAP Bar)")}
                            <p className="text-sm text-gray-500 mt-2 text-center italic">
                                Ranks the health indicators by their absolute impact on this prediction. Longer bars mean the AI considered these factors most critical for this patient.
                            </p>
                        </div>
                        <div className="lg:col-span-2">
                            {renderChartImage(result.lime, "Local Interpretation (LIME)")}
                            <p className="text-sm text-gray-500 mt-2 text-center italic">
                                Independent verification: Analyzing which specific features support a "High Risk" diagnosis versus those supporting a "Normal" diagnosis.
                            </p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}