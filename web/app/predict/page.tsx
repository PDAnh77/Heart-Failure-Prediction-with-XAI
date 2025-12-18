"use client"
import { useRouter } from "next/navigation";
import { FormEvent, useState, useRef, useEffect } from "react";
import { toast } from "react-toastify";
import { PredictionResult } from "@/types/prediction";
import { useAuth } from "@/context/authcontext";
import Image from "next/image";

export default function Predict() {
    const router = useRouter();
    const [submitting, setSubmitting] = useState(false);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const formRef = useRef<HTMLFormElement>(null);
    const topRef = useRef<HTMLDivElement>(null);
    const resultRef = useRef<HTMLDivElement>(null);
    const { user, loading: authLoading, logout } = useAuth();

    const RequiredMark = () => <span className="text-red-500">*</span>;

    // --- Sample patient for quick auto-fill testing ---
    const SAMPLE_PATIENTS = [
        {
            first_name: "Nguyen",
            last_name: "Van A",
            age: 40,
            sex: "M",
            chest_pain_type: "NAP",
            resting_bp: 130,
            cholesterol: 215,
            fasting_bs: 0,
            resting_ecg: "Normal",
            max_hr: 138,
            exercise_angina: "N",
            oldpeak: 0,
            st_slope: "Up",
            heart_disease: 0
        }
    ];

    const fillFormWithSample = (patient: any) => {
        if (!formRef.current) return;
        const form = formRef.current;

        (form.querySelector('#first-name') as HTMLInputElement).value = String(patient.first_name);
        (form.querySelector('#last-name') as HTMLInputElement).value = String(patient.last_name);
        (form.querySelector('#age') as HTMLInputElement).value = String(patient.age);
        (form.querySelector('#gender') as HTMLSelectElement).value = patient.sex;
        (form.querySelector('#chest-pain-type') as HTMLSelectElement).value = patient.chest_pain_type;
        (form.querySelector('#resting-bp') as HTMLInputElement).value = String(patient.resting_bp);
        (form.querySelector('#cholesterol') as HTMLInputElement).value = String(patient.cholesterol);
        (form.querySelector('#fasting-bs') as HTMLInputElement).value = patient.fasting_bs ? '130' : '95';
        (form.querySelector('#resting-ecg') as HTMLSelectElement).value = patient.resting_ecg;
        (form.querySelector('#max-hr') as HTMLInputElement).value = String(patient.max_hr);
        (form.querySelector('#exercise-angina') as HTMLSelectElement).value = patient.exercise_angina;
        (form.querySelector('#oldpeak') as HTMLInputElement).value = String(patient.oldpeak);
        (form.querySelector('#st-slope') as HTMLSelectElement).value = patient.st_slope;

        toast.info("Form auto-filled with sample data.");
    };

    const handleReset = () => {
        formRef.current?.reset();
        setResult(null);
        setSubmitting(false);
        toast.dismiss();

        topRef.current?.scrollIntoView({
            behavior: "smooth",
            block: "start",
        });

        toast.info("Form cleared.");
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
            toast.info("Checking authentication, try again in a moment.");
            return;
        }

        if (!user) {
            toast.warning("You must be logged in to submit a prediction.");
            router.push("/login");
            return;
        }

        setSubmitting(true);
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
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                credentials: "include",
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                if (res.status === 401) {
                    toast.error("Session expired. Please sign in again.");
                    logout();
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
                    onClick={() => {
                        fillFormWithSample(SAMPLE_PATIENTS[0]);
                        formRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
                    }}
                    className="rounded-md px-4 py-2 bg-gray-200 text-sm font-medium dark:text-black hover:bg-gray-300"
                >
                    Auto-fill sample
                </button>
            </div>
            <form ref={formRef} onSubmit={handleSubmit} className="mt-12 border-b border-gray-900/10 pb-12 px-2 sm:px-12 dark:border-white/10">
                <div className="mt-10 grid grid-cols-1 gap-x-6 gap-y-8 sm:grid-cols-6">

                    {/* --- First Name --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="first-name" className="block text-base/6 font-medium text-gray-900 dark:text-white">First name</label>
                        <div className="mt-2">
                            <input id="first-name" type="text" name="first-name" autoComplete="given-name" placeholder="e.g. John"
                                className="block w-full rounded-lg bg-white px-3 py-1.5 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500" />
                        </div>
                    </div>

                    {/* --- Last Name --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="last-name" className="block text-base/6 font-medium text-gray-900 dark:text-white">Last name</label>
                        <div className="mt-2">
                            <input id="last-name" type="text" name="last-name" autoComplete="family-name" placeholder="e.g. Doe"
                                className="block w-full rounded-lg bg-white px-3 py-1.5 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500" />
                        </div>
                    </div>

                    {/* --- Gender --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="gender" className="block text-base/6 font-medium text-gray-900 dark:text-white">Gender <RequiredMark /></label>
                        <div className="mt-2">
                            <select id="gender" required defaultValue="M" name="gender" className="block w-full rounded-lg bg-white px-3 py-2 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:focus:outline-indigo-500">
                                <option className="dark:bg-gray-800 dark:text-white" value="M">Male</option>
                                <option className="dark:bg-gray-800 dark:text-white" value="F">Female</option>
                            </select>
                        </div>
                    </div>

                    {/* --- Age --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="age" className="block text-base/6 font-medium text-gray-900 dark:text-white">Age <RequiredMark /></label>
                        <div className="mt-2">
                            <input id="age" required type="number" name="age" min={1} max={120} placeholder="e.g. 45"
                                className="block w-full rounded-lg bg-white px-3 py-1.5 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500" />
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Allowed range: <strong>1 - 120</strong> years.</p>
                    </div>

                    {/* --- Chest Pain Type --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="chest-pain-type" className="block text-base/6 font-medium text-gray-900 dark:text-white">Chest pain type <RequiredMark /></label>
                        <div className="mt-2">
                            <select id="chest-pain-type" required defaultValue="" name="chest-pain-type" className="block w-full rounded-lg bg-white px-3 py-2 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:focus:outline-indigo-500">
                                <option className="dark:bg-gray-800 dark:text-white" value="" disabled>Select the type of chest pain...</option>
                                <option className="dark:bg-gray-800 dark:text-white" value="TA">Typical Angina</option>
                                <option className="dark:bg-gray-800 dark:text-white" value="ATA">Atypical Angina</option>
                                <option className="dark:bg-gray-800 dark:text-white" value="NAP">Non-Anginal Pain</option>
                                <option className="dark:bg-gray-800 dark:text-white" value="ASY">Asymptomatic</option>
                            </select>
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Subjective description of pain reported by patient.</p>
                    </div>

                    {/* --- Resting BP --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="resting-bp" className="block text-base/6 font-medium text-gray-900 dark:text-white">Resting blood pressure <RequiredMark /></label>
                        <div className="mt-2">
                            <input id="resting-bp" required type="number" name="resting-bp" min={50} max={250} placeholder="e.g. 120"
                                className="block w-full rounded-lg bg-white px-3 py-1.5 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500" />
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Systolic BP in <strong>mmHg</strong> (Range: 50 - 250).</p>
                    </div>

                    {/* --- Cholesterol --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="cholesterol" className="block text-base/6 font-medium text-gray-900 dark:text-white">Cholesterol <RequiredMark /></label>
                        <div className="mt-2">
                            <input id="cholesterol" required type="number" name="cholesterol" min={0} max={600} placeholder="e.g. 210"
                                className="block w-full rounded-lg bg-white px-3 py-1.5 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500" />
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Serum cholesterol in <strong>mg/dl</strong> (Range: 0 - 600).</p>
                    </div>

                    {/* --- Fasting BS --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="fasting-bs" className="block text-base/6 font-medium text-gray-900 dark:text-white">Fasting blood sugar <RequiredMark /></label>
                        <div className="mt-2">
                            <input id="fasting-bs" required type="number" name="fasting-bs" min={0} max={500} placeholder="e.g. 95"
                                className="block w-full rounded-lg bg-white px-3 py-1.5 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500" />
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Measured in <strong>mg/dl</strong> after fasting (Range: 0 - 500).</p>
                    </div>

                    {/* --- Resting ECG --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="resting-ecg" className="block text-base/6 font-medium text-gray-900 dark:text-white">Resting ECG <RequiredMark /></label>
                        <div className="mt-2">
                            <select id="resting-ecg" required defaultValue="" name="resting-ecg" className="block w-full rounded-lg bg-white px-3 py-2 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:focus:outline-indigo-500">
                                <option className="dark:bg-gray-800 dark:text-white" value="" disabled>Select result...</option>
                                <option className="dark:bg-gray-800 dark:text-white" value="Normal">Normal</option>
                                <option className="dark:bg-gray-800 dark:text-white" value="ST">ST-T wave abnormality</option>
                                <option className="dark:bg-gray-800 dark:text-white" value="LVH">Left ventricular hypertrophy</option>
                            </select>
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Result based on ST-T wave or Estes criteria.</p>
                    </div>

                    {/* --- Max Heart Rate --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="max-hr" className="block text-base/6 font-medium text-gray-900 dark:text-white">Max heart rate <RequiredMark /></label>
                        <div className="mt-2">
                            <input id="max-hr" required type="number" name="max-hr" min={60} max={220} placeholder="e.g. 150"
                                className="block w-full rounded-lg bg-white px-3 py-1.5 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500" />
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Max HR achieved in <strong>bpm</strong> (Range: 60 - 220).</p>
                    </div>

                    {/* --- Exercise Angina --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="exercise-angina" className="block text-base/6 font-medium text-gray-900 dark:text-white">Exercise-induced angina <RequiredMark /></label>
                        <div className="mt-2">
                            <select id="exercise-angina" required defaultValue="" name="exercise-angina" className="block w-full rounded-lg bg-white px-3 py-2 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:focus:outline-indigo-500">
                                <option className="dark:bg-gray-800 dark:text-white" value="" disabled>Did patient have angina?</option>
                                <option className="dark:bg-gray-800 dark:text-white" value="Y">Yes</option>
                                <option className="dark:bg-gray-800 dark:text-white" value="N">No</option>
                            </select>
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">Pain specifically caused by exercise.</p>
                    </div>

                    {/* --- Oldpeak --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="oldpeak" className="block text-base/6 font-medium text-gray-900 dark:text-white">Oldpeak <RequiredMark /></label>
                        <div className="mt-2">
                            <input id="oldpeak" required type="number" name="oldpeak" step="0.1" min="0" max="6.2" placeholder="e.g. 1.5"
                                className="block w-full rounded-lg bg-white px-3 py-1.5 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500" />
                        </div>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">ST depression induced by exercise (Range: 0 - 6.2).</p>
                    </div>

                    {/* --- ST Slope --- */}
                    <div className="sm:col-span-3">
                        <label htmlFor="st-slope" className="block text-base/6 font-medium text-gray-900 dark:text-white">ST Slope <RequiredMark /></label>
                        <div className="mt-2">
                            <select id="st-slope" required defaultValue="" name="st-slope" className="block w-full rounded-lg bg-white px-3 py-2 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:focus:outline-indigo-500">
                                <option className="dark:bg-gray-800 dark:text-white" value="" disabled>Select the slope curve...</option>
                                <option className="dark:bg-gray-800 dark:text-white" value="Up">Upsloping</option>
                                <option className="dark:bg-gray-800 dark:text-white" value="Flat">Flat</option>
                                <option className="dark:bg-gray-800 dark:text-white" value="Down">Downsloping</option>
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
                <div ref={resultRef} className="mt-12 mb-12 animate-fade-in px-2 sm:px-8">
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
                            <p className="text-sm text-gray-500 mt-2 text-center italic dark:text-gray-200">
                                Visualizes how individual factors shift the prediction from the baseline.
                                <span className="font-bold text-red-500"> Red bars</span> indicate factors increasing heart failure risk,
                                while <span className="font-bold text-blue-500"> Blue bars</span> indicate factors decreasing the risk.
                            </p>
                        </div>
                        <div className="lg:col-span-2">
                            {renderChartImage(result.shap_bar, "Global Feature Importance (SHAP Bar)")}
                            <p className="text-sm text-gray-500 mt-2 text-center italic dark:text-gray-200">
                                Ranks the health indicators by their absolute impact on this prediction. Longer bars mean the AI considered these factors most critical for this patient.
                            </p>
                        </div>
                        <div className="lg:col-span-2">
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