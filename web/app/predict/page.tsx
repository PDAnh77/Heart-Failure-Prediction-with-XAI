"use client"

import { useRouter } from "next/navigation";
import { FormEvent } from "react";
import { toast } from "react-toastify";

export default function Predict() {
    const router = useRouter();

    const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        const token = localStorage.getItem("access_token");

        if (!token) {
            toast.warning("You must be logged in to submit a prediction.")
            router.push("/login");
            return;
        }
    };

    return (
        <div className="p-4">
            <h1 className="text-2xl font-bold">Patient Information</h1>
            <p className="mt-1 text-base/6 text-gray-600 dark:text-gray-400">Please provide accurate patient information for better prediction results.</p>

            <form onSubmit={handleSubmit} className="mt-16 border-b border-gray-900/10 pb-12 px-12 dark:border-white/10">
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
                            <select id="gender" defaultValue="M" className="block w-full rounded-lg bg-white px-3 py-2 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500">
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
                                id="age" type="number" name="age" min={1} max={120} 
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
                            <select id="chest-pain-type" defaultValue="" className="block w-full rounded-lg bg-white px-3 py-2 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500">
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
                                id="resting-bp" type="number" name="resting-bp" min={1} 
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
                                id="cholesterol" type="number" name="cholesterol" min={1} 
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
                                id="fasting-bs" type="number" name="fasting-bs" min={1} 
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
                            <select id="resting-ecg" defaultValue="" className="block w-full rounded-lg bg-white px-3 py-2 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500">
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
                                id="max-hr" type="number" name="max-hr" min={60} max={220} 
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
                            <select id="exercise-angina" defaultValue="" className="block w-full rounded-lg bg-white px-3 py-2 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500">
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
                                id="oldpeak" type="number" name="oldpeak" step="0.1" min="0" 
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
                            <select id="st-slope" defaultValue="" className="block w-full rounded-lg bg-white px-3 py-2 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-base/6 dark:bg-white/5 dark:text-white dark:outline-white/10 dark:placeholder:text-gray-500 dark:focus:outline-indigo-500">
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
                    <button type="button" className="rounded-md px-8 py-2 text-sm font-semibold text-gray-900 cursor-pointer hover:bg-gray-100 hover:text-indigo-600 dark:text-white dark:hover:bg-white/10 dark:hover:text-indigo-400 transition-colors">Reset</button>
                    <button type="submit" className="rounded-md bg-indigo-600 px-8 py-2 text-sm font-semibold text-white shadow-sm cursor-pointer hover:bg-indigo-500 focus-visible:outline focus-visible:outline-offset-2 focus-visible:outline-indigo-600">Predict</button>
                </div>
            </form>
        </div>
    )
}