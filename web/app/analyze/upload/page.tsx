"use client";
import { useState, ChangeEvent, DragEvent } from "react";
import { useAuth } from "@/context/authcontext"
import { useRouter } from "next/navigation";
import { FaUpload } from "react-icons/fa6";
import { FiFileText, FiLoader } from "react-icons/fi";
import { IoMdClose } from "react-icons/io";
import toast from "react-hot-toast";
import { api } from "@/lib/api";

export default function Upload() {
    // File states
    const [file, setFile] = useState<File | null>(null);
    const [isDragging, setIsDragging] = useState(false);
    
    // Upload & Dataset states
    const [isUploading, setIsUploading] = useState(false);
    const [datasetId, setDatasetId] = useState<string | null>(null);
    const [columns, setColumns] = useState<string[]>([]);
    const { user } = useAuth();
    const [selectedTarget, setSelectedTarget] = useState<string>("");
    
    const router = useRouter();
    const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0];
        validateAndSetFile(selectedFile);
    };

    const handleDrop = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragging(false);
        const droppedFile = e.dataTransfer.files[0];
        validateAndSetFile(droppedFile);
    };

    const validateAndSetFile = (selectedFile: File | undefined) => {
        if (selectedFile && selectedFile.name.endsWith('.csv')) {
            setFile(selectedFile);
            resetDatasetData();
        } else {
            toast.error("Please select a .csv file only");
        }
    };

    const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => {
        setIsDragging(false);
    };

    const removeFile = () => {
        setFile(null);
        resetDatasetData();
    };

    const resetDatasetData = () => {
        setDatasetId(null);
        setColumns([]);
        setSelectedTarget("");
    };

    // Handle File Upload API
    const handleUpload = async () => {
        if (!file) return;

        if (!user) {
            toast.error("Please sign in to continue.");
            router.push("/login");
            return;
        }

        setIsUploading(true);
        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await api.post("/datasets/upload", formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            });

            const { dataset_id, columns } = response.data;
            setDatasetId(dataset_id);
            setColumns(columns);
        } catch (error) {
            console.error("Upload error:", error);
            toast.error("An error occurred during upload. Please try again.");
        } finally {
            setIsUploading(false);
        }
    };

    // Handle Next Step / Proceed
    const handleProceed = () => {
        if (!selectedTarget) {
            toast.error("Please select a target column!");
            return;
        }
        
        toast.success(`Selected column: ${selectedTarget}. Proceeding...`);
        router.push(`/analyze/eda?id=${datasetId}&target=${encodeURIComponent(selectedTarget)}`);
    };

    return (
        <div className="p-4 h-full flex flex-col">
            <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Dataset Analysis & Feature Selection</h1>
                <p className="mt-1 text-base/6 text-gray-600 dark:text-gray-400">
                    Upload your dataset to generate insights and identify the most important features
                </p>
            </div>
            
            <div className="flex flex-col justify-center align-middle h-full items-center flex-1">
                <div className="w-[60%] min-w-[300px]">
                    
                    {/* Dropzone Area */}
                    <div
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        className={`relative border-2 border-dashed rounded-xl p-10 transition-all duration-200 flex flex-col items-center justify-center my-auto
                            ${isDragging
                                ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-900/20"
                                : "border-gray-300 bg-gray-50 dark:border-gray-500 dark:bg-gray-800/50 hover:border-gray-400 dark:hover:border-gray-500"
                            }`}
                    >
                        {!file ? (
                            <>
                                <div className="bg-indigo-100 dark:bg-indigo-900/50 p-4 rounded-full mb-4">
                                    <FaUpload className="w-8 h-8 text-indigo-600 dark:text-indigo-400" />
                                </div>
                                <p className="text-lg font-medium text-gray-700 dark:text-gray-300 text-center">
                                    Drag and drop your dataset here
                                </p>
                                <p className="text-sm text-gray-500 dark:text-gray-400 mb-4 text-center">
                                    Or click to upload from your computer
                                </p>
                                <input
                                    type="file"
                                    accept=".csv"
                                    onChange={handleFileChange}
                                    disabled={isUploading}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
                                />
                                <span className="px-4 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-200 rounded-lg text-sm font-semibold shadow-sm pointer-events-none">
                                    Choose .CSV File
                                </span>
                            </>
                        ) : (
                            <div className="w-full flex items-center justify-between bg-white dark:bg-gray-800 p-4 rounded-lg border border-green-200 dark:border-green-900/50 shadow-sm">
                                <div className="flex items-center space-x-4">
                                    <div className="p-2 bg-green-100 dark:bg-green-900/30 rounded-lg">
                                        <FiFileText className="w-6 h-6 text-green-600 dark:text-green-400" />
                                    </div>
                                    <div className="overflow-hidden">
                                        <p className="text-sm font-semibold text-gray-800 dark:text-gray-200 truncate max-w-[200px] sm:max-w-[300px]">
                                            {file.name}
                                        </p>
                                        <p className="text-xs text-gray-500 dark:text-gray-400">
                                            {(file.size / 1024 / 1024).toFixed(2)} MB
                                        </p>
                                    </div>
                                </div>
                                <div className="flex items-center space-x-2">
                                    <button
                                        onClick={removeFile}
                                        disabled={isUploading}
                                        className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full transition-colors hover:cursor-pointer disabled:opacity-50"
                                    >
                                        <IoMdClose className="w-5 h-5 text-red-500" />
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Upload Button */}
                    {file && !datasetId && (
                        <button 
                            onClick={handleUpload}
                            disabled={isUploading}
                            className="mt-6 w-full py-3 flex justify-center items-center gap-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-400 dark:disabled:bg-indigo-800 hover:cursor-pointer text-white font-bold rounded-lg transition-colors shadow-lg"
                        >
                            {isUploading ? (
                                <>
                                    <FiLoader className="w-5 h-5 animate-spin" />
                                    Uploading...
                                </>
                            ) : (
                                "Upload & Analyze Data"
                            )}
                        </button>
                    )}

                    {/* Target Column Selection Area */}
                    {datasetId && columns.length > 0 && (
                        <div className="mt-8 p-6 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl shadow-sm animate-in fade-in slide-in-from-bottom-4 duration-500">
                            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                                Select Target Column
                            </h3>
                            
                            <select
                                value={selectedTarget}
                                onChange={(e) => setSelectedTarget(e.target.value)}
                                className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500 outline-none transition-shadow hover:cursor-pointer"
                            >
                                <option value="" disabled>-- Select a column --</option>
                                {columns.map((col, index) => (
                                    <option key={index} value={col}>
                                        {col}
                                    </option>
                                ))}
                            </select>

                            <button 
                                onClick={handleProceed}
                                disabled={!selectedTarget}
                                className="mt-6 w-full py-3 hover:cursor-pointer bg-green-600 hover:bg-green-700 disabled:bg-gray-400 dark:disabled:bg-gray-700 disabled:hover:bg-gray-400 dark:disabled:hover:bg-gray-700 disabled:pointer-events-none text-white font-bold rounded-lg transition-colors shadow-lg"
                            >
                                Confirm & Proceed
                            </button>
                        </div>
                    )}

                </div>
            </div>
        </div>
    );
}