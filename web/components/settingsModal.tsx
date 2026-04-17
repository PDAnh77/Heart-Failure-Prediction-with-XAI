"use client";
import SettingRow from "@/components/settingRow";
import { useSettings } from "@/context/settingscontext";
import { useTheme } from "next-themes";
import { IoMdClose } from "react-icons/io";
import { useEffect, useState } from "react";

interface SettingsModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export default function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
    const [isVisible, setIsVisible] = useState(false);
    const { savePrediction, setSavePrediction } = useSettings();
    const { snowMode, setSnowMode } = useSettings();
    const { theme, setTheme } = useTheme();
    const isDarkMode = theme === 'dark';

    // Xử lý hiệu ứng transition khi mở modal
    useEffect(() => {
        if (isOpen) {
            const timer = setTimeout(() => setIsVisible(true), 10);
            return () => clearTimeout(timer);
        } else {
            setIsVisible(false);
        }
    }, [isOpen]);

    // Ngăn chặn cuộn nền khi modal đang mở
    useEffect(() => {
        if (isOpen) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = 'unset';
        }
        return () => {
            document.body.style.overflow = 'unset';
        };
    }, [isOpen]);

    // Đóng modal khi nhấn phím Escape
    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === "Escape") {
                onClose();
            }
        };
        if (isOpen) {
            document.addEventListener("keydown", handleKeyDown);
        }
        return () => {
            document.removeEventListener("keydown", handleKeyDown);
        };
    }, [isOpen, onClose]);

    if (!isOpen) return null;

    return (
        <div className="relative z-50" aria-labelledby="modal-title" role="dialog">
            {/* Backdrop (nền) */}
            <div
                className={`fixed inset-0 bg-gray-500/75 transition-opacity duration-300 ease-out ${isVisible ? "opacity-100" : "opacity-0"}`}
                onClick={onClose}
            >
                {/* Vùng Flex chứa Modal để canh giữa */}
                <div className="flex min-h-full items-center justify-center p-4 text-center sm:p-0">

                    {/* Modal Panel */}
                    <div
                        className={`relative transform overflow-hidden rounded-lg bg-white text-left shadow-xl transition-all sm:my-8 w-full sm:max-w-lg duration-300 ease-out ${isVisible ? "opacity-100 translate-y-0 sm:scale-100" : "opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"}`}
                        onClick={(e) => e.stopPropagation()} // Ngăn sự kiện click lan ra backdrop
                    >
                        <div className="bg-white px-4 pb-4 pt-5 sm:p-6 sm:pb-6 dark:bg-gray-800">

                            {/* Header & Nút đóng */}
                            <div className="flex items-center justify-between mb-5">
                                <h3 className="text-xl font-semibold leading-6 text-gray-900 dark:text-white" id="modal-title">
                                    Settings
                                </h3>
                                <button
                                    onClick={onClose}
                                    className="p-1.5 text-gray-400 hover:bg-gray-100 hover:text-gray-700 dark:hover:bg-gray-700 dark:text-gray-300 dark:hover:text-gray-50 rounded-full transition-colors hover:cursor-pointer"
                                    aria-label="Close settings"
                                >
                                    <IoMdClose className="w-6 h-6" />
                                </button>
                            </div>

                            {/* Danh sách Settings */}
                            <div className="divide-y divide-gray-200 dark:divide-gray-700">
                                <SettingRow
                                    title="Dark mode"
                                    description="Switch between light and dark themes to reduce eye strain."
                                    enabled={isDarkMode}
                                    setEnabled={(value) => setTheme(value ? 'dark' : 'light')}
                                />
                                <SettingRow
                                    title="Snowfall effect"
                                    description="Enable falling snow on the background."
                                    enabled={snowMode}
                                    setEnabled={setSnowMode}
                                />
                                <SettingRow
                                    title="Save prediction history"
                                    description="Store prediction results so you can review them later."
                                    enabled={savePrediction}
                                    setEnabled={setSavePrediction}
                                />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}