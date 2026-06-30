"use client";
import SettingRow from "@/components/settings/settingRow";
import { useSettings } from "@/context/settingscontext";
import { useTheme } from "next-themes";
import { IoMdClose, IoIosArrowDown, IoIosArrowUp } from "react-icons/io";
import { useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";

interface SettingsModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export default function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
    const [isVisible, setIsVisible] = useState(false);
    const [langOpen, setLangOpen] = useState(false);
    const langRef = useRef<HTMLDivElement>(null);

    const {
        savePrediction,
        setSavePrediction,
        snowMode,
        setSnowMode,
        language,
        setLanguage,
        isReady,
    } = useSettings();
    const { theme, setTheme } = useTheme();
    const isDarkMode = theme === "dark";
    const t = useTranslations("settings");

    // Xử lý hiệu ứng transition khi mở modal
    useEffect(() => {
        if (isOpen) {
            const timer = setTimeout(() => setIsVisible(true), 10);
            return () => clearTimeout(timer);
        } else {
            setIsVisible(false);
            setLangOpen(false); // Đóng dropdown khi modal đóng
        }
    }, [isOpen]);

    // Ngăn chặn cuộn nền khi modal đang mở
    useEffect(() => {
        if (isOpen) {
            document.body.style.overflow = "hidden";
        } else {
            document.body.style.overflow = "unset";
        }
        return () => {
            document.body.style.overflow = "unset";
        };
    }, [isOpen]);

    // Đóng modal khi nhấn phím Escape
    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === "Escape") {
                if (langOpen) {
                    setLangOpen(false); // Ưu tiên đóng dropdown trước
                } else {
                    onClose();
                }
            }
        };
        if (isOpen) {
            document.addEventListener("keydown", handleKeyDown);
        }
        return () => {
            document.removeEventListener("keydown", handleKeyDown);
        };
    }, [isOpen, onClose, langOpen]);

    // Đóng dropdown ngôn ngữ khi click ra ngoài
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (langRef.current && !langRef.current.contains(event.target as Node)) {
                setLangOpen(false);
            }
        };
        if (langOpen) {
            document.addEventListener("mousedown", handleClickOutside);
        }
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, [langOpen]);

    if (!isOpen) return null;

    const languageOptions: { value: "vi" | "en"; label: string }[] = [
        { value: "vi", label: t("languageVi") },
        { value: "en", label: t("languageEn") },
    ];

    const currentLanguageLabel =
        languageOptions.find((o) => o.value === language)?.label ?? language;

    return (
        <div className="relative z-50" aria-labelledby="modal-title" role="dialog">
            {/* Backdrop */}
            <div
                className={`fixed inset-0 bg-gray-500/75 transition-opacity duration-300 ease-out ${isVisible ? "opacity-100" : "opacity-0"
                    }`}
                onClick={onClose}
            >
                {/* Flex wrapper để canh giữa modal */}
                <div className="flex min-h-full items-center justify-center p-4 text-center sm:p-0">

                    {/* Modal Panel — bỏ overflow-hidden để dropdown không bị clip */}
                    <div
                        className={`relative transform rounded-lg bg-white text-left shadow-xl transition-all sm:my-8 w-full sm:max-w-lg duration-300 ease-out ${isVisible
                                ? "opacity-100 translate-y-0 sm:scale-100"
                                : "opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"
                            }`}
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="bg-white px-4 pb-4 pt-5 sm:p-6 sm:pb-6 dark:bg-gray-800 rounded-lg">

                            {/* Header & Nút đóng */}
                            <div className="flex items-center justify-between mb-5">
                                <h3
                                    className="text-xl font-semibold leading-6 text-gray-900 dark:text-white"
                                    id="modal-title"
                                >
                                    {t("title")}
                                </h3>
                                <button
                                    onClick={onClose}
                                    className="p-1.5 text-gray-400 hover:bg-gray-100 hover:text-gray-700 dark:hover:bg-gray-700 dark:text-gray-300 dark:hover:text-gray-50 rounded-full transition-colors hover:cursor-pointer"
                                    aria-label={t("closeAria")}
                                >
                                    <IoMdClose className="w-6 h-6" />
                                </button>
                            </div>

                            {/* Danh sách Settings */}
                            <div className="divide-y divide-gray-200 dark:divide-gray-700">

                                {/* Dark Mode */}
                                <SettingRow
                                    title={t("darkMode")}
                                    description=""
                                    enabled={isDarkMode}
                                    setEnabled={(value) => setTheme(value ? "dark" : "light")}
                                />

                                {/* Language — Custom Dropdown */}
                                <div className="flex items-center justify-between py-4">
                                    <div className="flex flex-col pr-4">
                                        <span className="text-base font-medium text-gray-900 dark:text-white">
                                            {t("language")}
                                        </span>
                                    </div>

                                    <div className="relative w-40 shrink-0" ref={langRef}>
                                        {/* Trigger Button */}
                                        <button
                                            type="button"
                                            disabled={!isReady}
                                            onClick={() => setLangOpen((prev) => !prev)}
                                            className="w-full flex items-center justify-between rounded-md bg-white px-3 py-2 text-sm text-gray-900 shadow-sm ring-1 ring-gray-300 hover:ring-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-gray-900 dark:text-gray-100 dark:ring-gray-600 dark:hover:ring-gray-500 transition-all"
                                            aria-haspopup="listbox"
                                            aria-expanded={langOpen}
                                        >
                                            <span>{currentLanguageLabel}</span>
                                            {langOpen
                                                ? <IoIosArrowUp className="w-4 h-4 ml-2 text-gray-500 dark:text-gray-400" />
                                                : <IoIosArrowDown className="w-4 h-4 ml-2 text-gray-500 dark:text-gray-400" />
                                            }
                                        </button>

                                        {/* Dropdown List — absolute, luôn hiển thị đúng vị trí */}
                                        {langOpen && (
                                            <ul
                                                role="listbox"
                                                className="absolute z-9999 mt-1 w-full rounded-md bg-white shadow-lg ring-1 ring-black/10 dark:bg-gray-800 dark:ring-gray-600 overflow-hidden"
                                            >
                                                {languageOptions.map((option) => (
                                                    <li
                                                        key={option.value}
                                                        role="option"
                                                        aria-selected={language === option.value}
                                                        onClick={() => {
                                                            setLanguage(option.value);
                                                            setLangOpen(false);
                                                        }}
                                                        className={`cursor-pointer px-3 py-2 text-sm select-none transition-colors
                                                            ${language === option.value
                                                                ? "bg-blue-50 text-blue-600 font-semibold dark:bg-blue-900/40 dark:text-blue-400"
                                                                : "text-gray-900 dark:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700"
                                                            }`}
                                                    >
                                                        {option.label}
                                                    </li>
                                                ))}
                                            </ul>
                                        )}
                                    </div>
                                </div>

                                {/* Snow Mode */}
                                <SettingRow
                                    title={t("snowfall")}
                                    description=""
                                    enabled={snowMode}
                                    setEnabled={setSnowMode}
                                />

                                {/* Save Prediction History */}
                                <SettingRow
                                    title={t("savePredictionHistory")}
                                    description=""
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