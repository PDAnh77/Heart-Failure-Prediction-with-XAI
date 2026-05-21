"use client";
import { useEffect, useState } from "react";
import { IoMdClose } from "react-icons/io";
import { FaLock } from "react-icons/fa6";
import { toast } from "react-hot-toast";
import { api } from "@/lib/api";
import { useTranslations } from "next-intl";

interface UpdatePasswordModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export default function UpdatePasswordModal({ isOpen, onClose }: UpdatePasswordModalProps) {
    const [isVisible, setIsVisible] = useState(false);
    const [newPassword, setNewPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const t = useTranslations("updatePassword");
    const tCommon = useTranslations("common");

    useEffect(() => {
        if (isOpen) {
            setNewPassword("");
            setConfirmPassword("");
            const timer = setTimeout(() => setIsVisible(true), 10);
            return () => clearTimeout(timer);
        } else {
            setIsVisible(false);
        }
    }, [isOpen]);

    const handleSave = async () => {
        if (!newPassword || !confirmPassword) {
            toast.error(t("errors.required"));
            return;
        }

        if (newPassword !== confirmPassword) {
            toast.error(t("errors.mismatch"));
            return;
        }

        try {
            setIsLoading(true);

            await api.put("/users/me/password", {
                password: newPassword,
            });

            toast.success(t("toast.success"));
            onClose();
        } catch (error: any) {
            console.error("Error updating password:", error);

            const errorMessage =
                error.response?.data?.message ||
                t("toast.genericError");

            toast.error(errorMessage);
        } finally {
            setIsLoading(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="relative z-60" aria-labelledby="modal-title" role="dialog">
            <div
                className={`fixed inset-0 bg-gray-500/75 transition-opacity duration-300 ease-out ${isVisible ? "opacity-100" : "opacity-0"}`}>
                <div className="flex min-h-full items-center justify-center p-4 text-center sm:p-0">
                    <div
                        className={`relative transform overflow-hidden rounded-2xl bg-white text-left shadow-xl transition-all sm:my-8 w-full sm:max-w-sm duration-300 ease-out ${isVisible
                            ? "opacity-100 translate-y-0 sm:scale-100"
                            : "opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"
                            }`}
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="bg-white px-4 pb-4 pt-5 sm:p-6 dark:bg-[#18181B]">
                            {/* Header */}
                            <div className="flex items-center justify-between mb-6">
                                <h3 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                                    <FaLock className="text-indigo-600 dark:text-indigo-400" />
                                    {t("title")}
                                </h3>
                                <button
                                    onClick={onClose}
                                    className="p-1.5 hover:cursor-pointer text-gray-400 hover:bg-gray-100 dark:hover:bg-white/10 rounded-full transition-colors"
                                >
                                    <IoMdClose className="w-6 h-6" />
                                </button>
                            </div>

                            {/* Body */}
                            <div className="space-y-4">
                                <div>
                                    <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1.5">
                                        {t("fields.newPassword")}
                                    </label>
                                    <input
                                        type="password"
                                        placeholder={t("fields.newPasswordPlaceholder")}
                                        value={newPassword}
                                        onChange={(e) => setNewPassword(e.target.value)}
                                        disabled={isLoading}
                                        className="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 dark:bg-[#1A1A1B] dark:border-[#FFFFFF1A] dark:text-white rounded-xl focus:ring-2 focus:ring-indigo-500 outline-none transition-all disabled:opacity-50"
                                    />
                                </div>

                                <div>
                                    <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1.5">
                                        {t("fields.confirmPassword")}
                                    </label>
                                    <input
                                        type="password"
                                        placeholder={t("fields.confirmPasswordPlaceholder")}
                                        value={confirmPassword}
                                        onChange={(e) => setConfirmPassword(e.target.value)}
                                        disabled={isLoading}
                                        className="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 dark:bg-[#1A1A1B] dark:border-[#FFFFFF1A] dark:text-white rounded-xl focus:ring-2 focus:ring-indigo-500 outline-none transition-all disabled:opacity-50"
                                    />
                                </div>
                            </div>

                            {/* Actions */}
                            <div className="mt-8 flex gap-3">
                                <button
                                    onClick={onClose}
                                    disabled={isLoading}
                                    className="flex-1 px-4 py-2.5 border hover:cursor-pointer border-gray-300 dark:border-[#FFFFFF1A] dark:text-white rounded-xl hover:bg-gray-50 dark:hover:bg-white/5 transition font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    {tCommon("cancel")}
                                </button>

                                <button
                                    onClick={handleSave}
                                    disabled={isLoading}
                                    className="flex-1 flex justify-center items-center px-4 py-2.5 bg-indigo-600 hover:cursor-pointer text-white rounded-xl hover:bg-indigo-700 shadow-lg shadow-indigo-500/20 transition font-semibold disabled:opacity-70 disabled:cursor-not-allowed"
                                >
                                    {isLoading ? (
                                        <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                                    ) : (
                                        t("actions.update")
                                    )}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}