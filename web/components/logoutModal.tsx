import { useEffect, useState } from "react";
import { FaRightFromBracket } from "react-icons/fa6";
import { useTranslations } from "next-intl";

interface LogoutModalProps {
    isOpen: boolean;
    onClose: () => void;
    onConfirm: () => void;
}

export default function LogoutModal({ isOpen, onClose, onConfirm }: LogoutModalProps) {
    const [isVisible, setIsVisible] = useState(false);
    const t = useTranslations("logoutModal");
    const tCommon = useTranslations("common");

    useEffect(() => {
        if (isOpen) {
            const timer = setTimeout(() => setIsVisible(true), 10);
            return () => clearTimeout(timer);
        } else {
            setIsVisible(false);
        }
    }, [isOpen]);

    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === "Escape") {
                onClose();
            }
        };
        if (isOpen) {
            document.addEventListener("keydown", handleKeyDown);
        }
        // Cleanup tránh memory leak
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
                onMouseDown={onClose}
            >
                <div className="flex min-h-full items-center justify-center p-4 text-center sm:items-center sm:p-0">
                    {/* Modal Panel */}
                    <div
                        className="relative transform overflow-hidden rounded-lg bg-white text-left shadow-xl transition-all sm:my-8 sm:w-full sm:max-w-lg" 
                        onMouseDown={(e) => e.stopPropagation()}
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="bg-white px-4 pb-4 pt-5 sm:p-6 sm:pb-4 dark:bg-gray-800">
                            <div className="sm:flex sm:items-start">
                                <div className="mx-auto flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-red-100 sm:mx-0 sm:h-10 sm:w-10">
                                    <FaRightFromBracket className="h-6 w-6 text-red-600" />
                                </div>
                                <div className="mt-3 text-center sm:ml-4 sm:mt-0 sm:text-left">
                                    <h3 className="text-base font-semibold leading-6 text-gray-900 dark:text-white" id="modal-title">
                                        {t("title")}
                                    </h3>
                                    <div className="mt-2">
                                        <p className="text-sm text-gray-500 dark:text-gray-300">
                                            {t("body")}
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="bg-gray-50 px-4 py-3 sm:flex sm:flex-row-reverse sm:px-6 dark:bg-gray-700">
                            <button
                                type="button"
                                onClick={onConfirm}
                                className="inline-flex w-full cursor-pointer justify-center rounded-xl bg-red-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-red-500 sm:ml-3 sm:w-auto"
                            >
                                {tCommon("confirm")}
                            </button>
                            <button
                                type="button"
                                onClick={onClose}
                                className="mt-3 inline-flex w-full cursor-pointer justify-center rounded-xl bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50 sm:mt-0 sm:w-auto dark:bg-gray-600 dark:text-white dark:ring-gray-500 dark:hover:bg-gray-500"
                            >
                                {tCommon("cancel")}
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}