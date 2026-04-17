"use client";
import { useEffect, useState, useRef } from "react";
import { IoMdClose, IoMdCamera } from "react-icons/io";
import { FaLock, FaRocket, FaUser, FaEnvelope } from "react-icons/fa6";
import { useAuth } from "@/context/authcontext";
import { toast } from "react-hot-toast";
import UpdatePasswordModal from "@/components/updatePasswordModal";

interface ProfileModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export default function ProfileModal({ isOpen, onClose }: ProfileModalProps) {
    const [isVisible, setIsVisible] = useState(false);
    const { user } = useAuth();
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Basic info states
    const [username, setUsername] = useState("");
    const [email, setEmail] = useState("");
    const [avatarPreview, setAvatarPreview] = useState<string | null>(null);

    // Trạng thái cho modal đổi mật khẩu
    const [isPasswordModalOpen, setIsPasswordModalOpen] = useState(false);

    useEffect(() => {
        if (user && isOpen) {
            setUsername(user.username || "");
            setEmail(user.email || "");
        }
    }, [user, isOpen]);

    useEffect(() => {
        if (isOpen) {
            const timer = setTimeout(() => setIsVisible(true), 10);
            return () => clearTimeout(timer);
        } else {
            setIsVisible(false);
        }
    }, [isOpen]);

    useEffect(() => {
        if (isOpen || isPasswordModalOpen) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = 'unset';
        }
        return () => { document.body.style.overflow = 'unset'; };
    }, [isOpen, isPasswordModalOpen]);

    // Đóng modal khi nhấn phím Escape
    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === "Escape") {
                // Nếu modal Update Password đang mở thì KHÔNG đóng Profile Modal
                if (!isPasswordModalOpen) {
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
    }, [isOpen, isPasswordModalOpen, onClose]);

    const handleAvatarClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onloadend = () => {
                setAvatarPreview(reader.result as string);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleSaveProfile = () => {
        // Call API update profile (avatar, etc.) logic here
        toast.success("Profile updated successfully!");
        onClose();
    };

    if (!isOpen) return null;

    return (
        <>
            <div className="relative z-50" aria-labelledby="modal-title" role="dialog">
                <div
                    className={`fixed inset-0 bg-gray-500/75 transition-opacity duration-300 ease-out ${isVisible ? "opacity-100" : "opacity-0"}`}
                    onClick={onClose}
                >
                    <div className="flex min-h-full items-center justify-center p-4 text-center sm:p-0">
                        <div
                            className={`relative transform overflow-hidden rounded-2xl bg-white text-left shadow-xl transition-all sm:my-8 w-full sm:max-w-md duration-300 ease-out ${isVisible ? "opacity-100 translate-y-0 sm:scale-100" : "opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"}`}
                            onClick={(e) => e.stopPropagation()}
                        >
                            <div className="bg-white px-4 pb-4 pt-5 sm:p-6 dark:bg-[#18181B]">
                                {/* Header */}
                                <div className="flex items-center justify-between mb-6">
                                    <h3 className="text-xl font-bold text-gray-900 dark:text-white">Profile</h3>
                                    <button onClick={onClose} className="p-1.5 hover:cursor-pointer text-gray-400 hover:bg-gray-100 dark:hover:bg-white/10 rounded-full transition-colors">
                                        <IoMdClose className="w-6 h-6" />
                                    </button>
                                </div>

                                <div className="space-y-6">
                                    {/* Avatar */}
                                    <div className="flex flex-col items-center">
                                        <div className="relative group cursor-pointer" onClick={handleAvatarClick}>
                                            <div className="w-24 h-24 rounded-full bg-indigo-600 flex items-center justify-center text-white text-3xl font-bold overflow-hidden shadow-inner">
                                                {avatarPreview ? (
                                                    <img src={avatarPreview} alt="Preview" className="w-full h-full object-cover" />
                                                ) : (
                                                    <FaRocket className="w-10 h-10 text-white" />
                                                )}
                                            </div>
                                            <div className="absolute inset-0 bg-black/40 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                                                <IoMdCamera className="text-white text-2xl" />
                                            </div>
                                            <input
                                                type="file"
                                                ref={fileInputRef}
                                                className="hidden"
                                                accept="image/*"
                                                onChange={handleFileChange}
                                            />
                                        </div>
                                    </div>

                                    {/* Basic info */}
                                    <div className="space-y-4">
                                        <div>
                                            <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1.5">
                                                <FaUser className="text-xs" /> Username
                                            </label>
                                            <input
                                                type="text"
                                                value={username}
                                                disabled
                                                className="w-full px-4 py-2.5 bg-gray-100 border border-gray-200 dark:bg-white/5 dark:border-[#FFFFFF1A] dark:text-gray-400 rounded-xl cursor-not-allowed"
                                            />
                                        </div>
                                        <div>
                                            <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1.5">
                                                <FaEnvelope className="text-xs" /> Email
                                            </label>
                                            <input
                                                type="email"
                                                value={email}
                                                disabled
                                                className="w-full px-4 py-2.5 bg-gray-100 border border-gray-200 dark:bg-white/5 dark:border-[#FFFFFF1A] dark:text-gray-400 rounded-xl cursor-not-allowed"
                                            />
                                        </div>

                                        {/* Manage Password Row */}
                                        <div>
                                            <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1.5">
                                                <FaLock className="text-xs" /> Password
                                            </label>
                                            <div className="flex items-center justify-between w-full px-4 py-2.5 bg-gray-100 border border-gray-200 dark:bg-white/5 dark:border-[#FFFFFF1A] rounded-xl">
                                                <span className="text-gray-400 dark:text-gray-500 tracking-widest font-bold">
                                                    ••••••••
                                                </span>
                                                <button
                                                    onClick={() => setIsPasswordModalOpen(true)}
                                                    className="text-sm font-bold hover:cursor-pointer text-indigo-600 dark:text-indigo-400 hover:underline"
                                                >
                                                    Edit
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Actions */}
                                <div className="mt-8 flex gap-3">
                                    <button
                                        onClick={onClose}
                                        className="flex-1 px-4 hover:cursor-pointer py-2.5 border border-gray-300 dark:border-[#FFFFFF1A] dark:text-white rounded-xl hover:bg-gray-50 dark:hover:bg-white/5 transition font-semibold"
                                    >
                                        Close
                                    </button>
                                    <button
                                        onClick={handleSaveProfile}
                                        className="flex-1 px-4 hover:cursor-pointer py-2.5 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 shadow-lg shadow-indigo-500/20 transition font-semibold"
                                    >
                                        Save Changes
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <UpdatePasswordModal
                isOpen={isPasswordModalOpen}
                onClose={() => setIsPasswordModalOpen(false)}
            />
        </>
    );
}