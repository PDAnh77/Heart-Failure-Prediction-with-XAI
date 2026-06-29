"use client";
import Link from "next/link";
import Image from 'next/image';
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState, useRef } from "react";
import { toast } from 'react-hot-toast';
import { useLocale, useTranslations } from "next-intl";
import LogoutModal from "@/components/modals/logoutModal";
import DeleteModal from "@/components/modals/deletePredictionModal";
import SettingsModal from "@/components/modals/settingsModal";
import ProfileModal from "@/components/modals/profileModal";
import { useAuth } from "@/context/authcontext"
import { TbLayoutSidebarFilled } from "react-icons/tb";
import { FaRocket, FaRegTrashCan } from "react-icons/fa6";
import { FiHome } from "react-icons/fi";
import { LuUserSearch, LuSettings, LuLogOut, LuLogIn, LuCircleUserRound, LuUsers, LuUser } from "react-icons/lu";
import { MdInsertChartOutlined } from "react-icons/md";
import { IoIosArrowForward, IoIosArrowDown } from "react-icons/io";
import { api } from "@/lib/api";
import { UnifiedHistoryItem } from "@/types/prediction";
import { PiHeartbeatFill } from "react-icons/pi";

export default function Sidebar() {
    const [open, setOpen] = useState(false);
    const [userMenuOpen, setUserMenuOpen] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);

    const pathname = usePathname();
    const [showPredictions, setShowPredictions] = useState(true);
    const [showLogoutModal, setShowLogoutModal] = useState(false);
    const [showSettingModal, setShowSettingModal] = useState(false);
    const [showProfileModal, setShowProfileModal] = useState(false);

    const { user, logout, newHistoryItem } = useAuth();
    const [result, setResult] = useState<UnifiedHistoryItem[] | null>(null);
    const router = useRouter();
    const [deleteItem, setDeleteItem] = useState<UnifiedHistoryItem | null>(null);
    const [isDeleting, setIsDeleting] = useState(false);
    const [offset, setOffset] = useState(0);
    const [hasMore, setHasMore] = useState(false);
    const [loadingMore, setLoadingMore] = useState(false);
    const t = useTranslations("sidebar");
    const tCommon = useTranslations("common");
    const locale = useLocale();
    const localeTag = locale === "vi" ? "vi-VN" : "en-US";

    const handleLogoutClick = () => {
        setShowLogoutModal(true);
    };

    useEffect(() => {
        if (!user) {
            setResult(null);
            return;
        }
        const loadPredictionHistory = async () => {
            try {
                const res = await api.get(`predictions/history/me?limit=12&offset=0`);
                setResult(res.data);
                setOffset(12);
                setHasMore(res.data.length === 12);
            } catch (error: any) {
                console.log(error);
            }
        }
        loadPredictionHistory();
    }, [user]);

    useEffect(() => {
        if (!newHistoryItem) return;

        setResult(prev => {
            if (!prev) return [newHistoryItem];
            if (prev.some(p => p.id === newHistoryItem.id)) return prev;

            return [newHistoryItem, ...prev];
        });
    }, [newHistoryItem]);

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setUserMenuOpen(false);
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    const handleLogout = async () => {
        try {
            api.post("auth/logout");
            logout();
            setShowLogoutModal(false);
            setOpen(false);
            router.push("/login");
        } catch (error: any) {
            console.log(error);
            toast.error(t("logoutFailed"));
        }
    };

    const handleConfirmDelete = async () => {
        if (!deleteItem) return;
        setIsDeleting(true);
        try {
            if (deleteItem.type === "batch") {
                await api.delete(`/predictions/batch/${deleteItem.id}`);
            } else {
                await api.delete(`/predictions/${deleteItem.id}`);
            }

            setResult((prevResult) => prevResult?.filter((item) => item.id !== deleteItem.id) || null);
            setDeleteItem(null);
        } catch (error) {
            toast.error("Failed to delete history");
        } finally {
            setIsDeleting(false);
        }
    };

    const formatDateTime = (dateString: string) => {
        const date = new Date(dateString);

        const parts = new Intl.DateTimeFormat(localeTag, {
            hour: '2-digit',
            minute: '2-digit',
            day: '2-digit',
            month: '2-digit',
            year: 'numeric',
        }).formatToParts(date);

        const hour = parts.find(p => p.type === 'hour')?.value;
        const minute = parts.find(p => p.type === 'minute')?.value;
        const day = parts.find(p => p.type === 'day')?.value;
        const month = parts.find(p => p.type === 'month')?.value;
        const year = parts.find(p => p.type === 'year')?.value;

        return `${hour}:${minute} - ${day}/${month}/${year}`;
    };

    const itemClass = (active: boolean) =>
        `rounded-xl transition ${active ? "bg-gray-100 font-semibold text-sm dark:text-white dark:bg-white/10" : "hover:bg-gray-100 text-sm dark:hover:bg-white/10"}`;

    const loadMoreItems = async () => {
        if (!hasMore || loadingMore) return;
        setLoadingMore(true);
        try {
            const res = await api.get(`predictions/history/me?limit=12&offset=${offset}`);
            const newItems: UnifiedHistoryItem[] = res.data || [];

            setResult((prev) => {
                const current = prev ?? [];
                const filteredNewItems = newItems.filter(
                    (newItem) => !current.some((oldItem) => oldItem.id === newItem.id)
                );
                return [...current, ...filteredNewItems];
            });

            setOffset((prevOffset) => prevOffset + 12);
            if (newItems.length < 12) {
                setHasMore(false);
            }
        } catch (error) {
            // Dừng việc gọi API liên tục nếu server báo lỗi khi hết page
            setHasMore(false);
        } finally {
            setLoadingMore(false);
        }
    }

    const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
        if (!hasMore || loadingMore) {
            return;
        }

        const { scrollTop, scrollHeight, clientHeight } = e.currentTarget;

        if (scrollHeight - scrollTop - clientHeight < 50) {
            loadMoreItems();
        }
    };

    return (
        <>
            {/* MOBILE HEADER */}
            <div className="lg:hidden flex items-center p-2 bg-white dark:bg-[#141516] border-b border-gray-200 dark:border-[#FFFFFF1A] sticky top-0 z-40">
                <button
                    className="p-2 rounded-xl cursor-pointer"
                    onClick={() => setOpen(true)}
                >
                    <TbLayoutSidebarFilled className="text-xl" />
                </button>
                <span className="font-bold ml-2">{tCommon("appName")}</span>
            </div>

            <div
                className={`
                    fixed inset-0 bg-black/50 z-40 lg:hidden 
                    transition-all duration-300 ease-in-out
                    ${open
                        ? "opacity-100 visible pointer-events-auto"
                        : "opacity-0 invisible pointer-events-none"
                    }
                `}
                onClick={() => setOpen(false)}
            />

            <aside className={`
                fixed inset-y-0 left-0 z-50 lg:z-40 w-72 p-2 transition-transform duration-300 ease-in-out bg-gray-50 dark:bg-[#141516] lg:bg-transparent lg:dark:bg-transparent
                ${open ? "translate-x-0" : "-translate-x-full"} 
                lg:static lg:translate-x-0 lg:min-h-screen
            `}>
                <div
                    className="bg-gray-50 rounded-xl h-full lg:border lg:border-gray-200 lg:shadow-md dark:bg-[#18181B] dark:border-[#FFFFFF1A] flex flex-col overflow-y-auto relative overscroll-none"
                    onScroll={handleScroll}
                >

                    {/* MENU TRÊN */}
                    <div className="p-2 space-y-2 sticky top-0 z-10 bg-gray-50 dark:bg-[#18181B]">
                        <div className="flex gap-1 p-2 my-4">
                            <PiHeartbeatFill className="text-xl text-red-500" />
                            <p className="font-bold text-sm">{tCommon("appName")}</p>
                        </div>
                        <ul className="space-y-2">
                            <li className={itemClass(pathname === "/")} onClick={() => setOpen(false)}>
                                <Link href="/" className="flex gap-2 p-2">
                                    <FiHome className="text-lg" />
                                    <span>{t("home")}</span>
                                </Link>
                            </li>
                            <li className={itemClass(pathname === "/predict/individual")} onClick={() => setOpen(false)}>
                                <Link href="/predict/individual" className="flex gap-2 p-2">
                                    <LuUserSearch className="text-lg" />
                                    <span>{t("predict")}</span>
                                </Link>
                            </li>
                            <li className={itemClass(pathname === "/predict/batch")} onClick={() => setOpen(false)}>
                                <Link href="/predict/batch" className="flex gap-2 p-2">
                                    <LuUsers className="text-lg" />
                                    <span>{t("predictBatch")}</span>
                                </Link>
                            </li>
                            <li className={itemClass(pathname.startsWith("/analyze"))} onClick={() => setOpen(false)}>
                                <Link href="/analyze/upload" className="flex gap-2 p-2">
                                    <MdInsertChartOutlined className="text-lg" />
                                    <span>{t("analyze")}</span>
                                </Link>
                            </li>
                        </ul>
                    </div>

                    {/* RECENT ANALYSES */}
                    <div className="flex-1 p-2">
                        <div
                            className="text-sm cursor-pointer flex items-center text-gray-500 dark:text-gray-300 mb-2"
                            onClick={() => setShowPredictions(!showPredictions)}
                        >
                            <span className="ml-2 mr-1">{t("recentPredictions")}</span>
                            {showPredictions ? <IoIosArrowDown /> : <IoIosArrowForward />}
                        </div>

                        {showPredictions && (
                            <ul className="space-y-1">
                                {result && result.length > 0 ? (
                                    [...result].map((item) => {
                                        const isBatch = item.type === "batch";
                                        const itemUrl = isBatch
                                            ? `/prediction-history/batch/${item.id}`
                                            : `/prediction-history/${item.id}`;

                                        const isActive = pathname === itemUrl;

                                        return (
                                            <li
                                                key={item.id}
                                                className={`${itemClass(isActive)} group relative flex items-center justify-between`}
                                                onClick={() => setOpen(false)}
                                            >
                                                <Link
                                                    href={itemUrl}
                                                    className="flex items-center p-2 gap-2 grow min-w-0"
                                                >
                                                    {isBatch ? (
                                                        <LuUsers className="text-gray-400 shrink-0" />
                                                    ) : (
                                                        <LuUser className="text-gray-400 shrink-0" />
                                                    )}
                                                    <div className="flex flex-col min-w-0">
                                                        <span className="truncate">{formatDateTime(item.created_at.toString())}</span>
                                                    </div>
                                                </Link>

                                                <div
                                                    className="p-2 rounded-full mr-2 opacity-0 pointer-events-none group-hover:opacity-100 group-hover:pointer-events-auto transition-opacity hover:bg-red-100 dark:hover:bg-red-800/30 cursor-pointer"
                                                    onClick={(e) => {
                                                        e.preventDefault();
                                                        e.stopPropagation();
                                                        setDeleteItem(item);
                                                    }}
                                                >
                                                    <FaRegTrashCan className="text-red-500" />
                                                </div>
                                            </li>
                                        );
                                    })
                                ) : (
                                    <li className="text-xs text-gray-400 mx-2">{tCommon("noRecords")}</li>
                                )}
                            </ul>
                        )}

                        {loadingMore && hasMore && (
                            <li className="text-xs text-gray-400 mx-2">{tCommon("loadingMore")}</li>
                        )}
                    </div>

                    {/* USER MENU */}
                    <div className="p-2 border-t border-gray-200 dark:border-[#FFFFFF1A] sticky bottom-0 z-20 bg-gray-50 dark:bg-[#18181B]" ref={dropdownRef}>

                        {/* THE DROPDOWN MENU */}
                        {userMenuOpen && (
                            <div className="absolute bottom-full left-2 right-2 mb-2 bg-white dark:bg-[#1A1A1B] border border-gray-200 dark:border-[#FFFFFF1A] rounded-2xl shadow-2xl overflow-hidden z-60 animate-in fade-in slide-in-from-bottom-2 duration-200">
                                <ul className="p-1">
                                    {user && (
                                        <>
                                            <li>
                                                <button
                                                    className="flex items-center transition w-full cursor-pointer gap-2 p-2 rounded-xl hover:bg-gray-100 dark:hover:bg-white/10 text-sm"
                                                    onClick={() => {
                                                        setUserMenuOpen(false);
                                                        setShowProfileModal(true);
                                                    }}
                                                >
                                                    <LuCircleUserRound className="text-lg" />
                                                    <span>{tCommon("profile")}</span>
                                                </button>
                                            </li>

                                            <div className="h-px bg-gray-100 dark:bg-white/5 my-1" />
                                        </>
                                    )}

                                    <li>
                                        <button
                                            className="flex items-center transition w-full cursor-pointer gap-2 p-2 rounded-xl hover:bg-gray-100 dark:hover:bg-white/10 text-sm"
                                            onClick={() => { setUserMenuOpen(false); setShowSettingModal(true); }}>
                                            <LuSettings className="text-lg" />
                                            <span>{tCommon("settings")}</span>
                                        </button>
                                    </li>

                                    <div className="h-px bg-gray-100 dark:bg-white/5 my-1" />

                                    {user ? (
                                        <li onClick={handleLogoutClick}>
                                            <div className="flex items-center gap-2 p-2 rounded-xl hover:bg-gray-100 dark:hover:bg-white/10 transition text-sm cursor-pointer">
                                                <LuLogOut className="text-lg" />
                                                <span>{tCommon("logout")}</span>
                                            </div>
                                        </li>
                                    ) : (
                                        <li>
                                            <Link href="/login"
                                                className="flex items-center gap-2 p-2 rounded-xl hover:bg-gray-100 dark:hover:bg-white/10 transition text-sm"
                                                onClick={() => { setUserMenuOpen(false); setOpen(false); }}>
                                                <LuLogIn className="text-lg" />
                                                <span>{tCommon("signIn")}</span>
                                            </Link>
                                        </li>
                                    )}
                                </ul>
                            </div>
                        )}

                        {/* TRIGGER BUTTON */}
                        <div
                            className={`flex items-center gap-2 p-2 rounded-xl transition cursor-pointer ${userMenuOpen ? 'bg-gray-100 dark:bg-white/10' : 'hover:bg-gray-100 dark:hover:bg-white/10'}`}
                            onClick={() => setUserMenuOpen(!userMenuOpen)}
                        >
                            <div className="w-8 h-8 rounded-xl bg-indigo-600 flex items-center justify-center text-white shrink-0 shadow-lg shadow-indigo-500/20 overflow-hidden">
                                {user?.avatar_url ? (
                                    <Image 
                                        src={user.avatar_url} 
                                        alt={user?.username || "User avatar"} 
                                        width={64} 
                                        height={64}
                                        className="w-full h-full object-cover"
                                    />
                                ) : (
                                    <FaRocket className="text-xs" />
                                )}
                            </div>
                            <div className="flex flex-col min-w-0">
                                <span className="text-sm font-bold truncate dark:text-white">
                                    {user ? (user.display_name || user.username) : tCommon("guest")}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            </aside>
            <DeleteModal
                isOpen={!!deleteItem}
                onClose={() => setDeleteItem(null)}
                onConfirm={handleConfirmDelete}
                isDeleting={isDeleting}
            />
            <LogoutModal
                isOpen={showLogoutModal}
                onClose={() => setShowLogoutModal(false)}
                onConfirm={handleLogout}
            />
            <SettingsModal
                isOpen={showSettingModal}
                onClose={() => setShowSettingModal(false)}
            />
            <ProfileModal
                isOpen={showProfileModal}
                onClose={() => setShowProfileModal(false)}
            />
        </>
    );
}
