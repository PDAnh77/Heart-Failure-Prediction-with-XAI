"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { toast } from 'react-hot-toast';
import LogoutModal from "@/components/modalLogout";
import DeleteModal from "@/components/deletePredictionModal";
import { useAuth } from "@/context/authcontext"
import { TbLayoutSidebarFilled } from "react-icons/tb";
import { FaMicroscope, FaGear, FaHouse, FaRightToBracket, FaRightFromBracket, FaRocket, FaRegTrashCan } from "react-icons/fa6";
import { IoIosArrowForward, IoIosArrowDown } from "react-icons/io";
import { api } from "@/lib/api";
import { PredictionHistoryBase } from "@/types/prediction";

export default function Sidebar() {
    const [open, setOpen] = useState(false);
    const pathname = usePathname();
    const [showPredictions, setShowPredictions] = useState(true);
    const [showLogoutModal, setShowLogoutModal] = useState(false);
    const { user, logout, newHistoryItem } = useAuth();
    const [result, setResult] = useState<PredictionHistoryBase[] | null>(null);
    const router = useRouter();
    const [deleteId, setDeleteId] = useState<string | null>(null);
    const [isDeleting, setIsDeleting] = useState(false);
    const [offset, setOffset] = useState(0);
    const [hasMore, setHasMore] = useState(false);
    const [loadingMore, setLoadingMore] = useState(false);

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
                const res = await api.get(`prediction-history?limit=12&offset=0`);
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

    // useEffect(() => {
    //     if (result !== null) {
    //         console.log(result);
    //     }
    // }, [result]);

    const handleLogout = async () => {
        try {
            api.post("auth/logout");
            logout();
            setShowLogoutModal(false);
            setOpen(false);
            router.push("/login");
        } catch (error: any) {
            console.log(error);
            toast.error("Logout failed.");
        }
    };

    const handleConfirmDelete = async () => {
        if (!deleteId) return;
        setIsDeleting(true);
        try {
            await api.delete(`/prediction-history/${deleteId}`);
            setResult((prevResult: any) => prevResult.filter((item: any) => item.id !== deleteId));
            setDeleteId(null);
        } catch (error) {
            console.log(error);
            toast.error("Failed to delete history:");
        } finally {
            setIsDeleting(false);
        }
    };

    const formatDateTime = (dateString: string) => {
        const date = new Date(dateString);

        const parts = new Intl.DateTimeFormat('vi-VN', {
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
            const res = await api.get(`prediction-history?limit=12&offset=${offset}`);
            const newItems: PredictionHistoryBase[] = res.data;

            setResult((prev) => {
                const current = prev ?? [];
                const filteredNewItems = newItems.filter(
                    (newItem) => !current.some((oldItem) => oldItem.id === newItem.id)
                );

                return [...current, ...filteredNewItems];
            });

            setOffset(offset + 12);

            if (newItems.length < 12) {
                setHasMore(false);
            }
        } catch (error) {
            console.log(error);
        } finally {
            setLoadingMore(false);
        }
    }

    const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
        const { scrollTop, scrollHeight, clientHeight } = e.currentTarget;

        if (scrollHeight - scrollTop - clientHeight < 50) {
            loadMoreItems();
        }
    };

    return (
        <>
            <div className="lg:hidden flex items-center p-2 bg-white dark:bg-[#141516] border-b border-gray-200 dark:border-[#FFFFFF1A] sticky top-0 z-40">
                <button
                    className="p-2 rounded-xl cursor-pointer"
                    onClick={() => setOpen(true)}
                >
                    <TbLayoutSidebarFilled className="text-xl" />
                </button>
                <span className="font-bold ml-2">Heart Failure Predict</span>
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
                <div className="bg-gray-50 rounded-xl h-full lg:border lg:border-gray-200 lg:shadow-md dark:bg-[#18181B] dark:border-[#FFFFFF1A] flex flex-col">
                    <div className="p-2 space-y-2">
                        <p className="font-bold p-2 my-4 text-sm">Heart Failure Predict</p>
                        <ul className="space-y-2">
                            <li className={itemClass(pathname === "/")} onClick={() => setOpen(false)}>
                                <Link href="/" className="flex gap-2 p-2">
                                    <FaHouse className="text-lg" />
                                    <span>Home</span>
                                </Link>
                            </li>
                            <li className={itemClass(pathname === "/predict")} onClick={() => setOpen(false)}>
                                <Link href="/predict" className="flex gap-2 p-2">
                                    <FaMicroscope className="text-lg" />
                                    <span>Predict</span>
                                </Link>
                            </li>
                            <li className={itemClass(pathname === "/settings")} onClick={() => setOpen(false)}>
                                <Link href="/settings" className="flex gap-2 p-2">
                                    <FaGear className="text-lg" />
                                    <span>Settings</span>
                                </Link>
                            </li>
                            {user ? (
                                <li className={itemClass(false)} onClick={handleLogoutClick}>
                                    <div className="flex gap-2 p-2 w-full cursor-pointer">
                                        <FaRightFromBracket className="text-lg" />
                                        <span>Logout</span>
                                    </div>
                                </li>
                            ) : (
                                <li className={itemClass(pathname === "/login")} onClick={() => setOpen(false)}>
                                    <Link href="/login" className="flex gap-2 p-2 w-full">
                                        <FaRightToBracket className="text-lg" />
                                        <span>Sign in</span>
                                    </Link>
                                </li>
                            )}
                        </ul>
                    </div>

                    <div className="flex-1 overflow-y-auto p-2" onScroll={handleScroll}>
                        <div
                            className="text-sm cursor-pointer flex items-center text-gray-500 dark:text-gray-300 mb-2"
                            onClick={() => setShowPredictions(!showPredictions)}
                        >
                            <span className="ml-2 mr-1">Recent analyses</span>
                            {showPredictions ? <IoIosArrowDown /> : <IoIosArrowForward />}
                        </div>

                        {showPredictions && (
                            <ul className="space-y-1">
                                {result && result.length > 0 ? (
                                    [...result].map((item) => {
                                        const isActive = pathname === `/prediction-history/${item.id}`;
                                        return (
                                            <li
                                                key={item.id}
                                                className={`${itemClass(isActive)} group relative flex items-center justify-between`}
                                                onClick={() => setOpen(false)}
                                            >
                                                <Link
                                                    href={`/prediction-history/${item.id}`}
                                                    className="flex flex-col p-2 gap-1 grow min-w-0"
                                                >
                                                    <div className="flex items-center gap-2">
                                                        <span>{formatDateTime(item.created_at.toString())}</span>
                                                    </div>
                                                </Link>

                                                <div
                                                    className=" p-2 rounded-full mr-2 opacity-0 pointer-events-none group-hover:opacity-100 group-hover:pointer-events-auto transition-opacity hover:bg-red-100 dark:hover:bg-red-800/30 cursor-pointer"
                                                    onClick={(e) => {
                                                        e.preventDefault();
                                                        e.stopPropagation();
                                                        setDeleteId(item.id);
                                                    }}
                                                >
                                                    <FaRegTrashCan className="text-red-500" />
                                                </div>
                                            </li>
                                        );
                                    })
                                ) : (
                                    <li className="text-xs text-gray-400 mx-2">No records found</li>
                                )}
                            </ul>
                        )}

                        {loadingMore && hasMore && (
                            <li className="text-xs text-gray-400 mx-2">Loading more...</li>
                        )}
                    </div>

                    <div className="p-2 border-t border-gray-200 dark:border-[#FFFFFF1A]">
                        <div className="flex items-center gap-3 p-2 rounded-xl hover:bg-gray-100 dark:hover:bg-white/10 transition">
                            <div className="w-8 h-8 rounded-xl bg-indigo-600 flex items-center justify-center text-white">
                                <FaRocket className="text-sm" />
                            </div>
                            <span className="text-sm font-bold">
                                {user ? user.username : "Guest"}
                            </span>
                        </div>
                    </div>
                </div>
            </aside>
            <DeleteModal
                isOpen={!!deleteId}
                onClose={() => setDeleteId(null)}
                onConfirm={handleConfirmDelete}
                isDeleting={isDeleting}
            />
            <LogoutModal
                isOpen={showLogoutModal}
                onClose={() => setShowLogoutModal(false)}
                onConfirm={handleLogout}
            />
        </>
    );
}
