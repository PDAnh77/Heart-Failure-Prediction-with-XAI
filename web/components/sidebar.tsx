"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";
import { toast } from 'react-toastify';
import LogoutModal from "@/components/modalLogout";
import { useAuth } from "@/context/authcontext"
import { TbLayoutSidebarFilled } from "react-icons/tb";
import { FaMicroscope, FaGear, FaHouse, FaRightToBracket, FaRightFromBracket, FaRocket } from "react-icons/fa6";
import { api } from "@/lib/api";

export default function Sidebar() {
    const [open, setOpen] = useState(false);
    const pathname = usePathname();
    const [showLogoutModal, setShowLogoutModal] = useState(false);
    const { user, logout } = useAuth();
    const router = useRouter();

    const handleLogoutClick = () => {
        setShowLogoutModal(true);
    };

    const handleLogout = async () => {
        try {
            const res = api.post("auth/logout")
            logout();
            setShowLogoutModal(false);
            setOpen(false);
            router.push("/login")
        } catch (error: any) {
            console.log(error);
            toast.error("Logout failed.");
        }
    };

    const itemClass = (active: boolean) =>
        `rounded-xl transition ${
            active 
                ? "bg-gray-100 font-bold dark:text-white dark:bg-white/10" 
                : "hover:bg-gray-100 font-normal dark:hover:bg-white/10"
            }`;

    return (
        <>
            <div className="lg:hidden flex items-center p-2 bg-background border-b border-sidebar-border sticky top-0 z-40">
                <button
                    className="p-2 rounded-xl cursor-pointer"
                    onClick={() => setOpen(true)}
                >
                    <TbLayoutSidebarFilled className="text-xl text-foreground" />
                </button>
                <span className="font-bold ml-2 text-foreground">Heart failure predict</span>
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
                fixed inset-y-0 left-0 z-50 lg:z-40 w-72 p-2 transition-transform duration-300 ease-in-out 
                bg-background dark:bg-sidebar-bg 
                ${open ? "translate-x-0" : "-translate-x-full"} 
                lg:static lg:translate-x-0 lg:min-h-screen
            `}>
                <div className="bg-sidebar rounded-xl h-full lg:border lg:border-sidebar-border lg:shadow-md flex flex-col justify-between relative">
                    <ul className="space-y-2 mt-8 pb-8 px-2 "> 
                        <p className="font-bold mb-4 mx-2 text-foreground">Heart failure predict</p>

                        <li className={itemClass(pathname === "/")} onClick={() => setOpen(false)}>
                            <Link href="/" className="flex items-center gap-2 p-2 text-foreground">
                                <FaHouse className="text-lg" />
                                <span>Home</span>
                            </Link>
                        </li>

                        <li className={itemClass(pathname === "/predict")} onClick={() => setOpen(false)}>
                            <Link href="/predict" className="flex items-center gap-2 p-2 text-foreground">
                                <FaMicroscope className="text-lg" />
                                <span>Predict</span>
                            </Link>
                        </li>

                        <li className={itemClass(pathname === "/setting")} onClick={() => setOpen(false)}>
                            <Link href="/setting" className="flex items-center gap-2 p-2 w-full text-foreground">
                                <FaGear className="text-lg" />
                                <span>Setting</span>
                            </Link>
                        </li>

                        {user ? (
                            <li className={itemClass(false)} onClick={handleLogoutClick}>
                                <div className="flex items-center gap-2 p-2 w-full hover:cursor-pointer">
                                    <FaRightFromBracket className="text-lg" />
                                    <span>Logout</span>
                                </div>
                            </li>
                        ) : (
                            <li className={itemClass(pathname === "/login")} onClick={() => setOpen(false)}>
                                <Link href="/login" className="flex items-center gap-2 p-2 w-full text-foreground">
                                    <FaRightToBracket className="text-lg" />
                                    <span>Sign in</span>
                                </Link>
                            </li>
                        )}
                    </ul>

                    <div className="p-2 pb-4">
                        <div className="flex items-center justify-between p-2 rounded-xl hover:bg-gray-100 dark:hover:bg-white/10 transition border border-transparent">
                            <div className="flex items-center gap-3">
                                <div className="w-8 h-8 rounded-xl bg-indigo-600 flex items-center justify-center text-white">
                                    <FaRocket className="text-sm" />
                                </div>
                                <div className="flex flex-col">
                                    <span className="text-sm font-bold text-foreground">
                                        {user ? user.username : "Guest"}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </aside>

            <LogoutModal
                isOpen={showLogoutModal}
                onClose={() => setShowLogoutModal(false)}
                onConfirm={handleLogout}
            />
        </>
    );
}
