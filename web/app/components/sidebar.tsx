"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { toast } from 'react-toastify';
import LogoutModal from "@/components/modalLogout";
import { FaMicroscope, FaGear, FaHouse, FaRightToBracket, FaRightFromBracket, FaRocket } from "react-icons/fa6";

export default function Sidebar() {
    const pathname = usePathname();
    const [username, setUsername] = useState<string | null>(null);
    const [showLogoutModal, setShowLogoutModal] = useState(false);

    useEffect(() => {
        const storedUser = localStorage.getItem("username");
        if (storedUser) {
            setUsername(storedUser);
        }
    }, []);

    const handleLogoutClick = () => {
        setShowLogoutModal(true);
    };

    const handleLogout = () => {
        localStorage.removeItem("access_token");
        localStorage.removeItem("token_type");
        localStorage.removeItem("username");
        setUsername(null);
        setShowLogoutModal(false);
        toast.success("Logout successful.");
    };

    const itemClass = (active: boolean) =>
        `rounded-xl transition 
     ${active ? "bg-gray-100 font-bold" : "hover:bg-gray-100 font-normal"}`;

    return (
        <aside className="w-72 min-h-screen p-2">
            <div className="bg-gray-50 rounded-lg h-full border border-gray-200 shadow-md flex flex-col justify-between">
                <ul className="space-y-2 mt-8 pb-8 px-2">
                    <p className="text-sm font-bold mb-4 mx-2">Heart failure predict</p>
                    <li className={itemClass(pathname === "/")}>
                        <Link href="/" className="flex items-center gap-2 p-2">
                            <FaHouse className="text-lg" />
                            <span>Home</span>
                        </Link>
                    </li>

                    <li className={itemClass(pathname === "/predict")}>
                        <Link href="/predict" className="flex items-center gap-2 p-2">
                            <FaMicroscope className="text-lg" />
                            <span>Predict</span>
                        </Link>
                    </li>

                    <li className={itemClass(pathname === "/setting")}>
                        <Link href="/setting" className="flex items-center gap-2 p-2 w-full">
                            <FaGear className="text-lg" />
                            <span>Setting</span>
                        </Link>
                    </li>

                    {username ? (
                        // Nếu đã đăng nhập -> Hiển thị nút Logout
                        <li className={itemClass(false)} onClick={handleLogoutClick}>
                            <div className="flex items-center gap-2 p-2 w-full hover:cursor-pointer">
                                <FaRightFromBracket className="text-lg" />
                                <span>Logout</span>
                            </div>
                        </li>
                    ) : (
                        // Nếu chưa đăng nhập -> Hiển thị nút Sign in
                        <li className={itemClass(pathname === "/login")}>
                            <Link href="/login" className="flex items-center gap-2 p-2 w-full">
                                <FaRightToBracket className="text-lg" />
                                <span>Sign in</span>
                            </Link>
                        </li>
                    )}
                </ul>

                <div className="p-2 pb-4">
                    <div className="flex items-center justify-between p-2 rounded-xl hover:bg-gray-200 transition border border-transparent hover:border-gray-200">

                        {/* Avatar & Name Group */}
                        <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-xl bg-black flex items-center justify-center text-white">
                                <FaRocket className="text-sm" />
                            </div>

                            <div className="flex flex-col">
                                <span className="text-sm font-bold text-gray-900">
                                    {username || "Guest"}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <LogoutModal 
                isOpen={showLogoutModal} 
                onClose={() => setShowLogoutModal(false)} 
                onConfirm={handleLogout} 
            />
        </aside>
    );
}
