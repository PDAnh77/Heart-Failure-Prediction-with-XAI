"use client";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { FiBarChart2, FiCpu } from "react-icons/fi";

export default function AnalyzeTabs() {
    const pathname = usePathname();
    const router = useRouter();
    const searchParams = useSearchParams();

    // Nếu không có datasetId (ví dụ đang ở trang /upload hoặc lỗi), không hiển thị Tab
    const datasetId = searchParams.get("id");
    if (!datasetId) return null;

    const tabs = [
        { name: "EDA", href: "/analyze/eda", icon: FiBarChart2 },
        { name: "Feature Selection", href: "/analyze/feature-selection", icon: FiCpu },
    ];

    const handleTabChange = (href: string) => {
        const params = new URLSearchParams(searchParams.toString());
        router.push(`${href}?${params.toString()}`);
    };

    return (
        <div className="flex border-b border-gray-200 dark:border-gray-700 px-4 mb-8 bg-white dark:bg-[#141516] sticky top-0 z-30">
            {tabs.map((tab) => {
                const isActive = pathname === tab.href;
                return (
                    <button
                        key={tab.name}
                        onClick={() => handleTabChange(tab.href)}
                        className={`flex hover:cursor-pointer items-center gap-1 px-4 py-2 mr-2 text-sm font-medium transition-all border-b-2 ${isActive
                                ? "border-[#4361EE] text-[#4361EE]"
                                : "border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                            }`}
                    >
                        <tab.icon className="text-lg" />
                        {tab.name}
                    </button>
                );
            })}
        </div>
    );
}