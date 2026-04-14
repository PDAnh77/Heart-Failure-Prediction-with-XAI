"use client";
import { useEffect, useState } from "react";
import Image from "next/image";
import { FiX } from "react-icons/fi";

interface ImageModalProps {
    imageUrl: string | null;
    onClose: () => void;
}

export default function ImageModal({ imageUrl, onClose }: ImageModalProps) {
    const [isVisible, setIsVisible] = useState(false);

    useEffect(() => {
        if (imageUrl) {
            document.body.style.overflow = 'hidden';
            const timer = setTimeout(() => setIsVisible(true), 10);
            return () => clearTimeout(timer);
        } else {
            // Khi đóng, trả lại scroll cho body và reset state
            document.body.style.overflow = 'unset';
            setIsVisible(false);
        }

        return () => {
            document.body.style.overflow = 'unset';
        };
    }, [imageUrl]);

    // Đóng modal lập tức khi nhấn phím Escape
    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === "Escape" && imageUrl) {
                onClose();
            }
        };
        if (imageUrl) {
            document.addEventListener("keydown", handleKeyDown);
        }
        return () => {
            document.removeEventListener("keydown", handleKeyDown);
        };
    }, [imageUrl, onClose]);

    if (!imageUrl) return null;

    return (
        <div className="relative z-100" aria-labelledby="modal-image" role="dialog">
            {/* Backdrop */}
            <div
                className={`fixed inset-0 bg-gray-500/75 backdrop-blur-sm transition-opacity duration-300 ease-out flex items-center justify-center p-4
                    ${isVisible ? "opacity-100" : "opacity-0"}`}
            >
                {/* Nút tắt */}
                <button
                    className="absolute hover:cursor-pointer top-6 right-6 text-gray-300 hover:text-white bg-gray-900/50 hover:bg-gray-900 p-2 rounded-full transition-colors z-110"
                    onClick={(e) => {
                        e.stopPropagation();
                        onClose();
                    }}
                >
                    <FiX className="w-8 h-8" />
                </button>

                {/* Vùng chứa ảnh phóng to với hiệu ứng scale lúc hiển thị lên */}
                <div
                    className={`relative w-full max-w-6xl h-[85vh] transform transition-all duration-300 ease-out
                        ${isVisible ? "opacity-100 scale-100 translate-y-0" : "opacity-0 scale-95 translate-y-4"}`}
                >
                    <Image
                        src={imageUrl}
                        alt="Expanded analysis chart"
                        fill
                        sizes="100vw"
                        className="object-contain"
                        priority
                    />
                </div>
            </div>
        </div>
    );
}