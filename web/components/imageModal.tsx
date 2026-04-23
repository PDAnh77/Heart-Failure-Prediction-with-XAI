"use client";
import { useEffect } from "react";
import Image from "next/image";
import { FiX } from "react-icons/fi";

interface ImageModalProps {
    imageUrl: string | null;
    onClose: () => void;
}

export default function ImageModal({ imageUrl, onClose }: ImageModalProps) {
    // Quản lý khóa cuộn trang (scroll) khi mở modal
    useEffect(() => {
        if (imageUrl) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = 'unset';
        }

        return () => {
            document.body.style.overflow = 'unset';
        };
    }, [imageUrl]);

    // Đóng modal khi nhấn phím Escape
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
        <div className="relative z-50" aria-labelledby="modal-image" role="dialog">
            {/* Backdrop: Tăng blur lên 'md' và dùng nền đen 70% để làm nổi bật ảnh lập tức */}
            <div
                className="fixed inset-0 bg-black/70 backdrop-blur-md flex items-center justify-center p-4"
                onClick={onClose} // Bấm ra ngoài ảnh để đóng
            >
                {/* Nút tắt */}
                <button
                    className="absolute hover:cursor-pointer top-6 right-6 text-gray-300 hover:text-white bg-gray-900/50 hover:bg-gray-900 p-2 rounded-full transition-colors z-50"
                    onClick={(e) => {
                        e.stopPropagation();
                        onClose();
                    }}
                >
                    <FiX className="w-8 h-8" />
                </button>

                {/* Vùng chứa ảnh: Bỏ hoàn toàn các class tạo hiệu ứng chuyển động */}
                <div
                    className="relative w-full max-w-6xl h-[85vh]"
                    onClick={(e) => e.stopPropagation()} // Ngăn việc bấm vào chính bức ảnh làm đóng modal
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