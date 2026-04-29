"use client";
import { useEffect } from "react";
import Image from "next/image";
import { FiX } from "react-icons/fi";

interface ImageModalProps {
    imageUrl: string | null;
    onClose: () => void;
}

export default function ImageModal({ imageUrl, onClose }: ImageModalProps) {
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
            <div
                className="fixed inset-0 bg-black/70 backdrop-blur-md flex items-center justify-center p-4"
                onMouseDown={onClose}
            >
                <button
                    className="absolute hover:cursor-pointer top-6 right-6 text-gray-300 hover:text-white bg-gray-900/50 hover:bg-gray-900 p-2 rounded-full transition-colors z-50"
                    onClick={(e) => {
                        e.stopPropagation();
                        onClose();
                    }}
                >
                    <FiX className="w-8 h-8" />
                </button>

                <div
                    className="relative w-full max-w-6xl h-[85vh]"
                    onMouseDown={(e) => e.stopPropagation()}
                    onClick={(e) => e.stopPropagation()}
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