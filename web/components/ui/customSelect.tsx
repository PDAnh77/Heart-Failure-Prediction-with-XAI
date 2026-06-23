"use client";
import { useState, useRef, useEffect } from "react";
import { IoIosArrowDown, IoIosArrowUp } from "react-icons/io";

export interface SelectOption {
    value: string;
    label: string;
}

export interface CustomSelectHandle {
    setValue: (v: string) => void;
}

export interface CustomSelectProps {
    id: string;
    name: string;
    options: SelectOption[];
    defaultValue?: string;
    placeholder?: string;
    isInvalid?: boolean;
    onChange?: (value: string) => void;
    selectRef?: React.RefObject<CustomSelectHandle | null>;
}

export default function CustomSelect({
    id,
    name,
    options,
    defaultValue = "",
    placeholder,
    isInvalid,
    onChange,
    selectRef,
}: CustomSelectProps) {
    const [open, setOpen] = useState(false);
    const [selected, setSelected] = useState(defaultValue);
    const containerRef = useRef<HTMLDivElement>(null);

    // Expose setValue để parent có thể set giá trị từ ngoài (autoFill, reset...)
    useEffect(() => {
        if (selectRef) {
            selectRef.current = {
                setValue: (v: string) => {
                    setSelected(v);
                    onChange?.(v);
                },
            };
        }
    }, [selectRef, onChange]);

    // Đóng dropdown khi click ra ngoài
    useEffect(() => {
        const handler = (e: MouseEvent) => {
            if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
                setOpen(false);
            }
        };
        if (open) document.addEventListener("mousedown", handler);
        return () => document.removeEventListener("mousedown", handler);
    }, [open]);

    const selectedLabel = options.find((o) => o.value === selected)?.label;

    const baseClass = `flex items-center justify-between w-full h-12 rounded-xl px-4 text-base outline-none transition cursor-pointer`;
    const stateClass = isInvalid
        ? `bg-red-50 text-red-800 border border-red-500 dark:bg-red-900/20 dark:text-red-200`
        : `bg-gray-100 text-gray-900 border border-transparent dark:bg-gray-800 dark:text-white`;

    return (
        <div className="relative" ref={containerRef}>
            {/* Hidden input để FormData lấy value khi submit */}
            <input type="hidden" name={name} value={selected} />

            {/* Trigger */}
            <button
                type="button"
                id={id}
                onClick={() => setOpen((prev) => !prev)}
                className={`${baseClass} ${stateClass} focus:ring-2 focus:ring-indigo-500 focus:bg-white dark:focus:bg-gray-900`}
                aria-haspopup="listbox"
                aria-expanded={open}
            >
                <span className={!selected ? "text-gray-400" : ""}>
                    {selectedLabel ?? placeholder ?? ""}
                </span>
                {open
                    ? <IoIosArrowUp className="w-4 h-4 shrink-0 text-gray-500 dark:text-gray-400" />
                    : <IoIosArrowDown className="w-4 h-4 shrink-0 text-gray-500 dark:text-gray-400" />
                }
            </button>

            {/* Dropdown */}
            {open && (
                <ul
                    role="listbox"
                    className="absolute z-30 mt-1 w-full rounded-xl bg-white shadow-lg ring-1 ring-black/10 dark:bg-gray-800 dark:ring-gray-600 overflow-hidden"
                >
                    {options.map((option) => (
                        <li
                            key={option.value}
                            role="option"
                            aria-selected={selected === option.value}
                            onClick={() => {
                                setSelected(option.value);
                                onChange?.(option.value);
                                setOpen(false);
                            }}
                            className={`cursor-pointer px-4 py-2.5 select-none transition-colors
                                ${selected === option.value
                                    ? "bg-indigo-50 text-indigo-600 font-semibold dark:bg-indigo-500/20 dark:text-indigo-300"
                                    : "text-gray-900 dark:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700"
                                }`}
                        >
                            {option.label}
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}