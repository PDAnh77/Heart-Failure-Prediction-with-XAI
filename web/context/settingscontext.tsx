"use client";
import React, { createContext, useContext, useEffect, useState } from 'react';

interface SettingsContextType {
    darkMode: boolean;
    setDarkMode: (val: boolean) => void;
    snowMode: boolean;
    setSnowMode: (val: boolean) => void;
}

const SettingsContext = createContext<SettingsContextType | undefined>(undefined);

export function SettingsProvider({ children }: { children: React.ReactNode }) {
    const [darkMode, setDarkMode] = useState(false);
    const [snowMode, setSnowMode] = useState(true);

    // Load cấu hình từ localStorage khi khởi chạy
    useEffect(() => {
        const savedDark = localStorage.getItem('darkMode') === 'true';
        const savedSnow = localStorage.getItem('snowMode') !== 'false'; // mặc định là true
        setDarkMode(savedDark);
        setSnowMode(savedSnow);
    }, []);

    // Xử lý logic Dark Mode (Thêm class 'dark' vào thẻ html)
    useEffect(() => {
        if (darkMode) {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
        localStorage.setItem('darkMode', darkMode.toString());
    }, [darkMode]);

    // Lưu trạng thái Snow Mode
    useEffect(() => {
        localStorage.setItem('snowMode', snowMode.toString());
    }, [snowMode]);

    return (
        <SettingsContext.Provider value={{ darkMode, setDarkMode, snowMode, setSnowMode }}>
            {children}
        </SettingsContext.Provider>
    );
}

export const useSettings = () => {
    const context = useContext(SettingsContext);
    if (!context) throw new Error("useSettings must be used within SettingsProvider");
    return context;
};