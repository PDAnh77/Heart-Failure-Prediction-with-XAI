'use client';
import { createContext, useContext, useEffect, useRef, useState } from 'react';
import Cookies from 'js-cookie';
import { SettingsContextType } from '@/types/settings_context';
import { useRouter } from 'next/navigation';

const SettingsContext = createContext<SettingsContextType | undefined>(undefined);

export const SettingsProvider = ({ children }: { children: React.ReactNode }) => {
    const [snowMode, setSnowMode] = useState(true);
    const [savePrediction, setSavePrediction] = useState(true);
    const [language, setLanguage] = useState<'vi' | 'en'>('en');
    const [isReady, setIsReady] = useState(false);
    const router = useRouter();
    const hasMountedRef = useRef(false);

    useEffect(() => {
        const snow = Cookies.get('snow_mode');
        const savePrediction = Cookies.get('save_prediction');
        const language = Cookies.get('language');
        if (snow !== null && snow !== undefined) {
            setSnowMode(snow === 'true');
        }
        if (savePrediction !== null && savePrediction !== undefined) {
            setSavePrediction(savePrediction === 'true');
        }
        if (language === 'vi' || language === 'en') {
            setLanguage(language);
        }
        setIsReady(true);
    }, []);

    useEffect(() => {
        Cookies.set('snow_mode', String(snowMode), { expires: 365 });
        Cookies.set('save_prediction', String(savePrediction), { expires: 365 });
        Cookies.set('language', language, { expires: 365 });
    }, [snowMode, savePrediction, language]);

    useEffect(() => {
        if (!isReady) return;
        if (!hasMountedRef.current) {
            hasMountedRef.current = true;
            return;
        }
        router.refresh();
    }, [language, isReady, router]);

    return (
        <SettingsContext.Provider
            value={{
                snowMode,
                savePrediction,
                language,
                setSnowMode,
                setSavePrediction,
                setLanguage,
                isReady,
            }}
        >
            {children}
        </SettingsContext.Provider>
    );
}

export const useSettings = () => {
    const context = useContext(SettingsContext);
    if (!context) {
        throw new Error('useSettings must be used within a SettingsProvider');
    }
    return context;
};
