'use client';
import { createContext, useContext, useEffect, useState } from 'react';
import Cookies from 'js-cookie';
import { SettingsContextType } from '@/types/settingscontext';

const SettingsContext = createContext<SettingsContextType | undefined>(undefined);

export const SettingsProvider = ({ children }: { children: React.ReactNode }) => {
    const [snowMode, setSnowMode] = useState(true);
    const [savePrediction, setSavePrediction] = useState(true);
    const [isReady, setIsReady] = useState(false);

    useEffect(() => {
        const snow = Cookies.get('snow_mode');
        const savePrediction = Cookies.get('save_prediction');
        if (snow !== undefined) {
            setSnowMode(snow === 'true');
        }
        if (savePrediction !== undefined) {
            setSavePrediction(savePrediction === 'true');
        }
        setIsReady(true);
    }, []);

    useEffect(() => {
        Cookies.set('snow_mode', String(snowMode), { expires: 365 });
        Cookies.set('save_prediction', String(savePrediction), { expires: 365 });
    }), [snowMode, savePrediction]

    return (
        <SettingsContext.Provider value={{ snowMode, savePrediction, setSnowMode, setSavePrediction, isReady }}>
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
