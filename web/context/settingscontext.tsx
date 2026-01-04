'use client';
import { createContext, useContext, useEffect, useState } from 'react';
import Cookies from 'js-cookie';
import { SettingsContextType } from '@/types/settingscontext';

const SettingsContext = createContext<SettingsContextType | undefined>(undefined);

export const SettingsProvider = ({ children }: { children: React.ReactNode }) => {
    const [snowMode, setSnowMode] = useState(true);
    const [isReady, setIsReady] = useState(false);

    useEffect(() => {
        const snow = Cookies.get('snow_mode');
        if (snow !== undefined) {
            setSnowMode(snow === 'true');
        }
        setIsReady(true);
    }, []);

    return (
        <SettingsContext.Provider value={{ snowMode, setSnowMode, isReady }}>
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
