"use client";
import SettingRow from "@/components/settingRow"
import { useSettings } from "@/context/settingscontext";
import { useTheme } from "next-themes";
import { useState } from "react";

export default function Settings() {
    const { savePrediction, setSavePrediction } = useSettings();
    const { snowMode, setSnowMode } = useSettings();
    const { theme, setTheme } = useTheme();
    const isDarkMode = theme === 'dark';

    return (
        <div className="p-4">
            <h1 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white">Customize Display</h1>
            <div className="max-w-xl divide-y divide-gray-200 dark:divide-gray-700">
                <SettingRow
                    title="Dark mode"
                    description="Switch between light and dark themes to reduce eye strain."
                    enabled={isDarkMode}
                    setEnabled={(value) => setTheme(value ? 'dark' : 'light')}
                />
                <SettingRow
                    title="Snowfall effect"
                    description="Enable winter animation with falling snow on the background."
                    enabled={snowMode}
                    setEnabled={setSnowMode}
                />
                <SettingRow
                    title="Save prediction history"
                    description="Store prediction results so you can review them later."
                    enabled={savePrediction}
                    setEnabled={setSavePrediction}
                />
            </div>
        </div>
    )
}