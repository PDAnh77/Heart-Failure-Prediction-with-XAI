"use client";
import { useState } from 'react'
import SettingRow from "@/components/settingRow"

export default function Settings() {
    const [darkMode, setDarkMode] = useState(false)
    const [snowMode, setSnowMode] = useState(true)

    return (
        <div className="p-4">
            <h1 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white">Customize Display</h1>
            <div className="max-w-xl divide-y divide-gray-200 dark:divide-gray-700">
                <SettingRow 
                    title="Dark Mode"
                    description="Switch between light and dark themes to reduce eye strain."
                    enabled={darkMode}
                    setEnabled={setDarkMode}
                />
                <SettingRow 
                    title="Snowfall Effect"
                    description="Enable winter animation with falling snow on the background."
                    enabled={snowMode}
                    setEnabled={setSnowMode}
                />
            </div>
        </div>
    )
}