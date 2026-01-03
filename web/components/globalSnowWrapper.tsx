"use client";
import { useSettings } from "@/context/settingscontext";
import Snow from "./snow";

export default function GlobalSnowWrapper() {
    const { snowMode } = useSettings();
    if (!snowMode) return null; // Nếu tắt snowMode thì không render canvas
    return <Snow />;
}