"use client";
import { useSettings } from "@/context/settingscontext";
import Snow from "../ui/snow";

export default function GlobalSnowWrapper() {
    const { snowMode } = useSettings();
    if (!snowMode) return null;
    return <Snow />;
}