import type { Metadata } from "next";

export const metadata: Metadata = {
    title: "Login",
    description: "Login to access the heart failure risk prediction system and analyze patient health data.",
};

export default function LoginLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return children;
}
