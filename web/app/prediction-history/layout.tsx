import type { Metadata } from "next";

export const metadata: Metadata = {
    title: {
        absolute: "Prediction History"
    },
    description:
        "View previously generated heart failure prediction results.",
    robots: {
        index: false,
        follow: false,
        noarchive: true,
        nosnippet: true,
        googleBot: {
            index: false,
            follow: false,
            noimageindex: true,
        },
    },
};

export default function PredictionHistoryLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return children;
}
