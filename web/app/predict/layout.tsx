import type { Metadata } from "next";

export const metadata: Metadata = {
    title: {
        absolute: "Heart Failure Prediction Tool"
    },
    description: "Enter patient clinical data to calculate heart failure risk using our AI-based prediction model.",
};

export default function PredictLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return children;
}
