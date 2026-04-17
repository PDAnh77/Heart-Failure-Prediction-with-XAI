import type { Metadata } from "next";

export const metadata: Metadata = {
    title: {
        absolute: "Dataset Analysis & Feature Selection"
    },
    description: "Upload your clinical dataset to perform exploratory data analysis (EDA), visualize patterns, and identify the most important features for heart failure prediction.",
};

export default function AnalyzeLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return children;
}