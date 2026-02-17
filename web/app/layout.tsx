import type { Metadata } from "next";
import { Inter, Geist_Mono } from "next/font/google";
import "./globals.css";

import Sidebar from "@/components/sidebar";
import { AuthProvider } from "@/context/authcontext";
import { SettingsProvider } from "@/context/settingscontext"; // Import mới
import GlobalSnowWrapper from "@/components/globalSnowWrapper"; // Component mới hỗ trợ logic ẩn/hiện
import { Analytics } from "@vercel/analytics/next"
import { SpeedInsights } from "@vercel/speed-insights/next"
import { ThemeProvider } from "next-themes";
import AppToaster from "@/components/appToaster";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
  weight: "500"
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "Heart Failure Predict",
    template: "Heart Failure Predict - %s",
  },
  description: "An application that predicts heart failure risk based on clinical and health data",
  keywords: [
    "heart disease predict",
    "heart disease prediction",
    "heart disease risk calculator",
    "heart failure predict",
    "cardiovascular disease prediction",
    "medical AI heart disease",
    "heart health assessment",
  ],
  icons: {
    icon: "/logo.png",
    apple: "/logo.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${geistMono.variable}`} suppressHydrationWarning>
      <head>
      </head>
      <body className={"antialiased"}>
        <AuthProvider>
          <ThemeProvider attribute="class" enableSystem defaultTheme="system">
            <SettingsProvider>
              <div className="flex flex-col lg:flex-row h-screen">
                <Sidebar />
                <main className="flex-1 p-4 h-full overflow-y-auto">
                  <GlobalSnowWrapper />
                  {children}
                  <AppToaster />
                  <Analytics />
                  <SpeedInsights />
                </main>
              </div>
            </SettingsProvider>
          </ThemeProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
