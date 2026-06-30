import type { Metadata } from "next";
import { Inter, Geist_Mono } from "next/font/google";
import "./globals.css";

import Sidebar from "@/components/layout/sidebar";
import { AuthProvider } from "@/context/authcontext";
import { SettingsProvider } from "@/context/settingscontext"; // Import mới
import GlobalSnowWrapper from "@/components/layout/globalSnowWrapper"; // Component mới hỗ trợ logic ẩn/hiện
import { Analytics } from "@vercel/analytics/next"
import { SpeedInsights } from "@vercel/speed-insights/next"
import { ThemeProvider } from "next-themes";
import AppToaster from "@/components/ui/appToaster";
import { NextIntlClientProvider } from "next-intl";
import { getLocale, getMessages } from "next-intl/server";

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
    default: "Heart Failure Analytics",
    template: "Heart Failure Analytics - %s",
  },
  description: "An application that predicts heart failure risk based on clinical and health data",
  keywords: [
    "heart failure analytics",
    "heart failure prediction",
    "heart disease risk prediction",
    "cardiovascular risk assessment",
    "clinical data analytics",
    "medical machine learning",
    "healthcare AI prediction",
    "heart disease risk calculator",
    "EDA medical dataset",
    "feature selection healthcare",
    "predictive analytics healthcare",
    "heart health prediction model",
  ],
  icons: {
    icon: "/logo.png",
    apple: "/logo.png",
  },
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const locale = await getLocale();
  const messages = await getMessages();

  return (
    <html lang={locale} className={`${inter.variable} ${geistMono.variable}`} suppressHydrationWarning>
      <head>
      </head>
      <body className={"antialiased"}>
        <NextIntlClientProvider locale={locale} messages={messages}>
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
        </NextIntlClientProvider>
      </body>
    </html>
  );
}
