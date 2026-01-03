import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

import Sidebar from "@/components/sidebar";
import { AuthProvider } from "@/context/authcontext";
import { SettingsProvider }  from "@/context/settingscontext"; // Import mới
import GlobalSnowWrapper from "@/components/globalSnowWrapper"; // Component mới hỗ trợ logic ẩn/hiện
import { Slide, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { Analytics } from "@vercel/analytics/next"
import { SpeedInsights } from "@vercel/speed-insights/next"

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Heart Failue Predict",
  description: "An application that predicts heart failure risk based on clinical and health data",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning> 
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              try {
                if (localStorage.getItem('darkMode') === 'true') {
                  document.documentElement.classList.add('dark');
                } else {
                  document.documentElement.classList.remove('dark');
                }
              } catch (_) {}
            `,
          }}
        />
      </head>
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <AuthProvider>
          <SettingsProvider>
            <div className="flex flex-col lg:flex-row h-screen">
             <Sidebar />
               <main className="flex-1 p-4 h-full overflow-y-auto">
                <GlobalSnowWrapper />
                 {children}
                <ToastContainer position="top-right" transition={Slide} autoClose={3000} hideProgressBar />
                <Analytics />
                <SpeedInsights />
              </main>
           </div>
          </SettingsProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
