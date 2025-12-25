import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

import Snow from "@/components/snow";
import Sidebar from "@/components/sidebar";
import { AuthProvider } from "@/context/authcontext";
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
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <AuthProvider>
          <div className="flex flex-col lg:flex-row h-screen">
            <Sidebar />
            <main className="flex-1 p-4 h-full overflow-y-auto">
              <Snow />
              {children}
              <ToastContainer position="top-right" transition={Slide} autoClose={3000} hideProgressBar />
              <Analytics />
              <SpeedInsights />
            </main>
          </div>
        </AuthProvider>
      </body>
    </html>
  );
}
