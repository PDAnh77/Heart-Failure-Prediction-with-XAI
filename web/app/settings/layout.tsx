import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Settings",
  robots: {
    index: false,
    follow: false,
  },
};

export default function SettingLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
