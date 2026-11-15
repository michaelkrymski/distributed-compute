import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Distributed Compute",
  description:
    "Team votes, model budgets, milestones, and training for a 25M parameter language model.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
