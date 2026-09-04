import Link from "next/link";
import WarrantyChat from "@/components/warranty/WarrantyChat";

export const metadata = {
  title: "Setup · Warranty · Delivery — Osaki & Titan",
  description:
    "Guided setup, warranty, and delivery help for your Osaki or Titan massage chair.",
};

export const viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover" as const,
  interactiveWidget: "resizes-content" as const,
};

export default function WarrantyPage() {
  return (
    <main className="flex h-dvh flex-col overflow-hidden bg-gray-50">
      {/* Page header */}
      <header className="shrink-0 border-b border-gray-200 bg-white px-4 py-3 shadow-sm">
        <div className="mx-auto flex max-w-2xl items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="text-sm text-gray-500 hover:text-gray-800"
              aria-label="Back to home"
            >
              ← Home
            </Link>
            <span className="text-gray-300">|</span>
            <div>
              <h1 className="text-base font-semibold text-gray-900">
                Setup · Warranty · Delivery
              </h1>
              <p className="text-xs text-gray-500">
                Guided help for Osaki &amp; Titan chairs
              </p>
            </div>
          </div>
          <Link
            href="/warranty/status"
            className="shrink-0 text-xs font-medium text-brand-700 hover:underline"
          >
            Check a case
          </Link>
        </div>
      </header>

      {/* Chat widget */}
      <div className="flex min-h-0 flex-1 flex-col">
        <WarrantyChat />
      </div>
    </main>
  );
}
