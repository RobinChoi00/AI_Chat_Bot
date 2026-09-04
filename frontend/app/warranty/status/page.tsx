import Link from "next/link";
import WarrantyCaseLookup from "@/components/warranty/WarrantyCaseLookup";

export const metadata = {
  title: "Check warranty case — Osaki & Titan",
  description:
    "Look up your Osaki or Titan warranty case with your case reference and email.",
};

export default async function WarrantyStatusPage({
  searchParams,
}: {
  searchParams: Promise<{ ref?: string }>;
}) {
  const { ref } = await searchParams;
  return (
    <main className="min-h-dvh bg-gray-50">
      <header className="border-b border-gray-200 bg-white px-4 py-3 shadow-sm">
        <div className="mx-auto flex max-w-md items-center gap-3">
          <Link
            href="/warranty"
            className="text-sm text-gray-500 hover:text-gray-800"
            aria-label="Back to warranty chat"
          >
            ← Chat
          </Link>
          <span className="text-gray-300">|</span>
          <div>
            <h1 className="text-base font-semibold text-gray-900">
              Check a warranty case
            </h1>
            <p className="text-xs text-gray-500">
              Enter your WR- reference and the email you used
            </p>
          </div>
        </div>
      </header>
      <div className="mx-auto max-w-md px-4 py-6">
        <WarrantyCaseLookup initialCaseReference={ref?.trim() || ""} />
      </div>
    </main>
  );
}
