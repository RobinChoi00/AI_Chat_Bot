import WarrantyChat from "@/components/warranty/WarrantyChat";

export const metadata = {
  title: "Setup · Warranty · Delivery — Osaki & Titan",
  description:
    "Guided setup and warranty help for Osaki and Titan massage chairs, plus delivery tracking with your order number and checkout email. If an update is unavailable, call Sales at +1-888-848-2630 ext. 2.",
};

export const viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover" as const,
  interactiveWidget: "resizes-content" as const,
};

export default async function WarrantyEmbedPage({
  searchParams,
}: {
  searchParams: Promise<{ store?: string }>;
}) {
  const { store } = await searchParams;
  return (
    <main className="flex h-dvh flex-col overflow-hidden bg-gray-50">
      <WarrantyChat embed storeDomain={store} />
    </main>
  );
}
