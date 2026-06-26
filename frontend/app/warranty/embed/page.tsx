import WarrantyChat from "@/components/warranty/WarrantyChat";

export const metadata = {
  title: "Setup · Warranty · Delivery — Osaki & Titan",
  description:
    "Guided setup, warranty, and delivery help for Osaki and Titan massage chairs.",
};

export const viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover" as const,
  interactiveWidget: "resizes-content" as const,
};

export default function WarrantyEmbedPage() {
  return (
    <main className="flex h-dvh flex-col overflow-hidden bg-gray-50">
      <WarrantyChat embed />
    </main>
  );
}
