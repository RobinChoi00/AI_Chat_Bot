import type { ReactNode } from "react";
import AdminHeader from "@/components/admin/AdminHeader";

export const metadata = {
  title: "Admin Portal — Titan / Osaki Warranty",
  robots: "noindex, nofollow",
};

export default function AdminLayout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen bg-gray-50">
      <AdminHeader />
      <main>{children}</main>
    </div>
  );
}
