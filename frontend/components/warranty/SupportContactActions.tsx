import type { SupportContact } from "@/lib/types";
import { resolveSupportContact } from "@/lib/warrantyContact";

export default function SupportContactActions({ contact }: { contact?: SupportContact }) {
  const resolved = resolveSupportContact(contact);
  return <div>
    <div className="mt-3 flex flex-wrap gap-3 text-sm font-medium">
      {resolved.phoneHref ? <a href={resolved.phoneHref} className="underline">Call {resolved.department === "sales" ? "Sales" : "Warranty"}: {resolved.phone}</a> : resolved.phone ? <span>{resolved.phone}</span> : null}
      {resolved.email && <a href={`mailto:${resolved.email}`} className="underline">Email our {resolved.department} team</a>}
    </div>
    {resolved.hours && <p className="mt-2 text-xs">Phone hours: {resolved.hours}</p>}
  </div>;
}
