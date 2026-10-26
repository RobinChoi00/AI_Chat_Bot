import type { SupportContact } from "@/lib/types";
import { resolveSupportContact } from "@/lib/warrantyContact";

export default function SupportContactActions({ contact, language = "en" }: { contact?: SupportContact; language?: "en" | "ko" | "es" }) {
  const resolved = resolveSupportContact(contact);
  const sales = resolved.department === "sales";
  const labels = language === "ko"
    ? { call: sales ? "구매·배송팀 전화" : "보증 지원팀 전화", email: sales ? "구매·배송팀에 이메일 보내기" : "보증 지원팀에 이메일 보내기", hours: "전화 상담 시간" }
    : language === "es"
      ? { call: sales ? "Llamar a Ventas" : "Llamar a Garantía", email: sales ? "Enviar correo a Ventas" : "Enviar correo a Garantía", hours: "Horario de atención telefónica" }
      : { call: `Call ${sales ? "Sales" : "Warranty"}`, email: `Email our ${resolved.department} team`, hours: "Phone hours" };
  return <div>
    <div className="mt-3 flex flex-wrap gap-3 text-sm font-medium">
      {resolved.phoneHref ? <a href={resolved.phoneHref} className="underline">{labels.call}: {resolved.phone}</a> : resolved.phone ? <span>{resolved.phone}</span> : null}
      {resolved.email && <a href={`mailto:${resolved.email}`} className="underline">{labels.email}</a>}
    </div>
    {resolved.hours && <p className="mt-2 text-xs">{labels.hours}: {resolved.hours}</p>}
  </div>;
}
