import type { WarrantyHandoff } from "@/lib/types";
import SupportContactActions from "./SupportContactActions";

export default function HumanHandoffNotice({ handoff, caseReference }: { handoff: WarrantyHandoff; caseReference?: string | null }) {
  const language = handoff.language === "ko" || handoff.language === "es" ? handoff.language : "en";
  const labels = language === "ko" ? {
    safety: "안전 안내 및 지원", request: "상담원 요청이 저장되었습니다", reference: "접수 번호",
    context: "문의하실 때 이 번호를 알려주시면 담당자가 상담 내용을 확인할 수 있습니다.",
    supportPending: "지원팀 알림 전송이 아직 확인되지 않았습니다.",
    callbackPending: "콜백 예약이나 이메일 알림 전송이 아직 확인되지 않았습니다.",
    contact: "아래 연락처로 지원팀에 직접 문의해 주세요.",
  } : language === "es" ? {
    safety: "Orientación de seguridad y asistencia", request: "Su solicitud de asistencia está guardada", reference: "Referencia del caso",
    context: "Comparta esta referencia al contactarnos para que el equipo pueda consultar sus respuestas.",
    supportPending: "No se ha confirmado el envío de una notificación al equipo.",
    callbackPending: "No se ha confirmado una devolución de llamada ni una notificación por correo.",
    contact: "Comuníquese directamente con nuestro equipo usando las opciones siguientes.",
  } : {
    safety: "Safety guidance and support", request: "Your request for a person is saved", reference: "Case reference",
    context: "Share this reference when you contact us so the team can see your answers.",
    supportPending: "A support notification has not been confirmed.",
    callbackPending: "A callback or email notification has not been confirmed.",
    contact: "Please use the contact options below to reach our team.",
  };
  return (
    <section className="rounded-xl border border-amber-200 bg-amber-50 p-4" aria-labelledby="human-handoff-heading" lang={language}>
      <h2 id="human-handoff-heading" className="text-sm font-semibold text-amber-950">{handoff.reason === "safety_hazard" ? labels.safety : labels.request}</h2>
      <p className="mt-1 text-sm text-amber-900">{handoff.message}</p>
      {caseReference && <p className="mt-2 text-sm text-amber-900">{labels.reference}: <strong>{caseReference}</strong>. {labels.context}</p>}
      {!handoff.notification_sent && <p className="mt-2 text-xs text-amber-900">{handoff.reason === "safety_hazard" ? labels.supportPending : labels.callbackPending} {labels.contact}</p>}
      <SupportContactActions contact={handoff} language={language} />
    </section>
  );
}
