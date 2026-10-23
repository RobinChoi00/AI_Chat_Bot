import type { WarrantyHandoff } from "@/lib/types";
import SupportContactActions from "./SupportContactActions";

export default function HumanHandoffNotice({ handoff, caseReference }: { handoff: WarrantyHandoff; caseReference?: string | null }) {
  return (
    <section className="rounded-xl border border-amber-200 bg-amber-50 p-4" aria-labelledby="human-handoff-heading">
      <h2 id="human-handoff-heading" className="text-sm font-semibold text-amber-950">{handoff.reason === "safety_hazard" ? "Safety guidance and support" : "Your request for a person is saved"}</h2>
      <p className="mt-1 text-sm text-amber-900">{handoff.message}</p>
      {caseReference && <p className="mt-2 text-sm text-amber-900">Case reference: <strong>{caseReference}</strong>. Share this reference when you contact us so the team can see your answers.</p>}
      {!handoff.notification_sent && <p className="mt-2 text-xs text-amber-900">{handoff.reason === "safety_hazard" ? "A support notification has not been confirmed." : "A callback or email notification has not been confirmed."} Please use the contact options below to reach our team.</p>}
      <SupportContactActions contact={handoff} />
    </section>
  );
}
