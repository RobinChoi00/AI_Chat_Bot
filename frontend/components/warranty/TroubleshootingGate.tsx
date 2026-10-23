"use client";

export type WarrantyIssueType = "installation" | "delivery" | "defect";

interface Props {
  mode: "troubleshooting" | "preparation";
  issueType?: WarrantyIssueType | string;
  stage: "review" | "outcome";
  stepCount: number;
  disabled?: boolean;
  onStepsCompleted: () => void;
  onResolved: () => void;
  onUnresolved: () => void;
  onUnableToAttempt: () => void;
  onDeferred: () => void;
}

function normalizeIssueType(value?: string): WarrantyIssueType {
  if (value === "installation" || value === "delivery") return value;
  return "defect";
}

export function resolutionGateLabels(
  issueType?: string,
  mode: "troubleshooting" | "preparation" = "troubleshooting"
): { resolved: string; unresolved: string; unable: string } {
  const issue = normalizeIssueType(issueType);
  if (issue === "installation") {
    return {
      resolved: "Yes — the chair is set up now",
      unresolved: "No — I still need install help",
      unable: "I can’t complete setup on my own",
    };
  }
  if (issue === "delivery") {
    return {
      resolved: "The delivery update answered my question",
      unresolved: "Continue to delivery review",
      unable: "I don’t have the photos or paperwork yet",
    };
  }
  const isPreparation = mode === "preparation";
  return {
    resolved: "Yes — it’s working now",
    unresolved: isPreparation
      ? "Yes — continue to team review"
      : "No — the issue is still there",
    unable: "I can’t safely complete these steps",
  };
}

export default function TroubleshootingGate({
  mode,
  issueType,
  stage,
  stepCount,
  disabled,
  onStepsCompleted,
  onResolved,
  onUnresolved,
  onUnableToAttempt,
  onDeferred,
}: Props) {
  const issue = normalizeIssueType(issueType);
  const isPreparation = mode === "preparation" || issue === "delivery";

  if (issue === "delivery") {
    return <section className="rounded-2xl border border-sky-200 bg-sky-50 p-4" aria-labelledby="delivery-review-title">
      <h2 id="delivery-review-title" className="text-base font-semibold text-sky-950">Send this delivery case for review?</h2>
      <p className="mt-1 text-sm text-sky-900">Photos and delivery paperwork help when available. You can continue without them.</p>
      <button type="button" onClick={onUnresolved} disabled={disabled} className="mt-4 min-h-[48px] w-full rounded-xl bg-sky-700 px-4 py-3 text-sm font-semibold text-white hover:bg-sky-800 disabled:opacity-50">Continue to delivery review</button>
      <button type="button" onClick={onDeferred} disabled={disabled} className="mt-2 min-h-[44px] w-full text-sm font-medium text-sky-900 underline underline-offset-2 disabled:opacity-50">Not yet — save for later</button>
    </section>;
  }

  const copy =
    issue === "installation"
      ? {
          reviewBadge: "Setup first",
          reviewTitle: "Review the installation guide",
          reviewBody:
            "Follow the resource only if its model label matches your chair. Tell us if you need help with the setup checks.",
          reviewButton: "I’ve reviewed the guide and tried the setup",
          unable: "I can’t complete setup on my own",
          outcomeTitle: "Is the chair set up correctly now?",
          outcomeBody:
            "If installation is complete, we’ll close this request as self-resolved.",
          resolved: "Yes — the chair is set up now",
          unresolved: "No — I still need install help",
        }
      : {
            reviewBadge: "Self-service first",
            reviewTitle: isPreparation
              ? "Complete the preparation above first"
              : "Try the troubleshooting steps above first",
            reviewBody: isPreparation
              ? "This gives the warranty team what they need for an accurate review of this product issue."
              : "Many product issues can be resolved here without waiting for service or shipping.",
            reviewButton: isPreparation
              ? "I’ve completed the preparation"
              : "I’ve tried all the steps",
            unable: "I can’t safely complete these steps",
            outcomeTitle: isPreparation
              ? "What would you like to do next?"
              : "Did the steps solve the issue?",
            outcomeBody: isPreparation
              ? "Continue only if you still need the warranty team to review this product issue."
              : "If the chair is working now, we’ll close this request as self-resolved.",
            resolved: "Yes — it’s working now",
            unresolved: isPreparation
              ? "Yes — continue to team review"
              : "No — the issue is still there",
          };

  if (stage === "review") {
    return (
      <section
        className="rounded-2xl border border-sky-200 bg-sky-50 p-4 shadow-sm"
        aria-labelledby="resolution-first-title"
      >
        <div className="mb-2 flex items-center justify-between gap-3">
          <span className="rounded-full bg-sky-700 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide text-white">
            {isPreparation ? "Team review" : "Try the guidance"}
          </span>
          <span className="text-xs font-medium text-sky-800">{copy.reviewBadge}</span>
        </div>
        <h2 id="resolution-first-title" className="text-base font-semibold text-sky-950">
          {copy.reviewTitle}
        </h2>
        <p className="mt-1 text-sm leading-relaxed text-sky-900">
          {stepCount > 0
            ? `${stepCount} recommended ${stepCount === 1 ? "step is" : "steps are"} shown above. `
            : "The recommended guide is shown above. "}
          {copy.reviewBody}
        </p>
        <button
          type="button"
          onClick={onStepsCompleted}
          disabled={disabled}
          className="mt-4 min-h-[48px] w-full rounded-xl bg-sky-700 px-4 py-3 text-sm font-semibold text-white transition hover:bg-sky-800 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50"
        >
          {copy.reviewButton}
        </button>
        <button
          type="button"
          onClick={onUnableToAttempt}
          disabled={disabled}
          className="mt-2 min-h-[40px] w-full px-3 py-2 text-xs font-medium text-sky-800 underline underline-offset-2 hover:text-sky-950 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {copy.unable}
        </button>
        <button type="button" onClick={onDeferred} disabled={disabled} className="min-h-[44px] w-full px-3 py-2 text-xs text-sky-800 underline underline-offset-2 disabled:opacity-50">Save for later</button>
      </section>
    );
  }

  return (
    <section
      className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4 shadow-sm"
      aria-labelledby="resolution-outcome-title"
    >
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="rounded-full bg-emerald-700 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide text-white">
          {isPreparation ? "Next step" : "Check the result"}
        </span>
        <span className="text-xs font-medium text-emerald-800">Check the result</span>
      </div>
      <h2 id="resolution-outcome-title" className="text-base font-semibold text-emerald-950">
        {copy.outcomeTitle}
      </h2>
      <p className="mt-1 text-sm text-emerald-900">{copy.outcomeBody}</p>
      <div className="mt-4 flex flex-col gap-2">
        <button
          type="button"
          onClick={isPreparation ? onUnresolved : onResolved}
          disabled={disabled}
          className="min-h-[48px] w-full rounded-xl bg-emerald-700 px-4 py-3 text-sm font-semibold text-white transition hover:bg-emerald-800 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isPreparation ? copy.unresolved : copy.resolved}
        </button>
        <button
          type="button"
          onClick={isPreparation ? onResolved : onUnresolved}
          disabled={disabled}
          className="min-h-[48px] w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm font-medium text-gray-700 transition hover:border-gray-400 hover:bg-gray-50 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isPreparation ? copy.resolved : copy.unresolved}
        </button>
        <button type="button" onClick={onDeferred} disabled={disabled} className="min-h-[44px] w-full px-3 py-2 text-xs text-emerald-900 underline underline-offset-2 disabled:opacity-50">Save for later</button>
      </div>
    </section>
  );
}
