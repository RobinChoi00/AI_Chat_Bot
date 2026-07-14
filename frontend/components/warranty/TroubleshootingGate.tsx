"use client";

interface Props {
  mode: "troubleshooting" | "preparation";
  stage: "review" | "outcome";
  stepCount: number;
  disabled?: boolean;
  onStepsCompleted: () => void;
  onResolved: () => void;
  onUnresolved: () => void;
  onUnableToAttempt: () => void;
}

export default function TroubleshootingGate({
  mode,
  stage,
  stepCount,
  disabled,
  onStepsCompleted,
  onResolved,
  onUnresolved,
  onUnableToAttempt,
}: Props) {
  const isPreparation = mode === "preparation";

  if (stage === "review") {
    return (
      <section
        className="rounded-2xl border border-sky-200 bg-sky-50 p-4 shadow-sm"
        aria-labelledby="resolution-first-title"
      >
        <div className="mb-2 flex items-center justify-between gap-3">
          <span className="rounded-full bg-sky-700 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide text-white">
            Step 1 of 2
          </span>
          <span className="text-xs font-medium text-sky-800">Self-service first</span>
        </div>
        <h2 id="resolution-first-title" className="text-base font-semibold text-sky-950">
          {isPreparation
            ? "Complete the preparation above first"
            : "Try the troubleshooting steps above first"}
        </h2>
        <p className="mt-1 text-sm leading-relaxed text-sky-900">
          {stepCount > 0
            ? `${stepCount} recommended ${stepCount === 1 ? "step is" : "steps are"} shown above. `
            : "The recommended guide is shown above. "}
          {isPreparation
            ? "This gives the warranty team what they need for an accurate review."
            : "Many issues can be resolved here without waiting for service or shipping."}
        </p>
        <button
          type="button"
          onClick={onStepsCompleted}
          disabled={disabled}
          className="mt-4 min-h-[48px] w-full rounded-xl bg-sky-700 px-4 py-3 text-sm font-semibold text-white transition hover:bg-sky-800 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isPreparation ? "I’ve completed the preparation" : "I’ve tried all the steps"}
        </button>
        <button
          type="button"
          onClick={onUnableToAttempt}
          disabled={disabled}
          className="mt-2 min-h-[40px] w-full px-3 py-2 text-xs font-medium text-sky-800 underline underline-offset-2 hover:text-sky-950 disabled:cursor-not-allowed disabled:opacity-50"
        >
          I can’t safely complete these steps
        </button>
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
          Step 2 of 2
        </span>
        <span className="text-xs font-medium text-emerald-800">Check the result</span>
      </div>
      <h2 id="resolution-outcome-title" className="text-base font-semibold text-emerald-950">
        {isPreparation ? "Do you still need team review?" : "Did the steps solve the issue?"}
      </h2>
      <p className="mt-1 text-sm text-emerald-900">
        {isPreparation
          ? "Continue only if you still need the warranty team to review the case."
          : "If the chair is working now, we’ll close this request as self-resolved."}
      </p>
      <div className="mt-4 flex flex-col gap-2">
        <button
          type="button"
          onClick={onResolved}
          disabled={disabled}
          className="min-h-[48px] w-full rounded-xl bg-emerald-700 px-4 py-3 text-sm font-semibold text-white transition hover:bg-emerald-800 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isPreparation ? "No — I’m all set" : "Yes — it’s working now"}
        </button>
        <button
          type="button"
          onClick={onUnresolved}
          disabled={disabled}
          className="min-h-[48px] w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm font-medium text-gray-700 transition hover:border-gray-400 hover:bg-gray-50 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isPreparation
            ? "Yes — continue to team review"
            : "No — the issue is still there"}
        </button>
      </div>
    </section>
  );
}
