"use client";

import { useState } from "react";
import type { StepEnrichment } from "@/lib/types";
import { formatEnrichmentSource, hasStepEnrichmentPanel } from "@/lib/warrantyHydration";

interface Props {
  enrichment: StepEnrichment | null;
}

export default function StepEnrichmentPanel({ enrichment }: Props) {
  const [expanded, setExpanded] = useState(true);

  if (!hasStepEnrichmentPanel(enrichment)) {
    return null;
  }

  const tips = enrichment?.tips ?? [];
  const sources = Array.from(
    new Map(
      (enrichment?.sources ?? []).map((source) => [formatEnrichmentSource(source), source])
    ).values()
  );

  return (
    <div className="mt-3 rounded-xl border border-sky-100 bg-sky-50/80 px-3 py-2 text-sm text-sky-950">
      <button
        type="button"
        onClick={() => setExpanded((open) => !open)}
        className="flex w-full items-center justify-between gap-2 text-left"
        aria-expanded={expanded}
      >
        <span className="font-medium">From past cases</span>
        <span className="text-xs text-sky-700">{expanded ? "Hide" : "Show"}</span>
      </button>

      {expanded && (
        <div className="mt-2 space-y-2 text-xs sm:text-sm">
          {enrichment?.top_match && (
            <p className="text-sky-900">
              <span className="font-medium">Related topic:</span> {enrichment.top_match}
            </p>
          )}

          {tips.length > 0 && (
            <ul className="list-disc space-y-1 pl-5 text-sky-900">
              {tips.map((tip, idx) => (
                <li key={`${idx}-${tip.slice(0, 24)}`}>{tip}</li>
              ))}
            </ul>
          )}

          {sources.length > 0 && (
            <div className="flex flex-wrap gap-1.5 pt-1">
              {sources.map((source) => (
                <span
                  key={source}
                  className="rounded-full bg-white px-2 py-0.5 text-[11px] font-medium text-sky-800 ring-1 ring-sky-200"
                >
                  {formatEnrichmentSource(source)}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
