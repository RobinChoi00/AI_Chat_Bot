import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { JourneyMetrics, SalesMetricsResponse, WarrantyMetricsResponse } from "@/lib/adminTypes";
import AdminJourneyPanel from "./AdminJourneyPanel";
import AdminMetricsDashboard from "./AdminMetricsDashboard";
import SalesMetricsDashboard from "./SalesMetricsDashboard";

function warranty(): WarrantyMetricsResponse {
  return {
    range: { days: null, all_time: true, start: "2026-10-23", end: "2026-10-23", abandon_threshold_hours: 6 },
    totals: { started: 0, reached_terminal: 0, completion_rate_pct: 0, contact_captured: 0, contact_rate_pct: 0, admin_decided: 0, resolved: 0, resolved_rate_pct: 0, self_service_started: 0, self_service_resolved: 0, self_service_resolution_rate_pct: 0, escalated_after_self_service: 0, abandoned: 0, abandoned_rate_pct: 0, median_turns_to_terminal: 0 },
    by_status: [], by_issue_type: [], by_domain: [], top_terminals: [], daily_started: [],
    purpose_counts: { repair: 0, delivery: 0, other: 0, unknown: 0 },
    repair: { consultations: 0, diagnosis_completed: 0, diagnosis_completion_rate_pct: null, latest_outcomes: { fixed: 0, not_fixed: 0, unable_to_attempt: 0, deferred: 0, unknown: 0 }, result_reported: 0, confirmed_fix_rate_all_pct: null, confirmed_fix_rate_reported_pct: null, followed_up: 0, reopened_after_fix: 0 },
    delivery: { consultations: 0, latest_lookup_available: 0, latest_lookup_unavailable: 0, lookup_not_attempted: 0, answered: 0, needs_review: 0, unable_to_attempt: 0, deferred: 0, unknown: 0 },
    handoffs: { requested: 0, by_department: [], by_reason: [] },
  };
}

function sales(): SalesMetricsResponse {
  return {
    range: { period: "all", days: null, start: null, end: "2026-10-23", start_date: null, end_date: "2026-10-23" },
    totals: { started: 0, engaged: 0, engagement_rate_pct: 0, recommended: 0, recommend_rate_pct: 0, nofit: 0, nofit_rate_pct: 0, handoffs: 0, handoff_rate_pct: 0, leads: 0, lead_rate_pct: 0, lead_forward_failed: 0, lead_forward_failure_rate_pct: 0, user_turns: 0, assistant_turns: 0 },
    by_status: [], by_intent: [], by_domain: [], by_channel: [], lead_delivery: [], daily: [], artifacts: { ok: true, models: 2, doorway_models: 1, files: {} },
    outcomes: {
      recommendation: { requested_sessions: 0, denominator_sessions: 0, recommended_sessions: 0, no_verified_match_sessions: 0, completed_sessions: 0, completion_rate_pct: null },
      human: { requested_sessions: 0, explicit_requested_sessions: 0, request_rate_pct: null, staff_confirmed_sessions: null, staff_confirmation_status: "unmeasured" },
      follow_up: { lead_sessions: 0, lead_records: 0, pending: 0, sent: 0, failed: 0, unknown: 0, submission_failure_rate_pct: null, staff_confirmed: null, transport: { configured: false, failure_code: "smtp_not_configured", acceptance_is_staff_confirmation: false } },
      conversation: { answer_provided_sessions: 0, active_waiting_sessions: 0, idle_missing_followup_sessions: 0, human_requested_sessions: 0, no_verified_match_sessions: 0, unknown_sessions: 0, idle_after_hours: 24 },
      quality: { repeated_unclear_sessions: 0, unclear_turns: 0, assistant_turns: 0, helpful: 0, not_helpful: 0, rated: 0, rated_sessions: 0, response_sessions: 0, unrated_sessions: 0, helpful_rate_pct: null, rating_coverage_pct: null },
    },
  };
}

describe("evidence-based admin outcomes", () => {
  it("shows empty sample rates as unmeasured rather than zero-percent success", () => {
    for (const html of [renderToStaticMarkup(<AdminMetricsDashboard data={warranty()} days={null} />), renderToStaticMarkup(<SalesMetricsDashboard data={sales()} />)]) {
      expect(html).not.toContain(">0%<");
      expect(html).not.toContain("NaN");
      expect(html).not.toContain("undefined");
      expect(html).toContain("—");
      expect(html).toContain("missing telemetry is unknown");
    }
  });

  it("keeps closure, inability, deferral and delivery distinct from actual repair fixes", () => {
    const data = warranty();
    data.totals.started = 100;
    data.totals.resolved = 80;
    data.totals.self_service_resolved = 80;
    data.purpose_counts = { repair: 10, delivery: 90, other: 0, unknown: 0 };
    data.repair = { ...data.repair!, consultations: 10, result_reported: 4, latest_outcomes: { fixed: 1, not_fixed: 3, unable_to_attempt: 2, deferred: 1, unknown: 3 } };
    data.delivery = { ...data.delivery!, consultations: 90, answered: 10, deferred: 20 };
    const html = renderToStaticMarkup(<AdminMetricsDashboard data={data} days={null} />);
    expect(html).toContain("1 confirmed fixes / 10 repair consultations");
    expect(html).toContain("1 fixed / 4 fixed-or-not-fixed results");
    expect(html).toContain(">10%<");
    expect(html).toContain(">25%<");
    expect(html).not.toContain(">80%<");
    expect(html).toContain("Unable to attempt");
    expect(html).toContain("Delivery deferred");
    expect(html).toContain("not a repair fix or proof of shipment delivery");
  });

  it("does not reconstruct missing outcome telemetry from historical completion fields", () => {
    const data = warranty();
    delete data.repair;
    delete data.delivery;
    const html = renderToStaticMarkup(<AdminMetricsDashboard data={data} days={null} />);
    expect(html).toContain("Latest repair outcomes are unavailable");
    expect(html).toContain("Purpose-specific delivery measurements are unavailable");
    expect(html).not.toContain(">Confirmed fixed / repair consultations<");
  });

  it("uses eligible recommendation and rated samples, while keeping handoff acceptance separate", () => {
    const data = sales();
    data.totals.started = 100;
    data.totals.handoffs = 70;
    data.outcomes!.recommendation = { requested_sessions: 4, denominator_sessions: 4, recommended_sessions: 2, no_verified_match_sessions: 1, completed_sessions: 3, completion_rate_pct: 75 };
    data.outcomes!.human = { requested_sessions: 2, explicit_requested_sessions: 1, request_rate_pct: 2, staff_confirmed_sessions: null, staff_confirmation_status: "unmeasured" };
    data.outcomes!.quality = { ...data.outcomes!.quality, helpful: 2, not_helpful: 1, rated: 3, rated_sessions: 2, response_sessions: 10, unrated_sessions: 8, helpful_rate_pct: 66.7, rating_coverage_pct: 20 };
    data.outcomes!.follow_up = { ...data.outcomes!.follow_up, lead_sessions: 2, lead_records: 3, sent: 1, failed: 1, pending: 1 };
    const html = renderToStaticMarkup(<SalesMetricsDashboard data={data} />);
    expect(html).toContain("3 / 4 eligible recommendation conversations");
    expect(html).toContain(">75%<");
    expect(html).toContain("2 helpful / 3 ratings");
    expect(html).toContain("20% of 10 conversations with an assistant reply had a rating");
    expect(html).toContain("staff acceptance unmeasured");
    expect(html).toContain("1 accepted / 2 completed send attempts");
    expect(html).toContain("Acceptance does not confirm inbox delivery or staff response");
    expect(html).toContain("smtp not configured");
    expect(html).not.toContain("sessions transferred");
    expect(html).not.toContain(">Engagement<");
  });

  it("displays authenticated staff observations as conversation counts without inferring missing activity", () => {
    const data = sales();
    data.outcomes!.human.staff_confirmed_sessions = 1;
    const html = renderToStaticMarkup(<SalesMetricsDashboard data={data} />);
    expect(html).toContain("1 conversations with authenticated staff acceptance observed");
    expect(html).toContain("other staff activity unknown");
  });
});

describe("recorded journey coverage", () => {
  it("distinguishes post-consent stage observations, unlinked sessions and unknown completion", () => {
    const journey: JourneyMetrics = { measured: true, measurement_start: "after_recording_consent", measurement_started_at: "2026-10-23T18:00:00Z", sessions_observed: 5, unlinked_sessions: 2, client_events: 7, server_events: 4, note: "No old events were backfilled.", stages: [{ stage: "evidence", shown: 5, completed: 2, skipped: 1, failed: 1, resumed: 1, requested: 0, without_completion: 2 }] };
    const html = renderToStaticMarkup(<AdminJourneyPanel journey={journey} />);
    expect(html).toContain("Measurement begins after recording consent");
    expect(html).toContain("2 unlinked observed sessions");
    expect(html).toContain("does not confirm abandonment");
    expect(html).toContain("not a verified helpful answer");
    expect(html).toContain("No old events were backfilled");
    expect(html).toContain("<table");
    expect(html).toContain("Without completion");
    expect(html).toContain("Skipped");
    expect(html).not.toContain("% success");
  });
});
