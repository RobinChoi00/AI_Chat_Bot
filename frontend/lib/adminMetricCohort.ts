export function metricCohort(raw: unknown): string {
  return typeof raw === "string" && /^[A-Za-z0-9_.:-]{1,160}$/.test(raw) ? raw : "all";
}

export function withMetricCohort(href: string, cohort: string): string {
  const [path, rawQuery] = href.split("?");
  const query = new URLSearchParams(rawQuery);
  if (cohort !== "all") query.set("cohort", cohort);
  return `${path}${query.size ? `?${query}` : ""}`;
}
