/** Missing measurements and empty samples are displayed separately from measured zero. */
export function metricCount(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "—" : String(value);
}

export function metricRate(numerator: number | null | undefined, denominator: number | null | undefined): string {
  if (numerator == null || denominator == null || !Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator <= 0) return "—";
  return `${Math.round(numerator / denominator * 1000) / 10}%`;
}

export function recordedRate(value: number | null | undefined, denominator: number | null | undefined): string {
  return value == null || denominator == null || denominator <= 0 || !Number.isFinite(value) ? "—" : `${value}%`;
}
