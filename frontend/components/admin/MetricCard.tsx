export default function MetricCard({ label, value, hint, alert = false }: {
  label: string; value: string; hint: string; alert?: boolean;
}) {
  return <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
    <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">{label}</p>
    <p className={`mt-1 text-2xl font-semibold ${alert ? "text-amber-800" : "text-gray-900"}`}>{value}</p>
    <p className="mt-1 text-xs text-gray-500">{hint}</p>
  </div>;
}
