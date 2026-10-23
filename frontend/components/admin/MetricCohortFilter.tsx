import type { MetricCohortMetadata } from "@/lib/adminTypes";

export default function MetricCohortFilter({ action, selected, metadata, hiddenFields = {} }: {
  action: string; selected: string; metadata?: MetricCohortMetadata; hiddenFields?: Record<string, string>;
}) {
  const reserved = ["all", "current", "legacy", "mixed"];
  const releases = Array.from(new Set([...(metadata?.available_releases ?? []), ...(!reserved.includes(selected) ? [selected] : [])])).filter((release) => !reserved.includes(release));
  return <section className="mb-5 rounded-xl border border-gray-200 bg-white p-4">
    <form action={action} method="get" className="flex flex-wrap items-end gap-3">
      {Object.entries(hiddenFields).map(([key, value]) => <input key={key} type="hidden" name={key} value={value} />)}
      <label className="min-w-0 flex-1 text-sm font-medium text-gray-700">Release cohort
        <select name="cohort" defaultValue={selected} className="mt-1 block w-full max-w-xl rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm">
          <option value="all">All stored cohorts</option>
          <option value="current">Current release</option>
          <option value="legacy">Legacy · release not recorded</option>
          <option value="mixed">Mixed releases</option>
          {releases.map((release) => <option value={release} key={release}>{release}</option>)}
        </select>
      </label>
      <button type="submit" className="rounded-lg bg-indigo-700 px-4 py-2 text-sm font-medium text-white">Apply cohort</button>
    </form>
    <p className="mt-3 text-xs text-gray-600">Historical conversations keep their original evidence. Legacy records have no verified release; mixed conversations span releases.</p>
    {metadata ? <>
      <p className="mt-1 break-all text-xs text-gray-500">Current release: {metadata.current_release || "Not recorded"} · Surface measurement started: {metadata.measurement_started_at?.replace("T", " ") || "No recorded events yet"}</p>
      {metadata.legacy_note && <p className="mt-1 text-xs text-gray-500">{metadata.legacy_note}</p>}
    </> : <p className="mt-1 text-xs text-amber-800">Release measurements are unavailable from this response. Do not infer a historical version from the current configuration.</p>}
  </section>;
}
