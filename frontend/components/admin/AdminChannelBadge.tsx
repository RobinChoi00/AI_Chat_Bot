"use client";

interface Props {
  channel: string | null | undefined;
}

export default function AdminChannelBadge({ channel }: Props) {
  const value = (channel || "").trim().toLowerCase();
  if (!value) {
    return <span className="text-gray-400">Web</span>;
  }
  if (value === "phone") {
    return (
      <span className="inline-flex items-center rounded-full bg-sky-100 px-2 py-0.5 text-xs font-semibold text-sky-800">
        📞 Phone
      </span>
    );
  }
  return (
    <span className="inline-flex items-center rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-700">
      {channel}
    </span>
  );
}
