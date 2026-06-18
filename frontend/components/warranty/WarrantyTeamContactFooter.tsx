import {
  WARRANTY_SUPPORT_EMAIL,
  WARRANTY_SUPPORT_HOURS,
  WARRANTY_SUPPORT_PHONE,
  WARRANTY_SUPPORT_PHONE_HREF,
} from "@/lib/warrantyContact";

interface Props {
  className?: string;
  compact?: boolean;
}

/** Phone + email + hours for warranty customers. */
export default function WarrantyTeamContactFooter({
  className = "",
  compact = false,
}: Props) {
  if (compact) {
    return (
      <p className={`text-center text-[10px] text-gray-500 ${className}`}>
        Warranty team:{" "}
        <a href={WARRANTY_SUPPORT_PHONE_HREF} className="text-brand-600 hover:underline">
          {WARRANTY_SUPPORT_PHONE}
        </a>
        {" · "}
        <a
          href={`mailto:${WARRANTY_SUPPORT_EMAIL}`}
          className="text-brand-600 hover:underline"
        >
          {WARRANTY_SUPPORT_EMAIL}
        </a>
      </p>
    );
  }

  return (
    <div
      className={`rounded-lg border border-gray-200 bg-white px-3 py-3 text-center ${className}`}
    >
      <p className="text-xs font-medium text-gray-700">Warranty team contact</p>
      <p className="mt-1.5 text-sm text-gray-800">
        <a
          href={WARRANTY_SUPPORT_PHONE_HREF}
          className="font-medium text-brand-700 hover:underline"
        >
          {WARRANTY_SUPPORT_PHONE}
        </a>
      </p>
      <p className="mt-1 text-sm">
        <a
          href={`mailto:${WARRANTY_SUPPORT_EMAIL}`}
          className="text-brand-600 hover:underline"
        >
          {WARRANTY_SUPPORT_EMAIL}
        </a>
      </p>
      <p className="mt-1.5 text-[10px] text-gray-500">Hours: {WARRANTY_SUPPORT_HOURS}</p>
    </div>
  );
}
