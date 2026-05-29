import Link from "next/link";

export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center px-4">
      <div className="w-full max-w-md text-center">
        {/* Brand header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900">
            Osaki &amp; Titan
          </h1>
          <p className="mt-1 text-sm text-gray-500">Premium Massage Chairs</p>
        </div>

        {/* Action cards */}
        <div className="space-y-3">
          <Link
            href="/warranty"
            className="flex items-center justify-between rounded-xl border border-gray-200 bg-white px-5 py-4 shadow-sm transition hover:border-brand-500 hover:shadow"
          >
            <div className="text-left">
              <p className="font-semibold text-gray-900">Warranty Support</p>
              <p className="text-sm text-gray-500">
                Report a defect, delivery issue, or installation problem
              </p>
            </div>
            <span className="ml-4 text-2xl">🛡️</span>
          </Link>

          <a
            href="https://www.osaki.com"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between rounded-xl border border-gray-200 bg-white px-5 py-4 shadow-sm transition hover:border-brand-500 hover:shadow"
          >
            <div className="text-left">
              <p className="font-semibold text-gray-900">Browse Chairs</p>
              <p className="text-sm text-gray-500">
                Explore our full catalog at osaki.com
              </p>
            </div>
            <span className="ml-4 text-2xl">💺</span>
          </a>
        </div>

        <p className="mt-8 text-xs text-gray-400">
          All warranty decisions are reviewed by our support team.
        </p>
      </div>
    </main>
  );
}
