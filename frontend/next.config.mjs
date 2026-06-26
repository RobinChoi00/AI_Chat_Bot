/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  async headers() {
    return [
      {
        source: "/warranty/embed",
        headers: [
          {
            key: "Content-Security-Policy",
            value: [
              "frame-ancestors 'self'",
              "https://*.myshopify.com",
              "https://osakiusa.com",
              "https://www.osakiusa.com",
              "https://*.osakichair.com",
              "https://titanchair.com",
              "https://www.titanchair.com",
              "https://osakimassagechair.com",
              "https://www.osakimassagechair.com",
            ].join(" "),
          },
        ],
      },
    ];
  },
  // Proxy /api/* to the FastAPI backend so the browser never needs CORS headers
  async rewrites() {
    const apiBase =
      process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
    return [
      {
        // Proxy customer-facing FastAPI endpoints (/api/v1/...)
        // Admin API calls go through Next.js Route Handlers (/api/admin/...)
        // which keep ADMIN_API_KEY server-side — do NOT add /admin/* here.
        source: "/api/v1/:path*",
        destination: `${apiBase}/api/v1/:path*`,
      },
    ];
  },
};

export default nextConfig;
