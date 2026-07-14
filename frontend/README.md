# Titan / Osaki — Next.js Frontend

Customer-facing warranty chat UI for Osaki and Titan massage chairs.

## Stack

| Layer | Technology |
|---|---|
| Framework | Next.js 15 (App Router) |
| Language | TypeScript |
| Styling | Tailwind CSS 3 |
| HTTP | Native `fetch` with streaming |
| State | React hooks only (no Redux/Zustand) |

---

## Quick Start

```bash
# 1. Install dependencies
cd frontend
npm install

# 2. Create your local env file
cp .env.local.example .env.local
# Edit .env.local and set NEXT_PUBLIC_API_BASE_URL

# 3. Run development server
npm run dev
# → http://localhost:3000
```

The FastAPI backend must be running at the URL configured in `.env.local`.

---

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx          Root layout (fonts, metadata)
│   ├── page.tsx            Home page (nav to warranty)
│   ├── globals.css         Tailwind base styles
│   └── warranty/
│       └── page.tsx        Warranty chat page
├── components/
│   └── warranty/
│       ├── WarrantyChat.tsx       Main chat widget (session, streaming, state)
│       ├── ChatMessageBubble.tsx  User / assistant message bubble
│       ├── AnswerOptions.tsx      Clickable answer buttons for warranty options
│       ├── EvidenceUploader.tsx   File upload widget for evidence
│       └── TicketStatusBadge.tsx  Status pill (In Progress, Under Review, …)
└── lib/
    ├── api.ts              All network calls (streamChat, uploadEvidence, getWarrantySession)
    └── types.ts            Shared TypeScript types
```

---

## API Endpoints Used

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/chat` | Send customer message, receive streaming response |
| `GET` | `/api/v1/warranty/session/{session_id}` | Get structured warranty ticket state (options, status) |
| `POST` | `/api/v1/warranty/{ticket_id}/troubleshooting-outcome` | Record self-service progress and resolution outcome |
| `POST` | `/api/v1/warranty/{ticket_id}/evidence` | Upload evidence file |
| `GET` | `/api/v1/warranty/{ticket_id}/evidence` | List uploaded evidence |

All calls are proxied through Next.js rewrites (`next.config.mjs`) so the browser never needs CORS headers.

---

## Safety Rules (enforced in UI)

- The chat interface **never** shows warranty approval, replacement, or technician dispatch promises.
- Team-review controls stay hidden until the customer confirms the troubleshooting or preparation steps were completed. The resolved outcome is visually primary; unresolved and safety-exception paths remain available without promising an approval.
- When `status === awaiting_admin_review`, a banner reads:
  > "Your case has been prepared for support team review. Final warranty decisions are handled by our support team."
- The backend `saved_path` field in evidence upload responses is stripped before display.
- `ADMIN_API_KEY` is never exposed to the browser.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | Yes | `http://localhost:8000` | FastAPI backend URL |

---

## Production Build

```bash
npm run build
npm start
# Runs on port 3000
```

For Docker:

```bash
docker build -f Dockerfile.frontend -t titan-frontend .
docker run -p 3000:3000 -e NEXT_PUBLIC_API_BASE_URL=http://your-backend:8000 titan-frontend
```

Or use the included `docker-compose.yml` from the project root.

---

## Streamlit (Legacy)

The original Streamlit frontend is archived at `legacy/streamlit/`.
It is **NOT** started in production. See `legacy/streamlit/README.md` for details.
