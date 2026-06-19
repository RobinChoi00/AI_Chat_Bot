"use client";

import type { ReactNode } from "react";
import type { ChatMessage } from "@/lib/types";

interface Props {
  message: ChatMessage;
  isStreaming?: boolean;
}

const LINK_RE = /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g;
const BOLD_RE = /\*\*([^*]+)\*\*/g;
const URL_RE = /(https?:\/\/[^\s<]+[^\s<.,;:!?)])/g;

function renderInline(text: string, keyPrefix: string) {
  const parts: ReactNode[] = [];
  let remaining = text;
  let idx = 0;

  while (remaining.length > 0) {
    LINK_RE.lastIndex = 0;
    BOLD_RE.lastIndex = 0;
    URL_RE.lastIndex = 0;

    const linkMatch = LINK_RE.exec(remaining);
    const boldMatch = BOLD_RE.exec(remaining);
    const urlMatch = URL_RE.exec(remaining);

    type MatchKind = "link" | "bold" | "url";
    let best: { kind: MatchKind; index: number; length: number; nodes: ReactNode } | null =
      null;

    if (linkMatch?.index !== undefined) {
      best = {
        kind: "link",
        index: linkMatch.index,
        length: linkMatch[0].length,
        nodes: (
          <a
            key={`${keyPrefix}-link-${idx}`}
            href={linkMatch[2]}
            target="_blank"
            rel="noopener noreferrer"
            className="font-medium text-brand-700 underline underline-offset-2 hover:text-brand-900"
          >
            {linkMatch[1]}
          </a>
        ),
      };
    }

    if (boldMatch?.index !== undefined) {
      if (!best || boldMatch.index < best.index) {
        best = {
          kind: "bold",
          index: boldMatch.index,
          length: boldMatch[0].length,
          nodes: (
            <strong key={`${keyPrefix}-bold-${idx}`} className="font-semibold">
              {boldMatch[1]}
            </strong>
          ),
        };
      }
    }

    if (urlMatch?.index !== undefined) {
      if (!best || urlMatch.index < best.index) {
        best = {
          kind: "url",
          index: urlMatch.index,
          length: urlMatch[0].length,
          nodes: (
            <a
              key={`${keyPrefix}-url-${idx}`}
              href={urlMatch[1]}
              target="_blank"
              rel="noopener noreferrer"
              className="break-all font-medium text-brand-700 underline underline-offset-2 hover:text-brand-900"
            >
              {urlMatch[1]}
            </a>
          ),
        };
      }
    }

    if (!best) {
      parts.push(remaining);
      break;
    }

    if (best.index > 0) {
      parts.push(remaining.slice(0, best.index));
    }
    parts.push(best.nodes);
    remaining = remaining.slice(best.index + best.length);
    idx += 1;
  }

  return parts;
}

export default function ChatMessageBubble({ message, isStreaming }: Props) {
  const isUser = message.role === "user";
  const lines = message.content.split("\n");

  return (
    <div className={`flex w-full ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && (
        <div className="mr-2 mt-0.5 flex-shrink-0">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-brand-600 text-sm text-white shadow-sm">
            🛡️
          </div>
        </div>
      )}

      <div
        className={`max-w-[92%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm sm:max-w-[85%] ${
          isUser
            ? "rounded-br-md bg-brand-600 text-white"
            : "rounded-bl-md bg-white text-gray-800 ring-1 ring-gray-100"
        }`}
      >
        {lines.map((line, i) => (
          <span key={i} className="block">
            {renderInline(line, `line-${i}`)}
          </span>
        ))}
        {isStreaming && (
          <span className="ml-1 inline-block h-3 w-1.5 animate-pulse rounded-sm bg-current opacity-70" />
        )}
      </div>

      {isUser && (
        <div className="ml-2 mt-0.5 flex-shrink-0">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gray-200 text-sm shadow-sm">
            👤
          </div>
        </div>
      )}
    </div>
  );
}
