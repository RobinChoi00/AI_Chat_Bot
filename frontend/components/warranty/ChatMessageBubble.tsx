import type { ChatMessage } from "@/lib/types";

interface Props {
  message: ChatMessage;
  isStreaming?: boolean;
}

export default function ChatMessageBubble({ message, isStreaming }: Props) {
  const isUser = message.role === "user";

  return (
    <div className={`flex w-full ${isUser ? "justify-end" : "justify-start"}`}>
      {/* Avatar */}
      {!isUser && (
        <div className="mr-2 mt-0.5 flex-shrink-0">
          <div className="flex h-7 w-7 items-center justify-center rounded-full bg-brand-600 text-sm text-white shadow-sm">
            🛡️
          </div>
        </div>
      )}

      {/* Bubble */}
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm shadow-sm ${
          isUser
            ? "rounded-br-sm bg-brand-600 text-white"
            : "rounded-bl-sm bg-white text-gray-800 ring-1 ring-gray-100"
        }`}
      >
        {/* Render newlines as line breaks */}
        {message.content.split("\n").map((line, i) => (
          <span key={i}>
            {line}
            {i < message.content.split("\n").length - 1 && <br />}
          </span>
        ))}
        {isStreaming && (
          <span className="ml-1 inline-block h-3 w-1.5 animate-pulse rounded-sm bg-current opacity-70" />
        )}
      </div>

      {/* User avatar */}
      {isUser && (
        <div className="ml-2 mt-0.5 flex-shrink-0">
          <div className="flex h-7 w-7 items-center justify-center rounded-full bg-gray-200 text-sm shadow-sm">
            👤
          </div>
        </div>
      )}
    </div>
  );
}
