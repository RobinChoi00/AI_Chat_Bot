"use client";

import { useCallback, useEffect, useState, useSyncExternalStore } from "react";
import {
  isSpeechSupported,
  messageContentToSpeech,
  speakText,
  stopSpeech,
  warmSpeechVoices,
} from "@/lib/speech";

interface Props {
  text: string;
  className?: string;
}

export default function SpeakButton({ text, className = "" }: Props) {
  const [speaking, setSpeaking] = useState(false);
  const supported = useSyncExternalStore(
    () => () => undefined,
    isSpeechSupported,
    () => false
  );

  useEffect(() => {
    warmSpeechVoices();
    return () => stopSpeech();
  }, []);

  const plain = messageContentToSpeech(text);

  const toggle = useCallback(() => {
    if (speaking) {
      stopSpeech();
      setSpeaking(false);
      return;
    }
    const started = speakText(text, {
      onEnd: () => setSpeaking(false),
      onError: () => setSpeaking(false),
    });
    if (started) setSpeaking(true);
  }, [text, speaking]);

  if (!supported || !plain) return null;

  return (
    <button
      type="button"
      onClick={toggle}
      aria-pressed={speaking}
      aria-label={speaking ? "Stop reading aloud" : "Read message aloud"}
      className={`inline-flex min-h-[36px] items-center gap-1.5 rounded-lg px-2 py-1 text-xs font-medium text-brand-700 transition hover:bg-brand-50 active:scale-[0.98] ${className}`}
    >
      <span aria-hidden>{speaking ? "⏹" : "🔊"}</span>
      <span>{speaking ? "Stop" : "Listen"}</span>
    </button>
  );
}
