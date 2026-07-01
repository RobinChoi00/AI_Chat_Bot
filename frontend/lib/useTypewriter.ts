"use client";

import { useEffect, useState } from "react";

interface Options {
  /** Full target text. */
  text: string;
  /** When false, the hook immediately returns the full text (no animation). */
  enabled: boolean;
  /** Characters revealed per tick. Defaults to 3. */
  chunkSize?: number;
  /** Milliseconds between ticks. Defaults to 22. */
  intervalMs?: number;
  /** Absolute cap so a very long response never drags on. Defaults to 2500ms. */
  maxDurationMs?: number;
}

/**
 * Reveal ``text`` character-by-character on the client — no backend required.
 *
 * The chunk size auto-scales when the full text is much longer than the max
 * duration so we never exceed roughly ``maxDurationMs`` even for a long
 * diagnosis message. Falls back to instant reveal when ``enabled`` is false
 * (used for hydrated / historical messages).
 */
export function useTypewriter({
  text,
  enabled,
  chunkSize = 3,
  intervalMs = 22,
  maxDurationMs = 2500,
}: Options): { visible: string; done: boolean } {
  const [visible, setVisible] = useState<string>(enabled ? "" : text);
  const [done, setDone] = useState<boolean>(!enabled);

  useEffect(() => {
    if (!enabled) {
      setVisible(text);
      setDone(true);
      return;
    }
    setVisible("");
    setDone(false);

    const total = text.length;
    if (total === 0) {
      setDone(true);
      return;
    }

    const budgetChunks = Math.ceil(maxDurationMs / intervalMs);
    const dynamicChunkSize = Math.max(chunkSize, Math.ceil(total / budgetChunks));

    let i = 0;
    const handle = window.setInterval(() => {
      i = Math.min(total, i + dynamicChunkSize);
      setVisible(text.slice(0, i));
      if (i >= total) {
        window.clearInterval(handle);
        setDone(true);
      }
    }, intervalMs);

    return () => window.clearInterval(handle);
  }, [text, enabled, chunkSize, intervalMs, maxDurationMs]);

  return { visible, done };
}
