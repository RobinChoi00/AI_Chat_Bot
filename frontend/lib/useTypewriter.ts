"use client";

import { useEffect, useState } from "react";

interface Options {
  /** Full target text. */
  text: string;
  /** When false, the hook immediately returns the full text (no animation). */
  enabled: boolean;
  /** Characters revealed per tick. Defaults to 1. */
  chunkSize?: number;
  /** Milliseconds between ticks. Defaults to 75. */
  intervalMs?: number;
  /** Absolute cap so a very long response never drags on. Defaults to 2500ms. */
  maxDurationMs?: number;
}

/**
 * Reveal ``text`` character-by-character on the client — with human-like punctuation pauses.
 */
export function useTypewriter({
  text,
  enabled,
  chunkSize = 1,
  intervalMs = 75,
  maxDurationMs = 2500,
}: Options): { visible: string; done: boolean } {
  const [animation, setAnimation] = useState<{
    target: string;
    visible: string;
    done: boolean;
  }>({ target: text, visible: enabled ? "" : text, done: !enabled });

  useEffect(() => {
    if (!enabled) {
      return;
    }

    const total = text.length;
    if (total === 0) {
      return;
    }

    // 기존의 자동 속도 조절(맥스 제한) 로직 유지
    const budgetChunks = Math.ceil(maxDurationMs / intervalMs);
    const dynamicChunkSize = Math.max(chunkSize, Math.ceil(total / budgetChunks));

    let currentIndex = 0;
    let timeoutId: number | undefined;

    function tick() {
      // 다음 글자 범위 계산
      const nextIndex = Math.min(total, currentIndex + dynamicChunkSize);
      const justRevealed = text.slice(currentIndex, nextIndex);
      
      currentIndex = nextIndex;
      const isDone = currentIndex >= total;
      setAnimation({
        target: text,
        visible: text.slice(0, currentIndex),
        done: isDone,
      });

      if (isDone) {
        return;
      }

      // --- 사람 같은 타이핑 리듬감 계산 (Human-like Typing Logic) ---
      let nextDelay = intervalMs;

      // 1. 미세한 랜덤 속도 부여 (기계적인 일정함 방지)
      // 기준 속도의 ±15ms 범위 내에서 무작위 변화를 줍니다.
      const randomVariance = Math.floor(Math.random() * 31) - 15;
      nextDelay = Math.max(10, nextDelay + randomVariance);

      // 2. 문장 부호가 포함되어 있다면 사람이 숨을 고르거나 말문이 막힌 것처럼 지연 추가
      // 이번에 출력된 글자의 마지막 부분에 마침표나 쉼표가 있는지 확인합니다.
      const lastChar = justRevealed.trim().slice(-1);
      
      if (/[.?!]/.test(lastChar)) {
        nextDelay += 450; // 마침표, 물음표, 느낌표 뒤에는 0.45초 추가 휴식
      } else if (/[,-]/.test(lastChar)) {
        nextDelay += 250; // 쉼표 뒤에는 0.25초 짧은 휴식
      }

      // 다음 글자 출력을 위해 재귀 호출
      timeoutId = window.setTimeout(tick, nextDelay);
    }

    // 타이핑 애니메이션 시작
    timeoutId = window.setTimeout(tick, intervalMs);

    return () => {
      if (timeoutId !== undefined) window.clearTimeout(timeoutId);
    };
  }, [text, enabled, chunkSize, intervalMs, maxDurationMs]);

  if (!enabled) {
    return { visible: text, done: true };
  }
  if (!text) {
    return { visible: "", done: true };
  }
  if (animation.target !== text) {
    return { visible: "", done: false };
  }
  return { visible: animation.visible, done: animation.done };
}
