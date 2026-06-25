/**
 * Browser text-to-speech helpers (Web Speech API).
 * Used for "Read aloud" on warranty assistant messages — no backend cost.
 */

export function isSpeechSupported(): boolean {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

/** Strip markdown so TTS reads natural sentences, not syntax. */
export function messageContentToSpeech(text: string): string {
  return text
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/https?:\/\/[^\s]+/g, "")
    .replace(/[🛡️👋💌📦🔧⚙️✓✗→]/g, "")
    .replace(/\n+/g, ". ")
    .replace(/\s+/g, " ")
    .trim();
}

let activeUtterance: SpeechSynthesisUtterance | null = null;

export function stopSpeech(): void {
  if (!isSpeechSupported()) return;
  window.speechSynthesis.cancel();
  activeUtterance = null;
}

export function speakText(
  text: string,
  callbacks?: { onEnd?: () => void; onError?: () => void }
): boolean {
  if (!isSpeechSupported()) return false;

  const plain = messageContentToSpeech(text);
  if (!plain) return false;

  stopSpeech();

  const utterance = new SpeechSynthesisUtterance(plain);
  utterance.lang = "en-US";
  utterance.rate = 1;

  const voices = window.speechSynthesis.getVoices();
  const enVoice =
    voices.find((v) => v.lang.startsWith("en") && v.localService) ??
    voices.find((v) => v.lang.startsWith("en"));
  if (enVoice) utterance.voice = enVoice;

  utterance.onend = () => {
    activeUtterance = null;
    callbacks?.onEnd?.();
  };
  utterance.onerror = () => {
    activeUtterance = null;
    callbacks?.onError?.();
  };

  activeUtterance = utterance;
  window.speechSynthesis.speak(utterance);
  return true;
}

/** iOS Safari loads voices asynchronously — warm up on first client mount. */
export function warmSpeechVoices(): void {
  if (!isSpeechSupported()) return;
  window.speechSynthesis.getVoices();
  window.speechSynthesis.onvoiceschanged = () => {
    window.speechSynthesis.getVoices();
  };
}
