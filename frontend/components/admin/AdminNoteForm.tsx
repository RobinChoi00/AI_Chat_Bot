"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { submitAdminNote } from "@/lib/adminApi";

interface Props {
  ticketId: string;
  currentNote: string | null;
}

export default function AdminNoteForm({ ticketId, currentNote }: Props) {
  const router = useRouter();
  const [note, setNote] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!note.trim()) return;

    setLoading(true);
    setError(null);
    setSuccess(false);

    try {
      await submitAdminNote(ticketId, {
        note: note.trim(),
        added_by: "admin",
      });
      setNote("");
      setSuccess(true);
      router.refresh();
      setTimeout(() => setSuccess(false), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add note.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-3">
      {/* Existing note */}
      {currentNote && (
        <div className="rounded-lg border border-gray-200 bg-gray-50 p-3">
          <p className="mb-1 text-xs font-medium text-gray-500 uppercase tracking-wide">
            Current Admin Note
          </p>
          <p className="whitespace-pre-wrap text-sm text-gray-800">{currentNote}</p>
        </div>
      )}

      {/* New note form */}
      <form onSubmit={handleSubmit} className="space-y-2">
        <textarea
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="Add an internal note (not visible to customer)…"
          rows={3}
          disabled={loading}
          className="w-full resize-y rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 placeholder-gray-400 focus:border-indigo-400 focus:outline-none focus:ring-1 focus:ring-indigo-400 disabled:opacity-60"
        />

        {error && (
          <p className="text-xs text-red-600">⚠️ {error}</p>
        )}
        {success && (
          <p className="text-xs text-green-600">✅ Note saved.</p>
        )}

        <button
          type="submit"
          disabled={!note.trim() || loading}
          className="rounded-lg bg-indigo-600 px-4 py-1.5 text-sm font-medium text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Saving…" : "Add Note"}
        </button>
      </form>
    </div>
  );
}
