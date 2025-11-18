"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { askRAG, fetchLectures, LectureItem } from "@/lib/api";

type Hit = {
  id: string | number;
  tag: string;
  score: number;
  metadata: any;
  text: string;
  citation?: string | null;
  file_url?: string | null;
};
type LectureImageAsset = {
  id: number;
  img_url: string;
  title: string;
  description?: string | null;
  notes?: string | null;
  lecture_key?: string | null;
  area_description?: any;
};
type Msg = { role: "user" | "assistant" | "system"; content: string };

const LOADING_LINES = [
  "Good question, lemme check the lectures…",
  "Found a reference in the lecture, looking for more mentions related to this…",
  "Gathering thoughts…",
  "Just a second, truly a confounding question…",
];

function formatCitationLabel(raw?: string | null) {
  if (!raw) return "";
  let trimmed = String(raw).trim();
  if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
    trimmed = trimmed.slice(1, -1).trim();
  }
  const withoutFileToken = trimmed
    .replace(/FILE_[A-Z0-9._-]+/gi, "")
    .replace(/^[\s:-]+|[\s:-]+$/g, "");
  return withoutFileToken.replace(/\s{2,}/g, " ").trim();
}

export default function Chat() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const [hits, setHits] = useState<Hit[]>([]);
  const [diag, setDiag] = useState<any>({});
  const [lectures, setLectures] = useState<LectureItem[]>([]);
  const [selectedLecture, setSelectedLecture] = useState<string>("");
  const [lectureImages, setLectureImages] = useState<LectureImageAsset[]>([]);

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // fetch lecture options on mount
    fetchLectures()
      .then(setLectures)
      .catch(() => setLectures([]));
  }, []);

  useEffect(() => {
    // auto-scroll to bottom when messages change
    scrollRef.current?.scrollTo({ top: 1e9, behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (!loading) return;
    setLoadingStep(0);
    const id = setInterval(() => {
      setLoadingStep((s) => (s + 1) % LOADING_LINES.length);
    }, 1600);
    return () => clearInterval(id);
  }, [loading]);

  const submit = async () => {
    const query = q.trim();
    if (!query || loading) return;

    setMessages((m) => [...m, { role: "user", content: query }]);
    setQ("");
    setLoading(true);
    setLectureImages([]);

    try {
      const opts: any = { use_global: true };
      if (selectedLecture) opts.force_lecture_key = selectedLecture;
      const data = await askRAG(query, opts);

      const answer = data.answer || "(no answer)";
      setMessages((m) => [...m, { role: "assistant", content: answer }]);
      setHits(data.hits || []);
      setDiag(data.diagnostics || {});
      setLectureImages(data.lecture_images || []);
    } catch (e: any) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: `Error: ${e?.message ?? "unknown error"}`,
        },
      ]);
      setHits([]);
      setDiag({});
      setLectureImages([]);
    } finally {
      setLoading(false);
    }
  };

  const detectedLecture = diag?.lecture_forced || diag?.lecture_detected || "";

  return (
    <div className="min-h-[100dvh] bg-neutral-950 text-neutral-100 relative">
      {/* Main content */}
      <div
        className={`transition duration-300 ${
          loading ? "blur-[2px] pointer-events-none" : ""
        }`}
      >
        {/* Header */}
        <header className="sticky top-0 z-10 border-b border-neutral-900 bg-neutral-950/70 backdrop-blur-md">
          <div className="mx-auto max-w-3xl px-4 py-3 flex items-center gap-3">
            <div className="h-6 w-6 rounded bg-gradient-to-br from-indigo-400 to-fuchsia-500" />
            <div className="font-semibold">ArcheoSensei</div>
            <div className="ml-auto text-xs text-neutral-400">
              {detectedLecture ? (
                <span>
                  Detected lecture:{" "}
                  <span className="px-2 py-0.5 rounded bg-neutral-800 text-indigo-300 font-mono">
                    {detectedLecture}
                  </span>
                </span>
              ) : (
                <span>Auto-detecting lecture…</span>
              )}
            </div>
          </div>
        </header>

        {/* Chat thread */}
        <div className="mx-auto max-w-3xl px-4">
          <div
            ref={scrollRef}
            className="pt-6 pb-40 overflow-auto"
            style={{ minHeight: "calc(100dvh - 160px)" }}
          >
            {!messages.length ? (
              <div className="mt-16 text-center text-neutral-400">
                <div className="text-2xl font-semibold mb-2">
                  Ask your course
                </div>
                <div className="text-sm">
                  It will auto-detect the lecture unless you restrict it.
                </div>
              </div>
            ) : (
              <div className="space-y-5">
                {messages.map((m, i) => (
                  <div key={i} className="flex">
                    <div
                      className={`max-w-[85%] rounded-2xl px-4 py-3 leading-relaxed shadow
                        ${
                          m.role === "user"
                            ? "ml-auto bg-indigo-600/90 text-white"
                            : "mr-auto bg-neutral-900/90 border border-neutral-800"
                        }`}
                      style={{ whiteSpace: "pre-wrap" }}
                    >
                      {m.content}
                    </div>
                  </div>
                ))}

                {/* Sources panel */}
                {hits.length > 0 && (
                  <div className="mt-6 border border-neutral-800 rounded-2xl p-3 bg-neutral-900/60">
                    <div className="text-xs font-semibold text-neutral-300 mb-2">
                      Sources
                    </div>
                    <p className="text-[11px] text-neutral-400 mb-2">
                      Click any citation inside the answer to open a popup that auto-searches
                      the lecture using the same snippet so you can verify it quickly.
                    </p>
                    <div className="space-y-2 max-h-[35vh] overflow-auto">
                      {hits.map((h, idx) => (
                        <div
                          key={idx}
                          className="text-xs border border-neutral-800 rounded-xl p-2"
                        >
                          <div className="font-mono text-neutral-300">
                            {(h.citation && formatCitationLabel(h.citation)) ||
                              "Context"}
                          </div>
                          <div className="text-neutral-400 line-clamp-3">
                            {h.text}
                          </div>
                          <div className="text-[10px] text-neutral-500 mt-1">
                            {h.metadata?.lecture_key && (
                              <>lecture: {h.metadata.lecture_key} · </>
                            )}
                            {h.metadata?.slide_no != null && (
                              <>slide: {h.metadata.slide_no}</>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {lectureImages.length > 0 && (
                  <div className="mt-6 border border-neutral-800 rounded-2xl p-3 bg-neutral-900/60">
                    <div className="flex items-center gap-2 text-xs font-semibold text-neutral-300 mb-2">
                      Visual references
                      <span className="text-[10px] text-neutral-500 font-normal">
                        pulled directly from annotated lecture assets
                      </span>
                    </div>
                    <div className="grid gap-3 sm:grid-cols-2">
                      {lectureImages.map((img) => {
                        const highlights = Array.isArray(img.area_description)
                          ? img.area_description.slice(0, 3)
                          : [];
                        return (
                          <div
                            key={`${img.id}-${img.img_url}`}
                            className="border border-neutral-800 rounded-xl overflow-hidden bg-neutral-950/60"
                          >
                            <div className="relative aspect-video bg-neutral-900">
                              <img
                                src={img.img_url}
                                alt={img.title}
                                loading="lazy"
                                className="h-full w-full object-cover"
                              />
                              <a
                                href={img.img_url}
                                target="_blank"
                                rel="noreferrer"
                                className="absolute bottom-2 right-2 text-[10px] bg-neutral-900/80 text-neutral-200 px-2 py-0.5 rounded-full border border-neutral-800"
                              >
                                open
                              </a>
                            </div>
                            <div className="p-3 space-y-2">
                              <div>
                                <div className="text-sm font-semibold text-neutral-100">
                                  {img.title}
                                </div>
                                {img.lecture_key && (
                                  <div className="text-[11px] text-neutral-400">
                                    {img.lecture_key}
                                  </div>
                                )}
                              </div>
                              {img.description && (
                                <div className="text-[11px] text-neutral-400 whitespace-pre-line">
                                  {img.description}
                                </div>
                              )}
                              {img.notes && (
                                <div className="text-[11px] text-neutral-300 bg-neutral-900/80 border border-neutral-800 rounded-lg p-2">
                                  {img.notes}
                                </div>
                              )}
                              {highlights.length > 0 && (
                                <div>
                                  <div className="text-[10px] uppercase tracking-wide text-neutral-500 mb-1">
                                    Highlights
                                  </div>
                                  <ul className="space-y-1 text-[11px] text-neutral-400">
                                    {highlights.map((area: any, idx: number) => (
                                      <li key={idx} className="flex gap-2">
                                        <span className="text-neutral-500">•</span>
                                        <span>
                                          <span className="font-medium text-neutral-300">
                                            {area?.label || `Region ${idx + 1}`}
                                          </span>
                                          {area?.help_text && (
                                            <span className="text-neutral-500"> — {area.help_text}</span>
                                          )}
                                        </span>
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Bottom composer (sticky) */}
        <div className="fixed inset-x-0 bottom-0 z-20 border-t border-neutral-900 bg-neutral-950/80 backdrop-blur-md">
          <div className="mx-auto max-w-3xl px-4 py-3">
            <div className="flex gap-2">
              {/* Lecture dropdown */}
              <select
                className="w-44 shrink-0 rounded-xl border border-neutral-800 bg-neutral-900 text-neutral-200 px-3 py-2 text-sm"
                value={selectedLecture}
                onChange={(e) => setSelectedLecture(e.target.value)}
              >
                <option value="">All lectures</option>
                {lectures.map((l) => (
                  <option key={l.lecture_key} value={l.lecture_key}>
                    {l.lecture_key} ({l.count})
                  </option>
                ))}
              </select>

              {/* Prompt input */}
              <input
                className="flex-1 rounded-xl border border-neutral-800 bg-neutral-900 px-4 py-3
                           placeholder-neutral-500 focus:outline-none focus:ring-1 focus:ring-neutral-700"
                placeholder="Ask anything about your lectures…"
                value={q}
                onChange={(e) => setQ(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && submit()}
                disabled={loading}
              />

              {/* Ask button */}
              <button
                onClick={submit}
                disabled={loading}
                className="shrink-0 rounded-xl px-4 py-3 bg-white text-black font-semibold
                           hover:opacity-90 disabled:opacity-50"
              >
                Ask
              </button>
            </div>
            <div className="text-[11px] text-neutral-500 mt-1">
              {selectedLecture
                ? `Restricted to ${selectedLecture}`
                : "Auto-detecting lecture"}
            </div>
          </div>
        </div>
      </div>

      {/* Full-screen loading overlay with blur + cycling text */}
      {loading && (
        <div className="pointer-events-none fixed inset-0 z-30 flex items-center justify-center">
          <div className="absolute inset-0 bg-neutral-950/40 backdrop-blur-sm" />
          <div className="relative bg-neutral-900/90 border border-neutral-800 rounded-2xl px-6 py-5 shadow-xl">
            <div className="animate-pulse text-neutral-200">
              {LOADING_LINES[loadingStep]}
            </div>
            <div className="mt-2 text-[11px] text-neutral-500 text-center">
              Thinking…
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
