"use client";

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  fetchLectures,
  LectureItem,
  startQueryJob,
  fetchQueryJob,
} from "@/lib/api";
import {
  Chat,
  Msg,
  Hit,
  FactCheckResult,
  loadChats,
  saveChats,
  loadActiveChatId,
  saveActiveChatId,
  createChat,
  appendMessage,
  replaceMessages,
  renameChat,
  deleteChat,
} from "@/lib/chat";

// ------- constants / helpers -------
type Phase =
  | "idle"
  | "sent"
  | "retrieving"
  | "llm"
  | "fact_ai"
  | "fact_claims"
  | "done";

const LOADING_LINES = [
  "Good question, lemme check the lectures…",
  "Found a reference in the lecture, looking for more mentions related to this…",
  "Gathering thoughts…",
  "Just a second, truly a confounding question…",
];

function backendBase() {
  return (
    typeof window === "undefined"
      ? process.env.BACKEND_URL_INTERNAL
      : process.env.NEXT_PUBLIC_BACKEND_URL
  ) as string;
}

function labelFromMeta(md: any) {
  if (!md) return "";
  if (md.slide_no != null && md.lecture_key) {
    const n = (md.lecture_key as string).split("_").pop();
    return `Lecture ${n} Slide ${md.slide_no}`;
  }
  if (md.source === "lecture_note" && md.lecture_key) {
    const n = (md.lecture_key as string).split("_").pop();
    return `Lecture ${n} Notes`;
  }
  if (md.source === "readings" && md.lecture_key) {
    const n = (md.lecture_key as string).split("_").pop();
    return `From Lecture ${n}`;
  }
  if (md.store === "global") return "Global";
  return md.source === "user_note" ? "User" : "";
}

// Turn "[Lecture N Slide M]" → markdown link: (cite:lec_N:M)
function toCiteLinks(text: string) {
  return text.replace(
    /\[Lecture\s+(\d+)\s+Slide\s+(\d+)\]/g,
    (_m, n, s) => `[Lecture ${n} Slide ${s}](cite:lec_${n}:${s})`
  );
}

function LectureRef({
  lecture,
  preview,
}: {
  lecture: number;
  preview: string;
}) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        title={`From Lecture ${lecture}: ${preview.slice(0, 120)}${
          preview.length > 120 ? "..." : ""
        }`}
        className="inline-flex items-center justify-center w-5 h-5 text-[10px] font-semibold rounded-full bg-indigo-500/90 text-white ml-1 hover:bg-indigo-400 focus:outline-none focus:ring-1 focus:ring-indigo-300"
      >
        L{lecture}
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="bg-neutral-900 text-neutral-100 rounded-xl p-6 shadow-lg text-center max-w-xs">
            <div className="text-sm mb-3 font-semibold">Lecture {lecture}</div>
            <p className="text-neutral-400 mb-4">Under Construction...</p>
            <button
              onClick={() => setOpen(false)}
              className="rounded-lg bg-indigo-500 text-white px-3 py-1 text-sm font-medium hover:bg-indigo-400"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </>
  );
}

// -----------------------------------

export default function ChatPage() {
  // sidebar state: chats in localStorage
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const activeChat = useMemo<Chat | null>(
    () => chats.find((c) => c.id === activeId) || null,
    [chats, activeId]
  );

  // prompt for new chat name
  const [namerOpen, setNamerOpen] = useState(false);
  const [namerValue, setNamerValue] = useState("");

  // UI / query state
  const [lectures, setLectures] = useState<LectureItem[]>([]);
  const [selectedLecture, setSelectedLecture] = useState<string>("");
  const [diag, setDiag] = useState<any>({});
  const [q, setQ] = useState("");

  // modal (source)
  const [popupSource, setPopupSource] = useState<{
    lecture_key: string;
    slide_no: number;
    text: string;
    image_url: string;
  } | null>(null);

  // HUD
  const [phase, setPhase] = useState<Phase>("idle");
  const [jobId, setJobId] = useState<string | null>(null);
  const [loadingLineIdx, setLoadingLineIdx] = useState(0);
  const [statusLine, setStatusLine] = useState("");
  const [retryCount, setRetryCount] = useState(0);
  const [maxRetries, setMaxRetries] = useState(3);
  const [factAiStatus, setFactAiStatus] = useState<"passed" | "failed" | null>(
    null
  );
  const [factClaimsStatus, setFactClaimsStatus] = useState<
    "passed" | "failed" | null
  >(null);
  const jobChatRef = useRef<string | null>(null);
  const progressPct =
    phase === "idle"
      ? 0
      : phase === "sent"
      ? 18
      : phase === "retrieving"
      ? 40
      : phase === "llm"
      ? 65
      : phase === "fact_ai"
      ? 82
      : phase === "fact_claims"
      ? 92
      : 100;

  const scrollRef = useRef<HTMLDivElement>(null);

  // load chats on mount
  useEffect(() => {
    const cs = loadChats();
    let id = loadActiveChatId();
    if (!cs.length) {
      setNamerOpen(true); // first run → ask for a name
    }
    setChats(cs);
    if (id && cs.some((c) => c.id === id)) setActiveId(id);
  }, []);

  // persist chats & active id
  useEffect(() => {
    saveChats(chats);
  }, [chats]);
  useEffect(() => {
    if (activeId) saveActiveChatId(activeId);
  }, [activeId]);

  // fetch lectures on mount
  useEffect(() => {
    fetchLectures()
      .then(setLectures)
      .catch(() => setLectures([]));
  }, []);

  // autoscroll
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: 1e9, behavior: "smooth" });
  }, [activeChat?.messages]);

  // HUD cycling
  useEffect(() => {
    if (phase === "idle" || phase === "done") return;
    const id = setInterval(
      () => setLoadingLineIdx((i) => (i + 1) % LOADING_LINES.length),
      1500
    );
    return () => clearInterval(id);
  }, [phase]);

  // actions: new chat
  function actionNewChat() {
    setNamerValue("");
    setNamerOpen(true);
  }
  function confirmCreateChat() {
    const name = namerValue.trim();
    if (!name) return;
    const c = createChat(name);
    const next = [c, ...chats];
    setChats(next);
    setActiveId(c.id);
    setNamerOpen(false);
  }

  // actions: select chat
  function selectChat(id: string) {
    setActiveId(id);
  }

  // actions: rename / delete (optional quick)
  function doRenameChat(id: string) {
    const name = prompt(
      "Rename chat:",
      chats.find((c) => c.id === id)?.name || ""
    );
    if (!name) return;
    setChats((prev) => renameChat(prev, id, name));
  }
  function doDeleteChat(id: string) {
    if (!confirm("Delete this chat?")) return;
    const next = deleteChat(chats, id);
    setChats(next);
    if (activeId === id) setActiveId(next[0]?.id || null);
  }

  // submit question
  const submit = async () => {
    if (!activeChat) {
      // force name prompt if no active chat exists
      actionNewChat();
      return;
    }
    const query = q.trim();
    if (!query || phase !== "idle") return;

    setChats((prev) =>
      appendMessage(prev, activeChat.id, { role: "user", content: query })
    );
    setQ("");
    setPhase("sent");
    setDiag({});
    setRetryCount(0);
    setMaxRetries(3);
    setStatusLine("");
    setFactAiStatus(null);
    setFactClaimsStatus(null);
    let jobStarted = false;
    try {
      const opts: any = { use_global: true };
      if (selectedLecture) opts.force_lecture_key = selectedLecture;
      const { job_id } = await startQueryJob(query, opts);
      jobChatRef.current = activeChat.id;
      setJobId(job_id);
      jobStarted = true;
    } catch (e: any) {
      setStatusLine("Encountered an error while generating answer.");
      setChats((prev) =>
        appendMessage(prev, activeChat.id, {
          role: "assistant",
          content: `Error: ${e?.message ?? "unknown error"}`,
        })
      );
    } finally {
      if (!jobStarted) {
        setTimeout(() => {
          setPhase("idle");
          setStatusLine("");
        }, 400);
      }
    }
  };

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    let handled = false;

    const phaseFromJob = (jobPhase: string): Phase => {
      switch (jobPhase) {
        case "retrieving":
          return "retrieving";
        case "llm":
          return "llm";
        case "fact_ai":
          return "fact_ai";
        case "fact_claims":
          return "fact_claims";
        case "done":
          return "done";
        case "queued":
        case "running":
        default:
          return "sent";
      }
    };

    const poll = async () => {
      try {
        const status = await fetchQueryJob(jobId);
        if (cancelled) return;
        setPhase(phaseFromJob(status.phase));
        if (typeof status.retry_count === "number") {
          setRetryCount(status.retry_count);
        }
        if (typeof status.max_attempts === "number") {
          setMaxRetries(status.max_attempts);
        }
        setStatusLine(status.message?.trim() || "");
        const aiStatus =
          status.fact_ai_status === "passed" || status.fact_ai_status === "failed"
            ? status.fact_ai_status
            : null;
        const claimsStatus =
          status.fact_claims_status === "passed" ||
          status.fact_claims_status === "failed"
            ? status.fact_claims_status
            : null;
        setFactAiStatus(aiStatus);
        setFactClaimsStatus(claimsStatus);

        if (!handled && status.status !== "running" && status.status !== "queued") {
          handled = true;
          const fact = status.fact_check as FactCheckResult | undefined;
          const limitedHits: Hit[] = (status.hits || []).slice(0, 3);

          if (status.status === "failed" && !status.message) {
            setStatusLine("AI response could not be validated after retries.");
          }

          const targetChatId = jobChatRef.current || activeChat?.id;
          if (targetChatId) {
            setChats((prev) =>
              appendMessage(prev, targetChatId, {
                role: "assistant",
                content: status.answer || "(no answer)",
                hits: limitedHits,
                diagnostics: status.diagnostics || {},
                fact_check: fact,
              })
            );
          }
          setDiag(status.diagnostics || {});
          setPhase("done");
          setJobId(null);
          jobChatRef.current = null;
          setTimeout(() => {
            if (!cancelled) {
              setPhase("idle");
              setStatusLine("");
            }
          }, 600);
        }
      } catch (err: any) {
        if (cancelled) return;
        setStatusLine("Lost connection to validation job.");
        setPhase("done");
        setJobId(null);
        jobChatRef.current = null;
        setTimeout(() => {
          if (!cancelled) {
            setPhase("idle");
          }
        }, 600);
      }
    };

    poll();
    const id = setInterval(poll, 900);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [jobId, activeChat?.id, setChats]);

  // open source
  async function openSourceByMeta(md: any) {
    if (!md?.lecture_key || md.slide_no == null) return;
    await openSource(md.lecture_key, md.slide_no);
  }
  async function openSource(lecture_key: string, slide_no: number) {
    const res = await fetch(
      `${backendBase()}/source/${lecture_key}/${slide_no}`,
      { cache: "no-store" }
    );
    if (!res.ok) return;
    const data = await res.json();
    setPopupSource(data);
  }

  const detectedLecture = diag?.lecture_forced || diag?.lecture_detected || "";
  const messages = activeChat?.messages || [];

  return (
    <div className="min-h-[100dvh] bg-neutral-950 text-neutral-100 relative flex">
      {/* Sidebar */}
      <aside className="w-64 border-r border-neutral-900 bg-neutral-950/80 backdrop-blur-sm p-3 hidden md:flex md:flex-col">
        <button
          onClick={actionNewChat}
          className="w-full mb-3 rounded-lg bg-white text-black font-semibold py-2 hover:opacity-90"
        >
          + New Chat
        </button>
        <div className="text-xs text-neutral-400 mb-2">Chats</div>
        <div className="flex-1 overflow-auto space-y-1">
          {chats.map((c) => (
            <div
              key={c.id}
              className={`group flex items-center justify-between gap-2 rounded-lg px-2 py-2 cursor-pointer
                          ${
                            activeId === c.id
                              ? "bg-neutral-800 text-white"
                              : "hover:bg-neutral-900 text-neutral-300"
                          }`}
              onClick={() => selectChat(c.id)}
              title={c.name}
            >
              <div className="truncate">{c.name}</div>
              <div className="opacity-0 group-hover:opacity-100 flex gap-2 text-xs">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    doRenameChat(c.id);
                  }}
                  className="hover:text-white"
                >
                  Rename
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    doDeleteChat(c.id);
                  }}
                  className="hover:text-red-300"
                >
                  Del
                </button>
              </div>
            </div>
          ))}
          {!chats.length && (
            <div className="text-neutral-600 text-xs">No chats yet.</div>
          )}
        </div>
      </aside>

      {/* Main */}
      <div className="flex-1 min-w-0">
        {/* Header */}
        <header className="sticky top-0 z-10 border-b border-neutral-900 bg-neutral-950/70 backdrop-blur-md">
          <div className="mx-auto max-w-3xl px-4 py-3 flex items-center gap-3">
            <div className="h-6 w-6 rounded bg-gradient-to-br from-indigo-400 to-fuchsia-500" />
            <div className="font-semibold truncate">
              {activeChat?.name || "ArcheoSensei"}
            </div>
            <div className="ml-auto text-xs text-neutral-400">
              {detectedLecture ? (
                <>
                  Detected lecture:{" "}
                  <span className="px-2 py-0.5 rounded bg-neutral-800 text-indigo-300 font-mono">
                    {detectedLecture}
                  </span>
                </>
              ) : (
                <>Auto-detecting lecture…</>
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
              <EmptyState />
            ) : (
              <div className="space-y-5">
                {messages.map((m, i) => (
                  <ChatTurn
                    key={i}
                    msg={m}
                    onOpenSourceMeta={openSourceByMeta}
                    onOpenToken={openSource}
                  />
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Bottom composer */}
        <Composer
          selectedLecture={selectedLecture}
          setSelectedLecture={setSelectedLecture}
          lectures={lectures}
          q={q}
          setQ={setQ}
          disabled={phase !== "idle" && phase !== "done"}
          onSubmit={submit}
        />
      </div>

      {/* HUD */}
      {phase !== "idle" && (
        <ProgressHUD
          progressPct={progressPct}
          line={statusLine || LOADING_LINES[loadingLineIdx]}
          phase={phase}
          retryCount={retryCount}
          maxRetries={maxRetries}
          factAiStatus={factAiStatus}
          factClaimsStatus={factClaimsStatus}
        />
      )}

      {/* Source modal */}
      {popupSource && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="bg-neutral-900 rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden shadow-lg relative">
            <button
              className="absolute top-3 right-3 text-neutral-400 hover:text-white"
              onClick={() => setPopupSource(null)}
            >
              ✕
            </button>
            <div className="flex flex-col md:flex-row h-full">
              <div className="flex-1 bg-black flex items-center justify-center">
                <img
                  src={popupSource.image_url}
                  alt={`Slide ${popupSource.slide_no}`}
                  className="object-contain max-h-[80vh]"
                />
              </div>
              <div className="w-full md:w-1/2 p-4 overflow-auto bg-neutral-950 border-l border-neutral-800">
                <h2 className="font-semibold mb-2">
                  Lecture {popupSource.lecture_key.replace("lec_", "")} – Slide{" "}
                  {popupSource.slide_no}
                </h2>
                <p className="text-neutral-300 whitespace-pre-wrap">
                  {popupSource.text}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* New Chat name modal */}
      {namerOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-5 w-[420px]">
            <div className="text-lg font-semibold mb-3">Enter Name of Chat</div>
            <input
              autoFocus
              value={namerValue}
              onChange={(e) => setNamerValue(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && confirmCreateChat()}
              placeholder="e.g., Week 3: Hashing & Trees"
              className="w-full rounded-xl border border-neutral-800 bg-neutral-950 px-3 py-2 mb-3 focus:outline-none focus:ring-1 focus:ring-neutral-700"
            />
            <div className="flex justify-end gap-2">
              <button
                className="px-4 py-2 rounded-xl border border-neutral-800 hover:bg-neutral-800/40"
                onClick={() => setNamerOpen(false)}
              >
                Cancel
              </button>
              <button
                className="px-4 py-2 rounded-xl bg-white text-black font-semibold disabled:opacity-50"
                disabled={!namerValue.trim()}
                onClick={confirmCreateChat}
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ---------- Presentational subcomponents ---------- */

function EmptyState() {
  return (
    <div className="mt-16 text-center text-neutral-400">
      <div className="text-2xl font-semibold mb-2">Ask your course</div>
      <div className="text-sm">
        Name your chat in the sidebar → “New Chat”. It will auto-detect the
        lecture unless you restrict it.
      </div>
    </div>
  );
}

function Composer({
  selectedLecture,
  setSelectedLecture,
  lectures,
  q,
  setQ,
  disabled,
  onSubmit,
}: {
  selectedLecture: string;
  setSelectedLecture: (v: string) => void;
  lectures: LectureItem[];
  q: string;
  setQ: (v: string) => void;
  disabled: boolean;
  onSubmit: () => void;
}) {
  return (
    <div className="fixed inset-x-0 bottom-0 z-20 border-t border-neutral-900 bg-neutral-950/80 backdrop-blur-md">
      <div className="mx-auto max-w-3xl px-4 py-3">
        <div className="flex gap-2">
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

          <input
            className="flex-1 rounded-xl border border-neutral-800 bg-neutral-900 px-4 py-3
                       placeholder-neutral-500 focus:outline-none focus:ring-1 focus:ring-neutral-700"
            placeholder="Ask anything about your lectures…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && onSubmit()}
            disabled={disabled}
          />

          <button
            onClick={onSubmit}
            disabled={disabled}
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
  );
}

/**
 * Renders a single message. Assistant answers are Markdown.
 * Citations like [Lecture N Slide M] become clickable (cite:lec_N:M).
 * Source list limited to 3 and each card is clickable → opens modal.
 */
function ChatTurn({
  msg,
  onOpenSourceMeta,
  onOpenToken,
}: {
  msg: Msg;
  onOpenSourceMeta: (md: any) => void;
  onOpenToken: (lec: string, slide: number) => void;
}) {
  const isUser = msg.role === "user";

  const AssistantMarkdown = ({ content }: { content: string }) => {
    const rewritten = toCiteLinks(content);
    return (
      <div className="prose prose-invert max-w-none prose-p:my-2 prose-li:my-1">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            a: ({ href, children, ...props }) => {
              if (href?.startsWith("cite:")) {
                const [, payload] = href.split("cite:");
                const [lecPart, slideStr] = payload.split(":"); // lec_1:3
                const lec = lecPart;
                const slide = Number(slideStr);
                return (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.preventDefault();
                      if (lec && Number.isFinite(slide))
                        onOpenToken(lec, slide);
                    }}
                    className="inline text-indigo-300 underline decoration-dotted hover:text-indigo-200"
                    {...props}
                  >
                    {children}
                  </button>
                );
              }
              return (
                <a {...props} href={href} className="text-indigo-300 underline">
                  {children}
                </a>
              );
            },
            text: ({ node, children }) => {
              // detect patterns like "[From Lecture 7]" or "[Lecture 3 Slide 2]"
              const textStr = String(children);
              const match = textStr.match(/\[From Lecture (\d+)\]/i);
              if (match) {
                const lectureNum = parseInt(match[1]);
                const previewText = msg.content.slice(0, 200); // basic context preview
                return (
                  <LectureRef lecture={lectureNum} preview={previewText} />
                );
              }
              return <>{children}</>;
            },

            code: ({ inline, className, children, ...props }) =>
              !inline ? (
                <pre className="bg-neutral-950 border border-neutral-800 rounded-lg p-3 overflow-auto">
                  <code className={className} {...props}>
                    {children}
                  </code>
                </pre>
              ) : (
                <code className="bg-neutral-800/70 rounded px-1.5 py-0.5">
                  {children}
                </code>
              ),
            table: (props) => (
              <table className="table-auto border-collapse" {...props} />
            ),
            th: (props) => (
              <th
                className="border border-neutral-700 px-2 py-1 bg-neutral-800"
                {...props}
              />
            ),
            td: (props) => (
              <td className="border border-neutral-800 px-2 py-1" {...props} />
            ),
          }}
        >
          {rewritten}
        </ReactMarkdown>
      </div>
    );
  };

  // Max 3 sources are already enforced at insertion time; this is just another safety slice
  const hits = (msg.hits || []).slice(0, 3);
  const factCheck = msg.fact_check;
  const factAttempts = factCheck?.attempts ?? [];
  const lastAttempt = factAttempts[factAttempts.length - 1];
  const claimCheck = lastAttempt?.claims_check;
  const aiConfidence = lastAttempt?.ai_check?.confidence;
  const claimScore = claimCheck?.score;
  const entailmentPct =
    typeof claimScore === "number"
      ? Math.round(Math.min(Math.max(claimScore, 0), 1) * 100)
      : null;
  const entailedClaims =
    typeof claimCheck?.entailed === "number" ? claimCheck.entailed : null;
  const totalClaims =
    typeof claimCheck?.total_claims === "number"
      ? claimCheck.total_claims
      : null;
  const entailmentTarget =
    typeof claimCheck?.threshold === "number"
      ? Math.round(Math.min(Math.max(claimCheck.threshold, 0), 1) * 100)
      : null;
  const failingClaim = claimCheck?.claims?.find(
    (c) => c && c.label && c.label.toLowerCase() !== "entailment"
  );
  const failingLabel = failingClaim?.label
    ? failingClaim.label.toUpperCase()
    : "";
  const failingSnippet = failingClaim?.claim
    ? failingClaim.claim.slice(0, 160)
    : "";
  const confidencePct =
    typeof aiConfidence === "number"
      ? Math.round(Math.min(Math.max(aiConfidence, 0), 1) * 100)
      : null;
  const maxAttemptsDisplay =
    factCheck?.max_attempts ??
    (factAttempts.length ? factAttempts.length : factCheck ? factCheck.retry_count + 1 : 1);

  return (
    <div className="flex">
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-3 leading-relaxed shadow ${
          isUser
            ? "ml-auto bg-indigo-600/90 text-white"
            : "mr-auto bg-neutral-900/90 border border-neutral-800"
        }`}
      >
        {isUser ? (
          <div className="whitespace-pre-wrap">{msg.content}</div>
        ) : (
          <AssistantMarkdown content={msg.content} />
        )}

        {!isUser && factCheck && (
          <div className="mt-3 border-t border-neutral-800 pt-2">
            <div className="text-xs font-semibold text-neutral-300 mb-1">
              Validation
            </div>
            {factCheck.status === "passed" ? (
              <div className="text-[11px] text-green-300">
                Passed fact checks
                {confidencePct !== null && ` · LLM confidence ${confidencePct}%`}
                {entailmentPct !== null && (
                  <>
                    {" "}· Evidence entailment {entailmentPct}%
                    {entailedClaims !== null && totalClaims !== null &&
                      ` (${entailedClaims}/${totalClaims} claims)`}
                    {entailmentTarget !== null && ` · target ${entailmentTarget}%`}
                  </>
                )}
              </div>
            ) : (
              <div className="text-[11px] text-yellow-300">
                Failed validation after {maxAttemptsDisplay} attempts.
                {factCheck.message ? ` ${factCheck.message}` : " Answer may be unreliable."}
              </div>
            )}
            {factCheck.status !== "passed" && entailmentPct !== null && (
              <div className="text-[11px] text-neutral-400 mt-1">
                Evidence entailment {entailmentPct}%
                {entailedClaims !== null && totalClaims !== null &&
                  ` (${entailedClaims}/${totalClaims} claims)`}
                {entailmentTarget !== null && ` · target ${entailmentTarget}%`}
              </div>
            )}
            {factCheck.status === "passed" && factCheck.retry_count > 0 && (
              <div className="text-[11px] text-neutral-400 mt-1">
                Validated after {factCheck.retry_count}{" "}
                {factCheck.retry_count === 1 ? "retry" : "retries"}.
              </div>
            )}
            {factCheck.status !== "passed" && failingSnippet && failingLabel && (
              <div className="text-[11px] text-red-300 mt-1">
                Key issue: “{failingSnippet}” → {failingLabel}
              </div>
            )}
          </div>
        )}

        {/* per-message sources (assistant only, max 3) */}
        {!isUser && hits.length > 0 && (
          <div className="mt-3 border-t border-neutral-800 pt-2">
            <div className="text-xs font-semibold text-neutral-300 mb-2">
              Sources
            </div>
            <div className="space-y-2">
              {hits.map((h, idx) => (
                <button
                  key={idx}
                  className="w-full text-left text-xs border border-neutral-800 rounded-xl p-2 hover:bg-neutral-800/40"
                  onClick={() => onOpenSourceMeta(h.metadata)}
                  title="Open slide"
                >
                  <div className="font-mono text-neutral-300">
                    {labelFromMeta(h.metadata) || h.tag} · {h.score.toFixed(3)}
                  </div>
                  <div className="text-neutral-400 line-clamp-3">{h.text}</div>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function ProgressHUD({
  progressPct,
  line,
  phase,
  retryCount,
  maxRetries,
  factAiStatus,
  factClaimsStatus,
}: {
  progressPct: number;
  line: string;
  phase: Phase;
  retryCount: number;
  maxRetries: number;
  factAiStatus: "passed" | "failed" | null;
  factClaimsStatus: "passed" | "failed" | null;
}) {
  return (
    <div className="fixed inset-0 z-30 flex items-end md:items-center justify-center p-4">
      <div className="absolute inset-0 bg-neutral-950/40 backdrop-blur-sm" />
      <div className="relative w-full md:w-[560px] rounded-2xl border border-neutral-800 bg-neutral-900/90 shadow-xl p-4">
        <div className="h-2 rounded-full bg-neutral-800 overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-indigo-400 to-fuchsia-500 transition-all duration-500"
            style={{ width: `${progressPct}%` }}
          />
        </div>
        <ul className="mt-4 space-y-2 text-sm">
          <Step
            done={phase !== "sent" && phase !== "idle"}
            active={phase === "sent"}
            label="Request sent to server"
          />
          <Step
            done={
              phase === "llm" ||
              phase === "fact_ai" ||
              phase === "fact_claims" ||
              phase === "done"
            }
            active={phase === "retrieving"}
            label="Finding context"
          />
          <Step
            done={
              phase === "fact_ai" || phase === "fact_claims" || phase === "done"
            }
            active={phase === "llm"}
            label="AI is generating answer..."
            spinner
          />
          <Step
            done={
              factAiStatus === "passed" ||
              phase === "fact_claims" ||
              phase === "done"
            }
            active={phase === "fact_ai"}
            label="Fact checking (strict LLM)"
            spinner
            status={factAiStatus}
          />
          <Step
            done={factClaimsStatus === "passed" || phase === "done"}
            active={phase === "fact_claims"}
            label="Fact checking (evidence entailment)"
            spinner
            status={factClaimsStatus}
          />
        </ul>
        <div className="mt-3 text-xs text-neutral-400 text-center min-h-[1.25rem]">
          {line}
        </div>
        <div className="mt-1 text-[11px] text-neutral-500 text-center">
          Retry {Math.max(0, retryCount)} / {Math.max(1, maxRetries)}
        </div>
      </div>
    </div>
  );
}

function Step({
  done,
  active,
  label,
  spinner,
  status,
}: {
  done: boolean;
  active: boolean;
  label: string;
  spinner?: boolean;
  status?: "passed" | "failed" | null;
}) {
  let icon: ReactNode;
  if (status === "failed") {
    icon = <CrossIcon />;
  } else if (done || status === "passed") {
    icon = <CheckIcon />;
  } else if (spinner && active) {
    icon = <Spinner />;
  } else {
    icon = <CircleIcon />;
  }

  const colorClass =
    status === "failed"
      ? "text-red-400"
      : done || status === "passed"
      ? "text-green-300"
      : active
      ? "text-neutral-200"
      : "text-neutral-500";

  return (
    <li className="flex items-center gap-3">
      <div className="w-5 h-5 flex items-center justify-center">{icon}</div>
      <div className={colorClass}>{label}</div>
    </li>
  );
}

function CheckIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-5 h-5 text-green-400">
      <path
        d="M20 6L9 17l-5-5"
        stroke="currentColor"
        strokeWidth="2"
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function CrossIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-5 h-5 text-red-400">
      <path
        d="M18 6L6 18M6 6l12 12"
        stroke="currentColor"
        strokeWidth="2"
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
function CircleIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-5 h-5 text-neutral-600">
      <circle
        cx="12"
        cy="12"
        r="9"
        stroke="currentColor"
        strokeWidth="2"
        fill="none"
      />
    </svg>
  );
}
function Spinner() {
  return (
    <svg viewBox="0 0 24 24" className="w-5 h-5 animate-spin text-neutral-300">
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="9"
        stroke="currentColor"
        strokeWidth="3"
        fill="none"
      />
      <path
        className="opacity-90"
        d="M21 12a9 9 0 0 1-9 9"
        stroke="currentColor"
        strokeWidth="3"
        fill="none"
      />
    </svg>
  );
}
