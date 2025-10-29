"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { visit } from "unist-util-visit";
import {
  fetchLectures,
  LectureItem,
  startQueryJob,
  fetchQueryJob,
  requestQuizQuestion,
  gradeQuizAnswer,
  QuizQuestionPayload,
  QuizGradeResponse,
  QuizQuestionType,
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

type QuizStage = "config" | "question";

type QuizConfig = {
  lecture_key: string;
  topic: string;
};

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

type ClaimCheckEntry = {
  claim: string;
  context?: string;
  context_index?: number;
  label?: string;
};

function createClaimHighlighter(claims: ClaimCheckEntry[]) {
  const entries = claims
    .filter((c) => c?.claim)
    .map((c) => ({
      ...c,
      claim: c.claim.trim(),
      label: c.label?.toLowerCase(),
    }))
    .filter((c) => c.claim.length > 0);

  if (!entries.length) {
    return null;
  }

  const sorted = [...entries].sort((a, b) => b.claim.length - a.claim.length);

  return () => {
    return (tree: any) => {
      visit(tree, "text", (node: any, index: number, parent: any) => {
        if (!parent || typeof node.value !== "string") return;
        if (parent.type === "code" || parent.type === "inlineCode") return;
        let segments = [
          { text: node.value, claim: null as ClaimCheckEntry | null },
        ];

        for (const entry of sorted) {
          const nextSegments: typeof segments = [];
          for (const seg of segments) {
            if (seg.claim || !seg.text) {
              nextSegments.push(seg);
              continue;
            }
            const needle = entry.claim;
            let remaining = seg.text;
            while (remaining) {
              const idx = remaining.indexOf(needle);
              if (idx === -1) {
                nextSegments.push({ text: remaining, claim: null });
                break;
              }
              const before = remaining.slice(0, idx);
              if (before) {
                nextSegments.push({ text: before, claim: null });
              }
              nextSegments.push({ text: needle, claim: entry });
              remaining = remaining.slice(idx + needle.length);
            }
          }
          segments = nextSegments;
        }

        if (segments.length === 1 && segments[0].claim === null) {
          return;
        }

        const newNodes = segments.map((seg) => {
          if (!seg.claim) {
            return { type: "text", value: seg.text };
          }
          const verified = seg.claim.label === "entailment";
          return {
            type: "claimHighlight",
            data: {
              hName: "span",
              hProperties: {
                className: `claim-chip ${
                  verified ? "claim-chip--verified" : "claim-chip--unverified"
                }`,
                tabIndex: 0,
                "data-tooltip": seg.claim.context || "",
              },
            },
            children: [{ type: "text", value: seg.text }],
          };
        });

        parent.children.splice(index, 1, ...newNodes);
        return index + newNodes.length;
      });
    };
  };
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
  const [validationModal, setValidationModal] = useState<{
    hits: Hit[];
    details?: string;
  } | null>(null);
  const jobChatRef = useRef<string | null>(null);
  const [quizOpen, setQuizOpen] = useState(false);
  const [quizStage, setQuizStage] = useState<QuizStage>("config");
  const [quizConfig, setQuizConfig] = useState<QuizConfig>({
    lecture_key: "",
    topic: "",
  });
  const [quizActiveConfig, setQuizActiveConfig] = useState<QuizConfig | null>(
    null
  );
  const [quizQuestion, setQuizQuestion] = useState<QuizQuestionPayload | null>(
    null
  );
  const [quizContext, setQuizContext] = useState("");
  const [quizHintVisible, setQuizHintVisible] = useState(false);
  const [quizAnswer, setQuizAnswer] = useState<string | string[]>("");
  const [quizEvaluation, setQuizEvaluation] =
    useState<QuizGradeResponse | null>(null);
  const [quizQuestionLoading, setQuizQuestionLoading] = useState(false);
  const [quizGrading, setQuizGrading] = useState(false);
  const [quizError, setQuizError] = useState("");
  const quizTypeIndexRef = useRef(0);
  const quizTypes: QuizQuestionType[] = [
    "true_false",
    "mcq_single",
    "short_answer",
    "mcq_multi",
  ];
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
    setValidationModal(null);
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

  const nextQuizType = () => {
    const idx = quizTypeIndexRef.current;
    const type = quizTypes[idx];
    quizTypeIndexRef.current = (idx + 1) % quizTypes.length;
    return type;
  };

  function handleOpenQuiz() {
    setQuizOpen(true);
    setQuizStage("config");
    setQuizConfig({
      lecture_key: selectedLecture || "",
      topic: "",
    });
    setQuizActiveConfig(null);
    setQuizQuestion(null);
    setQuizContext("");
    setQuizHintVisible(false);
    setQuizAnswer("");
    setQuizEvaluation(null);
    setQuizError("");
  }

  function handleCloseQuiz() {
    setQuizOpen(false);
    setQuizStage("config");
    setQuizConfig({ lecture_key: "", topic: "" });
    setQuizActiveConfig(null);
    setQuizQuestion(null);
    setQuizContext("");
    setQuizHintVisible(false);
    setQuizAnswer("");
    setQuizEvaluation(null);
    setQuizQuestionLoading(false);
    setQuizGrading(false);
    setQuizError("");
  }

  const generateQuiz = async (config: QuizConfig) => {
    if (quizQuestionLoading) return;
    setQuizQuestionLoading(true);
    setQuizError("");
    setQuizQuestion(null);
    setQuizEvaluation(null);
    setQuizHintVisible(false);
    setQuizAnswer("");
    try {
      const normalized: QuizConfig = {
        lecture_key: (config.lecture_key || "").trim(),
        topic: (config.topic || "").trim(),
      };
      const type = nextQuizType();
      const res = await requestQuizQuestion({
        lecture_key: normalized.lecture_key || undefined,
        topic: normalized.topic || undefined,
        question_type: type,
      });
      setQuizQuestion(res.question);
      setQuizContext(res.context || "");
      setQuizHintVisible(false);
      setQuizAnswer(res.question.question_type === "mcq_multi" ? [] : "");
      setQuizStage("question");
      setQuizActiveConfig(normalized);
    } catch (err: any) {
      setQuizError(err?.message ?? "Unable to generate quiz question");
    } finally {
      setQuizQuestionLoading(false);
    }
  };

  const handleQuizSubmit = async () => {
    if (!quizQuestion || quizGrading) return;
    const answerPayload = Array.isArray(quizAnswer)
      ? quizAnswer
      : quizAnswer.toString().trim();
    const hasAnswer = Array.isArray(answerPayload)
      ? answerPayload.length > 0
      : answerPayload.length > 0;
    if (!hasAnswer) {
      setQuizError("Please provide an answer before submitting.");
      return;
    }
    setQuizGrading(true);
    setQuizError("");
    try {
      const result = await gradeQuizAnswer({
        question: quizQuestion,
        context: quizContext,
        user_answer: answerPayload,
      });
      setQuizEvaluation(result);
    } catch (err: any) {
      setQuizError(err?.message ?? "Failed to grade answer");
    } finally {
      setQuizGrading(false);
    }
  };

  const handleQuizNext = () => {
    const base = quizActiveConfig || quizConfig;
    if (!base) return;
    generateQuiz(base);
  };

  const handleQuizHint = () => {
    setQuizHintVisible(true);
  };

  const quizAnswerHasValue = Array.isArray(quizAnswer)
    ? quizAnswer.length > 0
    : quizAnswer.toString().trim().length > 0;

  const quizTypeLabel = (type?: QuizQuestionType) => {
    switch (type) {
      case "true_false":
        return "True or False";
      case "mcq_single":
        return "Multiple Choice (single answer)";
      case "mcq_multi":
        return "Multiple Choice (choose all that apply)";
      case "short_answer":
        return "Short Answer";
      default:
        return "Question";
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
          status.fact_ai_status === "passed" ||
          status.fact_ai_status === "failed"
            ? status.fact_ai_status
            : null;
        const claimsStatus =
          status.fact_claims_status === "passed" ||
          status.fact_claims_status === "failed"
            ? status.fact_claims_status
            : null;
        setFactAiStatus(aiStatus);
        setFactClaimsStatus(claimsStatus);

        if (
          !handled &&
          status.status !== "running" &&
          status.status !== "queued"
        ) {
          handled = true;
          const fact = status.fact_check as FactCheckResult | undefined;
          const limitedHits: Hit[] = (status.hits || []).slice(0, 3);

          if (status.status === "failed" && !status.message) {
            setStatusLine("AI response could not be validated after retries.");
          }

          const jobFailed =
            status.status === "failed" || (fact?.status || "") === "failed";
          if (jobFailed) {
            setValidationModal({
              hits: limitedHits,
              details:
                status.message?.trim() ||
                fact?.message?.trim() ||
                "AI response could not be validated after retries.",
            });
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
        <button
          onClick={handleOpenQuiz}
          className="w-full mb-4 rounded-lg border border-neutral-800 text-neutral-200 font-semibold py-2 hover:bg-neutral-900"
        >
          🎯 Quiz Me
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

      {quizOpen && (
        <div className="fixed overflow-y-auto inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm px-4">
          <div className="w-full max-w-2xl rounded-2xl border border-neutral-800 bg-neutral-900 p-6 shadow-2xl">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-2xl font-semibold text-neutral-100">
                  Quiz Me
                </h2>
                <p className="text-sm text-neutral-400">
                  Choose a lecture or enter a topic to generate a quick practice
                  question.
                </p>
              </div>
              <button
                onClick={handleCloseQuiz}
                className="rounded-full border border-neutral-700 px-3 py-1 text-sm text-neutral-300 hover:bg-neutral-800"
              >
                Close
              </button>
            </div>

            {quizStage === "config" ? (
              <div className="mt-6 space-y-5">
                <div>
                  <label className="block text-sm font-medium text-neutral-300">
                    Choose a lecture
                  </label>
                  <select
                    className="mt-1 w-full rounded-lg border border-neutral-800 bg-neutral-950 px-3 py-2 text-sm text-neutral-200 focus:outline-none focus:ring-1 focus:ring-neutral-600"
                    value={quizConfig.lecture_key}
                    onChange={(e) =>
                      setQuizConfig((prev) => ({
                        ...prev,
                        lecture_key: e.target.value,
                      }))
                    }
                    onFocus={() => setQuizError("")}
                  >
                    <option value="">All lectures</option>
                    {lectures.map((l) => (
                      <option key={l.lecture_key} value={l.lecture_key}>
                        {l.lecture_key} ({l.count})
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-neutral-300">
                    Or type any topic
                  </label>
                  <input
                    value={quizConfig.topic}
                    onChange={(e) =>
                      setQuizConfig((prev) => ({
                        ...prev,
                        topic: e.target.value,
                      }))
                    }
                    placeholder="e.g. Stratigraphy basics"
                    className="mt-1 w-full rounded-lg border border-neutral-800 bg-neutral-950 px-3 py-2 text-sm text-neutral-200 focus:outline-none focus:ring-1 focus:ring-neutral-600"
                    onFocus={() => setQuizError("")}
                  />
                </div>

                {quizError && quizStage === "config" && (
                  <div className="rounded-md border border-red-500/60 bg-red-500/10 px-3 py-2 text-sm text-red-300">
                    {quizError}
                  </div>
                )}

                <div className="flex justify-end gap-3 pt-2">
                  <button
                    onClick={handleCloseQuiz}
                    className="rounded-lg border border-neutral-700 px-4 py-2 text-sm text-neutral-300 hover:bg-neutral-800"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={() => generateQuiz(quizConfig)}
                    disabled={
                      (!quizConfig.lecture_key && !quizConfig.topic.trim()) ||
                      quizQuestionLoading
                    }
                    className="rounded-lg bg-indigo-500 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {quizQuestionLoading ? "Generating…" : "Quiz Me"}
                  </button>
                </div>
              </div>
            ) : (
              <div className="mt-6 space-y-5">
                <div className="flex flex-wrap items-center gap-2 text-sm text-neutral-400">
                  {quizActiveConfig?.lecture_key && (
                    <span className="rounded-full bg-neutral-800 px-3 py-1 text-xs uppercase tracking-wide text-indigo-300">
                      {quizActiveConfig.lecture_key}
                    </span>
                  )}
                  {quizActiveConfig?.topic && (
                    <span className="rounded-full bg-neutral-800 px-3 py-1 text-xs text-neutral-200">
                      Topic: {quizActiveConfig.topic}
                    </span>
                  )}
                  <span className="rounded-full bg-neutral-800 px-3 py-1 text-xs text-neutral-200">
                    {quizTypeLabel(quizQuestion?.question_type)}
                  </span>
                </div>

                {quizQuestionLoading && !quizQuestion ? (
                  <div className="flex items-center justify-center rounded-xl border border-dashed border-neutral-700 py-16 text-neutral-400">
                    Generating your question…
                  </div>
                ) : quizQuestion ? (
                  <div className="space-y-5">
                    <div>
                      <div className="text-lg font-semibold text-neutral-100">
                        {quizQuestion.question_prompt}
                      </div>
                    </div>

                    {quizQuestion.question_type === "true_false" && (
                      <div className="space-y-3">
                        {["True", "False"].map((choice) => (
                          <label
                            key={choice}
                            className="flex cursor-pointer items-center gap-2 rounded-lg border border-neutral-800 bg-neutral-950 px-3 py-2 text-sm text-neutral-200 hover:border-neutral-600"
                          >
                            <input
                              type="radio"
                              className="h-4 w-4"
                              checked={
                                !Array.isArray(quizAnswer) &&
                                quizAnswer === choice
                              }
                              onChange={() => {
                                setQuizError("");
                                setQuizAnswer(choice);
                              }}
                            />
                            {choice}
                          </label>
                        ))}
                      </div>
                    )}

                    {quizQuestion.question_type === "mcq_single" && (
                      <div className="space-y-3">
                        {(quizQuestion.options || []).map((opt) => (
                          <label
                            key={opt}
                            className="flex cursor-pointer items-center gap-2 rounded-lg border border-neutral-800 bg-neutral-950 px-3 py-2 text-sm text-neutral-200 hover:border-neutral-600"
                          >
                            <input
                              type="radio"
                              className="h-4 w-4"
                              checked={
                                !Array.isArray(quizAnswer) && quizAnswer === opt
                              }
                              onChange={() => {
                                setQuizError("");
                                setQuizAnswer(opt);
                              }}
                            />
                            {opt}
                          </label>
                        ))}
                      </div>
                    )}

                    {quizQuestion.question_type === "mcq_multi" && (
                      <div className="space-y-3">
                        {(quizQuestion.options || []).map((opt) => {
                          const checked =
                            Array.isArray(quizAnswer) &&
                            quizAnswer.includes(opt);
                          return (
                            <label
                              key={opt}
                              className="flex cursor-pointer items-center gap-2 rounded-lg border border-neutral-800 bg-neutral-950 px-3 py-2 text-sm text-neutral-200 hover:border-neutral-600"
                            >
                              <input
                                type="checkbox"
                                className="h-4 w-4"
                                checked={checked}
                                onChange={() =>
                                  setQuizAnswer((prev) => {
                                    setQuizError("");
                                    const arr = Array.isArray(prev)
                                      ? [...prev]
                                      : [];
                                    if (arr.includes(opt)) {
                                      return arr.filter((v) => v !== opt);
                                    }
                                    arr.push(opt);
                                    return arr;
                                  })
                                }
                              />
                              {opt}
                            </label>
                          );
                        })}
                      </div>
                    )}

                    {quizQuestion.question_type === "short_answer" && (
                      <textarea
                        value={
                          Array.isArray(quizAnswer)
                            ? quizAnswer.join(", ")
                            : quizAnswer
                        }
                        onChange={(e) => {
                          setQuizError("");
                          setQuizAnswer(e.target.value);
                        }}
                        rows={4}
                        className="w-full rounded-lg border border-neutral-800 bg-neutral-950 px-3 py-2 text-sm text-neutral-200 focus:outline-none focus:ring-1 focus:ring-neutral-600"
                        placeholder="Write your answer here"
                      />
                    )}

                    <div className="flex flex-wrap items-center gap-3">
                      <button
                        onClick={handleQuizHint}
                        disabled={quizHintVisible || quizQuestionLoading}
                        className="rounded-lg border border-neutral-700 px-4 py-2 text-sm text-neutral-200 hover:bg-neutral-800 disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        Give me a hint
                      </button>
                      <button
                        onClick={handleQuizSubmit}
                        disabled={!quizAnswerHasValue || quizGrading}
                        className="rounded-lg bg-indigo-500 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        {quizGrading ? "Grading…" : "Submit answer"}
                      </button>
                      <button
                        onClick={handleQuizNext}
                        disabled={!quizEvaluation || quizQuestionLoading}
                        className="rounded-lg border border-neutral-700 px-4 py-2 text-sm text-neutral-200 hover:bg-neutral-800 disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        Go to next
                      </button>
                    </div>

                    {quizHintVisible && (
                      <div className="rounded-lg border border-amber-500/60 bg-amber-500/10 px-3 py-2 text-sm text-amber-200">
                        {quizQuestion.hint}
                      </div>
                    )}

                    {quizError && quizStage === "question" && (
                      <div className="rounded-md border border-red-500/60 bg-red-500/10 px-3 py-2 text-sm text-red-300">
                        {quizError}
                      </div>
                    )}

                    {quizEvaluation && quizQuestion && (
                      <div className="space-y-3 rounded-xl border border-neutral-700 bg-neutral-950 px-4 py-4">
                        <div className="flex items-center gap-3">
                          <div
                            className={`h-3 w-3 rounded-full ${
                              quizEvaluation.correct
                                ? "bg-emerald-400"
                                : "bg-red-400"
                            }`}
                          />
                          <div className="font-semibold text-neutral-100">
                            {quizEvaluation.assessment ||
                              (quizEvaluation.correct
                                ? "Great job!"
                                : "Keep practicing.")}
                          </div>
                          <div className="ml-auto text-sm text-neutral-400">
                            Score: {(quizEvaluation.score * 100).toFixed(0)}%
                          </div>
                        </div>
                        {quizEvaluation.good_points.length > 0 &&
                          quizQuestion.question_type === "short_answer" && (
                            <div>
                              <div className="text-xs font-semibold uppercase tracking-wide text-emerald-300">
                                Good points
                              </div>
                              <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-neutral-200">
                                {quizEvaluation.good_points.map(
                                  (point, idx) => (
                                    <li key={`good-${idx}`}>{point}</li>
                                  )
                                )}
                              </ul>
                            </div>
                          )}
                        {quizEvaluation.bad_points.length > 0 &&
                          quizQuestion.question_type === "short_answer" && (
                            <div>
                              <div className="text-xs font-semibold uppercase tracking-wide text-red-300">
                                Needs work
                              </div>
                              <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-neutral-200">
                                {quizEvaluation.bad_points.map((point, idx) => (
                                  <li key={`bad-${idx}`}>{point}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        <div className="rounded-lg bg-neutral-900 px-3 py-2 text-sm text-neutral-300">
                          <div className="text-xs uppercase tracking-wide text-neutral-500">
                            Explanation
                          </div>
                          <div>{quizQuestion.answer_explanation}</div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex items-center justify-center rounded-xl border border-dashed border-neutral-700 py-12 text-neutral-400">
                    Ready for your next question?
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* HUD */}
      {phase !== "idle" && phase !== "done" && (
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

      {validationModal && (
        <ValidationWarningModal
          hits={validationModal.hits}
          details={validationModal.details}
          onClose={() => setValidationModal(null)}
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

function ValidationWarningModal({
  hits,
  details,
  onClose,
}: {
  hits: Hit[];
  details?: string;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center p-4">
      <button
        type="button"
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
        aria-label="Close validation warning"
      />
      <div className="relative max-w-lg w-full rounded-2xl border border-red-500/40 bg-neutral-950/95 p-6 shadow-2xl">
        <div className="text-sm font-semibold uppercase tracking-wide text-red-300">
          PLEASE NOTE THAT THIS RESPONSE COULD NOT BE VERIFIED, PLEASE ENSURE TO
          VALIDATE YOURSELF.
        </div>
        {details && (
          <p className="mt-3 text-sm text-neutral-300 whitespace-pre-wrap">
            {details}
          </p>
        )}
        <div className="mt-4 text-xs font-semibold text-neutral-400">
          SOURCES USED:
        </div>
        <ul className="mt-2 space-y-2 text-sm text-neutral-200">
          {hits.length ? (
            hits.map((h, idx) => {
              const label = labelFromMeta(h.metadata) || h.tag;
              const filename = h.metadata?.filename;
              return (
                <li
                  key={idx}
                  className="border border-neutral-800 rounded-lg p-2"
                >
                  <div className="font-medium text-neutral-100">{label}</div>
                  {filename && (
                    <div className="text-xs text-neutral-400">
                      File: {filename}
                    </div>
                  )}
                  <div className="text-xs text-neutral-400 line-clamp-3 mt-1">
                    {h.text}
                  </div>
                </li>
              );
            })
          ) : (
            <li className="text-neutral-400 text-sm">No supporting sources.</li>
          )}
        </ul>
        <div className="mt-5 flex justify-end">
          <button
            onClick={onClose}
            className="text-sm font-semibold text-neutral-200 hover:text-white"
          >
            &lt;Click to Close&gt;
          </button>
        </div>
      </div>
    </div>
  );
}

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

  const AssistantMarkdown = ({
    content,
    claimCheck,
  }: {
    content: string;
    claimCheck?: { claims?: ClaimCheckEntry[] } | null;
  }) => {
    const rewritten = toCiteLinks(content);
    const highlightPlugin = useMemo(
      () => createClaimHighlighter(claimCheck?.claims ?? []),
      [claimCheck?.claims]
    );
    const remarkPlugins = useMemo(() => {
      const base: any[] = [remarkGfm];
      if (highlightPlugin) {
        base.push(highlightPlugin);
      }
      return base;
    }, [highlightPlugin]);
    return (
      <div className="prose prose-invert max-w-none prose-p:my-2 prose-li:my-1">
        <ReactMarkdown
          remarkPlugins={remarkPlugins}
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
    (factAttempts.length
      ? factAttempts.length
      : factCheck
      ? factCheck.retry_count + 1
      : 1);

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
          <AssistantMarkdown content={msg.content} claimCheck={claimCheck} />
        )}

        {!isUser && factCheck && (
          <div className="mt-3 border-t border-neutral-800 pt-2">
            <div className="text-xs font-semibold text-neutral-300 mb-1">
              Validation
            </div>
            {factCheck.status === "passed" ? (
              <div className="text-[11px] text-green-300">
                Passed fact checks
                {confidencePct !== null &&
                  ` · LLM confidence ${confidencePct}%`}
                {entailmentPct !== null && (
                  <>
                    {" "}
                    · Evidence entailment {entailmentPct}%
                    {entailedClaims !== null &&
                      totalClaims !== null &&
                      ` (${entailedClaims}/${totalClaims} claims)`}
                    {entailmentTarget !== null &&
                      ` · target ${entailmentTarget}%`}
                  </>
                )}
              </div>
            ) : (
              <div className="text-[11px] text-yellow-300">
                Failed validation after {maxAttemptsDisplay} attempts.
                {factCheck.message
                  ? ` ${factCheck.message}`
                  : " Answer may be unreliable."}
              </div>
            )}
            {factCheck.status !== "passed" && entailmentPct !== null && (
              <div className="text-[11px] text-neutral-400 mt-1">
                Evidence entailment {entailmentPct}%
                {entailedClaims !== null &&
                  totalClaims !== null &&
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
            {factCheck.status !== "passed" &&
              failingSnippet &&
              failingLabel && (
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
