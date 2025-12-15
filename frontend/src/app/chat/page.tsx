"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { User } from "@supabase/supabase-js";

import { AuthGate } from "@/components/AuthGate";
import {
  fetchInsights,
  fetchQueryJob,
  startQueryJob,
  type InsightsResponse,
  type QueryJobStatus,
} from "@/lib/api";
import { createChat } from "@/lib/chat";
import { updateActiveChatId, type UserProfile } from "@/lib/profile";

const shimmer =
  "before:absolute before:inset-0 before:-z-10 before:rounded-3xl before:bg-gradient-to-br before:from-white/10 before:via-white/0 before:to-white/5 before:opacity-60 before:blur-xl";

const glow =
  "after:absolute after:inset-[-1px] after:-z-20 after:rounded-3xl after:bg-[radial-gradient(circle_at_20%_20%,#6ee7ff33,transparent_40%),radial-gradient(circle_at_80%_0%,#c084fc33,transparent_36%),radial-gradient(circle_at_50%_80%,#f472b633,transparent_40%)] after:opacity-90";

type Message = {
  role: "user" | "assistant";
  content: string;
  meta?: QueryJobStatus | null;
};

type PanelProps = {
  title: string;
  subtitle?: string;
  children: ReactNode;
  actions?: ReactNode;
};

function GlassPanel({ title, subtitle, children, actions }: PanelProps) {
  return (
    <div
      className={`relative overflow-hidden rounded-3xl border border-white/10 bg-white/5 px-5 py-4 text-sm text-slate-100 shadow-[0_30px_60px_-40px_rgba(0,0,0,0.8)] backdrop-blur-xl ${shimmer} ${glow}`}
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-xs uppercase tracking-[0.35em] text-cyan-200/70">
            {title}
          </p>
          {subtitle && (
            <p className="text-sm font-semibold text-white/80">{subtitle}</p>
          )}
        </div>
        {actions}
      </div>
      <div className="mt-3 space-y-3">{children}</div>
    </div>
  );
}

function StatusPill({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/10 px-3 py-1 text-[11px] uppercase tracking-wide text-cyan-100/80 shadow-inner">
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-cyan-300 shadow-[0_0_12px_rgba(45,212,191,0.9)]" />
      {label}
    </span>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <div
      className={`group relative flex w-full items-start gap-3 rounded-2xl border border-white/10 px-4 py-3 backdrop-blur ${
        isUser
          ? "bg-cyan-500/10 text-cyan-50"
          : "bg-white/5 text-slate-100"
      }`}
    >
      <div
        className={`mt-1 h-8 w-8 flex-none rounded-xl border border-white/20 bg-gradient-to-br ${
          isUser
            ? "from-cyan-400/50 to-blue-500/30"
            : "from-violet-400/40 to-fuchsia-500/30"
        } shadow-[0_15px_40px_-25px_rgba(0,0,0,0.6)]`}
      />
      <div className="space-y-2 text-sm leading-relaxed">
        <ReactMarkdown
          className="prose prose-invert prose-p:my-2 prose-headings:text-white"
          remarkPlugins={[remarkGfm]}
        >
          {message.content}
        </ReactMarkdown>
        {message.meta?.fact_check?.status && (
          <div className="flex flex-wrap gap-2 text-[11px] text-white/70">
            <StatusPill label={`Fact check: ${message.meta.fact_check.status}`} />
            {message.meta.llm?.model && (
              <StatusPill label={`Model: ${message.meta.llm.model}` as string} />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function useJobPoller(onComplete: (status: QueryJobStatus) => void) {
  const timer = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);

  const poll = async (jobId: string) => {
    const status = await fetchQueryJob(jobId);
    if (status.status === "succeeded" || status.status === "failed") {
      onComplete(status);
      return;
    }
    timer.current = setTimeout(() => poll(jobId), 1100);
  };

  return poll;
}

function ChatExperience({
  user,
  profile,
  signOut,
}: {
  user: User;
  profile: UserProfile;
  signOut: () => Promise<void>;
}) {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Welcome to the new ArcheoSensei cockpit. Ask about your lectures or notes and I'll reason with verified context.",
    },
  ]);
  const [jobStatus, setJobStatus] = useState<QueryJobStatus | null>(null);
  const [sending, setSending] = useState(false);
  const [insights, setInsights] = useState<InsightsResponse | null>(null);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [insightsError, setInsightsError] = useState<string | null>(null);

  const chatSession = useMemo(() => createChat("Nebula Session"), []);
  const chatId = chatSession.id;
  const [chatName, setChatName] = useState(chatSession.name);

  useEffect(() => {
    updateActiveChatId(profile.id, chatId).catch((err) =>
      console.error("active chat update error", err)
    );
  }, [profile.id, chatId]);

  const pollJob = useJobPoller((status) => {
    setJobStatus(status);
    const answer =
      status.status === "failed"
        ? status.message || "The model hit a snag. Try again in a moment."
        : status.answer || "The model responded with an empty answer.";
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: answer, meta: status },
    ]);
    setSending(false);
  });

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || sending) return;
    setMessages((prev) => [...prev, { role: "user", content: trimmed }]);
    setInput("");
    setSending(true);
    try {
      const { job_id } = await startQueryJob(trimmed, {
        user_id: profile.id,
        chat_id: chatId,
        chat_name: chatName,
      });
      pollJob(job_id);
    } catch (err) {
      console.error("send error", err);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "I couldn't reach the reasoning engine. Try again shortly.",
        },
      ]);
      setSending(false);
    }
  };

  const fetchPersonalInsights = async () => {
    setInsightsError(null);
    setInsightsLoading(true);
    try {
      const data = await fetchInsights(profile.id);
      setInsights(data);
    } catch (err) {
      console.error("insights error", err);
      setInsightsError("I couldn't craft insights right now. Try again soon.");
    } finally {
      setInsightsLoading(false);
    }
  };

  useEffect(() => {
    fetchPersonalInsights();
  }, []);

  return (
    <div className="relative min-h-screen bg-[#030712] text-white">
      <div className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(circle_at_20%_20%,#14b8a680,transparent_25%),radial-gradient(circle_at_80%_0%,#a855f780,transparent_28%),radial-gradient(circle_at_50%_80%,#22d3ee40,transparent_35%)]" />
      <div className="absolute inset-0 -z-20 bg-gradient-to-b from-[#05070f] via-[#050710] to-[#01030a]" />

      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 pb-10 pt-8 lg:px-6">
        <header className="flex flex-col gap-4 rounded-3xl border border-white/10 bg-white/5 px-5 py-4 shadow-[0_40px_90px_-60px_rgba(0,0,0,0.8)] backdrop-blur-xl sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.45em] text-cyan-200/80">ArcheoSensei</p>
            <h1 className="text-2xl font-semibold text-white">Neo Aurora Interface</h1>
            <p className="text-sm text-white/70">
              Glassmorphic control center for your lectures, chats, and insights.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex flex-col text-right text-xs text-white/70">
              <span className="text-sm font-semibold text-white">
                {profile.display_name || user.email}
              </span>
              <span className="text-white/60">{user.email}</span>
            </div>
            <button
              onClick={() => signOut()}
              className="rounded-full border border-white/20 bg-white/10 px-4 py-2 text-xs font-semibold uppercase tracking-wide text-white/80 transition hover:border-white/40 hover:bg-white/20"
            >
              Sign Out
            </button>
          </div>
        </header>

        <div className="grid gap-5 lg:grid-cols-[320px,1fr,340px]">
          <div className="space-y-4">
            <GlassPanel title="Account" subtitle="Control surface">
              <div className="flex flex-col gap-3 text-sm text-white/80">
                <div className="flex items-center justify-between">
                  <span>Session name</span>
                  <input
                    value={chatName}
                    onChange={(e) => setChatName(e.target.value)}
                    className="w-40 rounded-xl border border-white/10 bg-white/10 px-3 py-2 text-xs text-white placeholder:text-white/40 focus:border-cyan-300/60 focus:outline-none"
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span>Active chat ID</span>
                  <code className="rounded-full bg-white/5 px-3 py-1 text-[11px] text-white/70">
                    {chatId.slice(0, 10)}
                  </code>
                </div>
                <p className="text-xs text-white/60">
                  Your conversations are linked to your account and synced to the insights engine.
                </p>
              </div>
            </GlassPanel>

            <GlassPanel title="Live Engine" subtitle="Model diagnostics">
              <div className="space-y-2 text-xs text-white/70">
                <div className="flex items-center justify-between">
                  <span>Phase</span>
                  <StatusPill label={jobStatus?.phase || "Idle"} />
                </div>
                <div className="flex items-center justify-between">
                  <span>State</span>
                  <span className="rounded-full bg-white/5 px-3 py-1 text-[11px] text-white/70">
                    {jobStatus?.status || (sending ? "Sending" : "Ready")}
                  </span>
                </div>
                {jobStatus?.llm?.model && (
                  <div className="flex items-center justify-between">
                    <span>Model</span>
                    <span className="text-[11px] text-cyan-100/90">
                      {jobStatus.llm.model}
                    </span>
                  </div>
                )}
              </div>
            </GlassPanel>
          </div>

          <div className="flex min-h-[520px] flex-col gap-4 rounded-3xl border border-white/10 bg-white/5 p-4 shadow-[0_40px_100px_-70px_rgba(0,0,0,0.9)] backdrop-blur-xl">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <div>
                <p className="text-xs uppercase tracking-[0.45em] text-cyan-200/80">
                  Conversational Cortex
                </p>
                <h2 className="text-lg font-semibold text-white">Ask anything</h2>
              </div>
              {sending ? <StatusPill label="Computing" /> : <StatusPill label="Standing by" />}
            </div>

            <div className="flex-1 space-y-3 overflow-y-auto pr-2">
              {messages.map((msg, idx) => (
                <MessageBubble key={`msg-${idx}-${msg.role}`} message={msg} />
              ))}
            </div>

            <div className="flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 p-3 shadow-inner">
              <input
                className="flex-1 rounded-2xl border border-white/10 bg-black/30 px-4 py-3 text-sm text-white placeholder:text-white/50 focus:border-cyan-300/60 focus:outline-none"
                placeholder="Ask for insights, lecture breakdowns, or quiz-worthy details"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
              />
              <button
                onClick={handleSend}
                disabled={sending}
                className="rounded-2xl border border-cyan-300/50 bg-gradient-to-br from-cyan-400/70 to-blue-500/60 px-4 py-3 text-xs font-semibold uppercase tracking-wide text-slate-950 shadow-[0_15px_50px_-30px_rgba(34,211,238,0.9)] transition hover:scale-[1.01] hover:border-cyan-200/80 disabled:opacity-60"
              >
                Launch
              </button>
            </div>
          </div>

          <div className="space-y-4">
            <GlassPanel
              title="My Insights"
              subtitle="AI pulse check"
              actions={
                <button
                  onClick={fetchPersonalInsights}
                  className="rounded-full border border-white/20 bg-white/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-wide text-white transition hover:border-white/40 hover:bg-white/20"
                  disabled={insightsLoading}
                >
                  Refresh
                </button>
              }
            >
              {insightsLoading && <p className="text-sm text-white/70">Synthesizing your study fingerprints…</p>}
              {insightsError && (
                <p className="text-sm text-rose-200/80">{insightsError}</p>
              )}
              {insights && !insightsLoading && (
                <div className="space-y-3 text-sm leading-relaxed text-white">
                  <ReactMarkdown
                    className="prose prose-invert prose-p:my-2 prose-ul:list-disc prose-li:marker:text-cyan-200"
                    remarkPlugins={[remarkGfm]}
                  >
                    {insights.summary}
                  </ReactMarkdown>
                  <div className="flex items-center justify-between text-[11px] text-white/60">
                    <span>Samples analyzed: {insights.sample_conversations}</span>
                    {insights.llm.model && <span>Model: {insights.llm.model}</span>}
                  </div>
                </div>
              )}
            </GlassPanel>

            <GlassPanel title="Quick Prompts" subtitle="Jumpstart ideas">
              <div className="flex flex-wrap gap-2">
                {["Summarize last lecture", "Highlight weak spots", "Suggest quiz questions"].map(
                  (prompt) => (
                    <button
                      key={prompt}
                      onClick={() => setInput(prompt)}
                      className="rounded-full border border-white/10 bg-white/10 px-3 py-2 text-xs text-white/80 transition hover:border-cyan-200/50 hover:bg-cyan-400/10"
                    >
                      {prompt}
                    </button>
                  )
                )}
              </div>
            </GlassPanel>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Page() {
  return (
    <AuthGate>
      {({ user, profile, signOut }) => (
        <ChatExperience user={user} profile={profile} signOut={signOut} />
      )}
    </AuthGate>
  );
}
