// frontend/src/lib/chats.ts
import { supabase } from "./supabase";
export type Hit = {
  id: string | number;
  tag: string;
  score: number;
  metadata: any;
  text: string;
  citation?: string | null;
  file_url?: string | null;
};
export type FactCheckAttempt = {
  attempt: number;
  needs_retry: boolean;
  directives?: string | null;
  ai_check: {
    passed: boolean;
    confidence: number;
    verdict: string;
    rationale: string;
  };
  claims_check: {
    score: number;
    entailed: number;
    total_claims: number;
    threshold: number;
    passed: boolean;
    claims: {
      claim: string;
      context: string;
      context_index: number;
      label: string;
      entailment_probability: number;
      neutral_probability: number;
      contradiction_probability: number;
      topic_index?: number;
      topic_share_context?: number;
      topic_share_answer?: number;
      max_cos_to_answer?: number;
    }[];
    details?: string;
  };
  answer_excerpt: string;
};

export type FactCheckResult = {
  status: "passed" | "failed";
  retry_count: number;
  threshold: number;
  max_attempts: number;
  attempts: FactCheckAttempt[];
  message?: string;
};

export type Msg = {
  role: "user" | "assistant";
  content: string;
  hits?: Hit[];
  diagnostics?: any;
  fact_check?: FactCheckResult;
};

export type Chat = {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  messages: Msg[];
};
export type ChatRow = {
  id: string;
  user_id: string;
  name: string;
  messages: Msg[];
  created_at: string;
  updated_at: string;
};

export function createChat(name: string): Chat {
  const now = Date.now();
  const id =
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `chat_${now}_${Math.random().toString(36).slice(2)}`;
  return {
    id,
    name,
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

export function renameChat(chats: Chat[], id: string, name: string): Chat[] {
  return chats.map((c) =>
    c.id === id ? { ...c, name, updatedAt: Date.now() } : c
  );
}

export function deleteChat(chats: Chat[], id: string): Chat[] {
  return chats.filter((c) => c.id !== id);
}

export function appendMessage(chats: Chat[], id: string, msg: Msg): Chat[] {
  return chats.map((c) =>
    c.id === id
      ? { ...c, messages: [...c.messages, msg], updatedAt: Date.now() }
      : c
  );
}

export function replaceMessages(
  chats: Chat[],
  id: string,
  messages: Msg[]
): Chat[] {
  return chats.map((c) =>
    c.id === id ? { ...c, messages, updatedAt: Date.now() } : c
  );
}

export async function fetchUserChats(userId: string): Promise<Chat[]> {
  const { data, error } = await supabase
    .from("user_chats")
    .select("*")
    .eq("user_id", userId)
    .order("updated_at", { ascending: false });

  if (error) {
    throw error;
  }

  return (data || []).map(normalizeChatRow);
}

export async function upsertUserChat(
  userId: string,
  chat: Chat
): Promise<void> {
  const payload = serializeChat(userId, chat);
  const { error } = await supabase.from("user_chats").upsert(payload);
  if (error) throw error;
}

export async function deleteUserChat(
  userId: string,
  chatId: string
): Promise<void> {
  const { error } = await supabase
    .from("user_chats")
    .delete()
    .eq("user_id", userId)
    .eq("id", chatId);
  if (error) throw error;
}

function normalizeChatRow(row: ChatRow): Chat {
  return {
    id: row.id,
    name: row.name,
    createdAt: row.created_at ? Date.parse(row.created_at) : Date.now(),
    updatedAt: row.updated_at ? Date.parse(row.updated_at) : Date.now(),
    messages: Array.isArray(row.messages) ? row.messages : [],
  };
}

function serializeChat(userId: string, chat: Chat) {
  return {
    id: chat.id,
    user_id: userId,
    name: chat.name,
    messages: chat.messages,
    created_at: new Date(chat.createdAt).toISOString(),
    updated_at: new Date(chat.updatedAt).toISOString(),
  } satisfies ChatRow;
}
