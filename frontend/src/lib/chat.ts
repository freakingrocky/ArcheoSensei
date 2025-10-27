// frontend/src/lib/chats.ts
export type Hit = { tag: string; score: number; metadata: any; text: string };
export type FactCheckAttempt = {
  attempt: number;
  temperature: number;
  needs_retry: boolean;
  ai_check: {
    passed: boolean;
    confidence: number;
    verdict: string;
    rationale: string;
  };
  ner_check: {
    score: number;
    matched_pairs: string[][];
    missing_pairs: string[][];
    answer_entities: string[];
    context_entities: string[];
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

const KEY = "archeo:chats:v1";
const KEY_ACTIVE = "archeo:activeChatId";

export function loadChats(): Chat[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw) as Chat[];
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}

export function saveChats(chats: Chat[]) {
  try {
    localStorage.setItem(KEY, JSON.stringify(chats));
  } catch {}
}

export function loadActiveChatId(): string | null {
  try {
    return localStorage.getItem(KEY_ACTIVE);
  } catch {
    return null;
  }
}

export function saveActiveChatId(id: string) {
  try {
    localStorage.setItem(KEY_ACTIVE, id);
  } catch {}
}

export function createChat(name: string): Chat {
  const now = Date.now();
  return {
    id: `chat_${now}_${Math.random().toString(36).slice(2)}`,
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
