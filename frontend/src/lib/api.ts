const backendBase = () =>
  (typeof window === "undefined"
    ? process.env.BACKEND_URL_INTERNAL
    : process.env.NEXT_PUBLIC_BACKEND_URL_BROWSER) as string;

export async function askRAG(question: string, opts?: any) {
  const res = await fetch(`${backendBase()}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: question, options: opts || {} }),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Backend error ${res.status}`);
  return res.json();
}

export async function startQueryJob(question: string, opts?: any) {
  const res = await fetch(`${backendBase()}/query/async`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: question, options: opts || {} }),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Backend error ${res.status}`);
  return res.json() as Promise<{ job_id: string }>;
}

export type QueryJobStatus = {
  job_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  phase: string;
  message?: string;
  retry_count?: number;
  max_attempts?: number;
  attempts?: any[];
  diagnostics?: any;
  hits?: any[];
  top_k?: number;
  context_len?: number;
  answer?: string;
  fact_check?: any;
  llm?: any;
  fact_ai_status?: "passed" | "failed";
  fact_claims_status?: "passed" | "failed";
};

export async function fetchQueryJob(jobId: string): Promise<QueryJobStatus> {
  const res = await fetch(`${backendBase()}/query/async/${jobId}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Job ${jobId} missing (${res.status})`);
  return res.json();
}

export type LectureItem = { lecture_key: string; count: number };

export async function fetchLectures(): Promise<LectureItem[]> {
  const res = await fetch(`${backendBase()}/lectures`, { cache: "no-store" });
  if (!res.ok) return [];
  const data = await res.json();
  return (data?.lectures ?? []) as LectureItem[];
}

export type QuizQuestionType =
  | "true_false"
  | "mcq_single"
  | "mcq_multi"
  | "short_answer";

export type QuizQuestionPayload = {
  question_type: QuizQuestionType;
  question_prompt: string;
  options?: string[] | null;
  correct_answer: string | string[];
  answer_rubric: string;
  hint: string;
  answer_explanation: string;
};

export type QuizQuestionResponse = {
  question: QuizQuestionPayload;
  context: string;
  lecture_key?: string | null;
  topic?: string | null;
};

export type QuizGradeResponse = {
  correct: boolean;
  score: number;
  assessment: string;
  good_points: string[];
  bad_points: string[];
};

export async function requestQuizQuestion(payload: {
  lecture_key?: string;
  topic?: string;
  question_type?: QuizQuestionType;
}): Promise<QuizQuestionResponse> {
  const res = await fetch(`${backendBase()}/quiz/question`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Quiz question error ${res.status}: ${text}`);
  }
  return res.json();
}

export async function gradeQuizAnswer(payload: {
  question: QuizQuestionPayload;
  context: string;
  user_answer: string | string[];
}): Promise<QuizGradeResponse> {
  const res = await fetch(`${backendBase()}/quiz/grade`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Quiz grading error ${res.status}: ${text}`);
  }
  return res.json();
}
