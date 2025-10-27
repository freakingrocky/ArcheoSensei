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
