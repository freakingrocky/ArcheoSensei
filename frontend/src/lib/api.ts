export async function askRAG(question: string, opts?: any) {
  const base =
    typeof window === "undefined"
      ? process.env.BACKEND_URL_INTERNAL
      : process.env.NEXT_PUBLIC_BACKEND_URL_BROWSER;
  const res = await fetch(`${base}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: question, options: opts || {} }),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Backend error ${res.status}`);
  return res.json();
}

export type LectureItem = { lecture_key: string; count: number };

export async function fetchLectures(): Promise<LectureItem[]> {
  const base =
    typeof window === "undefined"
      ? process.env.BACKEND_URL_INTERNAL
      : process.env.NEXT_PUBLIC_BACKEND_URL_BROWSER;
  const res = await fetch(`${base}/lectures`, { cache: "no-store" });
  if (!res.ok) return [];
  const data = await res.json();
  return (data?.lectures ?? []) as LectureItem[];
}
