# RAG Stack Monorepo

- Frontend: Next.js (TS) in `/frontend`
- Backend: FastAPI in `/backend`
- Worker: Python ingestion in `/worker`

## Dev (Docker)

```bash
cp .env.example .env
./scripts/dev-up.sh
```

## Environment variables

Authentication and Supabase storage rely on the following keys:

- `SUPABASE_URL` / `NEXT_PUBLIC_SUPABASE_URL`
- `SUPABASE_ANON_KEY` / `NEXT_PUBLIC_SUPABASE_ANON_KEY` (publishable key)
- `SUPABASE_SERVICE_ROLE_KEY` (secret key)
- `SUPABASE_JWT_SECRET` (legacy JWT key)
