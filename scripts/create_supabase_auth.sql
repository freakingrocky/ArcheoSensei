-- Create profile and chat tables for authenticated users
create table if not exists public.user_profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  active_chat_id uuid,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.user_chats (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.user_profiles(id) on delete cascade,
  name text not null,
  messages jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists user_chats_user_id_idx on public.user_chats(user_id);
create index if not exists user_chats_updated_at_idx on public.user_chats(updated_at desc);

-- Track updates automatically
create or replace function public.touch_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

create or replace trigger user_profiles_touch_updated_at
before update on public.user_profiles
for each row execute function public.touch_updated_at();

create or replace trigger user_chats_touch_updated_at
before update on public.user_chats
for each row execute function public.touch_updated_at();

alter table public.user_profiles enable row level security;
alter table public.user_chats enable row level security;

-- Allow users to manage only their own profile
create policy if not exists "Users can view own profile"
  on public.user_profiles for select
  using (auth.uid() = id);

create policy if not exists "Users can insert own profile"
  on public.user_profiles for insert
  with check (auth.uid() = id);

create policy if not exists "Users can update own profile"
  on public.user_profiles for update
  using (auth.uid() = id);

-- Allow users to manage only their own chats
create policy if not exists "Users can view own chats"
  on public.user_chats for select
  using (auth.uid() = user_id);

create policy if not exists "Users can insert own chats"
  on public.user_chats for insert
  with check (auth.uid() = user_id);

create policy if not exists "Users can update own chats"
  on public.user_chats for update
  using (auth.uid() = user_id);

create policy if not exists "Users can delete own chats"
  on public.user_chats for delete
  using (auth.uid() = user_id);
