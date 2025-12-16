import { supabase } from "./supabase";
import type { User } from "@supabase/supabase-js";

export type UserProfile = {
  id: string;
  display_name: string | null;
  active_chat_id: string | null;
  context: string | null;
  created_at?: string;
  updated_at?: string;
};

export async function ensureUserProfile(user: User): Promise<UserProfile> {
  const profilePayload = {
    id: user.id,
    display_name:
      user.user_metadata?.full_name || user.user_metadata?.name || user.email,
  };

  const { data, error } = await supabase
    .from("user_profiles")
    .upsert(profilePayload, { onConflict: "id" })
    .select()
    .single();

  if (error) {
    throw error;
  }

  return normalizeProfile(data);
}

export async function fetchUserProfile(userId: string): Promise<UserProfile | null> {
  const { data, error } = await supabase
    .from("user_profiles")
    .select("*")
    .eq("id", userId)
    .maybeSingle();

  if (error) {
    throw error;
  }

  return data ? normalizeProfile(data) : null;
}

export async function updateActiveChatId(
  userId: string,
  chatId: string | null
): Promise<void> {
  const { error } = await supabase
    .from("user_profiles")
    .update({ active_chat_id: chatId })
    .eq("id", userId);

  if (error) throw error;
}

export async function updateUserContext(
  userId: string,
  context: string
): Promise<UserProfile> {
  const { data, error } = await supabase
    .from("user_profiles")
    .update({ context })
    .eq("id", userId)
    .select()
    .single();

  if (error) throw error;

  return normalizeProfile(data);
}

function normalizeProfile(row: any): UserProfile {
  return {
    id: row.id,
    display_name: row.display_name ?? null,
    active_chat_id: row.active_chat_id ?? null,
    context: row.context ?? null,
    created_at: row.created_at,
    updated_at: row.updated_at,
  };
}
