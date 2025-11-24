"use client";

import { useEffect, useMemo, useState } from "react";
import type { User } from "@supabase/supabase-js";
import { supabase } from "@/lib/supabase";
import { ensureUserProfile, type UserProfile } from "@/lib/profile";
import Image from "next/image";
import Turnstile from "react-turnstile";

type AuthGateProps = {
  children: (ctx: {
    user: User;
    profile: UserProfile;
    signOut: () => Promise<void>;
  }) => React.ReactNode;
};

type SignInViewState = "loading" | "ready" | "error";

export function AuthGate({ children }: AuthGateProps) {
  const [user, setUser] = useState<User | null>(null);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [state, setState] = useState<SignInViewState>("loading");
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [captchaToken, setCaptchaToken] = useState<string | null>(null);
  const [captchaError, setCaptchaError] = useState<string | null>(null);
  const [captchaInstance, setCaptchaInstance] = useState(0);
  const [verifyingCaptcha, setVerifyingCaptcha] = useState(false);

  const turnstileSiteKey = process.env.NEXT_PUBLIC_TURNSTILE_SITE_KEY;
  useEffect(() => {
    let mounted = true;
    supabase.auth
      .getSession()
      .then(({ data }) => {
        if (!mounted) return;
        setUser(data.session?.user ?? null);
        setState("ready");
      })
      .catch((err) => {
        console.error("auth session error", err);
        if (mounted) setState("error");
      });

    const { data: listener } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        setUser(session?.user ?? null);
        setProfile(null);
        setState("ready");
      }
    );

    return () => {
      mounted = false;
      listener.subscription.unsubscribe();
    };
  }, []);

  useEffect(() => {
    if (!user) return;
    ensureUserProfile(user)
      .then((p) => setProfile(p))
      .catch((err) => {
        console.error("profile error", err);
        setError("Could not load your profile. Please try again.");
      });
  }, [user]);

  const verifyTurnstile = async () => {
    if (!captchaToken) {
      setCaptchaError("Please complete the verification.");
      return false;
    }

    setVerifyingCaptcha(true);
    setCaptchaError(null);
    try {
      const res = await fetch("/api/turnstile/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: captchaToken }),
      });
      const data = await res.json();
      if (!data?.success) {
        setCaptchaError("Verification failed. Please try again.");
        setCaptchaToken(null);
        setCaptchaInstance((n) => n + 1);
        return false;
      }
      return true;
    } catch (err) {
      console.error("turnstile verify error", err);
      setCaptchaError("Could not verify the challenge. Please retry.");
      setCaptchaToken(null);
      setCaptchaInstance((n) => n + 1);
      return false;
    } finally {
      setVerifyingCaptcha(false);
    }
  };

  const handleSignInWithEmail = async () => {
    setMessage(null);
    setError(null);
    const trimmed = email.trim();
    if (!trimmed) return;
    const captchaOk = await verifyTurnstile();
    if (!captchaOk) return;
    const { error: authError } = await supabase.auth.signInWithOtp({
      email: trimmed,
      options: { emailRedirectTo: window.location.href },
    });
    if (authError) {
      setError(authError.message);
    } else {
      setMessage("Check your email for a login link.");
    }
  };

  const handleProviderLogin = async (provider: "github" | "google") => {
    setError(null);
    const captchaOk = await verifyTurnstile();
    if (!captchaOk) return;
    const { error: authError } = await supabase.auth.signInWithOAuth({
      provider,
      options: { redirectTo: window.location.href },
    });
    if (authError) {
      setError(authError.message);
    }
  };

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    setUser(null);
    setProfile(null);
  };

  const ready = useMemo(() => Boolean(user && profile), [user, profile]);

  if (ready) {
    return (
      <>
        {children({ user: user!, profile: profile!, signOut: handleSignOut })}
      </>
    );
  }

  return (
    <div className="min-h-[100dvh] bg-neutral-950 text-neutral-100 flex items-center justify-center px-4">
      <div className="w-full max-w-md rounded-2xl border border-neutral-800 bg-neutral-900/70 p-6 shadow-xl">
        <div className="mb-3 text-lg font-semibold">Sign in to continue</div>
        <p className="text-sm text-neutral-400 mb-4">
          Use email, GitHub, or Google to access your profile and saved chats.
        </p>
        <label className="text-xs text-neutral-400">Email</label>
        <div className="mt-1 flex gap-2">
          <input
            className="flex-1 rounded-lg border border-neutral-800 bg-neutral-950 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-neutral-700"
            placeholder="you@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            disabled={state === "loading" || verifyingCaptcha}
          />
          <button
            onClick={handleSignInWithEmail}
            className="rounded-lg bg-white px-3 py-2 text-sm font-semibold text-black hover:opacity-90 disabled:opacity-50"
            disabled={state === "loading" || verifyingCaptcha}
          >
            Send link
          </button>
        </div>

        <div className="mt-4">
          <label className="block text-xs text-neutral-400">Verification</label>
          <div className="mt-2 rounded-lg border border-neutral-800 bg-neutral-950 p-3">
            <Turnstile
              key={captchaInstance}
              sitekey={turnstileSiteKey}
              onSuccess={(token) => {
                setCaptchaToken(token);
                setCaptchaError(null);
              }}
              onError={() => {
                setCaptchaToken(null);
                setCaptchaError("Verification failed. Please retry.");
              }}
              onExpire={() => {
                setCaptchaToken(null);
                setCaptchaError("Verification expired. Please retry.");
                setCaptchaInstance((n) => n + 1);
              }}
              options={{ theme: "dark" }}
            />
            {captchaError && (
              <div className="mt-2 text-xs text-rose-300">{captchaError}</div>
            )}
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-2">
          <button
            onClick={() => handleProviderLogin("github")}
            className="flex items-center justify-center gap-2 rounded-lg border border-neutral-800 bg-neutral-950 px-3 py-2 text-sm hover:border-neutral-700"
            disabled={state === "loading" || verifyingCaptcha}
          >
            <Image
              src="/github.svg"
              alt="GitHub"
              width={18}
              height={18}
              className="h-4 w-4"
            />
            Continue with GitHub
          </button>
          <button
            onClick={() => handleProviderLogin("google")}
            className="flex items-center justify-center gap-2 rounded-lg border border-neutral-800 bg-neutral-950 px-3 py-2 text-sm hover:border-neutral-700"
            disabled={state === "loading" || verifyingCaptcha}
          >
            <Image
              src="/google.svg"
              alt="Google"
              width={18}
              height={18}
              className="h-4 w-4"
            />
            Continue with Google
          </button>
        </div>

        {message && (
          <div className="mt-3 rounded-lg bg-emerald-900/40 px-3 py-2 text-xs text-emerald-200">
            {message}
          </div>
        )}
        {error && (
          <div className="mt-3 rounded-lg bg-rose-900/40 px-3 py-2 text-xs text-rose-200">
            {error}
          </div>
        )}
        {state === "error" && (
          <div className="mt-3 text-xs text-rose-300">
            Unable to connect to Supabase. Please refresh and try again.
          </div>
        )}
      </div>
    </div>
  );
}
