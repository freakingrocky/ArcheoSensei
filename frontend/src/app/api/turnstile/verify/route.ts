import { NextRequest, NextResponse } from "next/server";

const TURNSTILE_VERIFY_URL =
  "https://challenges.cloudflare.com/turnstile/v0/siteverify";

const secretKey =
  process.env.TURNSTILE_SECRET_KEY || "0x4AAAAAACCqHxKJDINsnaA0TEIgZb4jTNM";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const token = body?.token as string | undefined;

    if (!token) {
      return NextResponse.json(
        { success: false, error: "Missing verification token" },
        { status: 400 }
      );
    }

    const formData = new FormData();
    formData.append("secret", secretKey);
    formData.append("response", token);

    const res = await fetch(TURNSTILE_VERIFY_URL, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      return NextResponse.json(
        { success: false, error: "Verification failed" },
        { status: 500 }
      );
    }

    const data = await res.json();
    const success = Boolean(data?.success);

    return NextResponse.json({ success, result: data });
  } catch (err) {
    console.error("turnstile verify route error", err);
    return NextResponse.json(
      { success: false, error: "Unexpected verification error" },
      { status: 500 }
    );
  }
}
