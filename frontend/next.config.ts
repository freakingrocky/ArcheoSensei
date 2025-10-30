import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  typescript: {
    // Dangerously allow production builds to succeed even with type errors
    ignoreBuildErrors: true,
  },
};

export default nextConfig;
