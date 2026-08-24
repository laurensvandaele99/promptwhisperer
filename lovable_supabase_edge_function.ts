// Example Supabase Edge Function used by a Lovable app.
// Store MODEL_API_URL and MODEL_API_KEY as Supabase secrets, never in browser code.

import { serve } from "https://deno.land/std@0.224.0/http/server.ts";

serve(async (req) => {
  if (req.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  try {
    const { participant_id, prompt } = await req.json();
    if (!prompt || typeof prompt !== "string") {
      return new Response(JSON.stringify({ error: "prompt is required" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    const modelUrl = Deno.env.get("MODEL_API_URL");
    const modelKey = Deno.env.get("MODEL_API_KEY");
    if (!modelUrl || !modelKey) {
      throw new Error("MODEL_API_URL / MODEL_API_KEY not configured");
    }

    const started = Date.now();
    const response = await fetch(modelUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": modelKey,
      },
      body: JSON.stringify({ participant_id, prompt }),
    });

    const body = await response.json();
    if (!response.ok) {
      return new Response(JSON.stringify({ error: "model_api_failed", details: body }), {
        status: 502,
        headers: { "Content-Type": "application/json" },
      });
    }

    return new Response(
      JSON.stringify({
        ...body,
        edge_latency_ms: Date.now() - started,
      }),
      { headers: { "Content-Type": "application/json" } },
    );
  } catch (error) {
    return new Response(JSON.stringify({ error: String(error) }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
});
