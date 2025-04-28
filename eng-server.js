import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/community/vectorstores/pinecone";
import { initializeAgentExecutorWithOptions } from "langchain/agents";
import pkg from "@supabase/supabase-js";
const { createClient } = pkg;

export default async ({ req, res, log, error }) => {
  try {
    // === Environment Variables ===
    const {
      GOOGLE_API_KEY,
      PINECONE_API_KEY,
      SUPABASE_URL,
      SUPABASE_KEY,
      AUTH_SECRET,
    } = process.env;

    if (
      !GOOGLE_API_KEY ||
      !PINECONE_API_KEY ||
      !SUPABASE_URL ||
      !SUPABASE_KEY ||
      !AUTH_SECRET
    ) {
      throw new Error("Missing required environment variables");
    }

    const body = req.body || {};
    const { message, sessionId, authToken } = body;

    if (!message || !sessionId) {
      return res.json({ error: "Missing message or sessionId" }, 400);
    }

    if (authToken !== AUTH_SECRET) {
      return res.json(
        { error: "Back off motherfucker, you ain't authenticated" },
        401,
      );
    }

    // === Supabase Client ===
    const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

    // === Pinecone Setup ===
    const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });
    const pineconeIndex = pinecone.Index("english");

    const vectorStore = await PineconeStore.fromExistingIndex(
      new GoogleGenerativeAIEmbeddings({
        model: "text-embedding-004",
        apiKey: GOOGLE_API_KEY,
      }),
      { pineconeIndex },
    );

    // === Tools ===
    const tools = [
      // {
      //   name: "insights",
      //   description: "Fetch insights from previous chats",
      //   async func() {
      //     const { data, error } = await supabase.from("insights").select("*");
      //     if (error) throw new Error(error.message);
      //     return JSON.stringify(data);
      //   },
      // },
      {
        name: "English database",
        description: "Retrieve information to answer user queries.",
        async func(query) {
          const results = await vectorStore.similaritySearch(query, 5);
          return results.map((r) => r.pageContent).join("\n\n---\n");
        },
      },
    ];

    // === Chat Model ===
    const model = new ChatGoogleGenerativeAI({
      model: "gemini-2.0-flash",
      apiKey: GOOGLE_API_KEY,
      systemInstruction: {
        role: "system",
      },
    });

    const executor = await initializeAgentExecutorWithOptions(tools, model, {
      agentType: "zero-shot-react-description",
      verbose: true,
      returnIntermediateSteps: true,
    });

    // === Format Chat History ===
    const formatHistory = (history) =>
      history
        .map(
          (msg) =>
            `${msg.role === "user" ? "User" : "Assistant"}: ${msg.content}`,
        )
        .join("\n");

    // === Fetch & Format Chat History ===
    const { data: history, error: fetchError } = await supabase
      .from("eng-messages")
      .select("role, content")
      .eq("session_id", sessionId)
      .order("created_at", { ascending: false })
      .limit(5);

    if (fetchError) throw new Error(fetchError.message);
    const formattedHistory = formatHistory(history.reverse());

    const systemMsg =
      "System: You are an AI agent created by Arjav who answers questions related to English. Always answer in detail unless specified not to. Always prefer knowledge from the English database over any other source.";

    const finalInput = formattedHistory
      ? `${systemMsg}\n${formattedHistory}\nUser: ${message}`
      : `${systemMsg}\nUser: ${message}`;

    // === Run Agent ===
    const result = await executor.invoke({ input: finalInput });

    // === Store messages in Supabase ===
    const { error: insertError } = await supabase.from("eng-messages").insert([
      { session_id: sessionId, role: "user", content: message },
      { session_id: sessionId, role: "assistant", content: result.output },
    ]);

    if (insertError) throw new Error(insertError.message);

    // === Return Response ===
    return res.json({
      success: true,
      response: result.output,
      sessionId,
      timestamp: Date.now(),
    });
  } catch (err) {
    error("❌ Error:", err);
    return res.json({ error: err.message }, 500);
  }
};
