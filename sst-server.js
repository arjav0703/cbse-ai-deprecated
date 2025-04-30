import express from "express";
import dotenv from "dotenv";
import bodyParser from "body-parser";
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/community/vectorstores/pinecone";
import { initializeAgentExecutorWithOptions } from "langchain/agents";
import pkg from "@supabase/supabase-js";

const { createClient } = pkg;

dotenv.config();
const app = express();
app.use(bodyParser.json());

const {
  GOOGLE_API_KEY,
  PINECONE_API_KEY,
  SUPABASE_URL,
  SUPABASE_KEY,
  AUTH_SECRET,
  PORT = 3000,
} = process.env;

if (
  !GOOGLE_API_KEY ||
  !PINECONE_API_KEY ||
  !SUPABASE_URL ||
  !SUPABASE_KEY ||
  !AUTH_SECRET
) {
  console.error("❌ Missing required environment variables");
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

app.post("/sst", async (req, res) => {
  try {
    const { message, sessionId, authToken } = req.body;

    if (!message || !sessionId) {
      return res.status(400).json({ error: "Missing message or sessionId" });
    }

    if (authToken !== AUTH_SECRET) {
      return res
        .status(401)
        .json({ error: "Back off, you ain't authenticated" });
    }

    // === Pinecone Setup ===
    const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });
    const pineconeIndex = pinecone.Index("sst");

    const vectorStore = await PineconeStore.fromExistingIndex(
      new GoogleGenerativeAIEmbeddings({
        model: "text-embedding-004",
        apiKey: GOOGLE_API_KEY,
      }),
      { pineconeIndex },
    );

    // === Tools ===
    const tools = [
      {
        name: "SST database",
        description: "Retrieve information to answer user queries.",
        async func(query) {
          const results = await vectorStore.similaritySearch(query, 5);
          return results.map((r) => r.pageContent).join("\n\n---\n");
        },
      },
    ];

    // === Model ===
    const model = new ChatGoogleGenerativeAI({
      model: "gemini-2.0-flash",
      apiKey: GOOGLE_API_KEY,
      systemInstruction: {
        role: "system",
        content: `You are an AI agent who answers questions related to history, geography, political science and economics. Always prefer knowledge from the SST database over any other source. Prefer answering in detail and in the format of bullet points.`,
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

    const { data: history, error: fetchError } = await supabase
      .from("sst-messages")
      .select("role, content")
      .eq("session_id", sessionId)
      .order("created_at", { ascending: false })
      .limit(5);

    if (fetchError) throw new Error(fetchError.message);

    const formattedHistory = formatHistory(history.reverse());

    const systemMsg =
      "System: You are an AI agent created by arjav who answers questions related to history, geography, political science and economics. Always answer in detail and in the format of bullet points unless specified not to. Always use the SST database to answer user queries. If the answer cannot be found in the SST Database, tell the user to select other subject through the dropdown menu.";

    const finalInput = formattedHistory
      ? `${systemMsg}\n${formattedHistory}\nUser: ${message}`
      : `${systemMsg}\nUser: ${message}`;

    const result = await executor.invoke({ input: finalInput });

    const { error: insertError } = await supabase.from("sst-messages").insert([
      { session_id: sessionId, role: "user", content: message },
      { session_id: sessionId, role: "assistant", content: result.output },
    ]);

    if (insertError) throw new Error(insertError.message);

    return res.json({
      success: true,
      response: result.output,
      sessionId,
      timestamp: Date.now(),
    });
  } catch (err) {
    console.error("❌ Error:", err);
    return res.status(500).json({ error: err.message });
  }
});

// === Start Server ===
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
