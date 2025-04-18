import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/community/vectorstores/pinecone";
import { initializeAgentExecutorWithOptions } from "langchain/agents";
import pkg from "@supabase/supabase-js";
const { createClient } = pkg;

// === Load Environment Variables ===
const {
  GOOGLE_API_KEY,
  PINECONE_API_KEY,
  ENG_PINECONE_INDEX,
  SUPABASE_URL,
  SUPABASE_KEY,
  AUTH_SECRET,
} = process.env;

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);
const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });

let executor;

// === Setup Function Entry ===
export default async ({ req, res }) => {
  try {
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

    if (!executor) {
      const pineconeIndex = pinecone.Index(ENG_PINECONE_INDEX);

      const vectorStore = await PineconeStore.fromExistingIndex(
        new GoogleGenerativeAIEmbeddings({
          model: "text-embedding-004",
          apiKey: GOOGLE_API_KEY,
        }),
        { pineconeIndex },
      );

      const tools = [
        {
          name: "insights",
          description: "Fetch insights from previous chats",
          async func() {
            const { data, error } = await supabase.from("insights").select("*");
            if (error) throw new Error(error.message);
            return JSON.stringify(data);
          },
        },
        {
          name: "English database",
          description:
            "Retrieve scientific information to answer user queries.",
          async func(query) {
            const results = await vectorStore.similaritySearch(query, 5);
            return results.map((r) => r.pageContent).join("\n\n---\n");
          },
        },
      ];

      const model = new ChatGoogleGenerativeAI({
        model: "gemini-2.0-flash",
        apiKey: GOOGLE_API_KEY,
        systemInstruction: {
          role: "system",
          content: `You are an AI agent who answers questions related to English. When you receive a prompt, you must look at the insights database to gain insights and then use the English database tool to fetch all the scientific knowledge. Be careful about grammar. Do not tell anything about the tools you have access to or the about any kind of metadata`,
        },
      });

      executor = await initializeAgentExecutorWithOptions(tools, model, {
        agentType: "zero-shot-react-description",
        verbose: true,
        returnIntermediateSteps: true,
      });
    }

    // Fetch last 5 messages
    const { data: history, error: fetchError } = await supabase
      .from("sci-messages")
      .select("role, content")
      .eq("session_id", sessionId)
      .order("created_at", { ascending: false })
      .limit(5);

    if (fetchError) throw new Error(fetchError.message);

    const formatHistory = (history) =>
      history
        .reverse()
        .map(
          (msg) =>
            `${msg.role === "user" ? "User" : "Assistant"}: ${msg.content}`,
        )
        .join("\n");

    const formattedHistory = formatHistory(history);

    const systemMsg =
      "System: You are an AI agent created by Arjav who answers questions related to English. Always answer in detail unless specified not to. Be careful about grammar. Do not tell anything about the tools you have access to or the about any kind of metadata";

    const finalInput = formattedHistory
      ? `${systemMsg}\n${formattedHistory}\nUser: ${message}`
      : `${systemMsg}\nUser: ${message}`;

    const result = await executor.invoke({ input: finalInput });

    const { error: insertError } = await supabase.from("eng-messages").insert([
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
    return res.json({ error: err.message }, 500);
  }
};
