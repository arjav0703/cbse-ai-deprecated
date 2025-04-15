import "dotenv/config";
import express from "express";
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/community/vectorstores/pinecone";
import { initializeAgentExecutorWithOptions } from "langchain/agents";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import pkg from "@supabase/supabase-js";
const { createClient } = pkg;

// === Load Env Vars ===
const {
  GOOGLE_API_KEY,
  PINECONE_API_KEY,
  PINECONE_INDEX,
  SUPABASE_URL,
  SUPABASE_KEY,
  PORT,
} = process.env;

if (!GOOGLE_API_KEY) throw new Error("Missing GOOGLE_API_KEY");
if (!PINECONE_API_KEY) throw new Error("Missing PINECONE_API_KEY");
if (!PINECONE_INDEX) throw new Error("Missing PINECONE_INDEX");
if (!SUPABASE_URL) throw new Error("Missing SUPABASE_URL");
if (!SUPABASE_KEY) throw new Error("Missing SUPABASE_KEY");

console.log("✅ Environment variables validated");

// === Express App ===
const app = express();
app.use(express.json());

// === Pinecone Setup ===
const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });
const pineconeIndex = pinecone.Index(PINECONE_INDEX);

// === Supabase ===
const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// === Vector Store ===
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
    name: "insights",
    description: "Fetch insights from previous chats",
    async func() {
      const { data, error } = await supabase.from("insights").select("*");
      if (error) throw new Error(error.message);
      return JSON.stringify(data);
    },
  },
  {
    name: "feedback",
    description: "Store feedback into database",
    async func(input) {
      const { error } = await supabase
        .from("insights")
        .insert([{ feedback: input }]);
      if (error) throw new Error(error.message);
      return "Feedback stored successfully";
    },
  },
  {
    name: "vector_database",
    description: "Retrieve scientific information from the knowledge base",
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
    content: `You are Chemi, an AI science assistant. Follow these rules:
1. Check insights first for context about previous interactions
2. Use the science database for accurate information
3. Store all user feedback
4. Be concise but thorough in explanations`,
  },
});

// === Agent Executor ===
const executor = await initializeAgentExecutorWithOptions(tools, model, {
  agentType: "zero-shot-react-description",
  verbose: true,
  returnIntermediateSteps: true,
});

// === In-Memory Chat History ===
const chatHistory = new Map(); // { userId: [{ role, content }] }
const MAX_HISTORY_LENGTH = 20;

const getChatHistory = (userId) => {
  if (!chatHistory.has(userId)) {
    chatHistory.set(userId, []);
  }
  return chatHistory.get(userId);
};

const pruneOldMessages = (history) => {
  if (history.length > MAX_HISTORY_LENGTH) {
    history.splice(0, history.length - MAX_HISTORY_LENGTH);
  }
  return history;
};

const formatChatHistory = (history) => {
  return history
    .map(
      (msg) => `${msg.role === "user" ? "User" : "Assistant"}: ${msg.content}`,
    )
    .join("\n");
};

// === Webhook Endpoint ===
app.post("/webhook", async (req, res) => {
  try {
    const { message, userId = "default" } = req.body;
    const history = getChatHistory(userId);

    // Add user message to history
    history.push({ role: "user", content: message });
    pruneOldMessages(history);

    const formattedHistory = formatChatHistory(history.slice(0, -1)); // exclude current message
    const finalInput = formattedHistory
      ? `${formattedHistory}\nUser: ${message}`
      : `User: ${message}`;

    // Call agent
    const result = await executor.invoke({ input: finalInput });

    // Save AI response to history
    history.push({ role: "assistant", content: result.output });
    pruneOldMessages(history);

    res.json({
      reply: result.output,
      history: history.slice(-10), // optional
    });
  } catch (err) {
    console.error("Error:", err);
    res.status(500).json({ error: err.message });
  }
});

// === Health Check ===
app.get("/health", (req, res) => {
  res.status(200).json({ status: "healthy" });
});

// === Start Server ===
const serverPort = PORT || 3000;
app.listen(serverPort, () => {
  console.log(`🚀 Server running on port ${serverPort}`);
  console.log(`🔗 Endpoint: http://localhost:${serverPort}/webhook`);
});
