import 'dotenv/config';
import express from 'express';
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/community/vectorstores/pinecone';
import { HumanMessage } from '@langchain/core/messages';
import { initializeAgentExecutorWithOptions } from 'langchain/agents';
import pkg from '@supabase/supabase-js';
const { createClient } = pkg;

// Load environment variables
const { 
  GOOGLE_API_KEY,
  PINECONE_API_KEY,
  PINECONE_INDEX,
  SUPABASE_URL,
  SUPABASE_KEY,
  PORT
} = process.env;

// Validate required environment variables
if (!GOOGLE_API_KEY) throw new Error("Missing GOOGLE_API_KEY");
if (!PINECONE_API_KEY) throw new Error("Missing PINECONE_API_KEY");
if (!PINECONE_INDEX) throw new Error("Missing PINECONE_INDEX");
if (!SUPABASE_URL) throw new Error("Missing SUPABASE_URL");
if (!SUPABASE_KEY) throw new Error("Missing SUPABASE_KEY");

console.log("Environment variables validated");

// Initialize Express app
const app = express();
app.use(express.json());

// === Setup Pinecone ===
const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });
const pineconeIndex = pinecone.Index(PINECONE_INDEX);

// === Initialize Supabase ===
const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// === Initialize Vector Store ===
const vectorStore = await PineconeStore.fromExistingIndex(
  new GoogleGenerativeAIEmbeddings({
    model: 'text-embedding-004',
    apiKey: GOOGLE_API_KEY
  }),
  { pineconeIndex }
);

// === Define Tools ===
const tools = [
  {
    name: 'insights',
    description: 'Fetch insights from previous chats',
    async func() {
      const { data, error } = await supabase.from('insights').select('*');
      if (error) throw new Error(error.message);
      return JSON.stringify(data);
    }
  },
  {
    name: 'feedback',
    description: 'Store feedback into database',
    async func(input) {
      const { error } = await supabase.from('insights').insert([{ feedback: input }]);
      if (error) throw new Error(error.message);
      return 'Feedback stored successfully';
    }
  },
  {
    name: 'vector_database',
    description: 'Retrieve scientific information from the knowledge base',
    async func(query) {
      const results = await vectorStore.similaritySearch(query, 5);
      return results.map(r => r.pageContent).join('\n\n---\n');
    }
  }
];

// === Initialize AI Model ===
const model = new ChatGoogleGenerativeAI({
  model: 'gemini-1.5-flash',
  apiKey: GOOGLE_API_KEY,
  temperature: 0.7,
  systemInstruction: {
    role: 'system',
    content: `You are Chemi, an AI science assistant. Follow these rules:
    1. Check insights first for context about previous interactions
    2. Use the science database for accurate information
    3. Store all user feedback
    4. Be concise but thorough in explanations`
  }
});

// === Initialize Agent ===
const executor = await initializeAgentExecutorWithOptions(
  tools,
  model,
  {
    agentType: 'zero-shot-react-description', // This is the most compatible agent type
    verbose: true,
    returnIntermediateSteps: true
  }
);

// === Webhook Endpoint ===
app.post('/webhook', async (req, res) => {
  try {
    const { message } = req.body;
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }

    const result = await executor.invoke({
      input: message,
      chat_history: [] // You can maintain chat history here
    });

    res.json({ 
      reply: result.output,
      sources: result.intermediateSteps // Optional: include sources if needed
    });
  } catch (err) {
    console.error('Error processing request:', err);
    res.status(500).json({ 
      error: 'Internal server error',
      details: err.message 
    });
  }
});

// === Health Check ===
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'healthy' });
});

// === Start Server ===
const serverPort = PORT || 3000;
app.listen(serverPort, () => {
  console.log(`🚀 Server running on port ${serverPort}`);
  console.log(`🔗 Endpoint: http://localhost:${serverPort}/webhook`);
});