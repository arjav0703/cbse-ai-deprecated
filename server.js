import 'dotenv/config';
import express from 'express';
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { PineconeStore } from '@langchain/community/vectorstores/pinecone';
import { Pinecone } from '@pinecone-database/pinecone';
import { SupabaseClient } from '@supabase/supabase-js';
import { HumanMessage } from '@langchain/core/messages';
import { initializeAgentExecutorWithOptions } from 'langchain/agents';


// === Env Variables ===
const {
  PINECONE_API_KEY,
  PINECONE_INDEX,
  SUPABASE_URL,
  SUPABASE_KEY,
  GOOGLE_API_KEY,
  PORT
} = process.env;

// === Setup ===
const app = express();
app.use(express.json());

const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });
const pineconeIndex = pinecone.Index(PINECONE_INDEX);
const supabase = new SupabaseClient(SUPABASE_URL, SUPABASE_KEY);

// === Tools ===
const vectorStore = await PineconeStore.fromExistingIndex(
  new GoogleGenerativeAIEmbeddings({
    modelName: 'models/text-embedding-004',
    apiKey: GOOGLE_API_KEY
  }),
  { pineconeIndex }
);

const insightsTool = {
  name: 'insights',
  description: 'Fetch insights from previous chats',
  func: async () => {
    const { data } = await supabase.from('insights').select('*');
    return JSON.stringify(data);
  }
};

const feedbackTool = {
  name: 'feedback',
  description: 'Store feedback into database',
  func: async (input) => {
    await supabase.from('insights').insert([{ feedback: input }]);
    return 'Feedback stored';
  }
};

const scienceTool = {
  name: 'vector_database',
  description: 'Call this tool to retrieve scientific information',
  func: async (query) => {
    const results = await vectorStore.similaritySearch(query, 5);
    return results.map(r => r.pageContent).join('\n');
  }
};

// === Agent ===
const model = new ChatGoogleGenerativeAI({
  modelName: 'models/gemini-2.0-flash',
  apiKey: GOOGLE_API_KEY,
  systemMessage: `You are Chemi, an AI agent who answers questions related to science...`
});

const executor = await initializeAgentExecutorWithOptions(
  [insightsTool, scienceTool, feedbackTool],
  model,
  {
    agentType: 'openai-functions',
    verbose: true
  }
);

// === Webhook ===
app.post('/webhook', async (req, res) => {
  const userInput = req.body.message;
  const result = await executor.invoke({
    input: userInput,
    messages: [new HumanMessage(userInput)]
  });
  res.json({ reply: result.output });
});

// === Start Server ===
app.listen(PORT || 3000, () => {
  console.log(`Server running on port ${PORT}`);
});
