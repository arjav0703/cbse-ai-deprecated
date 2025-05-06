import "dotenv/config";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/community/vectorstores/pinecone";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";

// Pinecone client
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});
const index = pinecone.Index("activity");

// Gemini embeddings
const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GOOGLE_API_KEY,
  model: "embedding-001",
});

// vector store from the Pinecone index and namespace
const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
  pineconeIndex: index,
  namespace: "bihar",
});

// similarity search
const results = await vectorStore.similaritySearch("dance forms of bihar", 1);

console.log(results);
