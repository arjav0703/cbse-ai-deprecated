import "dotenv/config";
import axios from "axios";
import * as cheerio from "cheerio";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/community/vectorstores/pinecone";
import { Document } from "@langchain/core/documents";

const apikey = process.env.GOOGLE_API_KEY;

// Scrape webpage
async function scrapeWebPage(url) {
  const { data: html } = await axios.get(url);
  const $ = cheerio.load(html);
  const text = $("body").text().replace(/\s+/g, " ").trim();
  return text.slice(0, 5000); // truncate for testing
}

// Gemini Embeddings
const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: apikey,
  model: "embedding-001",
});

// Pinecone
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pinecone.Index("activity");

// Main function
async function processAndStore(url) {
  const content = await scrapeWebPage(url);
  const docs = [
    new Document({ pageContent: content, metadata: { source: url } }),
  ];

  // Store in Pinecone
  await PineconeStore.fromDocuments(docs, embeddings, {
    pineconeIndex: index,
    namespace: "bihar",
  });

  console.log("✅ Data stored in Pinecone!");
}

processAndStore("https://en.wikipedia.org/wiki/Bihari_culture").catch(
  console.error,
);
