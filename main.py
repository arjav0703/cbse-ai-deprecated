from fastapi import FastAPI, Request
from langchain.embeddings import GooglePalmEmbeddings
from langchain.chat_models import ChatGooglePalm
from langchain.vectorstores import Pinecone
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferWindowMemory
from supabase import create_client, Client
import pinecone
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # optional for Google setup

# Initialize services
app = FastAPI()
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
embedding = GooglePalmEmbeddings(model="models/text-embedding-004")
vector_store = Pinecone.from_existing_index(index_name=PINECONE_INDEX, embedding=embedding)

chat_model = ChatGooglePalm(model="models/gemini-2.0-flash")
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5, return_messages=True)

# Tools
def get_insights():
    result = supabase.table("insights").select("*").execute()
    return str(result.data)

def store_feedback(feedback: str):
    supabase.table("insights").insert({"feedback": feedback}).execute()
    return "Feedback stored."

def retrieve_scientific_info(query: str):
    return vector_store.similarity_search(query, k=5)

tools = [
    Tool(name="insights", func=lambda _: get_insights(), description="Gain insights from previous chats"),
    Tool(name="vector_database", func=retrieve_scientific_info, description="Retrieve scientific information"),
    Tool(name="feedback", func=store_feedback, description="Store feedback data")
]

agent = initialize_agent(
    tools=tools,
    llm=chat_model,
    agent="chat-zero-shot-react-description",
    memory=memory,
    verbose=True
)

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    message = data.get("message")
    response = agent.run(message)
    return {"response": response}
