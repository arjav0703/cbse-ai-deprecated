import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.vectorstores import Pinecone as PineconeStore
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.agents import initialize_agent, Tool
from pinecone import Pinecone, Index
from supabase import create_client, Client
import asyncio

app = FastAPI()

# === Environment Variables ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
AUTH_SECRET = os.getenv("AUTH_SECRET")

if not all([GOOGLE_API_KEY, PINECONE_API_KEY, SUPABASE_URL, SUPABASE_KEY, AUTH_SECRET]):
    raise EnvironmentError("Missing required environment variables")

# === Supabase Client ===
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# === Pinecone Setup ===
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index: Index = pc.Index("english")


@app.post("/chat")
async def chat(req: Request):
    try:
        body = await req.json()
        message = body.get("message")
        session_id = body.get("sessionId")
        auth_token = body.get("authToken")

        if not message or not session_id:
            raise HTTPException(status_code=400, detail="Missing message or sessionId")

        if auth_token != AUTH_SECRET:
            raise HTTPException(status_code=401, detail="Back off motherfucker, you ain't authenticated")

        # === Vector Store ===
        embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=GOOGLE_API_KEY)
        vectorstore = PineconeStore.from_existing_index(embedding=embeddings, index_name="english")

        # === Tools ===
        async def fetch_insights():
            response = supabase.table("insights").select("*").execute()
            if response.error:
                raise Exception(response.error.message)
            return str(response.data)

        async def english_database(query: str):
            results = vectorstore.similarity_search(query, k=5)
            return "\n\n---\n".join([r.page_content for r in results])

        tools = [
            Tool(
                name="insights",
                func=lambda _: asyncio.run(fetch_insights()),
                description="Fetch insights from previous chats"
            ),
            Tool(
                name="English database",
                func=lambda q: asyncio.run(english_database(q)),
                description="Retrieve information to answer user queries."
            )
        ]

        # === Chat Model ===
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY
        )

        agent = initialize_agent(
            tools,
            model,
            agent="zero-shot-react-description",
            verbose=True,
            return_intermediate_steps=True,
        )

        # === Fetch & Format Chat History ===
        history_resp = supabase.table("sci-messages") \
            .select("role, content") \
            .eq("session_id", session_id) \
            .order("created_at", desc=False) \
            .limit(5) \
            .execute()

        if history_resp.error:
            raise Exception(history_resp.error.message)

        def format_history(history):
            return "\n".join(
                [f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" for msg in history]
            )

        formatted_history = format_history(history_resp.data)
        system_msg = "System: You are an AI agent created by Arjav who answers questions related to English. Always answer in detail unless specified not to. Always use the English database to answer user queries"

        final_input = f"{system_msg}\n{formatted_history}\nUser: {message}" if formatted_history else f"{system_msg}\nUser: {message}"

        # === Run Agent ===
        result = agent.run(final_input)

        # === Store Messages ===
        insert_resp = supabase.table("eng-messages").insert([
            {"session_id": session_id, "role": "user", "content": message},
            {"session_id": session_id, "role": "assistant", "content": result}
        ]).execute()

        if insert_resp.error:
            raise Exception(insert_resp.error.message)

        return JSONResponse({
            "success": True,
            "response": result,
            "sessionId": session_id,
            "timestamp": int(asyncio.time.time() * 1000),
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
