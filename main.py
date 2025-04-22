import os
import time
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.agents import initialize_agent, Tool
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from supabase import create_client, Client

load_dotenv()
app = FastAPI()

# === Environment Variables ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
AUTH_SECRET = os.getenv("AUTH_SECRET")

if not all([GOOGLE_API_KEY, PINECONE_API_KEY, SUPABASE_URL, SUPABASE_KEY, AUTH_SECRET]):
    raise EnvironmentError("Missing required environment variables")

# === Clients Initialization ===
supabase: Client = create_client(str(SUPABASE_URL), str(SUPABASE_KEY))

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

# === Helper Functions ===
def format_history(history):
    return "\n".join(
        [f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
         for msg in history]
    )

def check_supabase_response(response):
    error = getattr(response, "error", None)
    if error:
        raise Exception(error.message)
    return response

async def fetch_insights():
    response = supabase.table("insights").select("*").execute()
    check_supabase_response(response)
    return str(response.data)

async def english_database(query: str):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name="science-9",
        embedding=embeddings,
        text_key="text"
    )

    results = vectorstore.similarity_search(query, k=5)
    return "\n\n---\n".join([r.page_content for r in results])

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        message = body.get("message")
        session_id = body.get("sessionId")
        auth_token = body.get("authToken")

        if not message or not session_id:
            raise HTTPException(status_code=400, detail="Missing message or sessionId")

        if auth_token != AUTH_SECRET:
            raise HTTPException(status_code=401, detail="Unauthorized")

        tools = [
            Tool(
                name="insights",
                func=lambda _: asyncio.run(fetch_insights()),
                description="Fetch insights from previous chats",
                coroutine=fetch_insights
            ),
            Tool(
                name="English_database",
                func=lambda q: asyncio.run(english_database(q)),
                description="Retrieve information to answer user queries about English",
                coroutine=english_database
            )
        ]

        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )

        agent = initialize_agent(
            tools,
            model,
            agent="zero-shot-react-description",
            verbose=True,
            handle_parsing_errors=True
        )

        history_resp = supabase.table("sci-messages") \
            .select("role, content") \
            .eq("session_id", session_id) \
            .order("created_at") \
            .limit(5) \
            .execute()

        check_supabase_response(history_resp)

        system_msg = "System: You are an AI English tutor created by Arjav. Answer questions about English language and literature in detail."
        formatted_history = format_history(history_resp.data) if history_resp.data else ""
        final_input = f"{system_msg}\n{formatted_history}\nUser: {message}" if formatted_history else f"{system_msg}\nUser: {message}"

        result = await agent.arun(final_input)

        insert_resp = supabase.table("eng-messages").insert([
            {"session_id": session_id, "role": "user", "content": message},
            {"session_id": session_id, "role": "assistant", "content": result}
        ]).execute()

        check_supabase_response(insert_resp)

        return JSONResponse({
            "success": True,
            "response": result,
            "sessionId": session_id,
            "timestamp": int(time.time() * 1000),
        })

    except HTTPException as he:
        return JSONResponse(
            status_code=he.status_code,
            content={"error": he.detail}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# Modified main function to accept context parameter
def main(context=None):
    return app
