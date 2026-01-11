import time, asyncio, functools, json, logging, os
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from .database import get_active_retriever
from datetime import datetime

logger = logging.getLogger("blossom_agent")

# Global cache for MCP tools to reduce latency
_CACHED_HOLIDAYS: Optional[str] = None
llm_client = ChatOpenAI(model=os.getenv("CHAT_MODEL_NAME", "gpt-4o-mini"))

class AgentState(TypedDict):
    message: str
    user_date: Optional[str]
    answer: str
    topic: Optional[str]
    history: list[dict]
    latency_ms: Optional[float]
    temperature: Optional[float]
    top_p: Optional[float]

def time_logger(func):
    """Observability decorator for node latency tracking."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        duration = (time.perf_counter() - start) * 1000
        if isinstance(result, dict): result["latency_ms"] = round(duration, 2)
        logger.info(f"⏱️ Node {func.__name__} latency: {duration:.2f}ms")
        return result
    return wrapper

async def call_mcp_holidays():
    """Fetches federal holiday data via MCP Server."""
    try:
        params = StdioServerParameters(command="python", args=["src/mcp_server.py"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                year = datetime.now().year
                result = await session.call_tool("get_federal_holidays", arguments={"year": year})
                return result.content[0].text
    except Exception as e:
        logger.error(f"MCP Connection Error: {e}")
        return "[]"

async def stream_llm_response(system_prompt, history, user_message, temp=0.2, top_p=0.9):
    """Internal helper for LLM streaming token generation."""
    messages = [SystemMessage(content=system_prompt)]
    # Sliding window for short-term memory
    for msg in history[-4:]:
        role = HumanMessage if msg["role"] == "user" else AIMessage
        messages.append(role(content=msg["content"]))
    
    messages.append(HumanMessage(content=user_message))
    async for chunk in llm_client.astream(messages, temperature=temp, top_p=top_p):
        if chunk.content: yield chunk.content

@time_logger
async def blossom_node(state: AgentState):
    global _CACHED_HOLIDAYS

    # 1. State extraction & temporal context (source: federal holiday API)
    user_msg = state["message"].strip()
    history = state.get("history", [])
    user_date_str = state.get("user_date", datetime.now().strftime("%Y-%m-%d"))
    current_dt = datetime.strptime(user_date_str, "%Y-%m-%d")
    day_name = current_dt.strftime("%A")

    # 2. Domain Gating Logic
    user_msg_lower = user_msg.lower()
    security_keywords = {
        "login", "sign in", "password", "lock", "access", "mfa", "verification", 
        "code", "otp", "remember", "device", "username", "unlock", "security"
    }
    is_in_scope = any(kw in user_msg_lower for kw in security_keywords) or not history

    # 3. Concurrent Retrieval (RAG + MCP)
    tasks = []
    if is_in_scope:
        tasks.append(get_active_retriever().ainvoke(user_msg))
    else:
        tasks.append(asyncio.sleep(0)) 

    if not _CACHED_HOLIDAYS:
        tasks.append(call_mcp_holidays())
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    docs = results[0] if is_in_scope and not isinstance(results[0], Exception) and results[0] != 0 else []
    
    # 4. Holiday parsing from MCP Cache
    holiday_name = None
    if _CACHED_HOLIDAYS:
        try:
            holidays = json.loads(_CACHED_HOLIDAYS)
            current_h = next((h for h in holidays if h.get("date") == user_date_str), None)
            if current_h: holiday_name = current_h.get("name")
        except: pass

    # 5. Prompt Construction (Grounding & Personality)
    context_text = "\n".join([
        f"[Source: {d.metadata.get('source')}, Page: {d.metadata.get('page')}]: {d.page_content}" 
        for d in docs
    ])
    
    system_prompt = (
        "You are Blossom, a premium banking assistant specialized in Login/Security. "
        f"Today is {day_name}, {user_date_str}. "
        f"{f'Current Holiday: {holiday_name}.' if holiday_name else ''} "
        "\nPOLICIES:\n"
        "- Only help with login, password, MFA, or security topics.\n"
        "- Vary your tone naturally based on user input (be helpful and warm).\n"
        "- If it is a weekend or holiday, mention that manual reviews resume next business day.\n"
        "- Mention '(source: federal holiday API)' when discussing dates or holidays.\n"
        "\nPROTOCOLS:\n"
        f"{context_text if docs else 'No specific protocol found. Use general security best practices.'}"
    )

    # 6. Response Synthesis
    chunks = [t async for t in stream_llm_response(system_prompt, history, user_msg)]
    full_response = "".join(chunks).strip()

    # 7. Metadata Attribution
    source_tag = ""
    if docs:
        primary = docs[0].metadata
        source_tag = f"\n\n—\nSource: {primary.get('source')} (Page {primary.get('page')})"

    return {
        "answer": f"{full_response}{source_tag}",
        "topic": "Security/Login",
        "history": (history + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": full_response}])[-10:],
        "timing_ms": 0
    }

# 8. Graph definition
builder = StateGraph(AgentState)
builder.add_node("blossom", blossom_node)
builder.set_entry_point("blossom")
builder.add_edge("blossom", END)
blossom_app = builder.compile()