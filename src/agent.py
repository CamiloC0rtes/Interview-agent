import time, asyncio, functools, json, logging, os, re
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

# -------------------------
# Decorators
# -------------------------
def time_logger(func):
    """Observability decorator for node latency tracking."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        duration = (time.perf_counter() - start) * 1000
        if isinstance(result, dict):
            result["latency_ms"] = round(duration, 2)
        logger.info(f"⏱️ Node {func.__name__} latency: {duration:.2f}ms")
        return result
    return wrapper

# -------------------------
# MCP / Holiday Utilities
# -------------------------
async def call_mcp_holidays() -> str:
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

async def fetch_holiday_name(user_date_str: str) -> Optional[str]:
    """Returns the holiday name for the given date, parsing plain text from MCP."""
    global _CACHED_HOLIDAYS
    if not _CACHED_HOLIDAYS:
        try:

            _CACHED_HOLIDAYS = await call_mcp_holidays()
        except Exception as e:
            logger.error(f"Error fetching holidays: {e}")
            return None

    try:
        lines = _CACHED_HOLIDAYS.strip().split('\n')
        for line in lines:
            if ":" in line:
                date_part, name_part = line.split(":", 1)
                if date_part.strip() == user_date_str:
                    return name_part.strip()
        return None
    except Exception as e:
        logger.error(f"Error parsing plain text holidays: {e}")
        return None
# -------------------------
# LLM Utilities
# -------------------------
async def stream_llm_response(system_prompt, history, user_message, temp=0.2, top_p=0.9):
    """Internal helper for LLM streaming token generation."""
    messages = [SystemMessage(content=system_prompt)]
    # Sliding window for short-term memory
    for msg in history[-4:]:
        role = HumanMessage if msg["role"] == "user" else AIMessage
        messages.append(role(content=msg["content"]))
    messages.append(HumanMessage(content=user_message))
    async for chunk in llm_client.astream(messages, temperature=temp, top_p=top_p):
        if chunk.content:
            yield chunk.content

# -------------------------
# Domain / Keyword Checks
# -------------------------
def check_greeting(user_msg: str) -> bool:
    """Detecta si el usuario está saludando o usando expresiones corteses."""
    greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
    polite_closings = {"thanks", "thank you", "thx", "ty", "appreciate it", "much appreciated"}
    msg_lower = user_msg.lower()
    return any(g in msg_lower for g in greetings.union(polite_closings))

def check_farewell(user_msg: str) -> bool:
    """Return True if the user wants to end the session."""
    reset_keywords = {"thanks", "thank you", "i'm done", "done", "no thanks", "stop"}
    user_msg_lower = user_msg.lower()
    return any(kw in user_msg_lower for kw in reset_keywords)

def determine_scope(user_msg: str) -> bool:
    """Return True if message is related to security/login."""
    security_keywords = {
    "login", "sign in", "password", "lock", "access", "mfa",
    "verification", "code", "otp", "remember device", "username",
    "unlock", "security", "can't login", "cannot login", "forgot password",
    "two factor", "2fa","unlocking", "unlock my account","user"
    }
    user_msg_lower = user_msg.lower()
    return any(re.search(rf'\b{kw}\b', user_msg_lower) for kw in security_keywords)

async def retrieve_docs(user_msg: str) -> list:
    """Fetch documents from the retriever if message is in scope."""
    try:
        return await get_active_retriever().ainvoke(user_msg)
    except Exception as e:
        logger.error(f"Error retrieving docs: {e}")
        return []

# -------------------------
# Prompt Construction
# -------------------------
def build_prompt(docs: list, day_name: str, user_date_str: str, holiday_name: Optional[str]) -> str:
    """Construct LLM prompt including policies, protocols, and sources."""
    context_text = "\n".join(
        f"[Source: {d.metadata.get('source')}, Page: {d.metadata.get('page')}]: {d.page_content}"
        for d in docs
    )
    holiday_message = (
        f"Current Holiday: {holiday_name}. Manual reviews will resume next business day."
        if holiday_name else
        "Today is not a federal holiday. Your password reset should proceed without delay."
    )
    return (
        "You are Blossom, a premium banking assistant specialized in Login/Security. "
        f"Today is {day_name}, {user_date_str}. {holiday_message}\n"
        "\nPOLICIES:\n"
        "- YOU ONLY RESPOND WITH THE CONTEXT GIVEN\n"
        "- Only help with login, password, MFA, or security topics.\n"
        "- Vary your tone naturally based on user input (be helpful and warm).\n"
        "- If it is a holiday, mention that manual reviews resume next business day.\n"
        "- Mention '(source: federal holiday API)' when discussing dates or holidays.\n"
        "\nPROTOCOLS:\n"
        f"{context_text if docs else 'I do not have access to that information. Please contact customer support for assistance or follow the next steps provided in your banking portal.'}"
    )

async def generate_farewell(user_msg: str, history: list) -> str:
    """Generate warm farewell message."""
    system_prompt = (
        "You are Blossom, a friendly banking assistant. "
        "The user has finished their session. "
        "Generate a short, warm, polite farewell message that encourages them "
        "to come back if they need help with login, passwords, or security."
    )
    chunks = [t async for t in stream_llm_response(system_prompt, history, user_msg)]
    return "".join(chunks).strip()

async def generate_out_of_scope_response(user_msg: str, history: list) -> str:
    """Genera una respuesta natural cuando el mensaje está fuera de scope."""
    system_prompt = (
        "You are Blossom, a helpful and polite banking assistant. "
        "The user asked something outside your allowed scope (only login, password, MFA, or security). "
        "Respond naturally, politely redirect them to customer support be concise and drastically "
        "and maintain a friendly tone."
    )
    chunks = [t async for t in stream_llm_response(system_prompt, history, user_msg)]
    return "".join(chunks).strip()

async def generate_greeting(user_msg: str, history: list) -> str:
    """Genera un saludo natural usando el LLM."""
    system_prompt = (
        "You are Blossom, a friendly and warm banking assistant. "
        "The user just greeted you or said something polite. "
        "Respond naturally, acknowledging their greeting, "
        "and mention that you help with login, password, MFA, or security topics."
    )
    chunks = [t async for t in stream_llm_response(system_prompt, history, user_msg)]
    return "".join(chunks).strip()

def should_allow_steps(user_msg: str, last_topic: Optional[str]) -> bool:
    steps_keywords = {"steps", "more", "instructions", "guide", "procedure"}
    msg_lower = user_msg.lower()
    
    asks_for_steps = any(kw in msg_lower for kw in steps_keywords)
    
    if asks_for_steps:
        if last_topic == "Security/Login" or last_topic == "MFA":
            return True
        return False  
        
    return True  


# -------------------------
# Core Agent Node
# -------------------------
@time_logger
async def blossom_node(state: AgentState):
    user_msg = state["message"].strip()
    history = state.get("history", [])
    user_date_str = state.get("user_date", datetime.now().strftime("%Y-%m-%d"))
    current_dt = datetime.strptime(user_date_str, "%Y-%m-%d")
    day_name = current_dt.strftime("%A")
    
    last_topic = state.get("topic")

    if check_farewell(user_msg):
        farewell_message = await generate_farewell(user_msg, history)
        return {"answer": farewell_message, "topic": None, "history": [], "latency_ms": 0}

    if not should_allow_steps(user_msg, last_topic):
        out_of_scope_response = await generate_out_of_scope_response(user_msg, history)
        return {
            "answer": out_of_scope_response, 
            "topic": "Out of Scope", 
            "history": (history + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": out_of_scope_response}])[-10:]
        }

    is_in_scope = determine_scope(user_msg)
    docs, holiday_name = await asyncio.gather(
        retrieve_docs(user_msg) if is_in_scope else asyncio.sleep(0, []),
        fetch_holiday_name(user_date_str)
    )

    if check_greeting(user_msg):
        greeting_response = await generate_greeting(user_msg, history)
        return {
            "answer": greeting_response,
            "topic": "Greeting",
            "history": (history + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": greeting_response}])[-10:]
        }

    if not is_in_scope or (is_in_scope and not docs):
        out_of_scope_response = await generate_out_of_scope_response(user_msg, history)
        return {
            "answer": out_of_scope_response,
            "topic": "Out of Scope",
            "history": (history + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": out_of_scope_response}])[-10:]
        }

    system_prompt = build_prompt(docs, day_name, user_date_str, holiday_name)
    chunks = [t async for t in stream_llm_response(system_prompt, history, user_msg)]
    full_response = "".join(chunks).strip()

    current_topic = "Security/Login"
    source_tag = ""
    if docs:
        primary = docs[0].metadata
        source_tag = f"\n\n—\nSource: {primary.get('source')} (Page {primary.get('page')})"
        current_topic = primary.get("tags", "Security/Login")

    return {
        "answer": f"{full_response}{source_tag}",
        "topic": current_topic,
        "history": (history + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": full_response}])[-10:],
        "latency_ms": 0
    }
# -------------------------
# Graph definition
# -------------------------
builder = StateGraph(AgentState)
builder.add_node("blossom", blossom_node)
builder.set_entry_point("blossom")
builder.add_edge("blossom", END)
blossom_app = builder.compile()
