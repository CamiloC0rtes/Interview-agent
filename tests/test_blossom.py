import pytest
import asyncio
import json
from src.agent import call_mcp_holidays, blossom_app
from src.database import get_active_retriever, run_ingestion

@pytest.fixture(scope="session", autouse=True)
def setup_database():
    run_ingestion(force_rebuild=False)

@pytest.mark.asyncio
async def test_mcp_tool_call():
    try:
        result = await call_mcp_holidays()
        assert isinstance(result, str)
        assert "2026" in result 
        print(f"\n✅ MCP Tool Test Passed")
    except Exception as e:
        pytest.fail(f"MCP Tool failed: {e}")

@pytest.mark.asyncio
async def test_retriever_observability():
    retriever = get_active_retriever()
    docs = await retriever.ainvoke("password reset steps")
    assert len(docs) > 0
    assert "source" in docs[0].metadata
    assert "page" in docs[0].metadata
    print(f"\n✅ Retriever Test Passed.")

@pytest.mark.asyncio
async def test_security_scope_guardrail():
    state = {
        "message": "I want to open a new savings account",
        "history": [],
        "user_date": "2026-01-11",
        "temperature": 0.0
    }
    response = await blossom_app.ainvoke(state)
    answer = response["answer"].lower()
    
    assert any(term in answer for term in ["support", "help center", "cannot help", "information"])
    print("\n✅ Security Guardrail Test Passed.")