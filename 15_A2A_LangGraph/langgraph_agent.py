import os
import asyncio
import logging
from uuid import uuid4
from typing import Optional

import httpx
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

# Load environment variables from .env file
load_dotenv()


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


A2A_BASE_URL_ENV = "A2A_BASE_URL"
DEFAULT_A2A_BASE_URL = "http://localhost:10000"


async def _a2a_send_message_async(query: str, base_url: Optional[str] = None) -> str:
    """Send a single-turn message to the A2A server and return a text summary.

    Falls back to returning the JSON response string if a concise text field cannot be located.
    """
    base = base_url or os.getenv(A2A_BASE_URL_ENV, DEFAULT_A2A_BASE_URL)

    # Increase timeout for LLM responses
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base)
        agent_card = await resolver.get_agent_card()

        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)

        payload = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": query}],
                "message_id": uuid4().hex,
            }
        }
        request = SendMessageRequest(id=str(uuid4()), params=MessageSendParams(**payload))

        response = await client.send_message(request)

        # Try to extract a readable text response; otherwise return JSON
        try:
            # Prefer final result artifact text if available
            # The exact schema may vary across SDK versions; attempt common paths safely
            root = response.root  # type: ignore[attr-defined]
            result = getattr(root, "result", None)
            if result is not None:
                # Some implementations store output text in result.output_text or similar
                text = getattr(result, "output_text", None)
                if isinstance(text, str) and text.strip():
                    return text.strip()

                # Or collect any message parts that are text
                output_messages = getattr(result, "output_messages", None)
                if output_messages:
                    texts: list[str] = []
                    for m in output_messages:
                        parts = getattr(m, "parts", []) or []
                        for p in parts:
                            if getattr(p, "kind", "") == "text":
                                t = getattr(getattr(p, "root", p), "text", None)
                                if isinstance(t, str):
                                    texts.append(t)
                    if texts:
                        return "\n".join(texts).strip()
        except Exception as e:  # noqa: BLE001
            logger.debug("Falling back to JSON response due to parse error: %s", e)

        return response.model_dump_json(indent=2, exclude_none=True)


@tool("a2a_send_message", return_direct=False)
def a2a_send_message(query: str, base_url: Optional[str] = None) -> str:
    """Use the internal A2A agent to answer the user's query.

    - query: The user question to send to the A2A server.
    - base_url: Optional override of the A2A server base URL. Defaults to env A2A_BASE_URL or http://localhost:10000.
    """
    try:
        return asyncio.run(_a2a_send_message_async(query, base_url))
    except RuntimeError:
        # If already inside an event loop, create a new task
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_a2a_send_message_async(query, base_url))


def build_langgraph_agent():
    """Create a simple ReAct-style LangGraph agent that can call the A2A server as a tool."""
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", os.getenv("TOOL_LLM_NAME", "gpt-4o-mini")),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("TOOL_LLM_URL", "https://api.openai.com/v1"),
        temperature=0,
    )
    tools = [a2a_send_message]
    return create_react_agent(llm, tools)


def run_once(question: str, base_url: Optional[str] = None) -> str:
    """Run a one-off query through the LangGraph agent using the A2A tool."""
    agent = build_langgraph_agent()

    system_instruction = (
        "You are a routing assistant. When the query requires using an internal agent, "
        "call the a2a_send_message tool with the full query. Return helpful, concise answers."
    )

    inputs = {
        "messages": [
            ("system", system_instruction),
            ("user", question if base_url is None else f"{question}\n\n[A2A_BASE_URL={base_url}]"),
        ]
    }

    responses = []
    for update in agent.stream(inputs, stream_mode="values"):
        # Capture the final AI response text if present
        try:
            messages = update.get("messages", [])
            if messages:
                last = messages[-1]
                content = getattr(last, "content", None)
                if isinstance(content, str) and content.strip():
                    responses.append(content.strip())
        except Exception:
            continue

    return responses[-1] if responses else "No response generated."


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LangGraph Agent that calls an A2A server via a tool.")
    parser.add_argument("question", type=str, nargs="+", help="Question to ask")
    parser.add_argument("--base-url", dest="base_url", type=str, default=None, help="A2A server base URL")
    args = parser.parse_args()

    question_text = " ".join(args.question)
    answer = run_once(question_text, base_url=args.base_url)
    print(answer)
