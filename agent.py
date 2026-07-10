"""
Shizaye Multilinks Support Agent — Groq Edition (Free Tier)
-------------------------------------------------------------
Single-agent architecture using Groq's free API (OpenAI-compatible
tool-calling) instead of the paid Anthropic API. Model: Llama 3.3 70B.
No cost, no credit card required.
"""

import json
from openai import OpenAI
from tools.faq_rag import search_faq
from tools.order_lookup import lookup_order

MODEL = "openai/gpt-oss-120b"  # free on Groq, reliable tool-calling

SYSTEM_PROMPT = """You are Shiza, the friendly customer support assistant for
Shizaye Multilinks, a telecoms retail business with 8 branches across Kaduna,
Nigeria. You help customers with questions about products, services, pricing,
SIM registration, warranties, and order status.

Guidelines:
- Be warm, concise, and professional.
- Use the search_faq tool for general questions about products, policies, or services.
- Use the lookup_order tool when a customer asks about a specific order or account status.
- If you don't have enough information to help, say so honestly and suggest visiting
  the nearest branch or contacting a human agent.
- Never make up prices, policies, or order statuses that were not returned by a tool.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_faq",
            "description": "Search the Shizaye Multilinks FAQ and product knowledge base for answers to general questions about services, pricing, policies, or products.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The customer's question, in their own words.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up a customer's order or account status by order ID or phone number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID, e.g. ORD1001",
                    },
                    "phone": {
                        "type": "string",
                        "description": "The customer's phone number, used if order ID is not available.",
                    },
                },
            },
        },
    },
]


def _execute_tool(tool_name: str, tool_input: dict) -> str:
    """Dispatch a tool call to the correct local function."""
    if tool_name == "search_faq":
        return search_faq(query=tool_input.get("query", ""))
    elif tool_name == "lookup_order":
        return lookup_order(
            order_id=tool_input.get("order_id"),
            phone=tool_input.get("phone"),
        )
    return f"Unknown tool: {tool_name}"


def run_agent(user_message: str, conversation_history: list, api_key: str) -> tuple:
    """
    Run one turn of the agent loop using Groq's free API.

    Args:
        user_message: The latest message from the customer.
        conversation_history: List of prior {role, content} messages (memory).
        api_key: Groq API key.

    Returns:
        (assistant_reply, updated_conversation_history)
    """
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    if not conversation_history:
        conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    messages = conversation_history + [{"role": "user", "content": user_message}]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        max_tokens=1000,
    )

    message = response.choices[0].message

    # Handle tool-call loop (model may call one or more tools before answering)
    while message.tool_calls:
        messages.append(message.model_dump(exclude_none=True))

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_input = json.loads(tool_call.function.arguments)
            result = _execute_tool(tool_name, tool_input)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            max_tokens=1000,
        )
        message = response.choices[0].message

    final_text = message.content or ""
    messages.append({"role": "assistant", "content": final_text})
    return final_text, messages
