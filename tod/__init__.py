import ollama
import asyncio
import logging

logger = logging.getLogger("tod")

MAIN_MODEL = "mistral-nemo:12b-instruct-2407-q8_0"
MATH_MODEL = "mathstral:7b-v0.1-q4_K_M"


async def summarize(client: ollama.AsyncClient, text: str) -> str:
    response = await client.chat(
        model=MAIN_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Summarize text as concisely and correctly as you can, and do not say anything else. Remove all LaTeX commands.",
            },
            {"role": "user", "content": text},
        ],
        options={"temperature": 0.0},
    )
    return response["message"]["content"]


async def do_math(client: ollama.AsyncClient, question: str) -> str:
    response = await client.chat(
        model=MATH_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Correctly and carefully solve the problem posed. Do not make up information. If the question is incomplete, just say so. Think step by step.",
            },
            {"role": "user", "content": question},
        ],
        options={"temperature": 0.0},
    )
    return response["message"]["content"]


async def do_chat(client: ollama.AsyncClient, messages, tools=None):
    response = await client.chat(
        model=MAIN_MODEL,
        messages=messages,
        tools=tools,
        options={"temperature": 0.3},
    )
    return response["message"]


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    client = ollama.AsyncClient()
    messages = [
        {
            "role": "system",
            "content": " ".join(
                [
                    "You are an AI assistant named Tod.",
                    "You will answer questions, but never make up information unless asked to do so.",
                    "If you do not know the answer, you will say so.",
                    "You will include all context and information when invoking tools.",
                    "You are friendly and warm and have a fatherly tone.",
                ]
            ),
        }
    ]

    while True:
        query = input("> ")
        messages.append({"role": "user", "content": query})
        response = await do_chat(
            client,
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "description": "Get the current weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "The name of the city",
                                },
                            },
                            "required": ["city"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "perform_mathematical_reasoning",
                        "description": "Reason about a mathematical or logical question",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "The question to reason about",
                                }
                            },
                        },
                        "required": ["question"],
                    },
                },
            ],
        )

        messages.append(response)

        if response.get("tool_calls"):
            for tool in response["tool_calls"]:
                logger.info("Using tool: %s", tool["function"]["name"])
                if tool["function"]["name"] == "perform_mathematical_reasoning":
                    math_response = await do_math(client, query)
                    logger.info("math_response: %s", repr(math_response))
                    summarized_response = await summarize(client, math_response)
                    logger.info("summarized_response: %s", repr(summarized_response))
                    messages.append({"role": "tool", "content": summarized_response})

            response = await do_chat(client, messages)
            messages.append(response)

        print()
        print(response["content"])


if __name__ == "__main__":
    asyncio.run(main())
