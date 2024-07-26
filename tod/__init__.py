import ollama
import asyncio

MAIN_MODEL = "mistral-nemo:latest"
MATH_MODEL = "wizard-math:13b-fp16"


async def summarize(client: ollama.AsyncClient, text: str) -> str:
    response = await client.chat(
        model=MAIN_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You will summarize text as concisely and correctly as you can, and not say anything else.",
            },
            {"role": "user", "content": text},
        ],
    )
    return response["message"]["content"]


async def main() -> None:
    client = ollama.AsyncClient()
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant named Tod. You will concisely answer questions, but never make up information unless asked to do so. If you do not know the answer, you will say so. You are friendly and warm and have a fatherly tone.",
        }
    ]

    while True:
        query = input("> ")
        messages.append({"role": "user", "content": query})
        response = await client.chat(
            model=MAIN_MODEL,
            messages=messages,
            # provide a weather checking tool to the model
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

        messages.append(response["message"])

        if response["message"].get("tool_calls"):
            for tool in response["message"]["tool_calls"]:
                print("Using tool: " + tool["function"]["name"])
                if tool["function"]["name"] == "get_current_weather":
                    response = "Sunny and 30 degrees Celsius"
                    messages.append({"role": "tool", "content": response})
                elif tool["function"]["name"] == "perform_mathematical_reasoning":
                    response = await client.generate(
                        model=MATH_MODEL,
                        prompt=tool["function"]["arguments"]["question"],
                    )
                    summarized_response = await summarize(client, response["response"])

                    messages.append({"role": "tool", "content": summarized_response})

            response = await client.chat(model=MAIN_MODEL, messages=messages)

        print(response["message"]["content"])


if __name__ == "__main__":
    asyncio.run(main())
