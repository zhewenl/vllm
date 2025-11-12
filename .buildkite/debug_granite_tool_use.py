#!/usr/bin/env python3
"""
Debug script to reproduce and investigate the granite-3.0-8b tool use failure on AMD.

Usage:
    python debug_granite_tool_use.py

This script mimics the failing test to capture what the model actually generates.
"""

import asyncio
import json
import openai


async def main():
    # Same configuration as the failing test
    client = openai.AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy",
    )

    # Same tools as in the test
    WEATHER_TOOL = {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for, e.g. 'San Francisco'",
                    },
                    "state": {
                        "type": "string",
                        "description": "must the two-letter abbreviation for the state that the city is in, e.g. 'CA' which would mean 'California'",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
            },
        },
    }

    SEARCH_TOOL = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet and get a summary of the top 10 webpages. Should only be used if you don't know the answer to a user query, and the results are likely to be able to be found with a web search",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "The term to use in the search. This should ideally be keywords to search for, not a natural-language question",
                    }
                },
                "required": ["search_term"],
            },
        },
    }

    MESSAGES_ASKING_FOR_TOOLS = [
        {"role": "user", "content": "What is the weather in Dallas, Texas in Fahrenheit?"}
    ]

    print("=" * 80)
    print("Testing granite-3.0-8b tool use on AMD")
    print("=" * 80)

    try:
        # Get model name
        models = await client.models.list()
        model_name = models.data[0].id
        print(f"\nModel: {model_name}")

        # Make the request (same as failing test)
        print("\n" + "=" * 80)
        print("Making chat completion request...")
        print("=" * 80)

        chat_completion = await client.chat.completions.create(
            messages=MESSAGES_ASKING_FOR_TOOLS,
            temperature=0,
            max_completion_tokens=100,
            model=model_name,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            logprobs=False,
        )

        # Print the full response
        print("\n" + "=" * 80)
        print("FULL RESPONSE:")
        print("=" * 80)
        print(json.dumps(chat_completion.model_dump(), indent=2))

        # Extract key information
        choice = chat_completion.choices[0]
        tool_calls = choice.message.tool_calls
        content = choice.message.content
        finish_reason = choice.finish_reason
        role = choice.message.role

        print("\n" + "=" * 80)
        print("KEY FIELDS:")
        print("=" * 80)
        print(f"Role: {role}")
        print(f"Finish Reason: {finish_reason}")
        print(f"Content: {repr(content)}")
        print(f"Tool Calls: {tool_calls}")
        print(f"Number of Tool Calls: {len(tool_calls) if tool_calls else 0}")

        # Check if test would pass
        print("\n" + "=" * 80)
        print("TEST VALIDATION:")
        print("=" * 80)

        try:
            assert role == "assistant", f"Expected role='assistant', got {role}"
            print("✓ Role is 'assistant'")
        except AssertionError as e:
            print(f"✗ {e}")

        try:
            assert tool_calls is not None, "Tool calls is None"
            print("✓ Tool calls is not None")
        except AssertionError as e:
            print(f"✗ {e}")
            return

        try:
            assert len(tool_calls) == 1, f"Expected 1 tool call, got {len(tool_calls)}"
            print(f"✓ Got {len(tool_calls)} tool call(s)")
        except AssertionError as e:
            print(f"✗ {e}")
            print("\n" + "=" * 80)
            print("FAILURE ANALYSIS:")
            print("=" * 80)
            print(f"Expected: 1 tool call")
            print(f"Got: {len(tool_calls)} tool calls")
            print(f"Tool calls list: {tool_calls}")
            print(f"\nThis matches the test failure: assert len(tool_calls) == 1 -> assert {len(tool_calls)} == 1")
            return

        # If we got here, check the tool call details
        tool_call = tool_calls[0]
        print(f"\n✓ Tool Call ID: {tool_call.id}")
        print(f"✓ Tool Call Type: {tool_call.type}")
        print(f"✓ Function Name: {tool_call.function.name}")
        print(f"✓ Function Arguments: {tool_call.function.arguments}")

        # Parse arguments
        try:
            args = json.loads(tool_call.function.arguments)
            print(f"\nParsed arguments:")
            print(f"  city: {args.get('city')}")
            print(f"  state: {args.get('state')}")
            print(f"  unit: {args.get('unit', 'N/A')}")

            # Check expected values
            assert args.get("city") == "Dallas", f"Expected city='Dallas', got {args.get('city')}"
            assert args.get("state") == "TX", f"Expected state='TX', got {args.get('state')}"
            print("\n✓ All validations passed! Test would PASS.")
        except Exception as e:
            print(f"\n✗ Argument validation failed: {e}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("\nStarting debug script...")
    print("Make sure vLLM server is running with:")
    print("  vllm serve ibm-granite/granite-3.0-8b-instruct \\")
    print("    --enable-auto-tool-choice \\")
    print("    --max-model-len 1024 \\")
    print("    --max-num-seqs 256 \\")
    print("    --enforce-eager \\")
    print("    --no-enable-prefix-caching \\")
    print("    --tool-call-parser granite \\")
    print("    --chat-template examples/tool_chat_template_granite.jinja \\")
    print("    --seed 0\n")

    asyncio.run(main())
