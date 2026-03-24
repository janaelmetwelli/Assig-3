import os
import json
import math
from openai import OpenAI
from dotenv import load_dotenv
import requests
from concurrent.futures import ThreadPoolExecutor 
import time 
# Load environment variables from .env file
load_dotenv()

# NEW CODE - USE THESE 3 LINES:
API_KEY = os.environ.get("GROQ_API_KEY")
BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "llama-3.1-8b-instant"

# Initialize the OpenAI client with custom base URL
# Replace with your API key or set it as an environment variable
client = OpenAI(
api_key=API_KEY,
base_url=BASE_URL,
)



SAFE_LOCALS = {
    "pi": math.pi, "e": math.e,
    "sqrt": math.sqrt, "abs": abs, "round": round,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "exp": math.exp,
}

def calculator(expression):
    if not isinstance(expression, str) or not expression.strip():
        return "Error: Invalid input"
    
    try:
        result = eval(expression, {"__builtins__": {}}, SAFE_LOCALS)
        if not isinstance(result, (int, float)):
            return "Error: Result is not a number"
        if result == float("inf") or result != result:  # inf or NaN
            return "Error: Result is undefined (inf or NaN)"
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except SyntaxError:
        return "Error: Invalid expression syntax"
    except NameError as e:
        return f"Error: Unknown name in expression — {e}"
    except (ValueError, TypeError) as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {str(e)}"

def get_current_weather(location):
    """Get the current weather for a location."""
    api_key = os.environ.get("WEATHER_API_KEY")
    url = (
        f"http://api.weatherapi.com/v1/current.json"
        f"?key={api_key}&q={location}&aqi=no"
        )
    response = requests.get(url)
    data = response.json()
    if "error" in data:
        return f"Error: {data['error']['message']}"
    weather_info = data["current"]
    return json.dumps(
        {
        "location": data["location"]["name"],
        "temperature_c": weather_info["temp_c"],
        "temperature_f": weather_info["temp_f"],
        "condition": weather_info["condition"]["text"],
        "humidity": weather_info["humidity"],
        "wind_kph": weather_info["wind_kph"],
        }
    )

def get_weather_forecast(location, days=3):
    api_key = os.environ.get("WEATHER_API_KEY")
    url = (
    f"http://api.weatherapi.com/v1/forecast.json"
    f"?key={api_key}&q={location}&days={days}&aqi=no"
    )
    response = requests.get(url)
    data = response.json()
    if "error" in data:
        return f"Error: {data['error']['message']}"
    forecast_days = data["forecast"]["forecastday"]
    forecast_data = []

    for day in forecast_days:
        forecast_data.append(
            {
            "date": day["date"],
            "max_temp_c": day["day"]["maxtemp_c"],
            "min_temp_c": day["day"]["mintemp_c"],
            "condition": day["day"]["condition"]["text"],
            "chance_of_rain": day["day"]["daily_chance_of_rain"],
            }
        )
    return json.dumps(
            {
            "location": data["location"]["name"],
            "forecast": forecast_data,
            }
        )

weather_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "The city and state, e.g., San Francisco, CA, or a country, e.g., France"
                            ),
                         }
                    },
                    "required": ["location"],
                    },
                },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": (
                "Get the weather forecast for a location for a specific number of days"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": (
                                "The city and state, e.g., San Francisco, CA, or a country, e.g., France"
                                ),
                        },
                        "days": {
                            "type": "integer",
                            "description": "The number of days to forecast (1-10)",
                            "minimum": 1,
                            "maximum": 10,
                            },
                        },
                        "required": ["location"],
                    },
            },
    },
]

available_functions = {
    "get_current_weather": get_current_weather,
    "get_weather_forecast": get_weather_forecast,
}

calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "The mathematical expression to evaluate, e.g., '2 + 2' or '5 * (3 + 2)'"
                        ),
                    }
                },
            "required": ["expression"],
        },
    },
}
cot_system_message = """You are a helpful assistant that can answer questions
about weather and perform calculations.
When responding to complex questions, please follow these steps:
1. Think step-by-step about what information you need.
2. Break down the problem into smaller parts.
3. Use the appropriate tools to gather information.
4. Explain your reasoning clearly.
5. Provide a clear final answer.
For example, if someone asks about temperature conversions or
comparisons between cities, first get the weather data, then use the
calculator if needed, showing your work.
"""

cot_tools = weather_tools + [calculator_tool]
available_functions["calculator"] = calculator



def serialize_tool_call(tool_call):
    """Convert tool call object to dictionary."""
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments
        }
    }


advanced_tools = cot_tools
advanced_system_message = """You are a helpful weather assistant that can use
weather tools and a calculator to solve multi-step problems.
Guidelines:
1. If the user asks about several independent locations, use multiple weather
tool calls in parallel when appropriate.
2. If a question requires several steps, continue using tools until the task is
completed.
3. If a tool fails, explain the issue clearly and continue safely when possible.
4. For complex comparison or calculation queries, prepare a structured final
response.
5. For queries that require complex calculations use a calculator tool call 
6. don't implement your own mathematical calculations always use a helper function like calculator tool call
"""

required_output_keys = [
    "query_type",
    "locations",
    "summary",
    "tool_calls_used",
    "final_answer",
]

structured_output_prompt = """For complex comparison or calculation queries,
return the final answer as a valid JSON object with exactly these keys:
- query_type
- locations
- summary
- tool_calls_used
- final_answer
Do not include markdown fences.
"""
def validate_structured_output(response_text):
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON output: {str(e)}")
    for key in required_output_keys:
        if key not in parsed:
            raise ValueError(f"Missing required key: {key}")
    if not isinstance(parsed["locations"], list):
        raise ValueError("'locations' must be a list")
    if not isinstance(parsed["tool_calls_used"], list):
        raise ValueError("'tool_calls_used' must be a list")
    return parsed



def process_messages(client, messages, tools=None, available_functions=None, max_iterations=5):
    tools = tools or []
    available_functions = available_functions or {}

    for _ in range(max_iterations):
        # Step 1: Send messages to the model
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=tools,
        )

        response_message = response.choices[0].message

        # Step 2: Append the model's response
        message_dict = {
            "role": response_message.role,
            "content": response_message.content,
        }
        if response_message.tool_calls:
            message_dict["tool_calls"] = [
                serialize_tool_call(tc) for tc in response_message.tool_calls
            ]
        messages.append(message_dict)

        # Step 3: If no tool calls, the model is done — return
        if not response_message.tool_calls:
            break

        # Step 4: Execute ALL tool calls in this round
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name

            # Handle unknown tool names safely
            if function_name not in available_functions:
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": f"Error: Unknown function '{function_name}'",
                })
                continue

            function_args = json.loads(tool_call.function.arguments)
            function_response = available_functions[function_name](**function_args)

            # Step 5: Append each tool result
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })

        # Loop continues — model reads tool results and decides what to do next

    return messages

def run_conversation(client, system_message="You are a helpful weather assistant."):
    messages = [{"role": "system", "content": system_message}]
    
    print("Weather Assistant: Hello! I can help you with weather information.")
    print("Ask me about the weather anywhere!")
    print("(Type 'exit' to end the conversation)\n")
    
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        
        messages.append({"role": "user", "content": user_input})
        messages = process_messages(client, messages, cot_tools, available_functions)
        
        # Print response
        last_message = messages[-1]
        if last_message["role"] == "assistant" and last_message.get("content"):
            print(f"\nWeather Assistant: {last_message['content']}\n")
    
    return messages
  
def execute_tool_safely(tool_call, available_functions):
    
    # ── Step 1: Check the tool name exists ────────────────────────────
    function_name = tool_call.function.name
    if function_name not in available_functions:
        return json.dumps({
            "success": False,
            "error": f"Unknown function: {function_name}",
        })

    # ── Step 2: Parse the arguments ───────────────────────────────────
    try:
        function_args = json.loads(tool_call.function.arguments)
        if "days" in function_args:
            function_args["days"] = int(function_args["days"])
    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON arguments: {str(e)}",
        })

    # ── Step 3: Call the function ─────────────────────────────────────
    try:
        result = available_functions[function_name](**function_args)
        return json.dumps({
            "success": True,
            "function_name": function_name,
            "result": result,
        })
    except TypeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid arguments: {str(e)}",
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Tool execution failed: {str(e)}",
        })
   
def get_structured_final_response(client, messages):
    structured_messages = messages + [
        {
            "role": "system",
            "content": structured_output_prompt,
        }
    ]
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=structured_messages,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content

    return validate_structured_output(content)

def execute_tools_sequential(tool_calls, available_functions):
    results = []
    
    for tool_call in tool_calls:
        safe_result = execute_tool_safely(tool_call, available_functions)
        
        tool_message = {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": tool_call.function.name,
            "content": safe_result,
        }
        results.append(tool_message)
    return results

def execute_tools_parallel(tool_calls, available_functions, max_workers=4):
    def run_single_tool(tool_call):
        return {
            "tool_call_id":tool_call.id,
            "role":"tool",
            "name":tool_call.function.name,
            "content": execute_tool_safely(tool_call,available_functions)
        }
    with ThreadPoolExecutor(
        max_workers=min(max_workers,len(tool_calls)),
    ) as executor:
        return list(executor.map(run_single_tool,tool_calls)) 

def compare_parallel_vs_sequential(tool_calls, available_functions):
    start = time.perf_counter()
    sequential_results = execute_tools_sequential(tool_calls,available_functions)
    sequential_time = time.perf_counter() - start
    start = time.perf_counter()
    parallel_results = execute_tools_parallel(tool_calls,available_functions)
    parallel_time = time.perf_counter() - start
    speedup = (
    sequential_time / parallel_time if parallel_time > 0 else None
    )
    return {
    "sequential_results": sequential_results,
    "parallel_results": parallel_results,
    "sequential_time": sequential_time,
    "parallel_time": parallel_time,
    "speedup": speedup,
    }


def process_messages_advanced(client, messages, tools=None, available_functions=None):
    """Send messages to the model and execute any returned tools in parallel."""
    tools = tools or []
    available_functions = available_functions or {}

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=tools,
    )

    response_message = response.choices[0].message

    # Append model response
    message_dict = {
        "role": response_message.role,
        "content": response_message.content,
    }
    if response_message.tool_calls:
        message_dict["tool_calls"] = [
            serialize_tool_call(tc) for tc in response_message.tool_calls
        ]
    messages.append(message_dict)

    # Execute tool calls in parallel if any
    if response_message.tool_calls:
        tool_results = execute_tools_parallel(
            response_message.tool_calls,
            available_functions,
        )
        messages.extend(tool_results)

    return messages, response_message

def run_conversation_advanced(client, system_message=advanced_system_message, max_iterations=5):
    messages = [{"role": "system", "content": system_message}]

    print("Advanced Weather Assistant: Hello! Ask me complex weather questions.")
    print("I can compare cities, perform calculations, and return structured outputs.")
    print("(Type 'exit' to end the conversation)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nAdvanced Weather Assistant: Goodbye! Have a great day!")
            break

        messages.append({"role": "user", "content": user_input})

        # Multi-step loop
        for _ in range(max_iterations):
            messages, response_message = process_messages_advanced(
                client, messages, advanced_tools, available_functions
            )

            # If no tool calls, model is done — print answer and break
            if not response_message.tool_calls:
                if response_message.content:
                    print(f"\nAdvanced Weather Assistant: {response_message.content}\n")
                    try:
                        structured = get_structured_final_response(client, messages)
                        print("\n--- Structured Output ---")
                        print(json.dumps(structured, indent=2))
                        print("-------------------------\n")
                    except (ValueError, Exception) as e:
                        print(f"(Structured output not available: {e})\n")
                break
        else:
            print("\nAdvanced Weather Assistant: I stopped after reaching the"
                  " maximum number of tool iterations.\n")

    return messages


TEST_PROMPTS = [
    "Compare the current weather in Cairo, Riyadh, and London.",
    "Which city is warmer right now: Paris, Rome, or Berlin?",
    "Give me a short comparison of the weather in Alexandria, Aswan, and Dubai.",
]


if __name__ == "__main__":
    
    choice = input(
    "Choose an agent type (1: Basic, 2: Chain of Thought, 3: Advanced 4: Parallel vs Sequential test): "
    )
    if choice == "1":
        run_conversation(client, "You are a helpful weather assistant.")
    elif choice == "2":
        run_conversation(client, cot_system_message)
    elif choice == "3":
        run_conversation_advanced(client, advanced_system_message)

    elif choice == "4":
        for prompt in TEST_PROMPTS:
            messages = [
                {"role": "system", "content": cot_system_message},
                {"role": "user",   "content": prompt},
            ]
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    tools=advanced_tools,
                )
            except Exception as e:
                print(f"API call failed: {e}")
                continue

            message = response.choices[0].message
            if message.tool_calls:
                result = compare_parallel_vs_sequential(message.tool_calls, available_functions)
                seq_results = result["sequential_results"]
                par_results = result["parallel_results"]
                seq_time    = result["sequential_time"]
                par_time    = result["parallel_time"]
                speedup     = result["speedup"]

                print(f"\nresults for seq: {seq_results}")
                print(f"\nresults for parallel: {par_results}")
                print(f"\nresults for seq time: {seq_time}")
                print(f"\nresults for parallel time: {par_time}")
                print(f"results for speedup: {speedup}")

    else:
        print("Invalid choice. Defaulting to Basic agent.")
        run_conversation(client, "You are a helpful weather assistant.")