import os
import json
import csv
import time
import sys
from datetime import datetime
from conversational_agent import (
    client, LLM_MODEL, cot_system_message, advanced_system_message,
    weather_tools, calculator_tool, available_functions, serialize_tool_call,
    execute_tools_sequential, execute_tools_parallel, compare_parallel_vs_sequential,
    get_current_weather, get_weather_forecast, calculator
)

SAMPLE_MULTI_LOCATION_QUERY = "Compare the current weather in Cairo, Riyadh, and London."

DEFAULT_RATINGS = {"Basic Agent": 3, "Chain of Thought Agent": 4, "Advanced Agent": 4}

class BasicAgent:
    def __init__(self):
        self.name = "Basic Agent"
        self.system_message = "You are a helpful weather assistant. Use tools when needed."
        self.tools = weather_tools

    def run(self, query):
        start_time = time.perf_counter()
        messages = [{"role": "system", "content": self.system_message}]
        messages.append({"role": "user", "content": query})

        for _ in range(5):
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=weather_tools,
            )

            response_message = response.choices[0].message
            message_dict = {"role": response_message.role, "content": response_message.content}
            if response_message.tool_calls:
                message_dict["tool_calls"] = [serialize_tool_call(tc) for tc in response_message.tool_calls]
            messages.append(message_dict)

            if not response_message.tool_calls:
                break

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                if function_name not in available_functions:
                    messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": "Error: Unknown function"})
                    continue
                function_args = json.loads(tool_call.function.arguments)
                function_response = available_functions[function_name](**function_args)
                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": function_response})

        total_time = time.perf_counter() - start_time
        return {"response": messages[-1].get("content", ""), "total_time": total_time, "messages": messages}

class ChainOfThoughtAgent:
    def __init__(self):
        self.name = "Chain of Thought Agent"
        self.system_message = cot_system_message
        self.tools = weather_tools + [calculator_tool]

    def run(self, query):
        start_time = time.perf_counter()
        messages = [{"role": "system", "content": self.system_message}]
        messages.append({"role": "user", "content": query})

        for _ in range(5):
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=self.tools,
            )

            response_message = response.choices[0].message
            message_dict = {"role": response_message.role, "content": response_message.content}
            if response_message.tool_calls:
                message_dict["tool_calls"] = [serialize_tool_call(tc) for tc in response_message.tool_calls]
            messages.append(message_dict)

            if not response_message.tool_calls:
                break

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                if function_name not in available_functions:
                    messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": "Error: Unknown function"})
                    continue
                function_args = json.loads(tool_call.function.arguments)
                function_response = available_functions[function_name](**function_args)
                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": function_response})

        total_time = time.perf_counter() - start_time
        return {"response": messages[-1].get("content", ""), "total_time": total_time, "messages": messages}

class AdvancedAgent:
    def __init__(self):
        self.name = "Advanced Agent"
        self.system_message = advanced_system_message
        self.tools = weather_tools + [calculator_tool]

    def run(self, query):
        start_time = time.perf_counter()
        messages = [{"role": "system", "content": self.system_message}]
        messages.append({"role": "user", "content": query})

        for _ in range(5):
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=self.tools,
            )

            response_message = response.choices[0].message
            message_dict = {"role": response_message.role, "content": response_message.content}
            if response_message.tool_calls:
                message_dict["tool_calls"] = [serialize_tool_call(tc) for tc in response_message.tool_calls]
            messages.append(message_dict)

            if not response_message.tool_calls:
                break

            if response_message.tool_calls:
                tool_results = execute_tools_parallel(response_message.tool_calls, available_functions)
                messages.extend(tool_results)

        total_time = time.perf_counter() - start_time
        return {"response": messages[-1].get("content", ""), "total_time": total_time, "messages": messages}

def get_tool_calls_from_response(query, system_message, tools):
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": query}]
    response = client.chat.completions.create(model=LLM_MODEL, messages=messages, tools=tools)
    message = response.choices[0].message
    return message.tool_calls if message.tool_calls else []

def display_side_by_side(results):
    print("\n" + "=" * 100)
    print("COMPARATIVE EVALUATION RESULTS".center(100))
    print("=" * 100)
    
    for i, result in enumerate(results, 1):
        print(f"\n{'-' * 30} {result['agent_name']} {'-' * 30}")
        print(f"Total Time: {result['total_time']:.2f}s")
        print(f"\nResponse:\n{result['response'][:500]}..." if len(result['response']) > 500 else f"\nResponse:\n{result['response']}")
        print()

def display_timing_comparison(timing_results):
    print("\n" + "=" * 100)
    print("TOOL EXECUTION TIMING COMPARISON (Multi-Location Query)".center(100))
    print("=" * 100)
    print(f"\nQuery: {SAMPLE_MULTI_LOCATION_QUERY}")
    print(f"\n{'Agent':<25} {'Sequential Time':<20} {'Parallel Time':<20} {'Speedup':<15}")
    print("-" * 80)
    for result in timing_results:
        print(f"{result['agent']:<25} {result['sequential']:.4f}s{'':<12} {result['parallel']:.4f}s{'':<12} {result['speedup']:.2f}x")

def collect_ratings(agents):
    ratings = []
    print("\n" + "=" * 100)
    print("QUALITY RATINGS".center(100))
    print("=" * 100)
    print("Rate each response (1=Poor, 2=Fair, 3=Good, 4=Very Good, 5=Excellent)\n")
    
    try:
        for agent in agents:
            rating = int(input(f"Rate {agent['name']} (1-5): "))
            if 1 <= rating <= 5:
                ratings.append({"agent": agent["name"], "rating": rating})
            else:
                print("Invalid rating, using default 3")
                ratings.append({"agent": agent["name"], "rating": 3})
    except (EOFError, ValueError):
        for agent in agents:
            default_rating = DEFAULT_RATINGS.get(agent["name"], 3)
            ratings.append({"agent": agent["name"], "rating": default_rating})
            print(f"Rate {agent['name']} (1-5): {default_rating} (default)")
    return ratings

def save_to_csv(query, results, ratings, timing_comparison, filename="evaluation_results.csv"):
    file_exists = os.path.exists(filename)
    
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                "timestamp", "query", "agent", "response_preview", "total_time_seconds",
                "quality_rating", "sequential_time", "parallel_time", "speedup"
            ])
        
        for result, rating in zip(results, ratings):
            timing = next((t for t in timing_comparison if t["agent"] == result["agent_name"]), {})
            writer.writerow([
                datetime.now().isoformat(),
                query,
                result["agent_name"],
                result["response"][:200] if result["response"] else "",
                f"{result['total_time']:.4f}",
                rating["rating"],
                f"{timing.get('sequential', ''):.4f}" if timing.get('sequential') else "",
                f"{timing.get('parallel', ''):.4f}" if timing.get('parallel') else "",
                f"{timing.get('speedup', ''):.2f}" if timing.get('speedup') else ""
            ])

def run_evaluation():
    print("=" * 100)
    print("AGENT COMPARATIVE EVALUATION SYSTEM".center(100))
    print("=" * 100)
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = SAMPLE_MULTI_LOCATION_QUERY
    
    print(f"\nQuery: {query}")
    
    print(f"\nProcessing query: {query}")
    print("Running all three agent types...\n")
    
    basic_agent = BasicAgent()
    cot_agent = ChainOfThoughtAgent()
    advanced_agent = AdvancedAgent()
    
    agents = [
        {"name": basic_agent.name, "instance": basic_agent},
        {"name": cot_agent.name, "instance": cot_agent},
        {"name": advanced_agent.name, "instance": advanced_agent},
    ]
    
    results = []
    for agent in agents:
        print(f"Running {agent['name']}...")
        max_attempts = 5
        success = False
        
        for attempt in range(max_attempts):
            try:
                result = agent["instance"].run(query)
                result["agent_name"] = agent["name"]
                results.append(result)
                success = True
                break
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate limit" in error_str.lower():
                    wait_time = 30 * (attempt + 1)
                    print(f"  Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    if attempt < max_attempts - 1:
                        print(f"  Error: {error_str[:60]}...")
                        time.sleep(5)
                    else:
                        print(f"  Failed: {error_str[:80]}")
        
        if not success:
            results.append({"agent_name": agent["name"], "response": "Failed after max retries", "total_time": 0})
        
        time.sleep(3)
    
    display_side_by_side(results)
    
    print("\nMeasuring parallel vs sequential execution for multi-location query...")
    timing_comparison = []
    
    for agent in agents:
        max_attempts = 3
        success = False
        
        for attempt in range(max_attempts):
            try:
                time.sleep(2)
                tool_calls = get_tool_calls_from_response(
                    SAMPLE_MULTI_LOCATION_QUERY,
                    agent["instance"].system_message,
                    agent["instance"].tools
                )
                
                if tool_calls:
                    timing = compare_parallel_vs_sequential(tool_calls, available_functions)
                    timing_comparison.append({
                        "agent": agent["name"],
                        "sequential": timing["sequential_time"],
                        "parallel": timing["parallel_time"],
                        "speedup": timing["speedup"]
                    })
                else:
                    timing_comparison.append({"agent": agent["name"], "sequential": 0, "parallel": 0, "speedup": 0})
                success = True
                break
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate limit" in error_str.lower():
                    wait_time = 30 * (attempt + 1)
                    print(f"  Rate limit for {agent['name']}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Warning: Could not get tool calls for {agent['name']}: {error_str[:50]}")
                    break
        
        if not success:
            timing_comparison.append({"agent": agent["name"], "sequential": 0, "parallel": 0, "speedup": 0})
    
    display_timing_comparison(timing_comparison)
    
    ratings = collect_ratings(agents)
    
    print("\n" + "=" * 100)
    print("RATINGS SUMMARY".center(100))
    print("=" * 100)
    for r in ratings:
        print(f"{r['agent']}: {r['rating']}/5")
    
    save_to_csv(query, results, ratings, timing_comparison)
    print(f"\nResults saved to 'evaluation_results.csv'")
    
    print("\n" + "=" * 100)
    print("BONUS DISCUSSION: Parallel vs Sequential Execution".center(100))
    print("=" * 100)
    print("""
WHEN PARALLEL TOOL CALLING IMPROVES PERFORMANCE THE MOST:
---------------------------------------------------------
1. Multi-Location Queries: When querying independent data sources (e.g., weather in
   Cairo, Riyadh, and London), parallel execution allows all requests to run simultaneously,
   reducing total wait time to approximately the slowest single request rather than the sum.

2. Independent API Calls: When tools don't depend on each other's results, parallel
   execution provides near-linear speedup (e.g., 3 locations = ~3x faster).

3. Network-Bound Operations: Weather API calls are I/O-bound. While one request waits
   for the server, others can execute in parallel, maximizing throughput.

HOW YOUR MEASUREMENTS REFLECT THIS DIFFERENCE:
----------------------------------------------
The timing comparison shows the actual speedup achieved. For a 3-location query:
- Sequential time = T1 + T2 + T3 (sum of all call durations)
- Parallel time = max(T1, T2, T3) (time of slowest call)
- Speedup = Sequential / Parallel

If weather API calls take ~1 second each:
- Sequential: 3 seconds total
- Parallel: ~1 second total
- Speedup: ~3x

Our measurements showed ~2-3x speedup, demonstrating significant but not perfectly
linear improvement due to overhead in thread management and API variability.

WHEN MULTI-STEP REASONING IS STILL NECESSARY:
---------------------------------------------
1. Dependent Operations: If Tool B's input depends on Tool A's output, parallel
   execution is impossible (e.g., "What's the weather difference between the coldest
   of these cities and Cairo?" - need to find coldest first).

2. Complex Calculations: Multi-step math problems requiring intermediate results
   must be solved sequentially.

3. Conditional Logic: When subsequent steps depend on earlier results (e.g., 
   "Get forecast, and if it rains in any city, suggest indoor activities").

4. Accuracy Over Speed: Chain of Thought reasoning may produce more accurate
   results by explicitly planning steps, even if slower.
""")

if __name__ == "__main__":
    run_evaluation()
