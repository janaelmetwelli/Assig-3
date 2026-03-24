# CSAI 422 — Assignment 3: Conversational Agent with Tool Use

## Table of Contents

1. [Project Overview](#project-overview)
2. [Setup Instructions](#setup-instructions)
3. [Implementation Documentation](#implementation-documentation)
4. [Example Conversations](#example-conversations)
5. [Analysis: Reasoning & Orchestration Strategies](#analysis)
6. [Challenges & Solutions](#challenges--solutions)
7. [Bonus: Evaluation System](#bonus-evaluation-system)

---

## Project Overview

This project implements a multi-mode conversational weather assistant that demonstrates **LLM tool calling**, **chain-of-thought reasoning**, **parallel tool execution**, and **structured output validation**. The assistant is powered by **LLaMA 3.3 70B** via the Groq API and supports four distinct operating modes of increasing complexity.

**Core capabilities:**
- Real-time weather lookup (current conditions and multi-day forecasts) via WeatherAPI
- Safe sandboxed mathematical calculator
- Three agent reasoning modes: Basic, Chain-of-Thought, and Advanced
- Parallel vs. sequential tool execution benchmarking

---

## Setup Instructions

### Prerequisites

- Python 3.12 or higher
- A Groq API key — obtain at [https://console.groq.com](https://console.groq.com)
- A WeatherAPI key — obtain at [https://www.weatherapi.com](https://www.weatherapi.com)

### 1. Install Dependencies

```bash
pip install openai python-dotenv requests
```

### 2. Configure Environment Variables

Create a `.env` file in the same directory as `conversational_agent.py`:

```
GROQ_API_KEY=your_groq_api_key_here
WEATHER_API_KEY=your_weatherapi_key_here
```

### 3. Run the Script

```bash
python conversational_agent.py
```

You will be prompted to select an agent mode:

```
Choose an agent type (1: Basic, 2: Chain of Thought, 3: Advanced, 4: Parallel vs Sequential test):
```

Type `exit`, `quit`, or `bye` at any time to end a conversation session.

---

## Implementation Documentation

### File Structure

```
conversational_agent.py       # Single-file implementation containing all agents
.env                          # API keys (not committed to version control)
```

### Key Components

#### Tool Definitions

Two weather tools and one calculator tool are registered as JSON schemas following the OpenAI function-calling format:

| Tool | Description |
|---|---|
| `get_current_weather(location)` | Fetches live weather conditions for a city |
| `get_weather_forecast(location, days)` | Fetches a 1–10 day forecast |
| `calculator(expression)` | Safely evaluates a mathematical expression |

The calculator uses a sandboxed `eval()` with a whitelist (`SAFE_LOCALS`) containing only safe math functions (`sqrt`, `sin`, `cos`, `log`, etc.) and explicitly sets `__builtins__` to `{}` to prevent code injection.

#### Agent 1 — Basic Agent (`run_conversation`)

A straightforward conversational loop backed by `process_messages()`. Maintains full message history across turns for multi-turn context. Uses `cot_tools` (weather + calculator). The agentic loop runs up to 5 iterations per query, executing tool calls and feeding results back to the model until it produces a final answer with no further tool calls.

#### Agent 2 — Chain-of-Thought Agent (`run_conversation` with `cot_system_message`)

Identical architecture to the Basic agent but uses a structured system prompt that instructs the model to:
1. Think step-by-step about what information is needed
2. Break the problem into smaller parts
3. Use tools to gather data
4. Explain its reasoning before giving a final answer

This produces more transparent, traceable responses for complex queries.

#### Agent 3 — Advanced Agent (`run_conversation_advanced`)

The most capable mode. Key differences from the Basic agent:

- **Parallel tool execution** — uses `execute_tools_parallel()` with `ThreadPoolExecutor` so multiple weather API calls for different cities run simultaneously instead of sequentially.
- **Multi-step loop** — iterates up to 5 rounds of tool use per query, allowing the model to chain dependent tool calls.
- **Structured output** — after the final answer, `get_structured_final_response()` makes a second API call with `response_format={"type": "json_object"}` to produce a validated JSON response with the keys: `query_type`, `locations`, `summary`, `tool_calls_used`, and `final_answer`. The output is validated by `validate_structured_output()` before being printed.

#### Mode 4 — Parallel vs. Sequential Benchmark (`compare_parallel_vs_sequential`)

Runs three multi-city comparison prompts automatically. For each prompt, the model's tool calls are executed both sequentially (one after another) and in parallel (all at once using `ThreadPoolExecutor`), with `time.perf_counter()` measuring wall-clock time for each approach. The speedup ratio is printed for each prompt.

---

## Example Conversations

### Agent 1 — Basic Agent

```
Weather Assistant: Hello! I can help you with weather information.
Ask me about the weather anywhere!
(Type 'exit' to end the conversation)

You: What is the weather like in Cairo right now?

Weather Assistant: The current weather in Cairo is mostly clear with a temperature of 15.2°C (59.4°F) and a humidity level of 59%. The wind speed is 18.7 km/h.



You: what is the weather in cairo and paris 

Weather Assistant: The current weather in Cairo is mostly clear with a temperature of 15.2°C (59.4°F) and a humidity level of 59%. The wind speed is 18.7 km/h.

```

---

### Agent 2 — Chain-of-Thought Agent

```
Weather Assistant: Hello! I can help you with weather information.
Ask me about the weather anywhere!
(Type 'exit' to end the conversation)

You: What is the temperature difference between Cairo and London right now?

Weather Assistant: The current temperature in Cairo is 13.0°C (55.4°F) and in London is 5.4°C (41.7°F). The temperature difference between Cairo and London right now is 7.6°C (13.7°F).

You: hat is 15% of 37.5?

Weather Assistant: 15% of 37.5 is 5.625.

You: What is the current temperature in Paris in Fahrenheit?

Weather Assistant: The current temperature in Paris is 46.9°F.

You: Which is warmer right now, Riyadh or Dubai, and by how many degrees?

Weather Assistant: The current temperature in Riyadh is 22.3°C (72.1°F) and in Dubai is 25.2°C (77.3°F). Dubai is warmer than Riyadh by 2.9°C (5.2°F).
```

---

### Agent 3 — Advanced Agent

```
You: Give me a short comparison of the weather in Alexandria, Aswan, and Dubai.

Advanced Weather Assistant: Based on the current weather, the order from warmest to coolest is:

1. Aswan, Egypt (24.9°C/76.7°F)
2. Dubai, UAE (23.8°C/74.9°F)
3. Alexandria, Egypt (15.1°C/59.2°F)

Please note that this comparison is based on a single snapshot of the current weather and does not account for any changes that may occur over time.


--- Structured Output ---
{
  "query_type": "weather_comparison",
  "locations": [
    "Alexandria",
    "Aswan",
    "Dubai"
  ],
  "summary": "Comparison of current weather in Alexandria, Aswan, and Dubai",
  "tool_calls_used": [
    "get_current_weather(location='Alexandria')",
    "get_current_weather(location='Aswan')",
    "get_current_weather(location='Dubai')"
  ],
  "final_answer": {
    "order": [
      "Aswan",
      "Dubai",
      "Alexandria"
    ],
    "temperature": [
      24.9,
      23.8,
      15.1
    ],
    "units": "C"
  }
}
-------------------------


---

### Mode 4 — Parallel vs. Sequential Benchmark

```
 results for seq: [{'tool_call_id': '03tv54wz6', 'role': 'tool', 'name': 'get_current_weather', 'content': '{"success": true, "function_name": "get_current_weather", "result": "{\\"location\\": \\"Cairo\\", \\"temperature_c\\": 19.4, \\"temperature_f\\": 66.9, \\"condition\\": \\"Partly cloudy\\", \\"humidity\\": 49, \\"wind_kph\\": 23.8}"}'}, {'tool_call_id': 'z3jhem0am', 'role': 'tool', 'name': 'get_current_weather', 'content': '{"success": true, "function_name": "get_current_weather", "result": "{\\"location\\": \\"Riyadh\\", \\"temperature_c\\": 20.2, \\"temperature_f\\": 68.4, \\"condition\\": \\"Moderate or heavy rain with thunder\\", \\"humidity\\": 78, \\"wind_kph\\": 11.9}"}'}, {'tool_call_id': 'yfpdcnqjn', 'role': 'tool', 'name': 'get_current_weather', 'content': '{"success": true, "function_name": "get_current_weather", "result": "{\\"location\\": \\"London\\", \\"temperature_c\\": 12.1, \\"temperature_f\\": 53.8, \\"condition\\": \\"Partly cloudy\\", \\"humidity\\": 82, \\"wind_kph\\": 30.2}"}'}]

 results for parallel: [{'tool_call_id': '03tv54wz6', 'role': 'tool', 'name': 'get_current_weather', 'content': '{"success": true, "function_name": "get_current_weather", "result": "{\\"location\\": \\"Cairo\\", \\"temperature_c\\": 19.4, \\"temperature_f\\": 66.9, \\"condition\\": \\"Partly cloudy\\", \\"humidity\\": 49, \\"wind_kph\\": 23.8}"}'}, {'tool_call_id': 'z3jhem0am', 'role': 'tool', 'name': 'get_current_weather', 'content': '{"success": true, "function_name": "get_current_weather", "result": "{\\"location\\": \\"Riyadh\\", \\"temperature_c\\": 20.2, \\"temperature_f\\": 68.4, \\"condition\\": \\"Moderate or heavy rain with thunder\\", \\"humidity\\": 78, \\"wind_kph\\": 11.9}"}'}, {'tool_call_id': 'yfpdcnqjn', 'role': 'tool', 'name': 'get_current_weather', 'content': '{"success": true, "function_name": "get_current_weather", "result": "{\\"location\\": \\"London\\", \\"temperature_c\\": 12.1, \\"temperature_f\\": 53.8, \\"condition\\": \\"Partly cloudy\\", \\"humidity\\": 82, \\"wind_kph\\": 30.2}"}'}]

 results for seq time: 0.4764696001075208

 results for parallel time: 0.14573190035298467
results for speedup : 3.26949418043297

 results for seq: [{'tool_call_id': '33tdh77n7', 'role': 'tool', 'name': 'get_current_weather', 'content': '{"success": true, "function_name": "get_current_weather", "result": "{\\"location\\": \\"Paris\\", \\"temperature_c\\": 17.1, \\"temperature_f\\": 62.8, \\"condition\\": \\"Overcast\\", \\"humidity\\": 45, \\"wind_kph\\": 18.4}"}'}, {'tool_call_id': '8zy35a807', 'role': 'tool', 'name': 'get_current_weather', 'content': '{"success": true, "function_name": "get_current_weather", "result": "{\\"location\\": \\"Rome\\", \\"temperature_c\\": 15.1, \\"temperature_f\\": 59.2, \\"condition\\": \\"Partly Cloudy\\", \\"humidity\\": 72, \\"wind_kph\\": 9.4}"}'}, {'tool_call_id': 'gbet9cq3a', 'role': 'tool', 'name': 'get_current_weather', 'content': '{"success": true, "function_name": "get_current_weather", "result": "{\\"location\\": \\"Berlin\\", \\"temperature_c\\": 14.3, \\"temperature_f\\": 57.7, \\"condition\\": \\"Cloudy\\", \\"humidity\\": 47, \\"wind_kph\\": 16.6}"}'}]

 results for parallel: [{'tool_call_id': '33tdh77n7', 'role': 'tool', 'name': 'get_current_weather', 'content': '{"success": true, "function_name": "get_current_weather", "result": "{\\"location\\": \\"Paris\\", \\"temperature_c\\": 17.1, \\"temperature_f\\": 62.8, \\"condition\\": \\"Overcast\\", \\"humidity\\": 45, \\"wind_kph\\": 18.4}"}'}, {'tool_call_id': '8zy35a807', 'role': 'tool', 'name': 'get_current_weather', 'content': '{"success": true, "function_name": "get_current_weather", "result": "{\\"location\\": \\"Rome\\", \\"temperature_c\\": 15.1, \\"temperature_f\\": 59.2, \\"condition\\": \\"Partly Cloudy\\", \\"humidity\\": 72, \\"wind_kph\\": 9.4}"}'}, {'tool_call_id': 'gbet9cq3a', 'role': 'tool', 'name': 'get_current_weather', 'content': '{"success": true, "function_name": "get_current_weather", "result": "{\\"location\\": \\"Berlin\\", \\"temperature_c\\": 14.3, \\"temperature_f\\": 57.7, \\"condition\\": \\"Cloudy\\", \\"humidity\\": 47, \\"wind_kph\\": 16.6}"}'}]

 results for seq time: 0.4657596000470221

 results for parallel time: 0.15678669977933168
results for speedup : 2.9706575921462224

 results for seq: [{'tool_call_id': 'esenbmgw8', 'role': 'tool', 'name': 'get_current_weather', 'content': '{"success": true, "function_name": "get_current_weather", "result": "{\\"location\\": \\"Alexandria\\", \\"temperature_c\\": 16.0, \\"temperature_f\\": 60.8, \\"condition\\": \\"Partly cloudy\\", \\"humidity\\": 68, \\"wind_kph\\": 17.3}"}'}, {'tool_call_id': '3nwdqqp2e', 'role': 'tool', 'name': 'get_current_weather', 'content': '{"success": true, "function_name": "get_current_weather", "result": "{\\"location\\": \\"Aswan\\", \\"temperature_c\\": 27.9, \\"temperature_f\\": 82.2, \\"condition\\": \\"Clear\\", \\"humidity\\": 20, \\"wind_kph\\": 25.9}"}'}, {'tool_call_id': '3b2he6ee1', 'role': 'tool', 'name': 'get_current_weather', 'content': '{"success": true, "function_name": "get_current_weather", "result": "{\\"location\\": \\"Dubai\\", \\"temperature_c\\": 23.9, \\"temperature_f\\": 75.0, \\"condition\\": \\"Clear\\", \\"humidity\\": 71, \\"wind_kph\\": 4.7}"}'}]

 results for parallel: [{'tool_call_id': 'esenbmgw8', 'role': 'tool', 'name': 'get_current_weather', 'content': '{"success": true, "function_name": "get_current_weather", "result": "{\\"location\\": \\"Alexandria\\", \\"temperature_c\\": 16.0, \\"temperature_f\\": 60.8, \\"condition\\": \\"Partly cloudy\\", \\"humidity\\": 68, \\"wind_kph\\": 17.3}"}'}, {'tool_call_id': '3nwdqqp2e', 'role': 'tool', 'name': 'get_current_weather', 'content': '{"success": true, "function_name": "get_current_weather", "result": "{\\"location\\": \\"Aswan\\", \\"temperature_c\\": 27.9, \\"temperature_f\\": 82.2, \\"condition\\": \\"Clear\\", \\"humidity\\": 20, \\"wind_kph\\": 25.9}"}'}, {'tool_call_id': '3b2he6ee1', 'role': 'tool', 'name': 'get_current_weather', 'content': '{"success": true, "function_name": "get_current_weather", "result": "{\\"location\\": \\"Dubai\\", \\"temperature_c\\": 23.9, \\"temperature_f\\": 75.0, \\"condition\\": \\"Clear\\", \\"humidity\\": 71, \\"wind_kph\\": 4.7}"}'}]

 results for seq time: 0.5905977999791503

 results for parallel time: 0.391584400087595
results for speedup : 1.5082260678592847

```

---

## Analysis

### How Reasoning Strategies Affected Response Quality

**Basic Agent** produced correct answers for simple, single-step queries (e.g., "What is the weather in Cairo?") but struggled with multi-step questions. When asked to compare two cities and calculate a difference, it would generate incorrect answers so it was unable to calcuate mathemitical values correclty until we added the calc tool function

**Chain-of-Thought Agent** produced significantly more reliable results for complex queries. By instructing the model to reason step-by-step, the model generated more reliable reesults espicaially for calculations and showed step by step it's throught process.
**Advanced Agent** handled the most demanding queries — comparing three or more cities simultaneously, chaining tool calls across multiple iterations, and producing machine-readable structured output. The structured JSON output is particularly useful for downstream applications, as the response can be parsed programmatically rather than requiring further text extraction.

### How Orchestration Strategies Affected Performance

**Sequential execution** processes each tool call one at a time. For a query comparing three cities, this means three separate HTTP requests to the weather API happen back and forth, making total latency roughly 3× a single request, which obviously makes it inefficient for real time application

**Parallel execution** using `ThreadPoolExecutor` dispatches all tool calls simultaneously. this makes it 
much more efficent the time diff between the parallel and sequential excution differed completely
as the parallel showed a much faster preformnace and still achieving the same quality of results 

### Trade-offs Summary

| Strategy | Response Quality | Latency | Transparency |
|---|---|---|---|
| Basic Agent | Adequate for simple queries | Fast | Low |
| Chain-of-Thought | High for complex queries | Moderate | High |
| Advanced + Parallel | Highest | Low (parallel) | High + structured |

---

## Challenges & Solutions

### Challenge 1 — Preventing Unsafe Calculator Expressions

The `calculator` tool uses Python's `eval()` function, which can execute arbitrary code if not constrained. Passing user-provided strings directly to `eval()` would be a serious security risk.

**Solution:** The calculator uses a restricted execution environment: `eval(expression, {"__builtins__": {}}, SAFE_LOCALS)`. Setting `__builtins__` to an empty dictionary removes access to all built-in Python functions (like `open`, `exec`, `import`). `SAFE_LOCALS` is a hand-curated whitelist containing only safe mathematical functions from the `math` module. Additional guards catch division by zero, `NaN`, `inf`, and invalid syntax before they reach the caller.

### Challenge 2 — Structured Output Validation Failures

The `get_structured_final_response()` function instructs the model to return a JSON object with five required keys. In early testing, the model occasionally returned JSON missing one of the required keys (e.g., omitting `tool_calls_used` for simple queries) or returned `locations` as a string instead of a list.

**Solution:** The `validate_structured_output()` function checks for all required keys and verifies the types of `locations` and `tool_calls_used`. The call is wrapped in a `try/except` block in `run_conversation_advanced()`, so if validation fails, the assistant prints the plain-text answer and a graceful fallback message rather than crashing.

### Challenge 3 — Agentic Loop Termination

Early versions of the multi-step loop did not have an iteration limit, creating a risk of the model calling tools indefinitely if it kept generating new tool calls without reaching a final answer.

**Solution:** Both `process_messages()` and `run_conversation_advanced()` enforce a `max_iterations=5` limit. The loop breaks as soon as the model returns a message with no tool calls, and if the limit is hit, a clear message is printed to the user explaining that the maximum number of iterations was reached.

### Challenge 4 — Function Definition Order

When `get_structured_final_response()` was moved earlier in the file to be defined before `run_conversation_advanced()`, a stray leading space was accidentally introduced before the `def` keyword on that line. This caused a Python `IndentationError` that prevented the entire script from running.

**Solution:** The indentation was corrected so the function is defined at column 0 (top-level scope). The fix was verified by parsing the file with Python's `ast` module before submission.
### Challenge 5 - The running out of free tokens provided by groq and OpenAI
first time ran out of openAI so i switched to working with groq 
when running with groq i ran out of free tokens so i was forced until it reset 
---

## Bonus: Evaluation System

### Overview

`evaluation_system.py` implements a structured comparative evaluation framework that automatically runs all three agents on the same query, measures their performance, collects quality ratings, saves results to CSV, and prints a discussion of parallel vs. sequential execution.

### How to Run

```bash
# Run with the default multi-location query
python evaluation_system.py

# Run with a custom query
python evaluation_system.py Compare the weather in Tokyo, Sydney, and New York
```

### File Structure

```
evaluation_system.py        # Evaluation framework
evaluation_results.csv      # Auto-generated results log (appended on each run)
```

### Implementation Details

The evaluation system defines three agent classes — `BasicAgent`, `ChainOfThoughtAgent`, and `AdvancedAgent` — each wrapping its own agentic loop. The key architectural difference is in `AdvancedAgent.run()`, which uses `execute_tools_parallel()` from the main module instead of the sequential loop used by the other two agents.

Each run:
1. Executes all three agents on the same query with up to 5 retry attempts per agent (with exponential backoff on rate limit errors)
2. Displays responses side by side using `display_side_by_side()`
3. Measures parallel vs. sequential tool execution timing for each agent using `compare_parallel_vs_sequential()`
4. Collects a 1–5 quality rating from the user (or applies defaults if running non-interactively)
5. Appends all results to `evaluation_results.csv` with timestamps

### Evaluation Results (from `evaluation_results.csv`)

The evaluation was run across three distinct queries:
- `"Compare the current weather in Cairo, Riyadh, and London."` (run multiple times)
- `"What is the weather in Paris?"`
- `"Compare weather in Paris and London"`

**Aggregated results across all runs:**

| Agent | Avg Response Time | Avg Quality Rating | Avg Tool Speedup |
|---|---|---|---|
| Basic Agent | 1.983s | 2.60 / 5 | 2.72x |
| Chain of Thought Agent | 0.932s | 3.80 / 5 | 2.74x |
| Advanced Agent | 0.822s | 4.00 / 5 | 1.61x |

**Sample raw data from `evaluation_results.csv`:**

| Query | Agent | Response Time | Rating | Sequential | Parallel | Speedup |
|---|---|---|---|---|---|---|
| Cairo, Riyadh, London | Basic Agent | 1.98s | 3 | 0.524s | 0.156s | 3.35x |
| Cairo, Riyadh, London | Chain of Thought | 1.29s | 4 | 0.421s | 0.155s | 2.72x |
| Cairo, Riyadh, London | Advanced Agent | 0.96s | 4 | 0.148s | 0.138s | 1.07x |
| What is the weather in Paris? | Basic Agent | 1.80s | 3 | 0.945s | 0.367s | 2.57x |
| What is the weather in Paris? | Chain of Thought | 0.68s | 4 | 0.283s | 0.210s | 1.35x |
| What is the weather in Paris? | Advanced Agent | 0.79s | 4 | 0.338s | 0.173s | 1.95x |

### Analysis of Results

**Response time:** The Advanced Agent was the fastest on average (0.822s) due to parallel tool execution collapsing multi-city API call latency. The Basic Agent was the slowest (1.983s) because it executes tool calls sequentially. Interestingly, the Chain of Thought Agent was faster than the Basic Agent despite more verbose reasoning — this is because it had fewer total API round trips on simpler queries.

**Quality ratings:** Ratings aligned with expectations — Basic Agent scored lowest (2.60/5) due to less structured and sometimes incomplete responses, while the Advanced Agent scored highest (4.00/5) for comprehensive, well-organised answers. The CoT Agent scored 3.80/5, slightly below Advanced, as its step-by-step format is occasionally verbose without adding useful information for simple queries.

**Speedup results:** For multi-city queries (3 locations), the Basic and CoT agents showed a ~2.7x speedup in the timing comparison section, as expected for 3 parallel I/O-bound calls. The Advanced Agent's speedup was lower (~1.61x on average) because it already uses parallel execution internally in its `run()` method, meaning the timing comparison's sequential run was measuring calls the agent would never actually make sequentially in practice.

### When Parallel Execution Helps Most

Parallel tool calling provides the greatest benefit when:

1. **Multiple independent locations are queried simultaneously** — three weather API calls for Cairo, Riyadh, and London run in parallel, reducing total time from ~3× a single call to ~1× (the slowest single call). The CSV confirms ~3x speedup for these queries.
2. **Tool calls are I/O-bound** — weather API calls spend most of their time waiting for a network response. The CPU is idle during this wait, so parallelism with `ThreadPoolExecutor` effectively overlaps the waiting time across all calls at no CPU cost.
3. **Tools are independent of each other** — parallel execution is only possible when Tool B does not need Tool A's output as its input.

### When Sequential (Multi-Step) Reasoning is Still Necessary

Parallel execution cannot replace sequential reasoning when:

1. **Tool calls are dependent** — for example, "Find the coldest city among Cairo, Riyadh, and London, then get its 5-day forecast" requires knowing the result of the first tool call before making the second.
2. **Intermediate calculations are required** — multi-step math problems where each step depends on the previous result must remain sequential.
3. **Conditional logic is involved** — for example, "Get the weather in Cairo, and if it is raining, also check Alexandria" cannot be parallelised because the second call depends on the outcome of the first.

In these cases, the multi-step agentic loop (up to 5 iterations) is essential, and the chain-of-thought reasoning strategy helps the model correctly plan and sequence its tool calls.
