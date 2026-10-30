import re
import json
import sqlite3
import mock_llm
from datetime import datetime, timezone
from pathlib import Path


from src.policy_retrieval import PolicyRetriever


ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "instamart.db"
LEDGER_PATH = ROOT / "outputs" / "refund_ledger.jsonl"

retriever = PolicyRetriever()


# Tool 1: Look up an order in SQLite
def get_order(order_id):
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        row = connection.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,)
        ).fetchone()

        if row is None:
            return {"error": "order not found"}

        return {
            "order_id": row["order_id"],
            "status": row["status"],
            "order_amount": row["order_amount"],
            "delivery_minutes": row["delivery_minutes"]
        }

    finally:
        connection.close()


# Tool 2: Search the policy documents
def search_policy(query):
    results = retriever.retrieve(query, k=3)

    return {"results": results}


# Tool 3: Calculate the refund amount
def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None or delivery_minutes <= 45:
        amount = 0.0
    else:
        amount = float(order_amount)

    return {"refund_inr": round(amount, 2)}


# Common function for recording refund actions
def record_action(order_id, refund_inr, tool_name):
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2)
    }

    with LEDGER_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return {"refund_inr": record["refund_inr"]}


# Tool 4: Issue an approved refund
def issue_refund(order_id, refund_inr):
    return record_action(order_id, refund_inr, "issue_refund")


# Tool 5: Escalate a refund for human approval
def escalate_to_human(order_id, refund_inr):
    return record_action(order_id, refund_inr, "escalate_to_human")


# Tool descriptions supplied to mock_llm
TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up an order's status, amount and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Search the company policy documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate the eligible refund amount.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"}
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Record an automatically approved refund.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Send a refund above INR 1000 for human approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]


# Map tool names to Python functions
TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

import csv
import time


AGENT_SYSTEM_PROMPT = """
AGENT

You are a Swiggy Instamart refund support agent.
Use the supplied tools to look up orders, check policies,
calculate eligible refunds, and issue or escalate refunds.
Never issue a refund above INR 1000 automatically.
Only completed orders are eligible for refunds.
"""

class SimulatedCrash(Exception):
    pass


def checkpoint_path(run_id):
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", run_id)
    return ROOT / "state" / f"{safe_id}.json"


def save_checkpoint(state):
    path = checkpoint_path(state["run_id"])
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)

    temporary_path.replace(path)


def load_checkpoint(run_id):
    path = checkpoint_path(run_id)

    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found for {run_id}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)




def continue_agent(state, crash_after_tools=None):
    messages = state["messages"]
    steps = state["steps"]
    tokens_used = state["tokens_used"]
    llm_calls = state["llm_calls"]
    max_iterations = state["max_iterations"]
    token_limit = state["token_limit"]
    time_limit_seconds = state["time_limit_seconds"]
    start_time = time.monotonic()

    def fallback(status, summary):
        return {
            "scenario_id": state["scenario_id"],
            "status": status,
            "final_answer": f"{status}: {summary}",
            "steps": steps,
            "llm_calls": llm_calls,
            "tokens_used": tokens_used
        }

    for iteration in range(llm_calls, max_iterations):

        if time.monotonic() - start_time >= time_limit_seconds:
            return fallback(
                "FALLBACK_TIMEOUT",
                "Request exceeded the processing time limit. Please hand over to a human."
            )

        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += response.usage.total_tokens

        state["llm_calls"] = llm_calls
        state["tokens_used"] = tokens_used

        if tokens_used > token_limit:
            return fallback(
                "FALLBACK_TOKEN_LIMIT",
                "Request exceeded the token budget. Please hand over to a human."
            )

        assistant_message = response.choices[0].message

        if not assistant_message.tool_calls:
            final_answer = assistant_message.content or "FALLBACK: No answer returned."

            return {
                "scenario_id": state["scenario_id"],
                "status": final_answer.split(":")[0].strip(),
                "final_answer": final_answer,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used
            }

        tool_calls = []

        for call in assistant_message.tool_calls:
            arguments = call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            tool_calls.append({
                "id": call.id,
                "name": call.name,
                "arguments": arguments
            })

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls
        })

        for call in tool_calls:
            tool_name = call["name"]
            arguments = call["arguments"]

            if tool_name not in TOOL_MAP:
                observation = {"error": "unknown tool"}
            else:
                observation = TOOL_MAP[tool_name](**arguments)

            step = {
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation
            }

            steps.append(step)

            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": tool_name,
                "content": json.dumps(observation)
            })

            # Save state immediately after every tool observation
            state["messages"] = messages
            state["steps"] = steps
            state["step_number"] = len(steps)
            state["executed_tools"] = steps.copy()
            state["tokens_used"] = tokens_used
            state["llm_calls"] = llm_calls

            save_checkpoint(state)

            # Simulate a crash only after the checkpoint is safely saved
            if (
                crash_after_tools is not None
                and len(steps) == crash_after_tools
            ):
                raise SimulatedCrash(
                    f"Simulated crash after tool {len(steps)}"
                )

    return fallback(
        "FALLBACK_MAX_ITERATIONS",
        "Maximum agent steps reached. Please hand over to a human."
    )


def run_agent(
    user_message,
    scenario_id="TEST",
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tools=None
):
    if run_id is None:
        run_id = scenario_id

    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "user_message": user_message,
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "steps": [],
        "step_number": 0,
        "executed_tools": [],
        "tokens_used": 0,
        "llm_calls": 0,
        "max_iterations": max_iterations,
        "token_limit": token_limit,
        "time_limit_seconds": time_limit_seconds
    }

    return continue_agent(state, crash_after_tools=crash_after_tools)


def resume_agent(run_id):
    state = load_checkpoint(run_id)
    return continue_agent(state)


def run_all_scenarios():
    scenario_file = ROOT / "data" / "raw" / "agent_scenarios.csv"
    output_file = ROOT / "outputs" / "agent_runs.jsonl"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = []

    with scenario_file.open("r", encoding="utf-8", newline="") as file:
        scenarios = csv.DictReader(file)

        for scenario in scenarios:
            result = run_agent(
                scenario["message"],
                scenario["scenario_id"]
            )

            results.append(result)

            record = {
                "scenario_id": result["scenario_id"],
                "status": result["status"],
                "final_answer": result["final_answer"],
                "llm_calls": result["llm_calls"],
                "tokens_used": result["tokens_used"],
                "steps": result["steps"]
            }

            with output_file.open("a", encoding="utf-8") as out:
                out.write(json.dumps(record) + "\n")

            print(
                f'{result["scenario_id"]}: '
                f'{result["status"]} | '
                f'{result["final_answer"]}'
            )

    return results

if __name__ == "__main__":
    run_all_scenarios()

