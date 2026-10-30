
from pathlib import Path
import json
import random
import re
import time

import pandas as pd
import mock_llm


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"

TICKETS_FILE = RAW_DIR / "tickets.csv"
SEGMENTS_FILE = OUTPUT_DIR / "customer_segments.csv"

MAX_INPUT_TOKENS = 400
TOKEN_BUDGET = 6000
BASE_DELAY = 0.05

CATEGORIES = {"billing", "delivery", "technical", "fraud"}

# Apply these patterns in this exact order.
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_PATTERN = re.compile(r"ORD-\d{6}")

REDACTION_PATTERNS = [
    ("email", EMAIL_PATTERN, "[EMAIL]"),
    ("phone", PHONE_PATTERN, "[PHONE]"),
    ("order_id", ORDER_PATTERN, "[ORDER_ID]"),
]

ZERO_SHOT_PROMPT = """
CLASSIFY the support ticket into exactly one allowed category:
billing, delivery, technical, fraud.

Return JSON only, using this form:
{"category": "...", "reason": "..."}

The category must be one of the four allowed categories.
Use the ticket text as evidence. Do not invent information.
""".strip()

FEW_SHOT_PROMPT = """
CLASSIFY the support ticket into exactly one allowed category:
billing, delivery, technical, fraud.

EXAMPLES:
Example 1:
Ticket: I was charged twice for my payment.
Reply: {"category": "billing", "reason": "duplicate charge"}

Example 2:
Ticket: My parcel is late and the rider has not arrived.
Reply: {"category": "delivery", "reason": "late delivery"}

Example 3:
Ticket: Someone hacked my account and made an unauthorized payment.
Reply: {"category": "fraud", "reason": "unauthorized account activity"}

Example 4:
Ticket: The app crashes whenever I try to log in.
Reply: {"category": "technical", "reason": "app crash during login"}

Reply to the actual ticket with JSON only, using this form:
{"category": "...", "reason": "..."}

The category must be one of: billing, delivery, technical, fraud.
Do not invent information.
""".strip()


# ============================================================
# PRIVACY: REDACT ALL TICKETS AND VERIFY
# ============================================================

def redact_text(text):
    """Apply email, phone and order ID redaction in the required order."""
    redacted = text
    counts = {}

    for name, pattern, placeholder in REDACTION_PATTERNS:
        redacted, count = pattern.subn(placeholder, redacted)
        counts[name] = count

    return redacted, counts


def prepare_redacted_tickets(tickets):
    redacted_texts = []
    totals = {"email": 0, "phone": 0, "order_id": 0}

    for text in tickets["text"]:
        redacted, counts = redact_text(str(text))

        redacted_texts.append(redacted)

        for name, count in counts.items():
            totals[name] += count

        # No original email, phone or order ID pattern may remain.
        for _, pattern, _ in REDACTION_PATTERNS:
            assert pattern.search(redacted) is None, (
                "Privacy check failed: sensitive pattern remains"
            )

    tickets = tickets.copy()
    tickets["redacted_text"] = redacted_texts

    print("\n========== PRIVACY AUDIT ==========")
    print("Email substitutions:", totals["email"])
    print("Phone substitutions:", totals["phone"])
    print("Order ID substitutions:", totals["order_id"])
    print("Total substitutions:", sum(totals.values()))

    leftovers = {
        name: sum(
            len(pattern.findall(text))
            for text in tickets["redacted_text"]
        )
        for name, pattern, _ in REDACTION_PATTERNS
    }

    print("Sensitive-pattern matches after redaction:", leftovers)
    assert all(count == 0 for count in leftovers.values())

    return tickets, totals


# ============================================================
# RESPONSE VALIDATION
# ============================================================

def validate_response(content):
    """
    Return (status, category).

    Valid JSON with an invalid category is rejected outright.
    Keyword fallback is only used when JSON parsing fails.
    """
    if not isinstance(content, str):
        return "flagged_invalid_output", ""

    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", content.lower())
        found = [word for word in words if word in CATEGORIES]

        if len(found) == 1:
            return "classified_keyword", found[0]

        return "flagged_invalid_output", ""

    # Valid JSON must have an allowed category.
    if isinstance(parsed, dict):
        category = parsed.get("category")

        if isinstance(category, str) and category in CATEGORIES:
            return "classified_json", category

    return "flagged_invalid_output", ""


# ============================================================
# RETRIES WITH EXPONENTIAL BACKOFF AND JITTER
# ============================================================

def call_with_retries(messages, ticket_id):
    """Try the model call, allowing at most three retries."""
    for retry_number in range(4):
        try:
            # The user message must contain only the redacted ticket text.
            user_content = messages[-1]["content"]
            assert user_content == messages[-1]["_redacted_ticket_text"]

            for _, pattern, _ in REDACTION_PATTERNS:
                assert pattern.search(user_content) is None

            # Remove our internal assertion-only field before the API call.
            api_messages = [
                {key: value for key, value in message.items()
                 if not key.startswith("_")}
                for message in messages
            ]

            return mock_llm.chat(api_messages)

        except mock_llm.TransientLLMError:
            if retry_number >= 3:
                print(
                    f"PERMANENT ERROR ticket={ticket_id} "
                    "after 3 retries"
                )
                return None

            wait = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"RETRY ticket={ticket_id} "
                f"retry={retry_number} wait={wait:.4f}s"
            )
            time.sleep(wait)

    return None


# ============================================================
# ROUTING
# ============================================================

HUMAN_REVIEW_STATUSES = {
    "skipped_empty",
    "flagged_long",
    "flagged_invalid_output",
    "budget_exhausted",
}


def premium_value(value):
    """Interpret CSV boolean values safely."""
    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true", "1", "yes", "y"
    }


def route_ticket(status, category, is_premium):
    if status in HUMAN_REVIEW_STATUSES:
        return "NORMAL", "human_review"

    if category == "fraud":
        return "HIGH", "security_team"

    if is_premium:
        return "HIGH", "premium_desk"

    return "NORMAL", "support_queue"


# ============================================================
# FULL PIPELINE
# ============================================================

def run_pipeline(tickets, segments, system_prompt, token_budget=None):
    """
    Run one complete independent pipeline.

    Reset is called immediately before each full run.
    Tickets are processed in their original file order.
    """
    mock_llm.reset()

    running_tokens = 0
    results = []

    segment_lookup = (
        segments.drop_duplicates("customer_id")
        .set_index("customer_id")
        .to_dict("index")
    )

    for _, ticket in tickets.iterrows():
        ticket_id = ticket["ticket_id"]
        customer_id = ticket["customer_id"]
        raw_text = str(ticket["text"])
        redacted_text = ticket["redacted_text"]

        category = ""
        tokens_used = 0
        status = ""

        # Guardrails use the original ticket text and run before any call.
        if not raw_text.strip():
            status = "skipped_empty"

        elif mock_llm.est_tokens(raw_text) > MAX_INPUT_TOKENS:
            status = "flagged_long"

        else:
            # The budget is checked before every model call, including retries.
            if token_budget is not None and running_tokens >= token_budget:
                status = "budget_exhausted"

            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": redacted_text,
                        "_redacted_ticket_text": redacted_text,
                    },
                ]

                # Retry function checks budget immediately before each call.
                response = None
                permanent_error = False

                for retry_number in range(4):
                    if (
                        token_budget is not None
                        and running_tokens >= token_budget
                    ):
                        status = "budget_exhausted"
                        break

                    try:
                        api_messages = [
                            {
                                key: value
                                for key, value in message.items()
                                if not key.startswith("_")
                            }
                            for message in messages
                        ]

                        # Explicit privacy assertions on the exact sent content.
                        assert api_messages[-1]["content"] == redacted_text

                        for _, pattern, _ in REDACTION_PATTERNS:
                            assert pattern.search(
                                api_messages[-1]["content"]
                            ) is None

                        response = mock_llm.chat(api_messages)
                        break

                    except mock_llm.TransientLLMError:
                        if retry_number >= 3:
                            permanent_error = True
                            print(
                                f"PERMANENT ERROR ticket={ticket_id} "
                                "after 3 retries"
                            )
                            break

                        wait = (
                            BASE_DELAY
                            * (2 ** retry_number)
                            * random.uniform(0.9, 1.15)
                        )

                        print(
                            f"RETRY ticket={ticket_id} "
                            f"retry={retry_number} wait={wait:.4f}s"
                        )
                        time.sleep(wait)

                if permanent_error:
                    status = "flagged_invalid_output"

                elif response is not None:
                    tokens_used = response.usage.total_tokens
                    running_tokens += tokens_used

                    content = response.choices[0].message.content
                    status, category = validate_response(content)

                elif not status:
                    status = "flagged_invalid_output"

        segment = segment_lookup.get(customer_id, {})
        segment_name = segment.get("segment_name", "")
        if pd.isna(segment_name):
            segment_name = ""

        is_premium = premium_value(segment.get("is_premium", False))

        priority, route = route_ticket(
            status, category, is_premium
        )

        results.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": tokens_used,
            "true_category": ticket["true_category"],
        })

    return pd.DataFrame(results), running_tokens


# ============================================================
# REPORTING
# ============================================================

OUTPUT_COLUMNS = [
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used",
]


def report_run(name, results, total_tokens):
    classified = results[
        results["status"].isin(
            {"classified_json", "classified_keyword"}
        )
    ]

    # "none" labels are not valid target categories.
    scored = classified[
        classified["true_category"].isin(CATEGORIES)
    ]

    correct = (
        scored["category"] == scored["true_category"]
    ).sum()

    accuracy = correct / len(scored) if len(scored) else 0
    per_ticket = (
        total_tokens / len(classified)
        if len(classified) else 0
    )

    print(f"\n========== {name} REPORT ==========")
    print("Status counts:")
    print(results["status"].value_counts().to_string())
    print("Tickets classified:", len(classified))
    print(
        "Accuracy on classified tickets:",
        f"{accuracy:.1%} ({correct}/{len(scored)})",
    )
    print("Total response tokens:", total_tokens)
    print("Tokens per classified ticket:", f"{per_ticket:.2f}")

    return accuracy, len(classified), per_ticket


# ============================================================
# MAIN
# ============================================================

def main():
    if not TICKETS_FILE.exists():
        raise FileNotFoundError(TICKETS_FILE)

    if not SEGMENTS_FILE.exists():
        raise FileNotFoundError(SEGMENTS_FILE)

    tickets = pd.read_csv(
        TICKETS_FILE,
        keep_default_na=False,
    )
    segments = pd.read_csv(SEGMENTS_FILE)

    print("Tickets loaded:", len(tickets))
    print("Customer segments loaded:", len(segments))

    # Redact all 164 tickets, including those later skipped or flagged.
    tickets, redaction_counts = prepare_redacted_tickets(tickets)

    assert len(tickets) == 164
    assert redaction_counts == {
        "email": 55,
        "phone": 91,
        "order_id": 132,
    }, f"Unexpected redaction counts: {redaction_counts}"

    # 1. Compare both prompts without a token budget.
    zero_results, zero_tokens = run_pipeline(
        tickets,
        segments,
        ZERO_SHOT_PROMPT,
        token_budget=None,
    )
    zero_accuracy, zero_count, zero_per_ticket = report_run(
        "ZERO-SHOT",
        zero_results,
        zero_tokens,
    )

    few_results, few_tokens = run_pipeline(
        tickets,
        segments,
        FEW_SHOT_PROMPT,
        token_budget=None,
    )
    few_accuracy, few_count, few_per_ticket = report_run(
        "FEW-SHOT",
        few_results,
        few_tokens,
    )

    # Confusion table for zero-shot classified tickets.
    zero_scored = zero_results[
        zero_results["status"].isin(
            {"classified_json", "classified_keyword"}
        )
        & zero_results["true_category"].isin(CATEGORIES)
    ]

    confusion = pd.crosstab(
        zero_scored["true_category"],
        zero_scored["category"],
    ).reindex(
        index=sorted(CATEGORIES),
        columns=sorted(CATEGORIES),
        fill_value=0,
    )

    print("\n========== ZERO-SHOT CONFUSION TABLE ==========")
    print("Rows = true category; columns = predicted category")
    print(confusion.to_string())

    # 2. Budget demonstration: start at 6000 tokens.
    budget_used = 6000

    budget_results, budget_tokens = run_pipeline(
        tickets,
        segments,
        ZERO_SHOT_PROMPT,
        token_budget=budget_used,
    )

    exhausted = int(
        (budget_results["status"] == "budget_exhausted").sum()
    )

    # Lower the budget if the initial prompt does not exhaust it.
    if exhausted == 0:
        budget_used = 5000
        budget_results, budget_tokens = run_pipeline(
            tickets,
            segments,
            ZERO_SHOT_PROMPT,
            token_budget=budget_used,
        )
        exhausted = int(
            (budget_results["status"] == "budget_exhausted").sum()
        )

    budget_results[OUTPUT_COLUMNS].to_csv(
        OUTPUT_DIR / "triage_budget_demo.csv",
        index=False,
    )

    print("\n========== BUDGET DEMO ==========")
    print("Configured token budget:", budget_used)
    print("Response tokens consumed:", budget_tokens)
    print("Tickets stopped by budget:", exhausted)
    print("Saved: outputs/triage_budget_demo.csv")

    assert exhausted >= 1, (
        "No tickets exhausted the budget; lower the budget and rerun."
    )

    # 3. Final deliverable: few-shot with a 100000-token budget.
    final_results, final_tokens = run_pipeline(
        tickets,
        segments,
        FEW_SHOT_PROMPT,
        token_budget=100000,
    )

    final_results[OUTPUT_COLUMNS].to_csv(
        OUTPUT_DIR / "triage_results.csv",
        index=False,
    )

    print("\n========== FINAL PART C OUTPUT ==========")
    print("Rows saved:", len(final_results))
    print("Response tokens consumed:", final_tokens)
    print("Saved: outputs/triage_results.csv")
    print("\nFinal status counts:")
    print(final_results["status"].value_counts().to_string())
    print("\nFinal routing counts:")
    print(final_results["route"].value_counts().to_string())

    assert len(final_results) == 164
    assert list(pd.read_csv(
        OUTPUT_DIR / "triage_results.csv"
    ).columns) == OUTPUT_COLUMNS

    print("\nPART C PIPELINE COMPLETED")


if __name__ == "__main__":
    main()
