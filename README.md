# Swiggy Instamart Customer Intelligence & Support Copilot

## Overview

This project builds a synthetic customer intelligence and support automation system for Swiggy Instamart. It combines data cleaning, exploratory data analysis, customer segmentation, lapse prediction, privacy-aware support ticket triage, policy retrieval, and an AI-assisted refund workflow.

The project is divided into four parts:

- **Part A:** Data cleaning, SQLite database creation, SQL analysis, and EDA.
- **Part B:** Customer segmentation and lapse prediction.
- **Part C:** Privacy-aware support ticket classification and routing.
- **Part D:** Retrieval-grounded policy assistant and refund support agent.

All data is synthetic. The supplied `mock_llm.py` is used for model interactions. No external LLM API, API key, or live customer data is required.

## Reproduce

Run these commands in Windows PowerShell from the project root directory.

1. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

2. Activate the environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the required packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Generate the synthetic input data:

   ```powershell
   python generate_data.py
   ```

5. Run Part A:

   ```powershell
   python src/audit_data.py
   python src/clean_data.py
   python src/run_sql.py
   python src/eda.py
   python src/validate_data.py
   ```

6. Run Part B:

   ```powershell
   python src/part_b.py
   python src/lapse_model.py
   ```

7. Run Part C:

   ```powershell
   python src/part_c.py
   ```

8. Run the Part D policy retrieval evaluation:

   ```powershell
   python src/policy_retrieval.py
   ```

9. Run the refund agent:

   ```powershell
   python -m src.refund_agent
   ```

The scripts generate the cleaned datasets, database, model files, evaluation results, agent logs, and checkpoint files in the project folders.

## Part A — Data Cleaning, SQL and EDA

### Data cleaning and database

The synthetic dataset contains customer, product, order, and support ticket records. The raw data includes deliberately introduced quality issues such as duplicate records, inconsistent category and city values, missing values, invalid ages, negative delivery times, and unparseable order amounts.

The data-cleaning pipeline standardizes and validates the input data before saving the cleaned customer, product, and order datasets.

Generated files:

- `data/clean/customers.csv`
- `data/clean/products.csv`
- `data/clean/orders.csv`
- `data/instamart.db`
- `sql/queries.sql`

The SQLite database contains the `customers`, `products`, and `orders` tables.

### Exploratory Data Analysis (EDA)

The synthetic dataset contains 12,309 cleaned orders, including 11,110 completed orders. Total completed-order revenue is ₹29,98,668.80. Hyderabad generated the highest revenue at ₹5,28,589.90, while Mumbai had the highest average completed order amount at ₹297.00.

The average completed order amount is ₹269.91, compared with a median of ₹135.00. The positive skewness of 4.61 indicates a right-skewed distribution, with some high-value orders increasing the mean. Using the IQR method, 897 orders exceed the upper outlier fence of ₹645.30. These orders should be investigated rather than automatically removed.

The correlation between order amount and discount percentage is weakly negative at −0.095. The correlation between order amount and quantity is weakly positive at 0.087. These relationships are small and do not establish causation. The city-and-category pivot table reconciles with SQL revenue at ₹29,98,668.80, confirming consistency between the analyses.

### Part A validation

The data validation script completed successfully and reported:

`ALL CHECKPOINTS PASSED`

The analysis and validation results are based on the generated synthetic dataset.

## Part B — Customer Segmentation and Lapse Prediction

### Customer segmentation

RFM (Recency, Frequency, Monetary) features were used to group customers with K-Means clustering. Four customer segments were selected.

| Customer segment | Customers |
|---|---:|
| Steady Shoppers | 436 |
| Premium High-Value Customers | 138 |
| Frequent Regulars | 419 |
| At-Risk Occasional Customers | 182 |

The segments help identify differences in customer purchasing frequency, spending, and recency. They can support targeted retention strategies.

### Lapse prediction

A customer lapse prediction workflow was developed using Logistic Regression and Random Forest models. Customer features were calculated using order history before September 2025, with September 2025 used as the prediction target period.

The saved balanced Logistic Regression model achieved:

- F1-score: 0.6972
- Recall: 85.39%
- Precision: 58.91%

Recall is useful in this context because the model aims to identify customers who may lapse. However, the precision result indicates that some customers predicted to lapse may not actually lapse.

Generated files:

- `outputs/customer_segments.csv`
- `outputs/kmeans_metrics.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

**Retention list limitation:** The retention list is illustrative, not a fully leakage-free evaluation of a real-world retention campaign. Premium customer status was derived from RFM segmentation using the full available order period, including September 2025, while the lapse model uses customer features calculated before September 2025. Future improvements should assign customer segments using only historical data available before the prediction date and validate the model on a separate future time period.

## Part C — Support Ticket Triage

Implemented a privacy-aware support ticket classification pipeline using the supplied `mock_llm.py`.

- **Input:** 164 synthetic support tickets.
- **Privacy:** Replaced 55 email addresses, 91 phone numbers, and 132 order IDs. No matching sensitive patterns remained after redaction.
- **Guardrails:** Skipped 4 empty tickets and flagged 4 long tickets without sending them to the model.
- **Classification:** Zero-shot accuracy was 85.7% (126/147), while few-shot accuracy was 100% (147/147) on classified tickets.
- **Output validation:** 135 responses were valid JSON, 12 were classified using keyword fallback, and 9 were flagged for human review.
- **Token usage:** Zero-shot used 13,144 response tokens (89.41 per classified ticket). Few-shot used 28,276 (192.35 per classified ticket), trading higher token usage for better classification accuracy in this mock experiment.
- **Budget demonstration:** A 6,000-token budget was configured for the zero-shot run. The run consumed 6,076 response tokens because the budget was checked before each model call. A total of 84 tickets were stopped from being processed after the budget was reached.
- **Routing:** Fraud tickets go to the security team; premium customers go to the premium desk; other successfully classified tickets go to the support queue. Empty, long, invalid-output, and budget-exhausted tickets are routed for human review.

Generated files:

- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`

**Validation boundary:** Privacy substitutions, guardrails, output parsing, and routing are implemented as code rules. Classification labels are model-generated and should be reviewed before production use. Tickets flagged for human review require manual handling. Results are based on synthetic data and a mock language model.

## Part D — Policy Assistant and Refund Agent

### Policy retrieval assistant

A retrieval-grounded policy assistant was implemented using the five policy documents stored in `data/policies/`:

- Refund policy
- Delivery policy
- Escalation policy
- Payment policy
- Privacy policy

The documents are split into overlapping text chunks and indexed using TF-IDF vectorization. Cosine similarity retrieves relevant policy passages for each question. The retrieved passages are supplied as context to the policy assistant implemented through `mock_llm.py`.

The evaluation contained 14 questions: 12 answerable questions and 2 questions that should receive an unanswerable response.

With a 10-word chunk overlap:

- 12/12 answerable questions retrieved the expected policy document in the top three results.
- 12/12 answerable questions returned the expected policy fact.
- 2/2 unanswerable questions received the expected fallback response.

A comparison with zero-overlap chunking retrieved the expected document for all 12 answerable questions, but returned the expected fact for only 10/12. This demonstrates how chunk overlap can help preserve useful policy context.

### Refund support agent

A tool-using refund agent was developed with a ReAct workflow. It uses five tools:

1. `get_order` — retrieves order information from SQLite.
2. `search_policy` — retrieves relevant refund policy passages.
3. `calculate_refund` — calculates the eligible refund amount.
4. `issue_refund` — records an approved automatic refund.
5. `escalate_to_human` — records a refund requiring human approval.

The agent checks the order status and delivery time before deciding what action to take.

The implemented refund rules are:

- Only completed orders are eligible for a refund.
- Orders delivered more than 45 minutes after the relevant threshold qualify for a full order-amount refund.
- Orders delivered within 45 minutes do not qualify for a late-delivery refund.
- Refunds up to ₹1,000 can be processed automatically.
- Refunds above ₹1,000 are escalated to a human support lead.

The workflow assumes the refund request is within the seven-day policy window.

### Agent scenario evaluation

Seven scenarios were tested:

| Scenario | Expected and observed outcome |
|---|---|
| S1 | REFUND — ₹128.25 |
| S2 | REFUND — ₹135.00 |
| S3 | ESCALATE — ₹1,999.00 |
| S4 | DENIED — delivery within 45 minutes |
| S5 | DENIED — cancelled order |
| S6 | NOT_FOUND — order does not exist |
| S7 | FALLBACK_MAX_ITERATIONS — repeated tool calls stopped safely |

The agent records its execution details in JSONL files, including tool activity, outcomes, token usage, and model-call counts.

### ReAct versus Plan-and-Execute

I selected the ReAct approach for the refund support agent because refund decisions depend on information discovered during execution. The agent looks up the order, retrieves the relevant policy, calculates the eligible refund, and then decides whether to issue or escalate it.

The measured results show that S1 required 5 model calls and 1,107 tokens, S5 required 3 calls and 463 tokens, and S6 required 2 calls and 156 tokens. S7 reached the six-call iteration limit and safely returned `FALLBACK_MAX_ITERATIONS`, using 3,252 tokens.

The tool observations demonstrate why adaptive execution is useful. S5 returned a cancelled order, so the agent denied the refund. S6 returned an order-not-found response, so the agent stopped without attempting a refund.

Plan-and-Execute could reduce repeated planning for predictable workflows, but it may require replanning when an order is missing, cancelled, or requires human approval. ReAct has additional model-call and token overhead, but it responds to tool observations and supports conditional decisions.

For this project, ReAct is more suitable because refund processing requires checking real order data and applying different outcomes safely.

### Safety and checkpointing

The agent includes execution safeguards:

- Maximum of six iterations.
- Token limit of 6,000.
- Execution timeout of 10 seconds.
- Fallback when an execution limit is reached.
- JSON checkpoints saved under `state/`.
- Resume support to continue an interrupted run from its saved state.

A simulated crash-and-resume test completed successfully. The agent resumed from its checkpoint and completed the refund workflow without repeating previously executed tools.

Generated files:

- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`
- `state/` — saved agent checkpoints

## Limitations and Next Steps

This project uses synthetic data and a supplied mock language model. Its results demonstrate a working prototype, not a production-ready customer support system.

Important limitations and future improvements include:

- Evaluate customer lapse prediction using a separate future time period and leakage-free customer segments.
- Test the classification pipeline with real-world-like, human-reviewed support ticket examples.
- Improve policy retrieval for questions that use wording different from the source documents.
- Add more comprehensive tests for malformed model responses, database failures, and interrupted agent execution.
- Use a production-grade model and securely managed credentials if the system is later deployed.
- Add authentication, authorization, audit controls, and human approval before connecting the refund workflow to a real payment system.
- Validate refund request dates against actual order and request timestamps instead of assuming the seven-day window is satisfied.

The project demonstrates an end-to-end approach to customer analytics and support automation, combining data engineering, machine learning, information retrieval, privacy safeguards, and tool-based agent workflows.