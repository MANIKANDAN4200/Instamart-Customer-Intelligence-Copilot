
from pathlib import Path
import re

import pandas as pd
import mock_llm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "data" / "policies"
EVAL_FILE = ROOT / "data" / "raw" / "policy_eval.csv"

CHUNK_SIZE = 40
OVERLAP = 10


def load_chunks(overlap=OVERLAP):
    """Read policy files and split them into overlapping word chunks."""
    chunks = []
    step = CHUNK_SIZE - overlap

    for path in sorted(POLICY_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        words = text.split()

        start = 0
        while start < len(words):
            chunk_words = words[start:start + CHUNK_SIZE]

            chunks.append({
                "source": path.name,
                "text": " ".join(chunk_words),
            })

            if start + CHUNK_SIZE >= len(words):
                break

            start += step

    return chunks


class PolicyRetriever:
    """TF-IDF search over policy document chunks."""

    def __init__(self, overlap=OVERLAP):
        self.chunks = load_chunks(overlap=overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in self.chunks]
        )

    def retrieve(self, query, k=3):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()

        ranked_indices = scores.argsort()[::-1][:k]

        return [
            {
                "source": self.chunks[i]["source"],
                "score": float(scores[i]),
                "text": self.chunks[i]["text"],
            }
            for i in ranked_indices
        ]


def answer(question, retriever=None):
    """Answer a policy question using retrieved context."""
    if retriever is None:
        retriever = DEFAULT_RETRIEVER

    results = retriever.retrieve(question, k=3)

    context_parts = []
    for result in results:
        context_parts.append(
            f"[{result['source']}]\n{result['text']}"
        )

    user_content = (
        "CONTEXT:\n"
        + "\n".join(context_parts)
        + "\nQUESTION:\n"
        + question
    )

    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT: Answer only using the supplied policy "
                "context. If the context does not contain the answer, "
                "say exactly: I could not find that information in the "
                "policy documents. Do not use outside knowledge."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    response = mock_llm.chat(messages)
    text = response.choices[0].message.content

    return {
        "answer": text,
        "sources": list(dict.fromkeys(
            result["source"] for result in results
        )),
    }


def evaluate(retriever, evaluation_df):
    """Measure retrieval coverage, answer grounding, and fallback."""
    answerable = evaluation_df[
        evaluation_df["expected_doc"].str.upper() != "NONE"
    ]
    unanswerable = evaluation_df[
        evaluation_df["expected_doc"].str.upper() == "NONE"
    ]

    retrieval_hits = 0
    fact_hits = 0
    details = []

    for _, row in evaluation_df.iterrows():
        question = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]

        results = retriever.retrieve(question, k=3)
        response = answer(question, retriever=retriever)
        answer_text = response["answer"]

        retrieval_hit = (
            expected_doc.upper() == "NONE"
            or any(r["source"] == expected_doc for r in results)
        )
        fact_hit = (
            expected_doc.upper() != "NONE"
            and bool(expected_fact)
            and expected_fact in answer_text
        )
        fallback_hit = (
            expected_doc.upper() == "NONE"
            and answer_text.startswith("I could not find")
        )

        if expected_doc.upper() != "NONE":
            retrieval_hits += int(retrieval_hit)
            fact_hits += int(fact_hit)

        details.append({
            "question": question,
            "expected_doc": expected_doc,
            "expected_fact": expected_fact,
            "retrieved_sources": ", ".join(
                r["source"] for r in results
            ),
            "answer": answer_text,
            "retrieval_hit": retrieval_hit,
            "fact_hit": fact_hit,
            "fallback_hit": fallback_hit,
        })

    fallback_hits = sum(
        row["fallback_hit"]
        for row in details
        if row["expected_doc"].upper() == "NONE"
    )

    print(f"Answerable questions: {len(answerable)}")
    print(
        f"Expected document in top 3: "
        f"{retrieval_hits}/{len(answerable)}"
    )
    print(
        f"Expected fact in answer: "
        f"{fact_hits}/{len(answerable)}"
    )
    print(
        f"Unanswerable fallback answers: "
        f"{fallback_hits}/{len(unanswerable)}"
    )

    return details


if __name__ == "__main__":
    evaluation_df = pd.read_csv(
        EVAL_FILE,
        keep_default_na=False,
    )

    DEFAULT_RETRIEVER = PolicyRetriever(overlap=10)

    print("========== CHUNK COUNTS ==========")
    print(pd.Series(
        [chunk["source"] for chunk in DEFAULT_RETRIEVER.chunks]
    ).value_counts().to_string())
    print(f"Total chunks: {len(DEFAULT_RETRIEVER.chunks)}")

    print("\n========== RETRIEVAL EXAMPLE ==========")
    for result in DEFAULT_RETRIEVER.retrieve(
        "refund policy late delivery", k=3
    ):
        print(
            f"\nSource: {result['source']}"
            f"\nScore: {result['score']:.4f}"
            f"\nText: {result['text']}"
        )

    print("\n========== EVALUATION: OVERLAP 10 ==========")
    overlap_details = evaluate(DEFAULT_RETRIEVER, evaluation_df)

    print("\n========== EVALUATION: OVERLAP 0 ==========")
    no_overlap_retriever = PolicyRetriever(overlap=0)
    no_overlap_details = evaluate(
        no_overlap_retriever, evaluation_df
    )

    print("\n========== QUESTIONS THAT FAIL WITHOUT OVERLAP ==========")
    for row in no_overlap_details:
        if (
            row["expected_doc"].upper() != "NONE"
            and not row["fact_hit"]
        ):
            print(f"\nQuestion: {row['question']}")
            print(f"Expected fact: {row['expected_fact']}")
            print(f"Answer: {row['answer']}")

    print("\n========== TF-IDF EXPLANATION ==========")
    print(
        "Term Frequency (TF) measures how often a term appears "
        "in a document chunk."
    )
    print(
        "Inverse Document Frequency (IDF) gives more weight to "
        "terms that appear in fewer chunks. TF-IDF can miss semantic "
        "matches when two sentences express the same meaning using "
        "different words."
    )
