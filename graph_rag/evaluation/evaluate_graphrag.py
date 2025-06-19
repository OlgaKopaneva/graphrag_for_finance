from tqdm import tqdm
import pandas as pd
import evaluate
from datasets import load_dataset
from graph_rag.core.generation import generate_answer
from graph_rag.scripts.run_retriever import EnhancedRetrieverPipeline

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

def safe_get(metric_result, key, default=0.0):
    return metric_result.get(key, default) if isinstance(metric_result, dict) else default

print("Loading dataset...")
dataset = load_dataset("yymYYM/stock_trading_QA")["train"]  # type: ignore
dataset = dataset.select(range(500))  # type: ignore

methods = ["regular", "graph_rag"]
k_values = [5, 10, 15]

results = []

print("Running evaluation...")

try:
    pipeline = EnhancedRetrieverPipeline(use_graph_rag=True)
except Exception as e:
    print(f"Error initializing retriever pipeline: {e}")
    exit(1)

for method in methods:
    for k in k_values:
        candidates = []
        references = []

        for row in tqdm(dataset, desc=f"{method}@k={k}"):
            question = row["question"]  # type: ignore
            gold = row["answer"]  # type: ignore
            try:
                retrieved = pipeline.search(question, k, method)
            except Exception as e:
                print(f"Retrieval failed for question: {question} — {e}")
                retrieved = []

            context = [chunk["text"] for chunk in retrieved]
            response = generate_answer(question, context)

            candidates.append(response)
            references.append([gold]) 

        # Metrics
        rouge_score = rouge.compute(predictions=candidates, references=[r[0] for r in references])
        bleu_score = bleu.compute(predictions=candidates, references=references)
        meteor_score = meteor.compute(predictions=candidates, references=[r[0] for r in references])

        scores = {
            "method": method,
            "k": k,
            "rouge1": safe_get(rouge_score, "rouge1"),
            "bleu": safe_get(bleu_score, "bleu"),
            "meteor": safe_get(meteor_score, "meteor")
        }

        results.append(scores)

df = pd.DataFrame(results)
print("\n Evaluation results:\n")
print(df)

df.to_csv("evaluation_results.csv", index=False)