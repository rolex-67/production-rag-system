import os
import json
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
)
from rag import ProductionRAG

load_dotenv()

def run_evaluation():
    print("Initializing RAG system for evaluation...")
    rag = ProductionRAG()
    
    # Dummy evaluation dataset (you would normally load this from a JSON/CSV file)
    eval_questions = [
        "What is the main topic of the document?",
        "What happens if the user violates the policy?"
    ]
    ground_truths = [
        ["The main topic is the course handbook and guidelines."],
        ["The user may face disciplinary action."]
    ]
    
    answers = []
    contexts = []
    
    print("Generating answers for evaluation dataset...")
    for q in eval_questions:
        res = rag.query(q)
        answers.append(res["answer"])
        contexts.append(res["context"])
        
    data = {
        "question": eval_questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    print("Running RAGAS evaluation...")
    # Evaluate using Faithfulness, Answer Relevancy, and Context Recall
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
        ]
    )
    
    print("\n--- Evaluation Results ---")
    print(result)
    
    # Save results to a file
    with open("eval_results.json", "w") as f:
        df = result.to_pandas()
        f.write(df.to_json(orient="records"))
    print("Results saved to eval_results.json")

if __name__ == "__main__":
    run_evaluation()
