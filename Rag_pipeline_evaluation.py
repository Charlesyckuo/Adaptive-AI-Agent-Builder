import time
import logging
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from config import (
    MONGODB_URI,
    COHERE_API_KEY,
    VOYAGE_API_KEY,
    OPENAI_API_KEY,
    CLAUDE_API_KEY,
    GEMINI_API_KEY
)
from embedding import EmbeddingService, RerankingService
from database import DatabaseConnection
from evaluation import Evaluator
from llm import LLMsService
from Ragas import RagasScoreEvaluator
from datasets import Dataset
from email_notification import EmailNotification

class PipelineEvaluator:
    def __init__(self, config):
        self.api_keys = {
            'COHERE_API_KEY': config["COHERE_API_KEY"],
            'VOYAGE_API_KEY': config["VOYAGE_API_KEY"],
            'OPENAI_API_KEY': config["OPENAI_API_KEY"],
            'CLAUDE_API_KEY': config["CLAUDE_API_KEY"],
            'GEMINI_API_KEY': config["GEMINI_API_KEY"],
        }
        self.embedding_service = EmbeddingService(self.api_keys)
        self.reranking_service = RerankingService(self.api_keys)
        self.llm_service = LLMsService(self.api_keys)

    def load_dataset(self, file_path):
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read CSV file: '{file_path}'")
            return df
        except Exception as e:
            logging.error(f"Failed to read CSV file: '{file_path}'. Error: {e}")
            raise

    def initialize_database(self, emb_model):
        collection_name = 'voyage' if emb_model == "voyage-multilingual-2" else 'e5'
        return DatabaseConnection(MONGODB_URI, 'RAGAS', collection_name)

    def save_results(self, results, filename="ragas_results.csv"):
        try:
            df_existing = pd.read_csv(filename)
        except FileNotFoundError:
            df_existing = pd.DataFrame(columns=results.keys())

        df_new = pd.DataFrame(results)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(filename, index=False)
        logging.info(f"Ragas Results saved to {filename}")

    def send_email_notification(self, results, column_names, score_df):
        email_sender = EmailNotification(
            from_email="kuo901105aa@gmail.com",
            from_password="fpar qkgq xssd nqxy",
        )
        subject = "RAG Pipeline Evaluation Completed (Ragas)"
        body = f"""
        RAG Pipeline Evaluation Completed!
        Notice: 
        1. Embedding top_k for phase-1 retrieval Reranking top_k for phase-2 retireval
        2. Generation time includes getting the similar vector, reranking retrieved context and LLMs generation
        3. Evaluation includes calculating RAGAS score of every question,answer,contexts pair
        4. USD Cost only includes LLMs generation and RAGAS evaluation (gpt-4o-mini) 

        Embedding model: {results["Embedding model"][0]}
        Reranking model: {results["Reranking model"][0]}
        Embedding top_k: {results["Emb_top_k"][0]}
        Reranking top_k: {results["Re_top_k"][0]}
        LLM: {results["LLM"][0]}

        Generation Time: {results["Generation Time"][0]:.2f} seconds
        Evaluation Time: {results["Evaluation Time"][0]:.2f} seconds
        Total Time: {results["Total Time"][0]:.2f} seconds
        USD Cost: {results["USD Cost"][0]:.4f} USD

        Ragas Scores:
        """
        for col in column_names:
            body += f"\t{col}: {score_df[col].iloc[0]:.4f}\n"

        email_sender.send_email(
            to_email=["kuo901105aa@gmail.com"],
            subject=subject,
            body=body,
        )

    def evaluate_pipeline(self, src_file, emb_model, re_model, emb_top_n, re_top_n, llm):
        logging.basicConfig(level=logging.INFO)
        start_time_overall = time.time()

        # Initialize components
        db_connection = self.initialize_database(emb_model)
        evaluator = Evaluator(db_connection, self.embedding_service, self.reranking_service)
        embeddings, contexts, context_ids = evaluator.retrieve_all_embeddings()
        dataset = {"user_input": [], "retrieved_contexts": [], "response": [], "reference": []}
        cost_count = 0

        # Load dataset
        df = self.load_dataset(src_file)

        # Process questions
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Running RAG Pipeline"):
            try:
                question = row["Question"]
                ground_truth = row["Answer"]

                question_vector = self.embedding_service.get_embedding_for_evaluation(
                    input_text=question, model_name=emb_model
                )
                top_k_similar_contexts = evaluator.find_top_k_similar_contexts(
                    question_vector, embeddings, context_ids, emb_top_n
                )
                candidate_ids = [context_id for _, context_id in top_k_similar_contexts]
                candidate_texts = [contexts[context_ids.index(context_id)] for context_id in candidate_ids]

                reranked_results = self.reranking_service.get_topk_from_reranking(
                    query=question,
                    candidate_text=candidate_texts,
                    top_n=re_top_n,
                    model_name=re_model,
                )

                reranked_ids = self.process_reranked_results(re_model, reranked_results, candidate_ids, re_top_n)
                reranked_texts = [contexts[context_ids.index(context_id)] for context_id in reranked_ids]

                llm_result, cost = self.llm_service.get_answer_from_llm(
                    query=question, candidate_text=reranked_texts, model_name=llm
                )

                cost_count += cost
                dataset["user_input"].append(question)
                dataset["retrieved_contexts"].append(reranked_texts)
                dataset["response"].append(llm_result)
                dataset["reference"].append(ground_truth)
            except Exception as e:
                logging.error(f"Failed to process QA entry #{index + 1}. Error: {e}")

        # Evaluate and save results
        return self.finalize_evaluation(dataset, cost_count, start_time_overall, emb_model, re_model, emb_top_n, re_top_n, llm)

    def process_reranked_results(self, re_model, reranked_results, candidate_ids, re_top_n):
        '''if re_model == "rerank-2":
            return [candidate_ids[result.index] for result in reranked_results]
        else:
            reranked_results = sorted(zip(reranked_results, candidate_ids), reverse=True)
            return [result[1] for result in reranked_results][:re_top_n]'''
        
        if re_model == "rerank-2":
            return [candidate_ids[result.index] for result in reranked_results]
        elif re_model == "BAAI/bge-reranker-v2-m3" or re_model == "BAAI/bge-reranker-large" or re_model == "BAAI/bge-reranker-base":
            reranked_results = sorted(zip(reranked_results, candidate_ids), reverse=True)
            return [result[1] for result in reranked_results][:re_top_n]
        else:
            return [candidate_ids[result["index"]] for result in reranked_results]

    def finalize_evaluation(self, dataset, cost_count, start_time, emb_model, re_model, emb_top_n, re_top_n, llm):
        finish_generation_time = time.time()
        generation_time = finish_generation_time - start_time

        # Evaluate with Ragas
        dataset = Dataset.from_dict(dataset)
        ragas_evaluator = RagasScoreEvaluator(dataset)
        score_df, evaluation_cost = ragas_evaluator.get_all_score_df()
        if score_df.empty:
            logging.error("Ragas evaluation returned an empty dataframe.")
        column_names = score_df.columns.tolist()

        finish_evaluation_time = time.time()
        evaluation_time = finish_evaluation_time - finish_generation_time
        total_time = time.time() - start_time

        final_cost = evaluation_cost + cost_count
        logging.info(f"Total Time: {total_time:.2f} seconds, Total Cost: {final_cost:.4f} USD")

        # Save results and send email
        results = {
            "Embedding model": [emb_model],
            "Reranking model": [re_model],
            "Emb_top_k": [emb_top_n],
            "Re_top_k": [re_top_n],
            "LLM": [llm],
            "Generation Time": [generation_time],
            "Evaluation Time": [evaluation_time],
            "Total Time": [total_time],
            "USD Cost": [final_cost],
        }
        for col in column_names:
            results[col] = [score_df[col].iloc[0]]

        self.save_results(results)
        self.send_email_notification(results, column_names, score_df)

        # Generate improved bar plot for RAGAS scores
        fig, ax = plt.subplots(figsize=(10, 6))

        # 提取 RAGAS 分數
        scores = [score_df[col].iloc[0] for col in column_names]

        # 設定顏色，對比分數高低
        colors = ['#4CAF50' if score >= 0.8 else '#FF7043' for score in scores]

        # 繪製柱狀圖
        bars = ax.bar(column_names, scores, color=colors, edgecolor='black')

        # 添加標籤（分數值）在每個柱頂
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02, f"{score:.2f}", ha='center', fontsize=10, color='black')

        # 圖表標題與軸標籤
        ax.set_title("RAGAS Scores", fontsize=16, fontweight='bold')
        ax.set_ylabel("Score", fontsize=14)
        ax.set_xticks(range(len(column_names)))
        ax.set_xticklabels(column_names, rotation=45, ha="right", fontsize=12)

        # 添加網格線（橫向）
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # 設定分數範圍
        ax.set_ylim(0, 1.1)  # 為分數標籤留出空間

        # 調整圖表布局
        plt.tight_layout()

        # Save plot as an image file
        output_path = "ragas_scores_plot.png"
        plt.savefig(output_path, format="png")
        plt.close(fig)

        return results, output_path



#pipeline_evaluation(src_file = "QA_1217FULLTEXT3_all_modified.csv", emb_model = "voyage-multilingual-2", re_model = "rerank-2", emb_top_n = 10, re_top_n = 3, llm = "gpt-4o")
#pipeline_evaluation(src_file = "QA_1217FULLTEXT3_all_modified_test.csv", emb_model = "voyage-multilingual-2", re_model = "rerank-2", emb_top_n = 10, re_top_n = 3, llm = "yentinglin/Llama-3-Taiwan-8B-Instruct")
#pipeline_evaluation(src_file = "QA_1217FULLTEXT3_all_modified_test.csv", emb_model = "voyage-multilingual-2", re_model = "rerank-2", emb_top_n = 10, re_top_n = 3, llm = "gpt-4o")
#pipeline_evaluation(src_file = "QA_1217FULLTEXT3_all_modified_test.csv", emb_model = "voyage-multilingual-2", re_model = "rerank-2", emb_top_n = 10, re_top_n = 3, llm = "gpt-4o-mini")
#pipeline_evaluation(src_file = "QA_1217FULLTEXT3_all_modified_test.csv", emb_model = "voyage-multilingual-2", re_model = "rerank-2", emb_top_n = 10, re_top_n = 3, llm = "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1")
#pipeline_evaluation(src_file = "QA_1217FULLTEXT3_all_modified_test.csv", emb_model = "voyage-multilingual-2", re_model = "rerank-2", emb_top_n = 10, re_top_n = 3, llm = "Qwen/Qwen1.5-7B-Chat")
#pipeline_evaluation(src_file = "QA_1217FULLTEXT3_all_modified_test.csv", emb_model = "voyage-multilingual-2", re_model = "rerank-2", emb_top_n = 10, re_top_n = 3, llm = "MediaTek-Research/Breeze-7B-Instruct-v1_0")
#pipeline_evaluation(src_file = "QA_1217FULLTEXT3_all_modified_test.csv", emb_model = "voyage-multilingual-2", re_model = "rerank-2", emb_top_n = 10, re_top_n = 3, llm = "mistralai/Mistral-7B-Instruct-v0.2")
#pipeline_evaluation(src_file = "QA_1217FULLTEXT3_all_modified_test.csv", emb_model = "voyage-multilingual-2", re_model = "rerank-2", emb_top_n = 10, re_top_n = 3, llm = "meta-llama/Meta-Llama-3-8B-Instruct")
#pipeline_evaluation(src_file = "QA_1217FULLTEXT3_all_modified_test.csv", emb_model = "voyage-multilingual-2", re_model = "rerank-2", emb_top_n = 10, re_top_n = 3, llm = "claude-3-opus-20240229")
#pipeline_evaluation(src_file = "QA_1217FULLTEXT3_all_modified_test.csv", emb_model = "voyage-multilingual-2", re_model = "rerank-2", emb_top_n = 10, re_top_n = 3, llm = "gemini-1.5-pro")

