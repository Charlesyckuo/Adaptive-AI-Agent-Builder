import time
import logging
import pandas as pd
from tqdm import tqdm
from config import (
    MONGODB_URI,
    COHERE_API_KEY,
    VOYAGE_API_KEY,
    JINA_API_KEY,
    OPENAI_API_KEY,
    NVIDIA_SNOWFLAKE_API_KEY,
    MISTRAL_API_KEY,
    AIML_API_KEY,
    GEMINI_API_KEY,
)
from embedding import EmbeddingService, RerankingService
from database import DatabaseConnection
from evaluation import Evaluator
from email_notification import EmailNotification

class RetrieverEvaluator:
    def __init__(self):
        self.api_keys = {
            "COHERE_API_KEY": COHERE_API_KEY,
            "VOYAGE_API_KEY": VOYAGE_API_KEY,
            "JINA_API_KEY": JINA_API_KEY,
            "OPENAI_API_KEY": OPENAI_API_KEY,
            "NVIDIA_SNOWFLAKE_API_KEY": NVIDIA_SNOWFLAKE_API_KEY,
            "MISTRAL_API_KEY": MISTRAL_API_KEY,
            "AIML_API_KEY": AIML_API_KEY,
            "GEMINI_API_KEY": GEMINI_API_KEY,
        }
        self.embedding_service = EmbeddingService(self.api_keys)
        self.reranking_service = RerankingService(self.api_keys)

    def save_results_to_csv(self, results, filename):
        """保存結果到 CSV 文件"""
        try:
            df_existing = pd.read_csv(filename)
        except FileNotFoundError:
            df_existing = pd.DataFrame(columns=results.keys())

        df_new = pd.DataFrame(results)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(filename, index=False)
        logging.info(f"Results saved to {filename}")
    
    def send_email_notification_1(self, results):
        email_sender = EmailNotification(
            from_email="kuo901105aa@gmail.com",
            from_password="fpar qkgq xssd nqxy",
        )
        subject = "Embedding Model Evaluation Completed"
        body = f"""
        Embedding Model Evaluation Completed!
        Notice: 
        1. Embedding time includes generating embeddings of contexts and push into database
        3. Evaluation include caculating mrr and hit rate of every question-context pair

        Embedding Model: {results["Embedding Model"][0]}
        Embedding Top_k: {results["Top_k"][0]}
        MRR: {results["MRR"][0]}
        Hit Rate: {results["Hit Rate"][0]}
        Embedding Time: {results["Embedding Time"][0]:.2f} seconds
        Evaluation Time: {results["Evaluation Time"][0]:.2f} seconds
        Total Time: {results["Total Time"][0]:.2f} seconds

        """
        email_sender.send_email(
            to_email=["kuo901105aa@gmail.com"],
            subject=subject,
            body=body,
        )

    def send_email_notification_2(self, results):
        email_sender = EmailNotification(
            from_email="kuo901105aa@gmail.com",
            from_password="fpar qkgq xssd nqxy",
        )
        subject = "Embedding and Reranker Model Evaluation Completed"
        body = f"""
        Embedding and Reranker Model Evaluation Completed!
        Notice: 
        1. Embedding Top_k was set to  10 
        2. Embedding time includes generating embeddings of contexts and push into database
        3. Evaluation include reranking time and caculating mrr and hit rate of every question-context pair

        Embedding Model: {results["Embedding Model"][0]}
        Reranking Model: {results["Reranking Model"][0]}
        MRR: {results["MRR"][0]}
        Hit Rate: {results["Hit Rate"][0]}
        Reranker Top_k: {results["Top_k"][0]}
        Embedding Time: {results["Embedding Time"][0]:.2f} seconds
        Evaluation Time: {results["Evaluation Time"][0]:.2f} seconds
        Total Time: {results["Total Time"][0]:.2f} seconds

        """
        email_sender.send_email(
            to_email=["kuo901105aa@gmail.com"],
            subject=subject,
            body=body,
        )

    def generate_embeddings(self, df, emb_model, db_connection):
        """為所有文檔生成嵌入並保存到數據庫"""
        start_time = time.time()
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing CSV file"):
            context = row["summary"]
            doc_id = row["id"]
            try:
                # 根據模型生成嵌入
                if emb_model in ["intfloat/multilingual-e5-large", "intfloat/multilingual-e5-small"]:
                    context = "passage: " + context
                    embedding = self.embedding_service.get_e5_embeddings(input_text=context, model=emb_model)
                elif emb_model in ["BAAI/bge-large-zh-v1.5"]:
                    embedding = self.embedding_service.get_bge_embeddings(input_text=context, model=emb_model)
                else:
                    embedding = self.embedding_service.get_voyage_embeddings(input_text=context, model=emb_model, input_type="document")

                # 保存到數據庫
                document = {"id": doc_id, "text": context, "embedding": embedding}
                db_connection.replace_document({"id": doc_id}, document)
            except Exception as e:
                logging.error(f"Failed to process row {index}: {e}")
        elapsed_time = time.time() - start_time
        return elapsed_time

    def evaluate_embeddings(self, df, emb_model, top_n, evaluator, embeddings, context_ids):
        """評估嵌入模型"""
        start_time = time.time()
        mrr_1, hit_rate_1, mrr_2, hit_rate_2 = evaluator.evaluate_only_emb(
            questions_df=df, embeddings=embeddings, context_ids=context_ids, embedding_model_name=emb_model, top_k=top_n
        )
        elapsed_time = time.time() - start_time
        return mrr_1, hit_rate_1, mrr_2, hit_rate_2, elapsed_time

    def evaluate_reranking(self, df, emb_model, emb_top_n, re_model, re_top_n, evaluator, embeddings, context_ids, texts):
        """評估 Reranking 模型"""
        start_time = time.time()
        mrr_1, hit_rate_1, mrr_2, hit_rate_2 = evaluator.evaluate_with_reranker(
            questions_df=df,
            emb_model=emb_model,
            re_model=re_model,
            embeddings=embeddings,
            context_ids=context_ids,
            texts=texts,
            top_k=emb_top_n,
            rerank_top_n=re_top_n,
        )
        elapsed_time = time.time() - start_time
        return mrr_1, hit_rate_1, mrr_2, hit_rate_2, elapsed_time

    def run(self, src_file, emb_model, emb_top_n, task, re_model, re_top_n, db_state):
        """主函數：執行嵌入和重排評估"""
        logging.basicConfig(level=logging.INFO)

        # 初始化數據庫
        db_connection = DatabaseConnection(MONGODB_URI, "RAGAS", "Emb_rerank")

        # 加載數據集
        try:
            df = pd.read_csv(src_file)
        except Exception as e:
            logging.error(f"Failed to read CSV file: {e}")
            return

        generate_embedding_time = 0

        # 嵌入生成
        if db_state == "yes":
            generate_embedding_time = self.generate_embeddings(df, emb_model, db_connection)

        # 嵌入評估
        evaluator = Evaluator(db_connection, self.embedding_service, self.reranking_service)
        embeddings, texts, context_ids = evaluator.retrieve_all_embeddings()

        if task == "only_emb":
            mrr_1, hit_rate_1, mrr_2, hit_rate_2, evaluation_time = self.evaluate_embeddings(df, emb_model, emb_top_n, evaluator, embeddings, context_ids)
            
            results_1 = {"Embedding Model": [emb_model], "MRR": [mrr_1], "Hit Rate": [hit_rate_1], "Top_k": [emb_top_n-2], "Embedding Time": [generate_embedding_time], "Evaluation Time": [evaluation_time], "Total Time": [generate_embedding_time + evaluation_time]}
            self.save_results_to_csv(results_1, "embedding_results.csv")
            self.send_email_notification_1(results_1)
            
            results_2 = {"Embedding Model": [emb_model], "MRR": [mrr_2], "Hit Rate": [hit_rate_2], "Top_k": [emb_top_n], "Embedding Time": [generate_embedding_time], "Evaluation Time": [evaluation_time], "Total Time": [generate_embedding_time + evaluation_time]}
            self.save_results_to_csv(results_2, "embedding_results.csv")
            self.send_email_notification_1(results_2)
        else:
            mrr_1, hit_rate_1, mrr_2, hit_rate_2, evaluation_time = self.evaluate_reranking(df, emb_model, emb_top_n, re_model, re_top_n, evaluator, embeddings, context_ids, texts)

            results_1 = {"Embedding Model": [emb_model], "Reranking Model": [re_model], "MRR": [mrr_1], "Hit Rate": [hit_rate_1], "Top_k": [re_top_n-2], "Embedding Time": [generate_embedding_time], "Evaluation Time": [evaluation_time], "Total Time": [generate_embedding_time + evaluation_time]}
            self.save_results_to_csv(results_1, "reranking_results.csv")
            self.send_email_notification_2(results_1)

            results_2 = {"Embedding Model": [emb_model], "Reranking Model": [re_model], "MRR": [mrr_2], "Hit Rate": [hit_rate_2], "Top_k": [re_top_n], "Embedding Time": [generate_embedding_time], "Evaluation Time": [evaluation_time], "Total Time": [generate_embedding_time + evaluation_time]}
            self.save_results_to_csv(results_2, "reranking_results.csv")
            self.send_email_notification_2(results_2)

        


'''    
    main(emb_model = "text-embedding-3-large", db_state = "yes", top_n=3, task = "only_emb", re_model= "no")
    main(emb_model = "text-embedding-3-large", db_state = "no", top_n=5, task = "only_emb", re_model= "no")
    main(emb_model = "text-embedding-ada-002", db_state = "yes", top_n=3, task = "only_emb", re_model= "no")
    main(emb_model = "text-embedding-ada-002", db_state = "no", top_n=5, task = "only_emb", re_model= "no")
    main(emb_model = "voyage-3", db_state = "yes", top_n=3, task = "only_emb", re_model= "no")
    main(emb_model = "voyage-3", db_state = "no", top_n=5, task = "only_emb", re_model= "no")
    main(emb_model = "voyage-multilingual-2", db_state = "yes", top_n=3, task = "only_emb", re_model= "no")
    main(emb_model = "voyage-multilingual-2", db_state = "no", top_n=5, task = "only_emb", re_model= "no")
    main(emb_model = "BAAI/bge-m3", db_state = "yes", top_n=3, task = "only_emb", re_model= "no")
    main(emb_model = "BAAI/bge-m3", db_state = "no", top_n=5, task = "only_emb", re_model= "no")
    main(emb_model = "BAAI/bge-large-zh-v1.5", db_state = "yes", top_n=3, task = "only_emb", re_model= "no")
    main(emb_model = "BAAI/bge-large-zh-v1.5", db_state = "no", top_n=5, task = "only_emb", re_model= "no")
    main(emb_model = "intfloat/multilingual-e5-large", db_state = "yes", top_n=3, task = "only_emb", re_model= "no")
    main(emb_model = "intfloat/multilingual-e5-large", db_state = "no", top_n=5, task = "only_emb", re_model= "no")
    main(emb_model = "dunzhang/stella-large-zh-v3-1792d", db_state = "yes", top_n=3, task = "only_emb", re_model= "no")
    main(emb_model = "dunzhang/stella-large-zh-v3-1792d", db_state = "no", top_n=5, task = "only_emb", re_model= "no")
    main(emb_model = "embed-multilingual-v3.0", db_state = "yes", top_n=3, task = "only_emb", re_model= "no")
    main(emb_model = "embed-multilingual-v3.0", db_state = "no", top_n=5, task = "only_emb", re_model= "no")
    main(emb_model = "jina-embeddings-v3", db_state = "yes", top_n=3, task = "only_emb", re_model= "no")
    main(emb_model = "jina-embeddings-v3", db_state = "no", top_n=5, task = "only_emb", re_model= "no")



    main(emb_model = "voyage-multilingual-2", db_state = "yes", top_n=3, task = "only_emb", re_model= "rerank-multilingual-v3.0")
    main(emb_model = "voyage-multilingual-2", db_state = "no", top_n=5, task = "only_emb", re_model= "rerank-multilingual-v3.0")
    main(emb_model = "voyage-multilingual-2", db_state = "yes", top_n=3, task = "only_emb", re_model= "rerank-2")
    main(emb_model = "voyage-multilingual-2", db_state = "no", top_n=5, task = "only_emb", re_model= "rerank-2")
    main(emb_model = "voyage-multilingual-2", db_state = "yes", top_n=3, task = "only_emb", re_model= "jina-reranker-v2-base-multilingual")
    main(emb_model = "voyage-multilingual-2", db_state = "no", top_n=5, task = "only_emb", re_model= "jina-reranker-v2-base-multilingual")
    main(emb_model = "voyage-multilingual-2", db_state = "yes", top_n=3, task = "only_emb", re_model= "BAAI/bge-reranker-v2-m3")
    main(emb_model = "voyage-multilingual-2", db_state = "no", top_n=5, task = "only_emb", re_model= "BAAI/bge-reranker-v2-m3")
    main(emb_model = "voyage-multilingual-2", db_state = "yes", top_n=3, task = "only_emb", re_model= "BAAI/bge-reranker-large")
    main(emb_model = "voyage-multilingual-2", db_state = "no", top_n=5, task = "only_emb", re_model= "BAAI/bge-reranker-large")

    main(emb_model = "intfloat/multilingual-e5-large", db_state = "yes", top_n=3, task = "only_emb", re_model= "rerank-multilingual-v3.0")
    main(emb_model = "intfloat/multilingual-e5-large", db_state = "no", top_n=5, task = "only_emb", re_model= "rerank-multilingual-v3.0")
    main(emb_model = "intfloat/multilingual-e5-large", db_state = "yes", top_n=3, task = "only_emb", re_model= "rerank-2")
    main(emb_model = "intfloat/multilingual-e5-large", db_state = "no", top_n=5, task = "only_emb", re_model= "rerank-2")
    main(emb_model = "intfloat/multilingual-e5-large", db_state = "yes", top_n=3, task = "only_emb", re_model= "jina-reranker-v2-base-multilingual")
    main(emb_model = "intfloat/multilingual-e5-large", db_state = "no", top_n=5, task = "only_emb", re_model= "jina-reranker-v2-base-multilingual")
    main(emb_model = "intfloat/multilingual-e5-large", db_state = "yes", top_n=3, task = "only_emb", re_model= "BAAI/bge-reranker-v2-m3")
    main(emb_model = "intfloat/multilingual-e5-large", db_state = "no", top_n=5, task = "only_emb", re_model= "BAAI/bge-reranker-v2-m3")
    main(emb_model = "intfloat/multilingual-e5-large", db_state = "yes", top_n=3, task = "only_emb", re_model= "BAAI/bge-reranker-large")
    main(emb_model = "intfloat/multilingual-e5-large", db_state = "no", top_n=5, task = "only_emb", re_model= "BAAI/bge-reranker-large")
'''

''' 
    ARGS: emb_model, db_state, top_n, task, re_model
    main(emb_model = "voyage-multilingual-2", db_state = "yes", top_n=5, task = "emb_re", re_model= "rerank-multilingual-v3.0")
    main(emb_model = "voyage-multilingual-2", db_state = "no", top_n=5, task = "emb_re", re_model= "rerank-2")
    main(emb_model = "voyage-multilingual-2", db_state = "no", top_n=5, task = "emb_re", re_model= "jina-reranker-v2-base-multilingual")
    main(emb_model = "voyage-multilingual-2", db_state = "no", top_n=5, task = "emb_re", re_model= "BAAI/bge-reranker-v2-m3")
    main(emb_model = "voyage-multilingual-2", db_state = "no", top_n=5, task = "emb_re", re_model= "BAAI/bge-reranker-large")

    main(emb_model = "intfloat/multilingual-e5-large", db_state = "yes", top_n=5, task = "emb_re", re_model= "rerank-multilingual-v3.0")
    main(emb_model = "intfloat/multilingual-e5-large", db_state = "no", top_n=5, task = "emb_re", re_model= "rerank-2")
    main(emb_model = "intfloat/multilingual-e5-large", db_state = "no", top_n=5, task = "emb_re", re_model= "jina-reranker-v2-base-multilingual")
    main(emb_model = "intfloat/multilingual-e5-large", db_state = "no", top_n=5, task = "emb_re", re_model= "BAAI/bge-reranker-v2-m3")
    main(emb_model = "intfloat/multilingual-e5-large", db_state = "no", top_n=5, task = "emb_re", re_model= "BAAI/bge-reranker-large")
''' 

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#170900 bge large  / multilingual-e5-large
#22640 stella-large-zh-v3-1792d