# evaluation.py
import logging
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from database import DatabaseConnection
from embedding import EmbeddingService, RerankingService
import pandas as pd

class Evaluator:
    def __init__(self, db_connection, embedding_service, reranking_service):
        self.db_connection = db_connection
        self.embedding_service = embedding_service
        self.reranking_service = reranking_service

    def retrieve_all_embeddings(self):
        # 从数据库中获取所有文档的嵌入和相关信息
        all_documents = list(self.db_connection.collection.find({}, {"embedding": 1, "text": 1, "id": 1}))
        logging.info(f"從資料庫檢索到 {len(all_documents)} 個Contexts")
        
        embeddings = [doc["embedding"] for doc in all_documents]
        texts = [doc["text"] for doc in all_documents]
        id = [doc["id"] for doc in all_documents]
        
        return embeddings, texts, id

    def find_top_k_similar_contexts(self, question_vector, embeddings, context_ids, top_k):
        similarities = cosine_similarity([question_vector], embeddings)[0]
        # 将相似度与 context_id 组成配对
        similarity_id_pairs = list(zip(similarities, context_ids))
        # 按相似度排序，取前 top_k 个
        top_k_similar_contexts = sorted(similarity_id_pairs, key=lambda x: x[0], reverse=True)[:top_k]
        return top_k_similar_contexts

    def evaluate_only_emb(self, questions_df, embeddings, context_ids, embedding_model_name, top_k):
        record_top_k = {"mrr": [], "hit_rate": []}
        record_top_k2 = {"mrr": [], "hit_rate": []}
        total_questions = len(questions_df)
        
        for index, row in tqdm(questions_df.iterrows(), total=total_questions, desc="Processing Questions"):
            question = row["Question"]
            real_id = row["id"]

            # 获取问题的嵌入向量
            question_vector = self.embedding_service.get_embedding_for_evaluation(
                input_text=question,
                model_name=embedding_model_name
            )
            
            # 找到最相似的 Top-K 文档
            top_k_similar_contexts = self.find_top_k_similar_contexts(
                question_vector, embeddings, context_ids, top_k
            )
            # 提取候选的 context_ids
            candidate_ids = [context_id for _, context_id in top_k_similar_contexts]
            candidate_ids_top_k2 = candidate_ids[:top_k-2]

            # 计算 MRR
            if real_id in candidate_ids:
                rank = candidate_ids.index(real_id) + 1
                mrr = 1 / rank
            else:
                mrr = 0
            hit_rate = 1 if real_id in candidate_ids else 0
            
            record_top_k["mrr"].append(mrr)
            record_top_k["hit_rate"].append(hit_rate)

            # 计算 MRR
            if real_id in candidate_ids_top_k2:
                rank = candidate_ids_top_k2.index(real_id) + 1
                mrr = 1 / rank
            else:
                mrr = 0
            hit_rate = 1 if real_id in candidate_ids_top_k2 else 0
            
            record_top_k2["mrr"].append(mrr)
            record_top_k2["hit_rate"].append(hit_rate)
  
        # 计算最终的 MRR 和 Hit Rate
        mrr_final_k2 = sum(record_top_k2["mrr"]) / total_questions
        hit_rate_final_k2 = sum(record_top_k2["hit_rate"]) / total_questions
        mrr_final = sum(record_top_k["mrr"]) / total_questions
        hit_rate_final = sum(record_top_k["hit_rate"]) / total_questions
        
        logging.info(f"Final MRR: {mrr_final_k2}")
        logging.info(f"Final Hit Rate: {hit_rate_final_k2}")
        logging.info(f"Final MRR: {mrr_final}")
        logging.info(f"Final Hit Rate: {hit_rate_final}")
        
        return mrr_final_k2, hit_rate_final_k2, mrr_final, hit_rate_final

    def evaluate_with_reranker(self, questions_df, emb_model, re_model, embeddings, context_ids, texts, top_k, rerank_top_n):
            record_top_k = {"mrr": [], "hit_rate": []}
            record_top_k2 = {"mrr": [], "hit_rate": []}
            total_questions = len(questions_df)

            for index, row in tqdm(questions_df.iterrows(), total=total_questions, desc="Evaluating with Re-ranker"):
                question = row["Question"]
                real_id = row["id"]

                try:
                    # 使用嵌入模型提取 Top-K 候選集
                    question_vector = self.embedding_service.get_embedding_for_evaluation(
                        input_text=question,
                        model_name=emb_model  # 替換為所需模型
                    )
                    top_k_similar_contexts = self.find_top_k_similar_contexts(
                        question_vector, embeddings, context_ids, top_k
                    )

                    candidate_ids = [context_id for _, context_id in top_k_similar_contexts]
                    candidate_texts = [texts[context_ids.index(context_id)] for context_id in candidate_ids]

                    print(candidate_ids)

                    # 使用 RerankingService 進行重新排序
                    reranked_results = self.reranking_service.get_topk_from_reranking(
                        query=question, 
                        candidate_text=candidate_texts, 
                        top_n=rerank_top_n,
                        model_name=re_model
                    )

                    # 檢查是否返回結果，避免空結果錯誤
                    if not reranked_results:
                        logging.warning(f"No reranked results for question: {question}")
                        continue

                    # 根據 reranked_results 提取排序後的 ID
                    if re_model == "rerank-2":
                        reranked_ids = [candidate_ids[result.index] for result in reranked_results]
                    elif re_model == "BAAI/bge-reranker-v2-m3" or re_model == "BAAI/bge-reranker-large" or re_model == "BAAI/bge-reranker-base":
                        reranked_results = sorted(zip(reranked_results, candidate_ids), reverse=True)
                        reranked_ids = [result[1] for result in reranked_results]
                        reranked_ids = reranked_ids[:rerank_top_n]
                    else:
                        reranked_ids = [candidate_ids[result["index"]] for result in reranked_results]
                    print(reranked_ids)
                    

                    reranked_ids_top_k2 = reranked_ids[:rerank_top_n-2]
                    '''
                    從top5取top3
                    '''
                     # 計算 MRR 和 Hit Rate
                    if real_id in reranked_ids_top_k2:
                        rank = reranked_ids_top_k2.index(real_id) + 1
                        mrr = 1 / rank
                        hit_rate = 1
                    else:
                        mrr = 0
                        hit_rate = 0

                    record_top_k2["mrr"].append(mrr)
                    print(mrr)
                    record_top_k2["hit_rate"].append(hit_rate)
                    print(hit_rate)

                    '''
                    top5
                    '''
                    # 計算 MRR 和 Hit Rate
                    if real_id in reranked_ids:
                        rank = reranked_ids.index(real_id) + 1
                        mrr = 1 / rank
                        hit_rate = 1
                    else:
                        mrr = 0
                        hit_rate = 0

                    record_top_k["mrr"].append(mrr)
                    print(mrr)
                    record_top_k["hit_rate"].append(hit_rate)
                    print(hit_rate)
 
                except Exception as e:
                    logging.error(f"Error processing question {real_id}: {e}")
                    continue

            # 計算最終的 MRR 和 Hit Rate
            mrr_final_top3 = sum(record_top_k2["mrr"]) / total_questions if total_questions else 0
            hit_rate_final_top3 = sum(record_top_k2["hit_rate"]) / total_questions if total_questions else 0
            mrr_final_top5 = sum(record_top_k["mrr"]) / total_questions if total_questions else 0
            hit_rate_final_top5 = sum(record_top_k["hit_rate"]) / total_questions if total_questions else 0

            logging.info(f"Re-ranker Final Top {rerank_top_n-2} MRR: {mrr_final_top3}")
            logging.info(f"Re-ranker Final Top {rerank_top_n-2} Hit Rate: {hit_rate_final_top3}")
            logging.info(f"Re-ranker Final Top {rerank_top_n} MRR: {mrr_final_top5}")
            logging.info(f"Re-ranker Final Top {rerank_top_n} Hit Rate: {hit_rate_final_top5}")

            return mrr_final_top3, hit_rate_final_top3, mrr_final_top5, hit_rate_final_top5
    

