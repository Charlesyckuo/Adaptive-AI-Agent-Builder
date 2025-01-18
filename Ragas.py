from ragas import evaluate, SingleTurnSample
from ragas.dataset_schema import SingleTurnSample
import os
import pandas as pd
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.cost import get_token_usage_for_openai
from ragas.metrics import (
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ContextEntityRecall,
    Faithfulness,
    NoiseSensitivity,
    SemanticSimilarity
)
from config import (
    OPENAI_API_KEY
)

#os.environ["OPENAI_API_KEY"] = "sk-None-NbeGLKsywBa66ykNkAjlT3BlbkFJrE6x3GLHFC0JI2a7vkSR"
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini-2024-07-18"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model = "text-embedding-3-large"))

class RagasScoreEvaluator:
    def __init__(self, dataset):
        """
        初始化 Ragas 評估器

        Args:
            dataset: 支援 RAGAS 的 `datasets.Dataset` 格式
        """
        self.dataset = dataset
    def get_all_score_df(self):
        """
        使用 Ragas 計算所有 QA 資料的分數，並回傳 DataFrame

        Returns:
            pd.DataFrame: 包含各個指標分數的 DataFrame
        """
        metrics = [
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            ContextEntityRecall(),
            NoiseSensitivity(), 
            Faithfulness(),
            ResponseRelevancy(),
            SemanticSimilarity()
        ]

        results = evaluate(
            dataset=self.dataset, 
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            show_progress = True,
            token_usage_parser = get_token_usage_for_openai
        )
        results.total_tokens()
        cost = results.total_cost(cost_per_input_token=0.150 / 1e6, cost_per_output_token=0.600 / 1e6)
        print(f"Evaluation Cost: {cost} USD")
        df = results.to_pandas()
        numeric_columns = df.select_dtypes(include=["number"])
        #print(numeric_columns.head(1))
        
        avg_scores = numeric_columns.mean().to_dict()
        #print(avg_scores)
        
        avg_df = pd.DataFrame([avg_scores])
        print(avg_df)


        return avg_df, cost

