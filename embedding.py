# embedding.py
import cohere
import voyageai
from mistralai import Mistral
import requests
import json
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
from FlagEmbedding import FlagReranker

class EmbeddingService:
    def __init__(self, api_keys):
        self.cohere_api_key = api_keys.get('COHERE_API_KEY')
        self.voyage_api_key = api_keys.get('VOYAGE_API_KEY')
        self.jina_api_key = api_keys.get('JINA_API_KEY')
        self.openai_api_key = api_keys.get('OPENAI_API_KEY')
        self.nvidia_snowflake_api_key = api_keys.get('NVIDIA_SNOWFLAKE_API_KEY')
        self.mistral_api_key = api_keys.get('MISTRAL_API_KEY')
        self.aiml_api_key = api_keys.get('AIML_API_KRY')
        self.gemini_api_key = api_keys.get('GEMINI_API_KEY')
        
        # 初始化客戶端
        self.cohere_client = cohere.Client(self.cohere_api_key) if self.cohere_api_key else None
        self.voyage_client = voyageai.Client(self.voyage_api_key) if self.voyage_api_key else None
        self.openai_client = OpenAI() if self.openai_api_key else None
        self.mistral_client = Mistral(api_key = self.mistral_api_key) if self.mistral_api_key else None
        self.nvidia_snowflake_client = OpenAI(api_key=self.nvidia_snowflake_api_key, base_url="https://integrate.api.nvidia.com/v1")
        #self.aiml_client = openai.OpenAI(base_url="https://api.aimlapi.com", api_key = self.aiml_api_key)

    '''
    input_type: "search_query" "search_document"
    modle: "embed-multilingual-v3.0" "embed-multilingual-light-v3.0"
    '''
    def get_cohere_embeddings(self, input_text, model, input_type):
        if not self.cohere_client:
            raise ValueError("找不到 Cohere API key")
        response = self.cohere_client.embed(
            texts=[input_text],
            model=model,
            input_type=input_type
        )
        response = response.embeddings
        return response[0]
    
    '''
    input_type: "query" \ "document"
    model: "voyage-multilingual-2" \ "voyage-3" \ "voyage-3-lite"
    '''
    def get_voyage_embeddings(self, input_text, model, input_type):
        if not self.voyage_client:
            raise ValueError("找不到 Voyage API key")
        result = self.voyage_client.embed(
            [input_text],
            model=model,
            input_type=input_type
        )
        return result.embeddings[0]

    '''
    input_type: "retrieval.query" \ "retrieval.passage"
    model: "jina-embeddings-v3"
    '''
    def get_jina_embeddings(self, input_text, model, input_type):  
        if not self.jina_api_key:
            raise ValueError("找不到 Jina AI API key")
        url = 'https://api.jina.ai/v1/embeddings'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.jina_api_key}'
        }
        data = {
            "model": model,
            "task": input_type,
            "dimensions": "1024",
            "late_chunking": True,
            "embedding_type": "float",
            "input": [input_text]
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code != 200:
            raise Exception(f"Jina AI API 請求失敗，狀態碼：{response.status_code}，返回內容：{response.text}")
        return response.json()["data"][0]["embedding"]

    '''
    model: "text-embedding-ada-002" \ "text-embedding-3-small" \ "text-embedding-3-large"
    '''
    def get_openai_embeddings(self, input_text, model):
        if not self.openai_api_key:
            raise ValueError("找不到 OpenAI API key")
        response = self.openai_client.embeddings.create(input=input_text, model=model)
        return response.data[0].embedding

    '''
    input_type: "query" \ "passage"
    model: "nvidia/nv-embedqa-e5-v5"
    '''
    def get_nvidia_embeddings(self, input_text, model, input_type):
        '''if not self.nvidia_snowflake_api_key:
            raise ValueError("找不到 NVIDIA Snowflake API key")'''
        response = self.nvidia_snowflake_client.embeddings.create(
            input=[input_text],
            model=model,
            encoding_format="float",
            extra_body={"input_type": input_type, "truncate": "NONE"}
        )
        return response.data[0].embedding
    
    '''
    input_type: "query" \ "passage"
    model: "BAAI/bge-small-zh-v1.5" \ "BAAI/bge-base-zh-v1.5" \ "BAAI/bge-large-zh-v1.5"
    '''
    def get_bge_embeddings(self, input_text, model): 
        model = SentenceTransformer(model)
        response = model.encode(input_text)
        response = response.tolist()
        return response
 
    '''
    other: 在input_text中加入query: ; passage: prefix
    '''
    def get_e5_embeddings(self, input_text, model): #    intfloat/multilingual-e5-large
        model = SentenceTransformer(model)
        response = model.encode(input_text)
        response = response.tolist()
        return response
    
    '''
    input_type: "query" \ "passage"
    model: "intfloat/e5-mistral-7b-instruct"
    '''
    def get_e5_mistral_embeddings(self, input_text, model, input_type): #    intfloat/multilingual-e5-large
        model = SentenceTransformer(model)
        if input_type == "query":
            response = model.encode(input_text, prompt_name="web_search_query")
        else:
            response = model.encode(input_text)
        response = response.tolist()
        return response
    
    '''
    model: "textembedding-gecko@001" \ "textembedding-gecko@003" \ "textembedding-gecko-multilingual@001" \"text-multilingual-embedding-002"
    '''
    def get_aiml_google_embeddings(self, input_text, model):
        response = self.aiml_client.embeddings.create(input = input_text, model = model)
        response = response.json()['data'][0]['embedding']
        return response
    
    '''
    model: "dunzhang/stella-mrl-large-zh-v3.5-1792d" \ "dunzhang/stella-large-zh-v3-1792d" \ "infgrad/stella-base-zh-v3-1792d" / "infgrad/stella-base-zh-v2" / "infgrad/stella-large-zh-v2"
    '''
    def get_stella_embeddings(self, input_text, model): 
        model = SentenceTransformer(model)
        response = model.encode(input_text)
        response = response.tolist()
        return response
        
    def get_embedding_for_evaluation(self, input_text, model_name):
        if model_name in ["embed-multilingual-light-v3.0", "embed-multilingual-v3.0"]:
            return self.get_cohere_embeddings(
                input_text=input_text,
                model=model_name,
                input_type="search_query"
            )
        elif model_name in ["voyage-multilingual-2", "voyage-3", "voyage-3-lite"]:
            return self.get_voyage_embeddings(
                input_text=input_text,
                model=model_name,
                input_type="query"
            )
        elif model_name == "jina-embeddings-v3":
            return self.get_jina_embeddings(
                input_text=input_text,
                model=model_name,
                input_type="retrieval.query"
            )
        elif model_name in ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]:
            return self.get_openai_embeddings(
                input_text=input_text,
                model=model_name
            )
        elif model_name in ["BAAI/bge-m3", "BAAI/bge-base-zh-v1.5", "BAAI/bge-large-zh-v1.5", "BAAI/bge-small-zh-v1.5"]:
            return self.get_bge_embeddings(
                input_text=input_text,
                model=model_name
            )
        elif model_name in ["intfloat/multilingual-e5-large", "intfloat/multilingual-e5-base", "intfloat/multilingual-e5-small"]:
            return self.get_e5_embeddings(
                input_text="query" + input_text,
                model=model_name
            )
        elif model_name == "intfloat/e5-mistral-7b-instruct":
            return self.get_e5_mistral_embeddings(
                input_text=input_text,
                model=model_name,
                input_type="query"
            )
        elif model_name in ["dunzhang/stella-mrl-large-zh-v3.5-1792d", "dunzhang/stella-large-zh-v3-1792d", "infgrad/stella-base-zh-v3-1792d", "infgrad/stella-base-zh-v2", "infgrad/stella-large-zh-v2"]:
            return self.get_stella_embeddings(
                input_text=input_text,
                model=model_name
            )
        elif model_name == "nvidia/nv-embedqa-e5-v5":
            return self.get_nvidia_embeddings(
                input_text=input_text,
                model=model_name,
                input_type="query"
            )
        else:
            raise ValueError(f"不支持: {model_name}")
'''
        elif model_name == "mistral-embed":
            return self.get_mistral_embeddings(
                input_text = input_text,
                model = model_name
            )
        elif model_name == "textembedding-gecko@001" or model_name == "textembedding-gecko@003" or model_name == "textembedding-gecko-multilingual@001" or model_name == "text-multilingual-embedding-002":
            return self.get_aiml_google_embeddings(
                input_text = input_text,
                model = model_name
            )
        elif model_name == "text-embedding-004":
            return self.get_gemini_embeddings(
                input_text = input_text,
                model = model_name
'''

'''
    def get_mistral_embeddings(self, input_text, model):
        response = self.mistral_client.embeddings.create(model = model, inputs = [input_text])
        response = response.json()['data'].embedding
        return response
'''

class RerankingService:
    def __init__(self, api_keys):
        self.cohere_api_key = api_keys.get('COHERE_API_KEY')
        self.voyage_api_key = api_keys.get('VOYAGE_API_KEY')
        self.jina_api_key = api_keys.get('JINA_API_KEY')

        # 初始化客戶端
        self.cohere_client = cohere.ClientV2(self.cohere_api_key) if self.cohere_api_key else None
        self.voyage_client = voyageai.Client(self.voyage_api_key) if self.voyage_api_key else None

    def get_jina_rerank(self, query, candidates, top_n, model="jina-reranker-v2-base-multilingual"):
        url = 'https://api.jina.ai/v1/rerank'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.jina_api_key}'
        }
        data = {
            "model": model,
            "query": query,
            "top_n": top_n,
            "documents": candidates
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Jina API 請求失敗，狀態碼：{response.status_code}，返回內容：{response.text}")
        # 打印回傳的 JSON 結構
        #print("Jina API response:", response.json())
        # 提取排序結果
        reranked_results = response.json().get("results", [])

        if not reranked_results:
            print("No reranked results found.")
        return reranked_results
    
    def get_cohere_rerank(self, query, candidates, top_n, model="rerank-multilingual-v3.0"):
        url = "https://api.cohere.com/v2/rerank"
        # 請求標頭
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cohere_api_key}"
        }

        # 請求的數據（與 curl 中的 --data 參數內容相同）
        data = {
            "model": model,
            "query": query,
            "top_n": top_n,
            "documents": candidates
        }

        # 發送 POST 請求
        response = requests.post(url, headers=headers, data=json.dumps(data))

        # 檢查是否成功
        if response.status_code == 200:
            print("Success! Response:")
            #print(response.json())  # 打印返回的 JSON 內容
            return response.json().get("results", [])
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(response.text)  # 打印錯誤訊息
    
    def get_voyage_rerank(self, query, candidates, top_n, model):
        response = self.voyage_client.rerank(
            query = query,
            documents = candidates,
            model = model, 
            top_k = top_n
        )
        return response.results
    
    def get_bge_rerank(self, query, candidates, top_n, model):
        reranker = FlagReranker(model, use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
        rerank_scores = reranker.compute_score([[query, doc] for doc in candidates], normalize=True)
        return rerank_scores

    def get_topk_from_reranking(self, query, candidate_text, top_n, model_name):
        if model_name == "rerank-multilingual-v3.0":
            return self.get_cohere_rerank(
                query=query,
                candidates = candidate_text,
                top_n = top_n,
                model=model_name
            )
        elif model_name in ["rerank-2", "rerank-2-lite"]:
            return self.get_voyage_rerank(
                query = query,
                candidates = candidate_text,
                model = model_name,
                top_n = top_n
            )
        elif model_name in ["jina-reranker-v2-base-multilingual"]:
            return self.get_jina_rerank(
                query=query,
                candidates = candidate_text,
                top_n= top_n,
                model=model_name
            )
        elif model_name in ["BAAI/bge-reranker-v2-m3", "BAAI/bge-reranker-large", "BAAI/bge-reranker-base"]:
            return self.get_bge_rerank(
                query=query,
                candidates = candidate_text,
                top_n = top_n,
                model=model_name
            )    
        else:
            raise ValueError(f"不支持: {model_name}")