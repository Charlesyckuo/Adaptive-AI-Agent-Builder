import time
import logging
import pandas as pd
from tqdm import tqdm
from config import (
    MONGODB_URI,
    COHERE_API_KEY,
    VOYAGE_API_KEY,
)
from embedding import EmbeddingService
from database import DatabaseConnection
from langchain.text_splitter import RecursiveCharacterTextSplitter

'''
用於將 voyage-multilingual-2 與 intfloat/multilingual-e5-large 寫入資料庫中
Table1 for voyage 
Table2 for e5
'''
def vector_ingestion(df, emb_model, table):
    '''
    df: 傳入的dataframe
    table: 指定寫入的table
    em_model: 使用哪個Embedding model 進行嵌入
    '''
    logging.basicConfig(level=logging.INFO)

    api_keys = {
        'COHERE_API_KEY': COHERE_API_KEY,
        'VOYAGE_API_KEY': VOYAGE_API_KEY,
    }
    embedding_service = EmbeddingService(api_keys)

    # 資料庫連線
    db_connection = DatabaseConnection(MONGODB_URI, 'RAGAS', table)

    # Start overall time for tracking
    start_time_overall = time.time()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

    current_id = 1
    current_context = ""
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing CSV file"):
        try:
            context = row['summary']
            source = row["filename"]
            if context != current_context:
                current_context = context

                # 使用 RecursiveCharacterTextSplitter 切割文本
                chunks = text_splitter.split_text(current_context)
                logging.info(f"檔案 '{source}' 的chunks數量: {len(chunks)}")
                
                for chunk in chunks:
                    # Model-specific logic for embedding generation
                    if emb_model == "intfloat/multilingual-e5-large":
                        chunk = "passage: " + chunk
                        embedding = embedding_service.get_e5_embeddings(input_text=chunk, model=emb_model)
                    else:
                        embedding = embedding_service.get_voyage_embeddings(input_text=chunk, model=emb_model, input_type="document")

                    # Store results in database
                    document = {
                        "id": current_id,
                        "context_num": current_id,
                        "text": chunk,
                        "embedding": embedding,
                        "source": source
                    }
                    db_connection.replace_document({"id": document["id"]}, document)

                    current_id += 1  # 每次插入都遞增 ID
            else:
                    logging.info(f"Context #{id} 跳過寫入資料庫")
        except Exception as e:
            logging.error(f"Failed to process QA entry #{index + 1}. Error: {e}")

    embeding_elapsed_time = time.time()-start_time_overall
    logging.info(f"Finish Processing CSV file: '{src_file}'")
    logging.info(f"Total Time Spent: {embeding_elapsed_time}")


# Read CSV file containing questions, contexts and Answer
src_file = "QA_1217FULLTEXT3_all_modified_for_ragas.csv"

try:
    df = pd.read_csv(src_file)
    logging.info(f"Successfully read CSV file: '{src_file}'")
except Exception as e:
    logging.error(f"Failed to read CSV file: '{src_file}'. Error: {e}")

vector_ingestion(df, "voyage-multilingual-2", "voyage")
vector_ingestion(df, "intfloat/multilingual-e5-large", "e5")