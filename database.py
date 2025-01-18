# database.py
from pymongo import MongoClient
import logging

class DatabaseConnection:
    def __init__(self, mongodb_uri, db_name, collection_name):
        try:
            self.client = MongoClient(mongodb_uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            logging.info(f"成功連線到MongoDB '{db_name}' 和集合 '{collection_name}'。")
        except Exception as e:
            logging.error("連線 MongoDB 失敗")
            logging.error(f"錯誤提示: {e}")
            raise e

    def replace_document(self, filter_condition, document):
        try:
            self.collection.replace_one(filter_condition, document, upsert=True)
            #logging.info(f"chunk（id: {document.get('id')}）替換成功")
        except Exception as e:
            logging.error("chuck替換失敗")
            logging.error(f"錯誤提示: {e}")
            raise e