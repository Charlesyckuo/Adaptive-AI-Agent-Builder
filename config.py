# config.py
import os
from dotenv import load_dotenv

# 加載env讀環境變量
load_dotenv()

# 獲取所有API和URI
VOYAGE_API_KEY = os.getenv('VOYAGE_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
JINA_API_KEY = os.getenv('JINA_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NVIDIA_SNOWFLAKE_API_KEY = os.getenv('NVIDIA_SNOWFLAKE_API_KEY')
MONGODB_URI = os.getenv('MONGODB_URI')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
AIML_API_KEY = os.getenv('AIML_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')

# 檢查所有環境變量是否存在
if not MONGODB_URI:
    raise ValueError("找不到 MONGODB_URI 環境變量，请在 .env 文件中設置")

if not VOYAGE_API_KEY:
    raise ValueError("找不到 VOYAGE_API_KEY 環境變量，請在 .env 文件中設置")

if not COHERE_API_KEY:
    raise ValueError("找不到 COHERE_API_KEY 環境變量，請在 .env 文件中設置")

if not JINA_API_KEY:
    raise ValueError("找不到 JINA_API_KEY 環境變量，請在 .env 文件中設置")

if not OPENAI_API_KEY:
    raise ValueError("找不到 OPENAI_API_KEY 環境變量，請在 .env 文件中設置")

if not NVIDIA_SNOWFLAKE_API_KEY:
    raise ValueError("找不到 NVIDIA_SNOWFLAKE_API_KEY 環境變量，請在 .env 文件中設置")

if not MISTRAL_API_KEY:
    raise ValueError("找不到 MISTRAL_API_KEY 環境變量，請在 .env 文件中設置")

if not AIML_API_KEY:
    raise ValueError("找不到 AIML_API_KEY 環境變量，請在 .env 文件中設置")

if not GEMINI_API_KEY:
    raise ValueError("找不到 GEMINI_API_KEY 環境變量，請在 .env 文件中設置")

if not CLAUDE_API_KEY:
    raise ValueError("找不到 CLUADE_API_KEY 環境變量，請在 .env 文件中設置")