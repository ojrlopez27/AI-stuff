from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("GPT_4O_MINI_ENDPOINT"),
    openai_api_version=os.getenv("GPT_4O_MINI_API_VERSION"),
    deployment_name="gpt-4o-mini",
    openai_api_key=os.getenv("GPT_4O_MINI_API_KEY"),
    openai_api_type="azure"
)