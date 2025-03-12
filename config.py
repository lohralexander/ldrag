import logging
import os

from openai import OpenAI


class Config:
    openai_api_key = os.getenv("ldrag_openai_apikey")
    chatgpt_client = OpenAI(api_key=openai_api_key)


logger = logging.getLogger("OntologyApp")
if not logger.hasHandlers():
    logger.addHandler(logging.NullHandler())
