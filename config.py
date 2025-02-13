import logging
import os

from openai import OpenAI
class Config:
    openai_api_key = os.getenv("gptkidz")
    chatgpt_client = OpenAI(api_key=openai_api_key)