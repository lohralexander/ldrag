FROM python:3.10-slim-buster

WORKDIR /app

COPY gpt_chatbot_ldrag/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ldrag /app/ldrag

COPY gpt_chatbot_ldrag/ontology/ontology_43.json ./ontology/ontology_43.json

COPY gpt_chatbot_ldrag /app/gpt_chatbot_ldrag

EXPOSE 5000

ENV PYTHONUNBUFFERED=1

ENV PYTHONPATH=/app

CMD ["python", "gpt_chatbot_ldrag/app.py"]
