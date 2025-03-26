FROM python:3.10-slim

WORKDIR /app

# Copy requirements file (assuming it's in the root directory)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python files from root directory
COPY pypi_ldrag/config.py gptconnector.py graphdbconnector.py ontology.py ontology_io.py retriever.py ./

# Copy the entire retriever directory structure
COPY retriever/ ./retriever/

EXPOSE 5000

# Run the app.py from the retriever directory
CMD ["python", "retriever/app.py"]