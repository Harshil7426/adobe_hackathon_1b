FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm && \
    python -m spacy download en_core_web_md

COPY local_models/ ./local_models/
COPY . .

CMD ["python", "main.py"]
