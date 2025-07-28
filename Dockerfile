# Use official Python base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements.txt and install all dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download and install spaCy models
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download en_core_web_md

# Copy local models if your code uses them
COPY local_models/ ./local_models/

# Copy the rest of the project files (code, etc.)
COPY . .

# Set the entry point (you can change this if needed)
CMD ["python", "main.py"]
