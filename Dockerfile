FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLP Models
RUN python -m spacy download en_core_web_sm
RUN python -c "import nltk; nltk.download('wordnet')"

# Copy the rest of your backend files
COPY . .

# Hugging Face Spaces requires port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
