FROM python:latest

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model files and application code
COPY app.py .
COPY linear_svm.pkl .
COPY hashing_vectorizer.pkl .

# Expose the port Streamlit runs on
EXPOSE 8501

# Set up entrypoint
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]