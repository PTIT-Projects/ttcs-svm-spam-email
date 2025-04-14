FROM python:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY linear_svm.pkl .
COPY vectorizer.pkl .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]