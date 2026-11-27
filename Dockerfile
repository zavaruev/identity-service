FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py ./
COPY static/ ./static/
COPY scripts/ ./scripts/
RUN mkdir -p /app/data
ENV PYTHONUNBUFFERED=1
EXPOSE 18794
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "18794"]
