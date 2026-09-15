FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY code ./code
COPY models ./models
WORKDIR /app/code
EXPOSE 8050
ENV HOST=0.0.0.0
CMD ["python", "app.py"]

