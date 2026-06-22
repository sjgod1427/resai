FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY flask_app.py llm_supervisor.py ./
COPY templates/ ./templates/
COPY data/ ./data/

ENV PORT=7860
EXPOSE 7860

CMD ["python", "flask_app.py"]
