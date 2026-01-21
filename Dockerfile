# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copia arquivos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Treina o modelo durante o build (ou você poderia copiar o modelo pronto)
RUN python src/train.py

# Comando para iniciar a API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
