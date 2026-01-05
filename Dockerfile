# Dockerfile para LSTM Stock Price Predictor API
FROM python:3.11-slim

WORKDIR /app

# Copiar requirements
COPY src/api/requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY src/api/ /app/api/
COPY data/models/ /app/models/

# Criar estrutura de diretórios
RUN mkdir -p /app/logs

# Expor porta
EXPOSE 8000

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Comando para iniciar
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
