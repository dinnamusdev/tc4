# Guia de Execução da API

## Instalação e Execução Local

### 1. Instalar Dependências
```bash
pip install -r src/api/requirements.txt
```

### 2. Rodar a API Localmente
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

A API estará disponível em: http://localhost:8000

### 3. Acessar Documentação Interativa
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Execução com Docker

### 1. Criar Imagem Docker
```bash
docker build -t lstm-stock-predictor-api:latest .
```

### 2. Rodar Container
```bash
docker run -p 8000:8000 lstm-stock-predictor-api:latest
```

### 3. Com Docker Compose (Recomendado)
```bash
docker-compose up -d
```

Para parar:
```bash
docker-compose down
```

---

## Endpoints da API

### 1. Health Check
**GET** `/health`

Verifica se a API e o modelo estão funcionando corretamente.

**Resposta (200):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "timestamp": "2025-01-01T12:00:00.000Z"
}
```

---

### 2. Informações do Modelo
**GET** `/info`

Retorna informações sobre o modelo LSTM utilizado.

**Resposta (200):**
```json
{
  "model_name": "BiLSTM with Attention",
  "architecture": "Bidirectional LSTM + Attention Layer",
  "input_shape": [70, 34],
  "sequence_length": 70,
  "num_features": 34,
  "metrics": {
    "r2_score": 0.9054,
    "mae": 16.48,
    "mape": 4.77,
    "rmse": 24.32
  },
  "features": [
    "Close Price", "Volume", "High", "Low", 
    "MA_5", "MA_10", "MA_20", "RSI", "MACD", "Bollinger_Upper", 
    "Bollinger_Lower", "ATR", "ADX", "OBV",
    "P&L Variables", "Volatility Proxies", "Factor Proxies",
    "Cross-Factor Interactions"
  ]
}
```

---

### 3. Fazer uma Previsão
**POST** `/predict`

Realiza uma previsão de preço futuro com base em dados históricos de preço brutos (OHLCV).

**Request:**
```json
{
  "historical_prices": [
    {
      "date": "2024-12-10",
      "close": 150.00,
      "high": 151.50,
      "low": 149.50,
      "open": 150.25,
      "volume": 1000000
    },
    {
      "date": "2024-12-11",
      "close": 151.00,
      "high": 152.00,
      "low": 150.00,
      "open": 150.75,
      "volume": 1100000
    },
    "... mais 68 dias de dados históricos ..."
  ],
  "n_steps": 1
}
```

**Explicação:**
- `historical_prices`: Array com **MÍNIMO 70 dias** de dados OHLCV brutos
  - `date`: Data no formato YYYY-MM-DD
  - `open`: Preço de abertura do dia
  - `high`: Preço máximo do dia
  - `low`: Preço mínimo do dia
  - `close`: Preço de fechamento do dia
  - `volume`: Volume de negociação
- `n_steps`: Quantos passos futuros prever (1-10 dias)

**Resposta (200):**
```json
{
  "prediction": 152.35,
  "confidence_interval": {
    "lower": 144.73,
    "upper": 159.97,
    "confidence_level": "95%"
  },
  "timestamp": "2025-01-01T12:00:00.000000",
  "model_info": {
    "model_name": "LSTM 5-Factor",
    "version": "1.0.0",
    "architecture": "2-layer LSTM (160→80 units)",
    "input_source": "Histórico OHLCV bruto",
    "features_calculated": 34,
    "sequence_length": 70
  }
}
```

**Fluxo Interno:**
1. ✅ Recebe histórico OHLCV bruto (70 dias)
2. ✅ Valida dados (formato, coerência)
3. ✅ Calcula 34 features técnicas automaticamente:
   - Close, MA_5, MA_10, MA_20
   - RSI, MACD, Bollinger Bands
   - ATR, ADX, OBV
   - Volatilidade, Volume, Tendência
   - E mais 20+ features técnicas
4. ✅ Normaliza as 34 features
5. ✅ Alimenta modelo LSTM (input: 70 timesteps × 34 features)
6. ✅ Modelo retorna previsão
7. ✅ Desnormaliza resultado para preço real
8. ✅ Calcula intervalo de confiança (±4.77% baseado em MAPE)
9. ✅ Retorna resultado em JSON

**Erros Possíveis:**
- `400 Bad Request`: 
  - Menos de 70 dias de histórico
  - Dados inválidos (high < low, etc)
  - n_steps fora do intervalo 1-10
- `503 Service Unavailable`: Modelo não carregado


---

### 4. Previsões em Lote
**POST** `/predict-batch`

Realiza múltiplas previsões de uma vez com históricos diferentes.

**Request:**
```json
{
  "predictions": [
    {
      "historical_prices": [
        {"date": "2024-12-10", "close": 150.00, "high": 151.50, "low": 149.50, "open": 150.25, "volume": 1000000},
        "... mais 69 dias ..."
      ],
      "n_steps": 1
    },
    {
      "historical_prices": [
        {"date": "2024-12-10", "close": 200.00, "high": 201.50, "low": 199.50, "open": 200.25, "volume": 2000000},
        "... mais 69 dias ..."
      ],
      "n_steps": 1
    }
  ]
}
```

**Resposta (200):**
```json
{
  "predictions": [
    {
      "prediction": 152.35,
      "timestamp": "2025-01-01T12:00:00.000000"
    },
    {
      "prediction": 202.45,
      "timestamp": "2025-01-01T12:00:01.000000"
    }
  ]
}
```

---

## Testando com cURL

### Health Check
```bash
curl -X GET http://localhost:8000/health
```

### Informações do Modelo
```bash
curl -X GET http://localhost:8000/info
```

### Fazer uma Previsão
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "historical_prices": [
      {"date": "2024-12-20", "close": 150.00, "high": 151.00, "low": 149.00, "volume": 1000000}
    ],
    "features_data": [[...34 features...]],
    "n_steps": 1
  }'
```

---

## Variáveis de Ambiente

Criar arquivo `.env` (opcional):
```
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
MODEL_PATH=/app/models/lstm_5factor_final_20260101_224540.keras
SCALER_PATH=/app/models/scaler_5factor_final_20260101_224540.pkl
```

---

## Monitoramento

### Logs
Os logs são salvos em `./logs/api.log`

### Verificar Status
```bash
curl http://localhost:8000/health
```

### Parar a API
```bash
# Local
Ctrl+C

# Docker
docker-compose down

# Container específico
docker stop lstm-stock-predictor-api
```

---

## Troubleshooting

### Erro: "Model not loaded"
- Verifique se os arquivos do modelo existem em `data/models/`
- Reinicie a API

### Erro: "Invalid features"
- Valide se você tem exatamente 34 features
- Verifique se os dados estão normalizados corretamente

### Erro: "Connection refused"
- Verifique se a porta 8000 está disponível
- Use: `lsof -i :8000` (Linux/Mac) ou `netstat -ano | findstr :8000` (Windows)

### Performance Lenta
- Verifique recursos disponíveis (CPU, RAM)
- Considere usar GPU com TensorFlow: `pip install tensorflow[and-cuda]`

---

## Próximos Passos

1. **Testes Automatizados**: Criar testes unitários com pytest
2. **Autenticação**: Implementar API keys/JWT
3. **Rate Limiting**: Limitar requisições por IP
4. **Caching**: Cachear previsões repetidas
5. **Monitoring**: Integrar Prometheus/Grafana
6. **Deploy em Cloud**: AWS/GCP/Azure/Heroku
