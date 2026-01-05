# API de Previsões LSTM

Aplicação FastAPI para servir modelo de previsão de preços de ações baseado em redes neurais LSTM.

## 📋 Conteúdo

- `main.py` - Aplicação FastAPI com todos endpoints
- `utils.py` - Funções auxiliares para validação e processamento
- `__init__.py` - Inicialização do pacote
- `requirements.txt` - Dependências Python

## 🚀 Quick Start

### Instalar Dependências
```bash
pip install -r requirements.txt
```

### Rodar Localmente
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Com Docker
```bash
docker build -t lstm-api . && docker run -p 8000:8000 lstm-api
```

## 📚 Documentação

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- Ver documentação completa em `../../docs/GUIA_EXECUCAO_API.md`

## 🔌 Endpoints

- `GET /` - Raiz com links
- `GET /health` - Status da API
- `GET /info` - Informações do modelo
- `POST /predict` - Previsão única
- `POST /predict-batch` - Múltiplas previsões

## 📊 Entrada Esperada

Para previsões, forneça:
- `historical_prices`: Lista com data, close, high, low, volume
- `features_data`: Array (70, 34) com features normalizadas
- `n_steps`: Número de passos a prever

## 📈 Saída

```json
{
  "predicted_price": 152.35,
  "confidence_interval": {
    "lower_bound": 145.25,
    "upper_bound": 159.45,
    "confidence_level": 0.9523
  },
  "metrics": {
    "error_margin": 3.57,
    "mape": 4.77
  },
  "timestamp": "2025-01-01T12:00:00.000Z"
}
```

## 🛠️ Estrutura de Código

### main.py
- Inicialização FastAPI com CORS
- Carregamento de modelo no startup
- Definição de modelos Pydantic
- Implementação de 5 endpoints
- Error handling e logging

### utils.py
- `DataValidator`: Valida estrutura de dados
- `DataNormalizer`: Normaliza/desnormaliza features
- `MetricsCalculator`: Calcula intervalos de confiança
- `HistoricalDataProcessor`: Processa dados históricos
- `LoggingUtils`: Setup de logging

## 🔒 Segurança

- CORS habilitado (customize conforme necessário)
- Validação de tipos com Pydantic
- Error handling completo
- Logging de requisições

## 📊 Métricas do Modelo

- Arquivo: `../../data/models/lstm_5factor_final_20260101_224540.keras`
- R² Score: 0.9054
- MAE: $16.48
- MAPE: 4.77%
- RMSE: $24.32

## 📦 Dependências

- FastAPI 0.104.1
- Uvicorn 0.24.0
- Pydantic 2.5.0
- Keras 3.0.0
- TensorFlow 2.15.0
- NumPy, Pandas, scikit-learn

## 📝 Exemplo de Uso

```python
import requests

# Health check
response = requests.get('http://localhost:8000/health')
print(response.json())

# Fazer previsão
data = {
    "historical_prices": [
        {"date": "2024-12-20", "close": 150.0, "high": 151.0, "low": 149.0, "volume": 1000000}
    ],
    "features_data": [[0.0] * 34 for _ in range(70)],
    "n_steps": 1
}
response = requests.post('http://localhost:8000/predict', json=data)
print(response.json())
```

## 🐳 Docker

### Build
```bash
docker build -t lstm-stock-predictor-api:latest .
```

### Run
```bash
docker run -p 8000:8000 lstm-stock-predictor-api:latest
```

### Com Compose
```bash
docker-compose up -d
```

## 🧪 Testes

Rodar testes de todos endpoints:
```bash
python ../../test_api.py
```

## 📊 Monitoring

Logs são salvos em `../../logs/api.log`

Verificar status:
```bash
curl http://localhost:8000/health
```

## 💡 Próximas Melhorias

- [ ] Autenticação com API keys
- [ ] Rate limiting
- [ ] Caching de respostas
- [ ] Database para histórico
- [ ] Prometheus metrics
- [ ] WebSocket para streaming

## 📖 Documentação Completa

Ver `../../docs/GUIA_EXECUCAO_API.md`
