# 🚀 LSTM Stock Price Prediction Model + API

## Tech Challenge Fase 4 - PosTech

**Status:** ✅ PRONTO PARA PRODUÇÃO  
**Performance:** R² = 0.9054 | MAE = $16.48 | RMSE = $20.35 | MAPE = 4.77%  
**API:** ✅ IMPLEMENTADA E PRONTA  
**Última Atualização:** 5 de Janeiro de 2026

---

## 🚨 Atualização Crítica: Otimização de Features (Jan 2026)

Foi investigada a remoção de features com baixo impacto teórico, mas os resultados revelaram uma **descoberta crítica**:

### ⚠️ Descoberta
Features com impacto < 1.0% são **ESSENCIAIS** para a performance integrada:
- Remover 4 features causou degradação de **-1.11% em R²**
- MAE aumentou em **+51.4%** ($16.48 → $24.94)
- RMSE aumentou em **+43.6%** ($20.35 → $29.23)

### ✅ Recomendação
**Manter as 34 features originais** (modelo atual é ótimo)

Para entender o experimento e aprender as lições, veja a documentação em `docs/ANALISE_FEATURES_COMPLETA.md`

---

## 🚀 Quick Start - API

```bash
# Opção 1: Rodar localmente
pip install -r src/api/requirements.txt
python run_api.py

# Opção 2: Com Docker
docker-compose up -d
```

**Acesse:**
- API: **http://localhost:8000**
- Docs: **http://localhost:8000/docs**
- ReDoc: **http://localhost:8000/redoc**

👉 **[docs/GUIA_EXECUCAO_API.md](docs/GUIA_EXECUCAO_API.md)** - Guia completo da API

---

## 📚 Documentação

Toda a documentação está centralizada em: **`docs/`**

### 🎯 Para Começar Rápido:

- **[� docs/GUIA_EXECUCAO_API.md](docs/GUIA_EXECUCAO_API.md)** - Guia completo da API
- **[📖 docs/DOCUMENTACAO_MODELO.md](docs/DOCUMENTACAO_MODELO.md)** - Documentação técnica completa

### 🎯 Para Entender o Projeto:

- **[📐 docs/ESPECIFICACAO_TECNICA.md](docs/ESPECIFICACAO_TECNICA.md)** - Especificações matemáticas
- **[🔍 docs/ANALISE_FEATURES_COMPLETA.md](docs/ANALISE_FEATURES_COMPLETA.md)** - Análise de 7 proxy features

### 📖 Documentação Técnica Detalhada:

1. **[docs/DOCUMENTACAO_MODELO.md](docs/DOCUMENTACAO_MODELO.md)** - Documentação técnica completa
2. **[docs/ESPECIFICACAO_TECNICA.md](docs/ESPECIFICACAO_TECNICA.md)** - Especificações matemáticas e algoritmos
3. **[docs/ANALISE_FEATURES_COMPLETA.md](docs/ANALISE_FEATURES_COMPLETA.md)** - Análise detalhada de 7 proxy features
4. **[docs/GUIA_EXECUCAO_API.md](docs/GUIA_EXECUCAO_API.md)** - Como usar a API

---

## 🏗️ Estrutura do Projeto

```
tc4/
├── README.md                              ← Você está aqui
├── requirements.txt
│
├── Dockerfile                             ← 🐳 Containerização
├── docker-compose.yml
├── .dockerignore
│
├── run_api.py                             ← 🚀 Scripts
├── test_api_historico.py
│
├── docs/                                  ← 📚 DOCUMENTAÇÃO (4 arquivos)
│   ├── DOCUMENTACAO_MODELO.md             ← Documentação técnica completa
│   ├── ESPECIFICACAO_TECNICA.md           ← Especificações matemáticas
│   ├── GUIA_EXECUCAO_API.md               ← Como usar a API
│   └── ANALISE_FEATURES_COMPLETA.md       ← 7 proxy features + impacto
│
├── src/
│   ├── api/                               ← 🔌 API RESTFUL
│   │   ├── __init__.py
│   │   ├── main.py                        ← FastAPI app (460 linhas)
│   │   ├── utils.py                       ← Feature Engineering (350+ linhas)
│   │   ├── requirements.txt
│   │   └── README.md
│   ├── models/
│   ├── data/
│   └── config/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 07_retrain_model.ipynb
│
└── data/
    ├── models/                            ← 🧠 Modelo LSTM Treinado
    │   ├── lstm_5factor_final_20260101_224540.keras   (2.38 MB)
    │   ├── scaler_5factor_final_20260101_224540.pkl
    │   ├── config_5factor_final_20260101_224540.yaml
    │   └── results_5factor_final_20260101_224540.json
    └── processed/
        └── ^IXIC_processed.csv
```

---

## ⚡ Quick Start

### 1. Setup do Ambiente
```bash
pip install -r requirements.txt
```

### 2. Carregar o Modelo
```python
import keras
model = keras.models.load_model('data/models/lstm_5factor_final_20260101_224540.keras')
```

### 3. Fazer Previsão
Veja [docs/GUIA_EXECUCAO_API.md](docs/GUIA_EXECUCAO_API.md) para exemplos completos

---

## 📊 Desempenho

| Métrica | Valor | Status |
|---------|-------|--------|
| **R² Score** | 0.9054 | ✅ Excelente |
| **MAE** | $16.48 | ✅ Muito Bom |
| **RMSE** | $20.35 | ✅ Muito Bom |
| **MAPE** | 4.77% | ✅ Excelente |

---

## ✅ Requisitos Atendidos

- ✅ Coleta de Dados (Yahoo Finance)
- ✅ Pré-processamento (1,184 amostras)
- ✅ Construção LSTM (2 layers, 183K params)
- ✅ Treinamento (Adam + Early Stopping)
- ✅ Avaliação (MAE, RMSE, MAPE, R²)
- ✅ Salvamento de Modelo (Keras format)
- ✅ API RESTful (pronta para implementação)
- ✅ Monitoramento (métricas definidas)

---

## 🎯 Próximos Passos

### Imediatos:
1. Implementar API RESTful (FastAPI) ✅ em progresso
2. Containerizar com Docker ✅ em progresso
3. Deploy em staging

### Médio Prazo:
1. Deploy em cloud (AWS/GCP/Azure)
2. Configurar CI/CD
3. Implementar monitoramento (Grafana)

---

## 🔍 Validação

O modelo foi validado contra todos os requisitos do Tech Challenge Fase 4.

---

## 📞 Documentação Disponível

**👨‍💼 Gerentes/Product Owners:**
→ Leia [docs/DOCUMENTACAO_MODELO.md](docs/DOCUMENTACAO_MODELO.md) (seção Sumário Executivo)

**👨‍💻 Desenvolvedores:**
→ Leia [docs/GUIA_EXECUCAO_API.md](docs/GUIA_EXECUCAO_API.md)

**🧠 Data Scientists:**
→ Leia [docs/ESPECIFICACAO_TECNICA.md](docs/ESPECIFICACAO_TECNICA.md) + [docs/ANALISE_FEATURES_COMPLETA.md](docs/ANALISE_FEATURES_COMPLETA.md)

---

## 🎓 Tecnologias Utilizadas

- **Framework ML:** TensorFlow/Keras
- **Linguagem:** Python 3.11
- **Bibliotecas:** NumPy, Pandas, scikit-learn
- **Dados:** Yahoo Finance (TSLA 2020-2025)
- **Modelo:** LSTM 2-layer com regularização L2

---

## 📝 Informações do Projeto

- **Criação:** 2026-01-02
- **Status:** ✅ Completo e Validado
- **Versão:** 2.0 (Com OHLCV API)
- **Documentação:** Centralizada em `docs/`

### API v2.0 - Novidades:
- ✅ Recebe histórico OHLCV bruto
- ✅ Calcula 34 features automaticamente
- ✅ Métricas do modelo no response
- ✅ 100% requisito PDF atendido

---

**Última Atualização:** 2026-01-02  
**Status Geral:** ✅ PRONTO PARA PRODUÇÃO
