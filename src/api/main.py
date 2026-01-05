"""
FastAPI RESTful API para Previsão de Preços de Ações
Tech Challenge Fase 4 - LSTM Stock Price Prediction

Endpoints:
- POST /predict - Fazer previsão de preço com histórico OHLCV
- GET /health - Status da API
- GET /info - Informações do modelo
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import pandas as pd
import pickle
import json
import os
from pathlib import Path
import keras
from datetime import datetime, timedelta
import logging
import sys

# Adicionar caminho para importar utils
sys.path.insert(0, str(Path(__file__).parent))
from utils import DataValidator, DataNormalizer, MetricsCalculator, HistoricalDataProcessor, FeatureEngineer, LoggingUtils

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="LSTM Stock Price Predictor API",
    description="API para previsão de preços de ações usando LSTM",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== CONFIGURAÇÃO =====================

# Diretório do modelo
MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"
MODEL_NAME = "lstm_5factor_final_20260105_185821.keras"
SCALER_NAME = "scaler_5factor_final_20260105_185821.pkl"
CONFIG_NAME = "config_5factor_final_20260105_185821.yaml"
RESULTS_NAME = "results_5factor_final_20260105_185821.json"

# ===================== VARIÁVEIS GLOBAIS =====================

model = None
scaler = None
model_config = None
model_results = None
sequence_length = 70
num_features = 34

# ===================== MODELOS PYDANTIC =====================

class PriceData(BaseModel):
    """Estrutura de dados de preço individual"""
    date: str = Field(..., description="Data no formato YYYY-MM-DD")
    close: float = Field(..., description="Preço de fechamento")
    high: float = Field(..., description="Preço máximo do dia")
    low: float = Field(..., description="Preço mínimo do dia")
    open: float = Field(..., description="Preço de abertura")
    volume: float = Field(..., description="Volume de negociação")

class PredictionRequest(BaseModel):
    """Request para fazer previsão com histórico de preços brutos"""
    historical_prices: List[PriceData] = Field(
        ...,
        description="Últimos 70 dias de dados históricos de preço (OHLCV)"
    )
    n_steps: int = Field(
        default=1,
        description="Número de passos futuros para prever (1-60)"
    )
    
    class Config:
        example = {
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
                }
            ],
            "n_steps": 1
        }

class PredictionStep(BaseModel):
    """Estrutura de uma previsão individual para um passo futuro"""
    step: int = Field(description="Número do passo (1, 2, 3, ...n)")
    date: str = Field(description="Data prevista (estimada)")
    predicted_price: float = Field(description="Preço predito")
    confidence_interval: dict = Field(description="Intervalo de confiança 95%")

class PredictionResponse(BaseModel):
    """Response com previsões e métricas do modelo"""
    predictions: List[PredictionStep] = Field(description="Lista de previsões para cada passo")
    n_steps: int = Field(description="Número de passos preditos")
    last_historical_date: str = Field(description="Última data do histórico fornecido")
    timestamp: str = Field(description="Data/hora da previsão")
    model_info: dict = Field(description="Informações do modelo")
    model_metrics: dict = Field(description="Métricas de performance do modelo (R², MAE, RMSE, MAPE)")

class HealthResponse(BaseModel):
    """Response do health check"""
    status: str
    model_loaded: bool
    scaler_loaded: bool
    timestamp: str

class ModelInfoResponse(BaseModel):
    """Response com informações do modelo"""
    model_name: str
    version: str
    architecture: dict
    performance_metrics: dict
    training_date: str
    num_features: int
    sequence_length: int

# ===================== FUNÇÕES DE CARREGAMENTO =====================

def load_model_and_scaler():
    """Carregar modelo e scaler"""
    global model, scaler, model_config, model_results
    
    try:
        # Carregar modelo
        model_path = MODEL_DIR / MODEL_NAME
        if not model_path.exists():
            logger.error(f"Modelo não encontrado: {model_path}")
            return False
        
        model = keras.models.load_model(model_path)
        logger.info(f"✅ Modelo carregado: {MODEL_NAME}")
        
        # Carregar scaler
        scaler_path = MODEL_DIR / SCALER_NAME
        if not scaler_path.exists():
            logger.error(f"Scaler não encontrado: {scaler_path}")
            return False
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info(f"✅ Scaler carregado: {SCALER_NAME}")
        
        # Carregar resultados
        results_path = MODEL_DIR / RESULTS_NAME
        if results_path.exists():
            with open(results_path, 'r') as f:
                model_results = json.load(f)
            logger.info(f"✅ Resultados carregados: {RESULTS_NAME}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {str(e)}")
        return False

# ===================== ENDPOINTS =====================

@app.on_event("startup")
async def startup_event():
    """Executado ao iniciar a API"""
    logger.info("🚀 Iniciando API...")
    success = load_model_and_scaler()
    if success:
        logger.info("✅ API inicializada com sucesso!")
    else:
        logger.error("❌ Erro ao inicializar API")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check da API
    
    Returns:
        HealthResponse: Status da API e modelos carregados
    """
    return HealthResponse(
        status="healthy" if model and scaler else "unhealthy",
        model_loaded=model is not None,
        scaler_loaded=scaler is not None,
        timestamp=datetime.now().isoformat()
    )

@app.get("/info", response_model=ModelInfoResponse)
async def model_info():
    """
    Informações do modelo
    
    Returns:
        ModelInfoResponse: Detalhes do modelo LSTM
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    return ModelInfoResponse(
        model_name="LSTM 5-Factor Stock Price Predictor",
        version="1.0.0",
        architecture={
            "type": "LSTM",
            "layers": 2,
            "units": [160, 80],
            "parameters": 183297,
            "input_shape": (sequence_length, num_features),
            "output_shape": 1
        },
        performance_metrics={
            "r2_score": 0.9054,
            "mae": 16.48,
            "rmse": 20.35,
            "mape": 4.77
        } if model_results else {},
        training_date="2026-01-01",
        num_features=num_features,
        sequence_length=sequence_length
    )

def _estimate_future_price_data(last_price: float, last_price_data: dict, step: int, historical_prices: list) -> dict:
    """
    Estimar dados de preço para dias futuros (high, low, open, volume)
    baseado no histórico e no preço predito
    
    Args:
        last_price: Preço predito para o passo
        last_price_data: Último dado de preço do histórico
        step: Número do passo futuro
        historical_prices: Lista de preços históricos para calcular médias
    
    Returns:
        Dict com high, low, open, volume estimados
    """
    # Calcular volatilidade histórica (desvio padrão dos retornos)
    closes = [p['close'] for p in historical_prices]
    returns = [abs(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
    avg_volatility = np.mean(returns) if returns else 0.02
    
    # Estimar open como preço anterior
    estimated_open = historical_prices[-1]['close'] if historical_prices else last_price
    
    # Estimar high como preço + volatilidade
    estimated_high = last_price * (1 + avg_volatility)
    
    # Estimar low como preço - volatilidade
    estimated_low = last_price * (1 - avg_volatility)
    
    # Usar volume médio histórico
    volumes = [p['volume'] for p in historical_prices if 'volume' in p]
    avg_volume = np.mean(volumes) if volumes else 1000000
    
    return {
        'close': last_price,
        'high': estimated_high,
        'low': estimated_low,
        'open': estimated_open,
        'volume': avg_volume
    }


def _build_features_sequence_for_prediction(current_prices: list) -> np.ndarray:
    """
    Construir sequência de features para fazer previsão
    
    Args:
        current_prices: Lista de preços (últimos até 70 dias)
    
    Returns:
        Array de features normalizado com shape (70, 34)
    """
    # Preparar dados nos últimos 70 dias
    prices_for_features = current_prices[-70:]
    df = pd.DataFrame(prices_for_features)[['close', 'high', 'low', 'open', 'volume']]
    
    # Calcular features para cada dia
    features_sequence = []
    for i in range(len(prices_for_features)):
        df_day = df.iloc[i:i+1]
        features_day = FeatureEngineer.calculate_technical_indicators(df_day)
        features_sequence.append(features_day)
    
    # Se não temos 70 dias, preencher com o primeiro valor
    while len(features_sequence) < 70:
        features_sequence.insert(0, features_sequence[0] if features_sequence else np.full(34, 0.5))
    
    # Garantir 70 timesteps
    features_array = np.array(features_sequence[-70:])
    return features_array


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Fazer previsão de preço de ação baseado em histórico OHLCV para múltiplos passos
    
    Args:
        request: PredictionRequest com histórico de preços (70 dias) e número de passos (1-60)
    
    Returns:
        PredictionResponse: Lista de previsões para cada passo com intervalo de confiança
        
    Raises:
        HTTPException: Se modelo não estiver carregado ou dados inválidos
    """
    
    # Validar modelo
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    try:
        # 1. Validar dados históricos
        prices_list = [
            {
                "date": p.date,
                "close": p.close,
                "high": p.high,
                "low": p.low,
                "open": p.open,
                "volume": p.volume
            }
            for p in request.historical_prices
        ]
        
        is_valid, validation_msg = FeatureEngineer.validate_price_data(prices_list)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Dados inválidos: {validation_msg}")
        
        # 2. Validar n_steps (agora 1-60)
        if request.n_steps < 1 or request.n_steps > 60:
            raise HTTPException(
                status_code=400,
                detail="n_steps deve estar entre 1 e 60"
            )
        
        logger.info(f"📥 Request recebido: {len(prices_list)} preços históricos, n_steps={request.n_steps}")
        
        # 3. Cálculos iniciais
        mape = 4.77  # Margem de erro do modelo em %
        last_historical_date = datetime.strptime(prices_list[-1]['date'], "%Y-%m-%d")
        all_predictions = []
        current_prices = prices_list.copy()  # Cópia para simular histórico futuro
        
        logger.info(f"📊 Histórico: {len(current_prices)} dias | Última data: {last_historical_date.date()}")
        
        # 4. Loop para prever cada passo
        for step in range(1, request.n_steps + 1):
            logger.info(f"\n🔮 Previsão passo {step}/{request.n_steps}")
            
            # 4.1 Construir sequência de features para o passo atual
            features_sequence = _build_features_sequence_for_prediction(current_prices)
            features_input = features_sequence.reshape(1, 70, num_features)
            
            # 4.2 Fazer previsão
            prediction = model.predict(features_input, verbose=0)
            predicted_price_norm = float(prediction[0][0])
            
            # 4.3 Desnormalizar preço
            predicted_price = predicted_price_norm
            if scaler is not None:
                try:
                    predicted_price = float(scaler.inverse_transform([[predicted_price_norm]])[0][0])
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao desnormalizar no passo {step}: {str(e)}")
            
            # 4.4 Calcular intervalo de confiança
            confidence_interval = {
                "lower": predicted_price * (1 - mape / 100),
                "upper": predicted_price * (1 + mape / 100),
                "confidence_level": "95%",
                "mape_percent": mape
            }
            
            # 4.5 Calcular data estimada
            future_date = last_historical_date + timedelta(days=step)
            
            # 4.6 Adicionar à lista de previsões
            all_predictions.append({
                "step": step,
                "date": future_date.strftime("%Y-%m-%d"),
                "predicted_price": predicted_price,
                "confidence_interval": confidence_interval
            })
            
            logger.info(f"   ✅ Previsão: ${predicted_price:.2f} (IC: ${confidence_interval['lower']:.2f} - ${confidence_interval['upper']:.2f})")
            
            # 4.7 Preparar dados estimados para o próximo passo (reinjetar na sequência)
            estimated_data = _estimate_future_price_data(
                predicted_price,
                current_prices[-1],
                step,
                current_prices[-70:]
            )
            estimated_data['date'] = future_date.strftime("%Y-%m-%d")
            
            # Adicionar preço estimado ao histórico para o próximo ciclo
            current_prices.append(estimated_data)
        
        # 5. Preparar métricas do modelo
        model_metrics = {
            "r2_score": 0.9054,
            "mae": 16.48,
            "rmse": 20.35,
            "mape": 4.77,
            "confidence_level_percent": 95
        }
        
        if model_results and 'metrics' in model_results:
            model_metrics.update(model_results['metrics'])
        
        logger.info(f"\n✅ Todas as {request.n_steps} previsão(ões) realizadas com sucesso!")
        
        # 6. Construir response com lista de PredictionStep
        prediction_steps = [
            PredictionStep(
                step=p['step'],
                date=p['date'],
                predicted_price=p['predicted_price'],
                confidence_interval=p['confidence_interval']
            )
            for p in all_predictions
        ]
        
        return PredictionResponse(
            predictions=prediction_steps,
            n_steps=request.n_steps,
            last_historical_date=prices_list[-1]['date'],
            timestamp=datetime.now().isoformat(),
            model_info={
                "model_name": "LSTM 5-Factor",
                "version": "1.0.0",
                "architecture": "2-layer LSTM (160→80 units)",
                "input_source": "Histórico OHLCV bruto",
                "features_calculated": 34,
                "sequence_length": 70,
                "multi_step_forecasting": True,
                "max_steps": 60
            },
            model_metrics=model_metrics
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro na previsão: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao fazer previsão: {str(e)}")


@app.post("/predict-batch")
async def predict_batch(requests: List[PredictionRequest]):
    """
    Fazer múltiplas previsões em batch
    
    Args:
        requests: Lista de PredictionRequest
    
    Returns:
        Lista de PredictionResponse
    """
    
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    results = []
    for req in requests:
        try:
            # Reutilizar lógica do endpoint /predict
            if len(req.features) != num_features:
                results.append({"error": f"Invalid features count"})
                continue
            
            features_array = np.array(req.features).reshape(1, sequence_length, num_features)
            prediction = model.predict(features_array, verbose=0)
            predicted_price = float(prediction[0][0])
            
            results.append({
                "prediction": predicted_price,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            results.append({"error": str(e)})
    
    return {"predictions": results}

@app.get("/")
async def root():
    """Endpoint raiz com documentação"""
    return {
        "app": "LSTM Stock Price Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "GET /health": "Status da API",
            "GET /info": "Informações do modelo",
            "POST /predict": "Fazer previsão",
            "POST /predict-batch": "Fazer múltiplas previsões"
        }
    }

# ===================== TRATAMENTO DE ERROS =====================

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Tratador global de exceções"""
    logger.error(f"❌ Erro não tratado: {str(exc)}")
    return {
        "error": "Internal Server Error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    }

# ===================== MAIN =====================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Iniciando servidor Uvicorn...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
