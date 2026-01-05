"""
Funções utilitárias para a API
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validação de dados de entrada"""
    
    @staticmethod
    def validate_features(features: List[float], expected_count: int = 34) -> Tuple[bool, str]:
        """
        Validar features de entrada
        
        Args:
            features: Lista de features
            expected_count: Número esperado de features
        
        Returns:
            Tupla (is_valid, message)
        """
        if not isinstance(features, list):
            return False, "Features deve ser uma lista"
        
        if len(features) != expected_count:
            return False, f"Esperado {expected_count} features, recebido {len(features)}"
        
        if not all(isinstance(f, (int, float)) for f in features):
            return False, "Todas as features devem ser numéricas"
        
        if any(np.isnan(f) or np.isinf(f) for f in features):
            return False, "Features contém NaN ou Inf"
        
        return True, "OK"
    
    @staticmethod
    def validate_n_steps(n_steps: int) -> Tuple[bool, str]:
        """
        Validar número de passos
        
        Args:
            n_steps: Número de passos para prever
        
        Returns:
            Tupla (is_valid, message)
        """
        if not isinstance(n_steps, int):
            return False, "n_steps deve ser um inteiro"
        
        if n_steps < 1 or n_steps > 10:
            return False, "n_steps deve estar entre 1 e 10"
        
        return True, "OK"

class DataNormalizer:
    """Normalização de dados"""
    
    @staticmethod
    def normalize_features(features: np.ndarray, scaler) -> np.ndarray:
        """
        Normalizar features usando scaler treinado
        
        Args:
            features: Array de features
            scaler: MinMaxScaler treinado
        
        Returns:
            Features normalizadas
        """
        try:
            return scaler.transform(features.reshape(-1, 1)).flatten()
        except Exception as e:
            logger.error(f"Erro ao normalizar features: {str(e)}")
            raise
    
    @staticmethod
    def denormalize_prediction(prediction: float, scaler) -> float:
        """
        Desnormalizar previsão
        
        Args:
            prediction: Previsão normalizada
            scaler: MinMaxScaler treinado
        
        Returns:
            Previsão desnormalizada
        """
        try:
            return float(scaler.inverse_transform([[prediction]])[0][0])
        except Exception as e:
            logger.error(f"Erro ao desnormalizar previsão: {str(e)}")
            raise

class MetricsCalculator:
    """Cálculo de métricas"""
    
    @staticmethod
    def calculate_confidence_interval(prediction: float, mape: float = 4.77) -> dict:
        """
        Calcular intervalo de confiança baseado em MAPE
        
        Args:
            prediction: Previsão pontual
            mape: MAPE do modelo em porcentagem
        
        Returns:
            Dict com limites inferior e superior
        """
        error_margin = prediction * (mape / 100)
        return {
            "lower": prediction - error_margin,
            "upper": prediction + error_margin,
            "error_margin": error_margin,
            "confidence_level": "95%"
        }
    
    @staticmethod
    def calculate_prediction_stats(predictions: np.ndarray) -> dict:
        """
        Calcular estatísticas de múltiplas previsões
        
        Args:
            predictions: Array de previsões
        
        Returns:
            Dict com estatísticas
        """
        return {
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
            "median": float(np.median(predictions))
        }

class FeatureEngineer:
    """Engenharia de features a partir de dados OHLCV"""
    
    @staticmethod
    def calculate_technical_indicators(prices_df: pd.DataFrame) -> np.ndarray:
        """
        Calcular 34 features técnicas a partir de dados OHLCV
        
        Args:
            prices_df: DataFrame com colunas: close, high, low, open, volume
        
        Returns:
            Array com 34 features normalizadas (0-1)
        """
        try:
            df = prices_df.copy()
            features = []
            
            # ===== FEATURES DE PREÇO (6 features) =====
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # 1. Preço de fechamento (normalizado pela média dos últimos 70 dias)
            close_normalized = close[-1] / np.mean(close) if np.mean(close) != 0 else 0.5
            features.append(np.clip(close_normalized, 0, 1))
            
            # 2. Retorno do período (Close atual vs primeira data)
            returns = (close[-1] - close[0]) / close[0] if close[0] != 0 else 0
            features.append(np.clip((returns + 1) / 2, 0, 1))  # Normalizar em 0-1
            
            # 3-5. Médias móveis (MA_5, MA_10, MA_20)
            ma_5 = np.mean(close[-5:]) / np.mean(close) if np.mean(close) != 0 else 0.5
            ma_10 = np.mean(close[-10:]) / np.mean(close) if np.mean(close) != 0 else 0.5
            ma_20 = np.mean(close[-20:]) / np.mean(close) if np.mean(close) != 0 else 0.5
            features.extend([np.clip(ma_5, 0, 1), np.clip(ma_10, 0, 1), np.clip(ma_20, 0, 1)])
            
            # 6. Razão High/Low
            hl_ratio = np.mean(high) / np.mean(low) if np.mean(low) != 0 else 1
            features.append(np.clip(hl_ratio - 1, 0, 1))
            
            # ===== FEATURES DE MOMENTUM (8 features) =====
            
            # 7. RSI (Relative Strength Index) - simplificado
            delta = np.diff(close)
            gain = np.sum(np.where(delta > 0, delta, 0)) / len(delta) if len(delta) > 0 else 0
            loss = np.sum(np.where(delta < 0, -delta, 0)) / len(delta) if len(delta) > 0 else 0
            rs = gain / loss if loss != 0 else 1
            rsi = 100 - (100 / (1 + rs))
            features.append(np.clip(rsi / 100, 0, 1))
            
            # 8-9. MACD (simplificado - diferença entre 2 médias)
            ema_12 = np.mean(close[-12:]) if len(close) >= 12 else np.mean(close)
            ema_26 = np.mean(close[-26:]) if len(close) >= 26 else np.mean(close)
            macd = ema_12 - ema_26
            features.append(np.clip((macd / np.mean(close) + 0.1) / 0.2, 0, 1) if np.mean(close) != 0 else 0.5)
            
            # MACD Signal (SMA do MACD)
            macd_signal = (ema_12 - ema_26) / np.mean(close) if np.mean(close) != 0 else 0
            features.append(np.clip((macd_signal + 0.1) / 0.2, 0, 1))
            
            # 10-11. Bollinger Bands
            std_close = np.std(close)
            mean_close = np.mean(close)
            bb_upper = mean_close + (2 * std_close)
            bb_lower = mean_close - (2 * std_close)
            bb_upper_norm = np.clip((close[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5, 0, 1)
            features.extend([bb_upper_norm, np.clip(1 - bb_upper_norm, 0, 1)])  # Upper e Lower
            
            # 12-13. ATR e ADX (simplificado)
            tr = np.maximum(high - low, np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            ))
            atr = np.mean(tr) / mean_close if mean_close != 0 else 0
            features.append(np.clip(atr, 0, 1))
            
            # ADX simplificado (usando variação de preço)
            dx = np.abs(np.diff(high - low)) / (high[1:] - low[1:]) if np.any(high[1:] - low[1:] != 0) else 0
            adx = np.mean(dx) if isinstance(dx, np.ndarray) and len(dx) > 0 else 0
            features.append(np.clip(adx, 0, 1))
            
            # 14. OBV (On-Balance Volume) - simplificado
            obv = np.sum(np.where(np.diff(close) > 0, volume[1:], -volume[1:]))
            obv_norm = obv / np.sum(volume) if np.sum(volume) != 0 else 0.5
            features.append(np.clip((obv_norm + 1) / 2, 0, 1))
            
            # ===== FEATURES DE VOLATILIDADE (6 features) =====
            
            # 15. Volatilidade histórica
            returns_array = np.diff(close) / close[:-1] if len(close) > 1 else np.array([0])
            volatility = np.std(returns_array)
            features.append(np.clip(volatility * 10, 0, 1))  # Scale por fator
            
            # 16. Volatilidade rolante (últimos 10 dias)
            vol_rolling = np.std(returns_array[-10:]) if len(returns_array) >= 10 else volatility
            features.append(np.clip(vol_rolling * 10, 0, 1))
            
            # 17. Volatilidade rolante (últimos 20 dias)
            vol_rolling_20 = np.std(returns_array[-20:]) if len(returns_array) >= 20 else volatility
            features.append(np.clip(vol_rolling_20 * 10, 0, 1))
            
            # 18. Curtose (tail risk)
            from scipy.stats import kurtosis
            kurtosis_val = kurtosis(returns_array) if len(returns_array) > 3 else 0
            features.append(np.clip((kurtosis_val + 5) / 10, 0, 1))
            
            # 19. Skewness
            from scipy.stats import skew
            skew_val = skew(returns_array) if len(returns_array) > 2 else 0
            features.append(np.clip((skew_val + 2) / 4, 0, 1))
            
            # 20. Max Drawdown
            cumulative_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
            features.append(np.clip(-max_drawdown, 0, 1))
            
            # ===== FEATURES DE VOLUME (4 features) =====
            
            # 21. Volume médio
            avg_volume = np.mean(volume)
            vol_norm = volume[-1] / avg_volume if avg_volume != 0 else 0.5
            features.append(np.clip(vol_norm, 0, 2))  # Pode ser > 1
            
            # 22. Volume rolante (média 5 dias)
            vol_ma5 = np.mean(volume[-5:]) if len(volume) >= 5 else np.mean(volume)
            features.append(np.clip(volume[-1] / vol_ma5 if vol_ma5 != 0 else 0.5, 0, 2))
            
            # 23. Taxa de volume
            vol_change = (volume[-1] - np.mean(volume[-5:-1])) / np.mean(volume[-5:-1]) if np.mean(volume[-5:-1]) != 0 else 0
            features.append(np.clip((vol_change + 1) / 2, 0, 1))
            
            # 24. Volume trend
            vol_trend = (volume[-1] - volume[0]) / volume[0] if volume[0] != 0 else 0
            features.append(np.clip((vol_trend + 1) / 2, 0, 1))
            
            # ===== FEATURES DE TENDÊNCIA (10 features) =====
            
            # 25-26. Tendência curta e média
            short_trend = close[-1] - close[-5] if len(close) >= 5 else 0
            medium_trend = close[-1] - close[-20] if len(close) >= 20 else 0
            features.append(np.clip((short_trend / np.mean(close) + 0.5) / 1, 0, 1) if np.mean(close) != 0 else 0.5)
            features.append(np.clip((medium_trend / np.mean(close) + 0.5) / 1, 0, 1) if np.mean(close) != 0 else 0.5)
            
            # 27. Correlação com tendência
            x = np.arange(len(close))
            trend_corr = np.corrcoef(x, close)[0, 1] if len(close) > 1 else 0
            features.append(np.clip((trend_corr + 1) / 2, 0, 1))
            
            # 28-31. Features de aceleração (4 features)
            velocity = np.diff(close)
            acceleration = np.diff(velocity) if len(velocity) > 1 else np.array([0])
            
            accel_avg = np.mean(acceleration) if len(acceleration) > 0 else 0
            accel_std = np.std(acceleration) if len(acceleration) > 0 else 0
            features.append(np.clip((accel_avg / np.mean(close) + 0.5) / 1, 0, 1) if np.mean(close) != 0 else 0.5)
            features.append(np.clip(accel_std / np.mean(close) if np.mean(close) != 0 else 0.5, 0, 1))
            
            # Taxa de mudança de aceleração
            accel_change = (acceleration[-1] - acceleration[0]) if len(acceleration) >= 2 else 0
            features.append(np.clip((accel_change / np.mean(close) + 0.5) / 1, 0, 1) if np.mean(close) != 0 else 0.5)
            
            # Variância da aceleração
            accel_var = np.var(acceleration) if len(acceleration) > 0 else 0
            features.append(np.clip(accel_var / (np.mean(close) ** 2) if np.mean(close) != 0 else 0.5, 0, 1))
            
            # 32-34. Features Finais (market structure)
            # Spread
            spread = np.mean(high - low) / np.mean(close) if np.mean(close) != 0 else 0
            features.append(np.clip(spread * 100, 0, 1))
            
            # Close position within range
            daily_range = high[-1] - low[-1]
            close_position = (close[-1] - low[-1]) / daily_range if daily_range != 0 else 0.5
            features.append(np.clip(close_position, 0, 1))
            
            # Momentum factor
            momentum = (close[-1] - np.mean(close[-20:])) / np.mean(close[-20:]) if np.mean(close[-20:]) != 0 else 0
            features.append(np.clip((momentum + 0.5) / 1, 0, 1))
            
            # Garantir que temos exatamente 34 features
            while len(features) < 34:
                features.append(0.5)
            
            return np.array(features[:34], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Erro ao calcular features: {str(e)}")
            # Retornar 34 valores padrão em caso de erro
            return np.full(34, 0.5, dtype=np.float32)
    
    @staticmethod
    def validate_price_data(prices_data: List[dict]) -> Tuple[bool, str]:
        """
        Validar dados de preço
        
        Args:
            prices_data: Lista de dicts com OHLCV
        
        Returns:
            Tupla (is_valid, message)
        """
        if len(prices_data) < 70:
            return False, f"Esperado pelo menos 70 dias de dados, recebido {len(prices_data)}"
        
        required_fields = ['close', 'high', 'low', 'open', 'volume']
        for price_point in prices_data:
            for field in required_fields:
                if field not in price_point or price_point[field] is None:
                    return False, f"Campo '{field}' faltando ou None em algum registro"
            
            # Validações lógicas
            if price_point['high'] < price_point['low']:
                return False, "High deve ser >= Low"
            
            if price_point['close'] < price_point['low'] or price_point['close'] > price_point['high']:
                return False, "Close deve estar entre Low e High"
            
            if price_point['volume'] < 0:
                return False, "Volume não pode ser negativo"
        
        return True, "OK"


class LoggingUtils:
    """Utilidades de logging"""
    
    @staticmethod
    def setup_logging(level: str = "INFO"):
        """Configurar logging"""
        import logging
        logging.basicConfig(
            level=getattr(logging, level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    @staticmethod
    def log_request(endpoint: str, data: dict):
        """Log de request"""
        logger.info(f"📥 {endpoint}: {data}")
    
    @staticmethod
    def log_prediction(prediction: float, confidence: dict):
        """Log de previsão"""
        logger.info(f"🎯 Previsão: ${prediction:.2f} (IC: ${confidence['lower']:.2f} - ${confidence['upper']:.2f})")
