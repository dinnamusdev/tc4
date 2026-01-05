"""
Processamento de dados integrado com Feature Registry
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from sklearn.preprocessing import MinMaxScaler
from .cache_manager import get_stock_data
from .feature_engineer import TeslaFeatureEngineer, engineer_tesla_features
from ..models.feature_registry import get_feature_registry

logger = logging.getLogger(__name__)


class StockDataProcessor:
    """Processador de dados com suporte a features modulares"""
    
    def __init__(self, symbol: str = "TSLA", start_date: str = "2020-01-01", 
                 end_date: str = "2024-12-31"):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data = None
        self.processed_data = None
        self.feature_engineer = TeslaFeatureEngineer(symbol)
        self.feature_registry = get_feature_registry()
        
        logger.info(f"StockDataProcessor inicializado para {symbol}")
    
    def load_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Carrega dados usando cache manager"""
        logger.info(f"Carregando dados para {self.symbol}")
        self.raw_data = get_stock_data(
            self.symbol, 
            self.start_date, 
            self.end_date, 
            force_refresh
        )
        return self.raw_data
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona indicadores técnicos"""
        df = df.copy()
        
        # 1. Retornos
        df['Returns'] = df['Close'].pct_change()
        
        # 2. Médias móveis
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # 3. Bandas de Bollinger
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # 4. RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 5. Volume
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # 6. Volatilidade
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # 7. Gap de abertura
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # 8. Range do dia
        df['Daily_Range'] = (df['High'] - df['Low']) / df['Low']
        
        # 9. Posição relativa
        df['Close_vs_MA20'] = (df['Close'] - df['MA_20']) / df['MA_20']
        df['Close_vs_MA50'] = (df['Close'] - df['MA_50']) / df['MA_50']
        
        return df    
    
    def process_with_features(self, enable_competition: bool = False, 
                            enable_policy: bool = False) -> pd.DataFrame:
        """
        Processa dados com features modulares
        
        Args:
            enable_competition: Ativar features de competição
            enable_policy: Ativar features de política
        
        Returns:
            DataFrame processado
        """
        # Carregar dados se necessário
        if self.raw_data is None:
            self.load_data()
        
        # Configurar features baseado nos parâmetros
        if enable_competition:
            self.feature_registry.enable_feature_group('competition')
            logger.info("Features de competição ativadas")
        
        if enable_policy:
            self.feature_registry.enable_feature_group('policy')
            logger.info("Features de política ativadas")
        
        # Aplicar engenharia de features
        logger.info("Aplicando engenharia de features...")
        self.processed_data = self.feature_engineer.engineer_all_features(self.raw_data)
        
        # Gerar relatório
        report = self.feature_engineer.get_feature_report(self.processed_data)
        logger.info(f"Features processadas: {report['feature_validation']['total_active']} ativas")
        
        return self.processed_data
    
    def prepare_for_lstm(self, target_column: str = 'Close',
                        sequence_length: int = 60,
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15) -> Tuple:
        """
        Prepara dados para treinamento LSTM
        
        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_names)
        """
        if self.processed_data is None:
            raise ValueError("Dados não processados. Chame process_with_features() primeiro.")
        
        # Obter features ativas do registry
        feature_names = self.feature_registry.get_active_features()
        
        # Verificar se target está nas features ativas
        if target_column not in feature_names:
            feature_names.append(target_column)
        
        # Filtrar colunas disponíveis
        available_features = [f for f in feature_names if f in self.processed_data.columns]
        
        if target_column not in available_features:
            raise ValueError(f"Target column '{target_column}' não encontrada nos dados")
        
        logger.info(f"Preparando LSTM com {len(available_features)} features")
        logger.info(f"Features: {available_features}")
        
        # Separar dados
        data = self.processed_data[available_features].copy()
        data = data.dropna()
        
        # Índice da target
        target_idx = available_features.index(target_column)
        
        # Normalizar
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Criar sequências
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, target_idx])
        
        X, y = np.array(X), np.array(y)
        
        # Split temporal
        train_size = int(len(X) * train_ratio)
        val_size = int(len(X) * val_ratio)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        logger.info(f"Dataset split:")
        logger.info(f"  Treino: {X_train.shape[0]} sequências")
        logger.info(f"  Validação: {X_val.shape[0]} sequências")
        logger.info(f"  Teste: {X_test.shape[0]} sequências")
        logger.info(f"  Features por timestep: {X_train.shape[2]}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, scaler, available_features
    
    def get_feature_importance_data(self, model, X_test, feature_names) -> Dict:
        """
        Calcula importância das features usando perturbação
        
        Args:
            model: Modelo treinado
            X_test: Dados de teste
            feature_names: Nomes das features
        
        Returns:
            Dicionário com importância das features
        """
        from sklearn.metrics import mean_squared_error
        
        # Previsão baseline
        y_pred_baseline = model.predict(X_test).flatten()
        
        feature_importance = {}
        
        for i, feature_name in enumerate(feature_names):
            # Perturbar feature i
            X_perturbed = X_test.copy()
            
            # Adicionar ruído à feature
            noise = np.random.normal(0, 0.1, X_perturbed.shape[:2])
            X_perturbed[:, :, i] += noise
            
            # Prever com dados perturbados
            y_pred_perturbed = model.predict(X_perturbed).flatten()
            
            # Calcular aumento no erro
            mse_baseline = mean_squared_error(y_pred_baseline, y_pred_baseline)  # 0
            mse_perturbed = mean_squared_error(y_pred_baseline, y_pred_perturbed)
            
            # Importância proporcional ao aumento do erro
            importance = mse_perturbed / (mse_baseline + 1e-10)
            feature_importance[feature_name] = importance
        
        # Normalizar importâncias
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
        
        # Registrar no feature registry
        self.feature_registry.record_feature_importance(
            model_name="lstm_tesla_predictor",
            feature_importance=feature_importance
        )
        
        return feature_importance


    # Função de conveniência
    def process_tesla_data(symbol: str = "TSLA", start_date: str = "2020-01-01",
                        end_date: str = "2024-12-31", enable_all_features: bool = False) -> Tuple:
        """
        Pipeline completo de processamento
        
        Returns:
            (processed_df, lstm_data, feature_names)
        """
        processor = StockDataProcessor(symbol, start_date, end_date)
        
        # Carregar dados
        processor.load_data()
        
        # Processar com features (ativar todas se solicitado)
        processed_df = processor.process_with_features(
            enable_competition=enable_all_features,
            enable_policy=enable_all_features
        )
        
        # Preparar para LSTM
        lstm_data = processor.prepare_for_lstm()
        
        return processed_df, lstm_data, processor.feature_registry.get_active_features()
        
    def process_pipeline(self, force_refresh: bool = False) -> Tuple:
        """
        Pipeline completo de processamento
        
        Returns:
            (processed_df, train_val_test_data, scaler)
        """
        # 1. Carregar dados
        raw_df = self.load_data(force_refresh)
        
        # 2. Adicionar features
        processed_df = self.add_technical_indicators(raw_df)
        
        # 3. Armazenar dados processados
        self.processed_data = processed_df
        
        # 4. Preparar para LSTM
        lstm_data = self.prepare_for_lstm()
        
        return processed_df, lstm_data