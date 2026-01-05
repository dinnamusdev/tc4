"""
Engine de features para os 5 fatores da Tesla
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import sys
from pathlib import Path

# Adicionar diretório raiz ao path se necessário
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.feature_registry import get_feature_registry
from src.data.cache_manager import get_cache_manager

logger = logging.getLogger(__name__)


class TeslaFeatureEngineer:
    """
    Engenheiro de features para os 5 fatores da Tesla
    """
    
    def __init__(self, symbol: str = "TSLA"):
        self.symbol = symbol
        self.feature_registry = get_feature_registry()
        self.competitor_symbols = ['NIO', 'RIVN', 'LCID', 'FSR']  # Competidores EV
        self.market_indices = ['^GSPC', '^IXIC', '^DJI', '^VIX']  # Índices de mercado
        
        logger.info(f"TeslaFeatureEngineer inicializado para {symbol}")
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todas as transformações de features ativas
        
        Args:
            df: DataFrame com dados brutos da Tesla
            
        Returns:
            DataFrame com todas as features
        """
        logger.info("Iniciando engenharia de features...")
        
        # Fazer cópia para não modificar original
        df_enhanced = df.copy()
        
        # 1. Core features (técnicas básicas)
        df_enhanced = self._add_core_features(df_enhanced)
        
        # 2. Execution features (Fator 1)
        df_enhanced = self._add_execution_features(df_enhanced)
        
        # 3. Sentiment features (Fator 2)
        df_enhanced = self._add_sentiment_features(df_enhanced)
        
        # 4. Macro features (Fator 3)
        df_enhanced = self._add_macro_features(df_enhanced)
        
        # 5. Competition features (Fator 4) - se ativado
        if 'competition' in self.feature_registry.feature_groups:
            if self.feature_registry.feature_groups['competition']['enabled']:
                df_enhanced = self._add_competition_features(df_enhanced)
        
        # 6. Policy features (Fator 5) - se ativado
        if 'policy' in self.feature_registry.feature_groups:
            if self.feature_registry.feature_groups['policy']['enabled']:
                df_enhanced = self._add_policy_features(df_enhanced)
        
        # Remover NaN criados pelas features
        df_enhanced = df_enhanced.dropna()
        
        logger.info(f"Features aplicadas: {len(df_enhanced.columns)} colunas")
        
        # Validar features
        validation = self.feature_registry.validate_features(df_enhanced)
        logger.info(f"Coverage de features: {validation['coverage']:.1%}")
        
        if validation['missing']:
            logger.warning(f"Features faltando: {validation['missing']}")
        
        return df_enhanced
    
    def _add_core_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features técnicas core"""
        df = df.copy()
        
        # Retornos
        if 'Close' in df.columns:
            df['Returns'] = df['Close'].pct_change()
        
        # Médias móveis
        for window in [5, 20, 50]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        
        # RSI
        if 'Returns' in df.columns:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatilidade
        if 'Returns' in df.columns:
            df['Volatility_20d'] = df['Returns'].rolling(window=20).std()
        
        # Volume features
        if 'Volume' in df.columns:
            df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        
        # Range do dia
        if all(col in df.columns for col in ['High', 'Low']):
            df['Daily_Range_Pct'] = (df['High'] - df['Low']) / df['Low']
        
        return df
    
    def _add_execution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fator 1: Features de execução operacional"""
        df = df.copy()
        
        # Proxy para crescimento de entregas (usando volume como proxy)
        if 'Volume' in df.columns:
            df['Delivery_Growth_YoY'] = df['Volume'].pct_change(periods=252)  # ~1 ano
        
        # Score de surpresa de produção (simplificado)
        # Em produção real, integrar com earnings calendar
        df['Production_Beat_Score'] = 0.0
        
        # Sazonalidade de earnings (Tesla reporta ~Jan, Apr, Jul, Oct)
        df['Earnings_Season'] = 0
        earnings_months = [1, 4, 7, 10]
        for month in earnings_months:
            df.loc[df.index.month == month, 'Earnings_Season'] = 1
        
        # Trend de margens (proxy com price/volume)
        if all(col in df.columns for col in ['Close', 'Volume']):
            price_volume_trend = df['Close'].rolling(20).mean() / df['Volume'].rolling(20).mean()
            df['Margin_Trend_3m'] = price_volume_trend.pct_change(63)  # 3 meses
        
        # Delivery surprise (volatility around expected dates)
        df['Delivery_Surprise'] = df['Volatility_20d'] * df['Earnings_Season']
        
        return df
    
    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fator 2: Features de sentimento de mercado"""
        df = df.copy()
        
        # Short_Interest_Proxy: Estimativa de posições curtas baseada em volume e volatilidade
        if all(col in df.columns for col in ['Volatility_20d', 'Volume_Ratio']):
            df['Short_Interest_Proxy'] = df['Volatility_20d'] * df['Volume_Ratio']
        
        # Sentimento de retail (momentum de curto prazo)
        if 'Returns' in df.columns:
            df['Retail_Sentiment_5d'] = df['Returns'].rolling(5).mean()
        
        # Volume spikes (indica interesse anormal)
        if 'Volume_Ratio' in df.columns:
            df['Volume_Spike_Ratio'] = df['Volume_Ratio'].rolling(5).max()
        
        # Fear_Greed_Score: Sentimento geral de mercado baseado em RSI e volatilidade
        fear_greed_components = []
        
        if 'RSI' in df.columns:
            # RSI muito alto = greed, muito baixo = fear
            rsi_score = np.where(df['RSI'] > 70, -1, np.where(df['RSI'] < 30, 1, 0))
            fear_greed_components.append(rsi_score)
        
        if 'Volatility_20d' in df.columns:
            # Alta volatilidade = fear
            vol_score = -df['Volatility_20d'] / df['Volatility_20d'].rolling(60).mean()
            fear_greed_components.append(vol_score)
        
        if fear_greed_components:
            df['Fear_Greed_Score'] = np.mean(fear_greed_components, axis=0)
        
        return df
    
    def _add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fator 3: Features macroeconômicas"""
        df = df.copy()
        
        try:
            # Usar cache manager ao invés de yfinance
            cache_manager = get_cache_manager('../data/cache')
            
            start_date = df.index[0].strftime('%Y-%m-%d')
            end_date = df.index[-1].strftime('%Y-%m-%d')
            
            logger.info(f"Carregando dados macro do cache: {start_date} a {end_date}")
            
            # 1. S&P 500
            try:
                sp500 = cache_manager.get_stock_data('^GSPC', start_date, end_date)
                
                if sp500 is not None and not sp500.empty:
                    # Alinhar índices
                    sp500_aligned = sp500['Close'].reindex(df.index, method='ffill')
                    
                    # Retorno do S&P 500
                    df['SP500_Return'] = sp500_aligned.pct_change()
                    
                    # Beta da Tesla vs S&P (rolling 60 dias)
                    returns_cov = df['Returns'].rolling(60).cov(df['SP500_Return'])
                    sp500_var = df['SP500_Return'].rolling(60).var()
                    df['TSLA_Beta_60d'] = returns_cov / sp500_var
                    
                    # Score de sensibilidade a juros
                    if 'TSLA_Beta_60d' in df.columns and 'Volatility_20d' in df.columns:
                        df['Rate_Sensitivity_Score'] = df['TSLA_Beta_60d'] * df['Volatility_20d']
                    
                    # Correlação com mercado
                    df['Market_Correlation'] = df['Returns'].rolling(60).corr(df['SP500_Return'])
                    
                    logger.info(f"✅ S&P 500: {len(sp500)} registros carregados")
            except Exception as e:
                logger.warning(f"Erro ao carregar S&P 500: {e}")
            
            # 2. NASDAQ e VIX
            for idx_symbol, idx_name in [('^IXIC', 'NASDAQ'), ('^VIX', 'VIX')]:
                try:
                    idx_data = cache_manager.get_stock_data(idx_symbol, start_date, end_date)
                    if idx_data is not None and not idx_data.empty:
                        idx_aligned = idx_data['Close'].reindex(df.index, method='ffill')
                        df[f'{idx_symbol}_Return'] = idx_aligned.pct_change()
                        logger.info(f"✅ {idx_name}: {len(idx_data)} registros carregados")
                except Exception as e:
                    logger.debug(f"Erro ao carregar {idx_symbol}: {e}")
            
        except Exception as e:
            logger.warning(f"Erro ao adicionar features macro: {e}")
            # Fallback: criar features dummy
            df['SP500_Return'] = 0
            df['TSLA_Beta_60d'] = 2.0  # Beta típico da Tesla
            df['Rate_Sensitivity_Score'] = 1.0
            df['Market_Correlation'] = 0.5
        
        return df
    
    def _add_competition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fator 4: Features de competição (requer dados externos)"""
        df = df.copy()
        
        try:
            cache_manager = get_cache_manager('../data/cache')
            
            start_date = df.index[0].strftime('%Y-%m-%d')
            end_date = df.index[-1].strftime('%Y-%m-%d')
            
            logger.info(f"Carregando dados de competidores do cache")
            
            # Baixar dados de competidores
            competitor_data = {}
            for symbol in self.competitor_symbols:
                try:
                    data = cache_manager.get_stock_data(symbol, start_date, end_date)
                    if data is not None and not data.empty:
                        competitor_data[symbol] = data['Close']
                        logger.info(f"✅ Competidor {symbol}: {len(data)} registros carregados")
                except Exception as e:
                    logger.debug(f"Erro ao carregar {symbol}: {e}")
            
            if competitor_data:
                # Criar DataFrame de competidores
                comp_df = pd.DataFrame(competitor_data)
                comp_df = comp_df.reindex(df.index, method='ffill')
                
                # Retorno médio dos competidores
                comp_returns = comp_df.pct_change().mean(axis=1)
                df['Competitor_Avg_Return'] = comp_returns
                
                # Gap de performance Tesla vs Competidores
                if 'Returns' in df.columns:
                    df['Performance_Gap'] = df['Returns'] - df['Competitor_Avg_Return']
                
                # ETF de EV como proxy
                try:
                    ev_etf = cache_manager.get_stock_data('IDRV', start_date, end_date)
                    if ev_etf is not None and not ev_etf.empty:
                        ev_etf_aligned = ev_etf['Close'].reindex(df.index, method='ffill')
                        df['EV_Sector_Return'] = ev_etf_aligned.pct_change()
                        
                        # Market share proxy
                        if 'Returns' in df.columns:
                            df['EV_Market_Share_Proxy'] = df['Returns'].rolling(20).mean() - \
                                                         df['EV_Sector_Return'].rolling(20).mean()
                        
                        logger.info(f"✅ EV ETF (IDRV): {len(ev_etf)} registros carregados")
                except Exception as e:
                    logger.debug(f"Erro ao carregar ETF EV: {e}")
            
            # New_EV_Models_Proxy: Lançamentos de novos modelos (sazonal por trimestre)
            # Contador de novos modelos com lançamentos trimestrais
            df['New_EV_Models_Proxy'] = np.where(df.index.month.isin([1, 4, 7, 10]), 1, 0)
            
            # BYD_Growth_Proxy: Taxa de crescimento do maior competidor
            # Usar dados do setor como proxy para crescimento de BYD
            df['BYD_Growth_Proxy'] = df['Competitor_Avg_Return'].rolling(90).mean() if 'Competitor_Avg_Return' in df.columns else 0
            
        except Exception as e:
            logger.warning(f"Erro ao adicionar features de competição: {e}")
            # Fallback
            df['Competitor_Avg_Return'] = 0
            df['Performance_Gap'] = 0
            df['EV_Market_Share_Proxy'] = 0
            df['New_EV_Models_Proxy'] = 0
            df['BYD_Growth_Proxy'] = 0
        
        return df
    
    def _add_policy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fator 5: Features de políticas (baseadas em timeline histórica)"""
        df = df.copy()
        
        # 1. EV Tax Credit Status (histórico real)
        # Períodos onde Tesla foi elegível para crédito $7,500
        df['EV_Tax_Credit_Active'] = 0
        
        # Timeline histórica:
        # - 2010-2018: Elegível
        # - 2019-2022-08: Não elegível (atingiu 200k limite)
        # - 2022-08-presente: Elegível novamente (IRA)
        
        # Período 1: Até final 2018
        mask1 = df.index <= '2018-12-31'
        df.loc[mask1, 'EV_Tax_Credit_Active'] = 1
        
        # Período 2: Gap (não elegível)
        mask2 = (df.index > '2018-12-31') & (df.index < '2022-08-16')
        df.loc[mask2, 'EV_Tax_Credit_Active'] = 0
        
        # Período 3: IRA reinstated
        mask3 = df.index >= '2022-08-16'
        df.loc[mask3, 'EV_Tax_Credit_Active'] = 1
        
        # 2. China Tariff Level (proxy histórico)
        df['China_Tariff_Level'] = 0.0
        
        # Timeline aproximada:
        tariff_periods = [
            ('2010-01-01', '2017-12-31', 0.0),   # Pré-trade war
            ('2018-01-01', '2019-12-31', 0.25),  # Trump tariffs fase 1
            ('2020-01-01', '2021-12-31', 0.30),  # Fase 2
            ('2022-01-01', '2023-12-31', 0.35),  # Biden mantém/aumenta
            ('2024-01-01', '2024-12-31', 0.40),  # Aumentos recentes
        ]
        
        for start, end, level in tariff_periods:
            mask = (df.index >= start) & (df.index <= end)
            df.loc[mask, 'China_Tariff_Level'] = level
        
        # 3. FSD Regulatory Score (progresso ao longo do tempo)
        df['FSD_Regulatory_Score'] = 0.0
        
        # Marcar milestones de FSD
        fsd_milestones = [
            ('2015-10-01', 0.1),  # Autopilot v1
            ('2016-10-01', 0.2),  # Autopilot v2
            ('2019-04-01', 0.3),  # FSD computer
            ('2020-10-01', 0.4),  # FSD beta release
            ('2021-07-01', 0.5),  # FSD v9
            ('2022-11-01', 0.6),  # FSD wide release
            ('2023-07-01', 0.7),  # FSD v11
            ('2024-01-01', 0.8),  # FSD supervision removed (some areas)
        ]
        
        current_score = 0.0
        for date, score in sorted(fsd_milestones, key=lambda x: x[0]):
            mask = df.index >= date
            df.loc[mask, 'FSD_Regulatory_Score'] = score
            current_score = score
        
        # 4. Policy Uncertainty Index (proxy com election periods)
        df['Policy_Uncertainty_Index'] = 0.0
        
        # Aumentar perto de eleições
        election_dates = ['2012-11-06', '2016-11-08', '2020-11-03', '2024-11-05']
        
        for election in election_dates:
            election_date = pd.Timestamp(election)
            
            # 6 meses antes e 1 mês depois da eleição
            start_uncertainty = election_date - pd.Timedelta(days=180)
            end_uncertainty = election_date + pd.Timedelta(days=30)
            
            mask = (df.index >= start_uncertainty) & (df.index <= end_uncertainty)
            
            # Máximo 30 dias antes da eleição
            days_to_election = (election_date - df.index).days
            uncertainty_score = np.where(
                days_to_election > 0,
                np.minimum(1.0, (180 - days_to_election) / 150),  # Rampa até eleição
                np.maximum(0.0, 1.0 - (-days_to_election / 30))   # Decaimento após
            )
            
            df.loc[mask, 'Policy_Uncertainty_Index'] = np.maximum(
                df.loc[mask, 'Policy_Uncertainty_Index'],
                uncertainty_score[mask]
            )
        
        return df
    
    def get_feature_report(self, df: pd.DataFrame) -> Dict:
        """Gera relatório de features"""
        validation = self.feature_registry.validate_features(df)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'total_rows': len(df),
            'date_range': {
                'start': df.index[0].strftime('%Y-%m-%d'),
                'end': df.index[-1].strftime('%Y-%m-%d')
            },
            'feature_validation': validation,
            'feature_groups': self.feature_registry.get_feature_groups_status(),
            'feature_categories': self._categorize_features(df),
            'missing_data_stats': self._get_missing_data_stats(df),
        }
        
        return report
    
    def _categorize_features(self, df: pd.DataFrame) -> Dict:
        """Categoriza features por tipo"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        categories = {
            'price_related': [col for col in numeric_cols if any(kw in col.lower() 
                              for kw in ['price', 'close', 'open', 'high', 'low'])],
            'volume_related': [col for col in numeric_cols if 'volume' in col.lower()],
            'technical_indicators': [col for col in numeric_cols if any(kw in col.lower()
                                    for kw in ['ma', 'rsi', 'volatility', 'beta'])],
            'returns_related': [col for col in numeric_cols if any(kw in col.lower()
                                 for kw in ['return', 'growth', 'change', 'pct'])],
            'sentiment_scores': [col for col in numeric_cols if any(kw in col.lower()
                                   for kw in ['sentiment', 'score', 'index', 'proxy'])],
            'policy_regulatory': [col for col in numeric_cols if any(kw in col.lower()
                                   for kw in ['policy', 'tariff', 'credit', 'regulatory'])],
        }
        
        # Adicionar 'other' após criar todas as outras categorias
        all_categorized = sum(categories.values(), [])
        categories['other'] = [col for col in numeric_cols if col not in all_categorized]
        
        return {k: len(v) for k, v in categories.items()}
    
    def _get_missing_data_stats(self, df: pd.DataFrame) -> Dict:
        """Estatísticas de dados faltantes"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        missing_stats = {}
        for col in numeric_cols:
            missing_pct = df[col].isnull().sum() / len(df) * 100
            if missing_pct > 0:
                missing_stats[col] = {
                    'missing_pct': round(missing_pct, 2),
                    'first_valid': df[col].first_valid_index(),
                    'last_valid': df[col].last_valid_index(),
                }
        
        return missing_stats


# Função de conveniência
def engineer_tesla_features(df: pd.DataFrame, symbol: str = "TSLA") -> pd.DataFrame:
    """Função conveniente para engenharia de features"""
    engineer = TeslaFeatureEngineer(symbol)
    return engineer.engineer_all_features(df)