"""
Sistema de Registro e Gerenciamento de Features
"""
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureRegistry:
    """
    Registro centralizado de features com suporte a grupos modulares
    """
    
    def __init__(self, config_path: str = "src/config/model_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.feature_groups = {}
        self.feature_descriptions = {}
        self.feature_importance_history = []
        
        self._initialize_registry()
        logger.info(f"Feature Registry inicializado com {len(self.get_active_features())} features ativas")
    
    def _load_config(self) -> Dict:
        """Carrega configuração do arquivo YAML"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Erro ao carregar config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Configuração padrão se arquivo não existir"""
        return {
            'model': {
                'features': {
                    'core': {'enabled': True, 'features': ['Close', 'Volume', 'Returns']},
                    'execution': {'enabled': True, 'features': []},
                    'sentiment': {'enabled': True, 'features': []},
                    'macro': {'enabled': True, 'features': []},
                    'competition': {'enabled': False, 'features': []},
                    'policy': {'enabled': False, 'features': []},
                }
            }
        }
    
    def _initialize_registry(self):
        """Inicializa o registro com as features da config"""
        for group_name, group_config in self.config['model']['features'].items():
            self.register_feature_group(
                name=group_name,
                features=group_config.get('features', []),
                enabled=group_config.get('enabled', False),
                priority=group_config.get('priority', 99),
                description=group_config.get('description', '')
            )
    
    def register_feature_group(self, name: str, features: List[str], 
                              enabled: bool = True, priority: int = 99,
                              description: str = "") -> None:
        """
        Registra um novo grupo de features
        
        Args:
            name: Nome do grupo
            features: Lista de nomes de features
            enabled: Se o grupo está ativo
            priority: Prioridade (menor = mais importante)
            description: Descrição do grupo
        """
        self.feature_groups[name] = {
            'features': features,
            'enabled': enabled,
            'priority': priority,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        # Registrar descrições individuais
        for feature in features:
            self.feature_descriptions[feature] = {
                'group': name,
                'description': f"Feature do grupo {name}",
                'type': self._infer_feature_type(feature)
            }
        
        logger.info(f"Grupo '{name}' registrado com {len(features)} features")
    
    def _infer_feature_type(self, feature_name: str) -> str:
        """Infere o tipo da feature baseado no nome"""
        feature_name_lower = feature_name.lower()
        
        if any(keyword in feature_name_lower for keyword in ['return', 'growth', 'rate', 'change']):
            return 'percentage'
        elif any(keyword in feature_name_lower for keyword in ['price', 'value', 'amount']):
            return 'currency'
        elif any(keyword in feature_name_lower for keyword in ['volume', 'count', 'ratio']):
            return 'numeric'
        elif any(keyword in feature_name_lower for keyword in ['score', 'index', 'sentiment']):
            return 'score'
        elif any(keyword in feature_name_lower for keyword in ['active', 'status', 'flag']):
            return 'binary'
        else:
            return 'numeric'
    
    def enable_feature_group(self, group_name: str) -> None:
        """Ativa um grupo de features"""
        if group_name in self.feature_groups:
            self.feature_groups[group_name]['enabled'] = True
            self.feature_groups[group_name]['updated_at'] = datetime.now().isoformat()
            logger.info(f"Grupo '{group_name}' ativado")
        else:
            logger.warning(f"Grupo '{group_name}' não encontrado")
    
    def disable_feature_group(self, group_name: str) -> None:
        """Desativa um grupo de features"""
        if group_name in self.feature_groups:
            self.feature_groups[group_name]['enabled'] = False
            self.feature_groups[group_name]['updated_at'] = datetime.now().isoformat()
            logger.info(f"Grupo '{group_name}' desativado")
        else:
            logger.warning(f"Grupo '{group_name}' não encontrado")
    
    def add_feature_to_group(self, group_name: str, feature_name: str, 
                            description: str = "") -> None:
        """Adiciona uma feature a um grupo existente"""
        if group_name in self.feature_groups:
            if feature_name not in self.feature_groups[group_name]['features']:
                self.feature_groups[group_name]['features'].append(feature_name)
                self.feature_groups[group_name]['updated_at'] = datetime.now().isoformat()
                
                self.feature_descriptions[feature_name] = {
                    'group': group_name,
                    'description': description,
                    'type': self._infer_feature_type(feature_name)
                }
                logger.info(f"Feature '{feature_name}' adicionada ao grupo '{group_name}'")
            else:
                logger.warning(f"Feature '{feature_name}' já existe no grupo '{group_name}'")
        else:
            logger.error(f"Grupo '{group_name}' não encontrado")
    
    def get_active_features(self) -> List[str]:
        """Retorna todas as features ativas, ordenadas por prioridade"""
        active_features = []
        
        # Ordenar grupos por prioridade
        sorted_groups = sorted(
            [(name, data) for name, data in self.feature_groups.items() 
             if data['enabled']],
            key=lambda x: x[1]['priority']
        )
        
        for group_name, group_data in sorted_groups:
            active_features.extend(group_data['features'])
        
        return active_features
    
    def get_feature_groups_status(self) -> Dict:
        """Retorna status de todos os grupos de features"""
        status = {}
        for group_name, group_data in self.feature_groups.items():
            status[group_name] = {
                'enabled': group_data['enabled'],
                'feature_count': len(group_data['features']),
                'priority': group_data['priority'],
                'last_updated': group_data['updated_at']
            }
        return status
    
    def record_feature_importance(self, model_name: str, 
                                 feature_importance: Dict[str, float]) -> None:
        """
        Registra a importância das features para análise histórica
        
        Args:
            model_name: Nome do modelo
            feature_importance: Dicionário feature->importance
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'model': model_name,
            'feature_importance': feature_importance,
            'active_features': self.get_active_features()
        }
        self.feature_importance_history.append(record)
        
        # Manter apenas últimos 100 registros
        if len(self.feature_importance_history) > 100:
            self.feature_importance_history = self.feature_importance_history[-100:]
    
    def get_feature_importance_trend(self, feature_name: str) -> List[Dict]:
        """Retorna histórico de importância para uma feature específica"""
        trend = []
        for record in self.feature_importance_history:
            if feature_name in record['feature_importance']:
                trend.append({
                    'timestamp': record['timestamp'],
                    'importance': record['feature_importance'][feature_name],
                    'model': record['model']
                })
        return trend
    
    def export_config(self, export_path: str) -> None:
        """Exporta configuração atual para arquivo"""
        export_data = {
            'feature_groups': self.feature_groups,
            'feature_descriptions': self.feature_descriptions,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Configuração exportada para {export_path}")
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Valida se todas as features ativas existem no DataFrame
        
        Returns:
            Dict com missing features e available features
        """
        active_features = self.get_active_features()
        available_features = [f for f in active_features if f in df.columns]
        missing_features = [f for f in active_features if f not in df.columns]
        
        return {
            'available': available_features,
            'missing': missing_features,
            'total_active': len(active_features),
            'coverage': len(available_features) / len(active_features) if active_features else 0
        }


# Singleton para uso global
_feature_registry: Optional[FeatureRegistry] = None

def get_feature_registry(config_path: str = "src/config/model_config.yaml") -> FeatureRegistry:
    """Retorna instância singleton do FeatureRegistry"""
    global _feature_registry
    if _feature_registry is None:
        _feature_registry = FeatureRegistry(config_path)
    return _feature_registry