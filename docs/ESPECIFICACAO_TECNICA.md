# 🔬 Especificação Técnica Detalhada - Modelo LSTM TSLA

---

## 📐 Arquitetura Detalhada

### Fluxo de Dados Completo

```
ENTRADA (Batch Size B)
    ↓
    └─ Shape: (B, 70, 34)
       • B = número de sequências no batch
       • 70 = timesteps (dias históricos)
       • 34 = features
       • Valores: Normalizados [0, 1]

    ↓ LSTM Layer 1 (160 unidades)
    ├─ Input: (B, 70, 34)
    ├─ Processamento:
    │  ├─ 160 células LSTM independentes
    │  ├─ Cada célula processa todos os 70 timesteps
    │  ├─ Cada célula mantém estado interno (memory cell)
    │  ├─ Return sequences = True → saída inclui todos os timesteps
    ├─ Output: (B, 70, 160)
    ├─ L2 Regularization (0.001):
    │  └─ Penalty = 0.001 * sum(weights²)
    └─ Parâmetros:
       └─ ~120K (160 * (34 + 160) * 4)

    ↓ Dropout Layer 1 (28%)
    ├─ Durante treinamento: Remove 28% das ativações aleatoriamente
    ├─ Durante inferência: Usa média ponderada
    └─ Efeito: Regularização, evita co-adaptação

    ↓ LSTM Layer 2 (80 unidades)
    ├─ Input: (B, 70, 160)
    ├─ Processamento:
    │  ├─ 80 células LSTM independentes
    │  ├─ Processa 160 features de entrada
    │  └─ Return sequences = False → saída é last timestep
    ├─ Output: (B, 80)
    ├─ L2 Regularization (0.001)
    └─ Parâmetros:
       └─ ~60K (80 * (160 + 80) * 4)

    ↓ Dropout Layer 2 (28%)
    ├─ Remove 28% das 80 ativações
    └─ Output: (B, 80)

    ↓ Dense Layer (32 unidades, ReLU)
    ├─ Transformação não-linear:
    │  └─ output[i] = max(0, sum(input[j] * weight[i,j]) + bias[i])
    ├─ ReLU: max(0, x) → introduz não-linearidade
    ├─ Output: (B, 32)
    ├─ L2 Regularization (0.001)
    └─ Parâmetros:
       └─ ~2.6K (32 * (80 + 1))

    ↓ Dropout Layer 3 (14%)
    ├─ Remove 14% das 32 ativações
    └─ Output: (B, 32)

    ↓ Output Layer (1 unidade, Linear)
    ├─ Transformação linear:
    │  └─ output = sum(input[i] * weight[i]) + bias
    ├─ Linear: Sem função de ativação (regressão contínua)
    ├─ Output: (B, 1)
    ├─ Valor: [0, 1] normalizado
    └─ Parâmetros:
       └─ ~33 (1 * (32 + 1))

SAÍDA (Preço Previsto)
    ├─ Shape: (B, 1)
    ├─ Valores: [0, 1] normalizado
    └─ Pós-processamento: Inverter com scaler
       └─ Preço real = scaler.inverse_transform()
```

---

## 🧠 Célula LSTM: Como Funciona

### Mecanismo Interno

```
LSTM Cell em Timestep t:
┌─────────────────────────────────────────────────┐
│                                                 │
│  Inputs:                                        │
│  • x_t: vetor de entrada (34 features)          │
│  • h_{t-1}: hidden state anterior (160)         │
│  • c_{t-1}: cell state anterior (160)           │
│                                                 │
│  Operações:                                     │
│  1. Input Gate: i_t = σ(W_i[h_{t-1}, x_t] + b_i)
│     → Controla quanto da entrada entra na célula
│                                                 │
│  2. Forget Gate: f_t = σ(W_f[h_{t-1}, x_t] + b_f)
│     → Controla quanto do passado é esquecido
│                                                 │
│  3. Candidate: C̃_t = tanh(W_c[h_{t-1}, x_t] + b_c)
│     → Nova informação candidata
│                                                 │
│  4. Cell State: c_t = f_t * c_{t-1} + i_t * C̃_t
│     → Memória de longo prazo (elemento-wise mult)
│                                                 │
│  5. Output Gate: o_t = σ(W_o[h_{t-1}, x_t] + b_o)
│     → Controla saída
│                                                 │
│  6. Hidden State: h_t = o_t * tanh(c_t)
│     → Saída para próximo timestep
│                                                 │
└─────────────────────────────────────────────────┘

Onde:
  σ = sigmoid (0-1, controla fluxo)
  tanh = (-1 a 1, pode "esquecer" valores antigos)
  * = multiplicação elemento-wise
  W, b = pesos e vieses (aprendidos durante treinamento)
```

### Vantagem sobre RNN Simples

| Aspecto | RNN Simples | LSTM |
|--------|-----------|------|
| **Janela efetiva** | ~5-10 passos | ~70+ passos |
| **Vanishing gradient** | Severo | Mitigado |
| **Memória longa** | Difícil | Fácil |
| **Dependências** | Curtas | Longas e curtas |

---

## 📊 Process de Normalização com MinMaxScaler

### Transformação Forward (Treinamento)

Para cada feature f:
```
X_normalized[f] = (X_raw[f] - X_min[f]) / (X_max[f] - X_min[f])
```

**Exemplo com Close:**
```
X_raw['Close'] = [150, 200, 250, 300, ...]
X_min['Close'] = 150 (mínimo histórico)
X_max['Close'] = 300 (máximo histórico)

X_normalized['Close'] = [(150-150)/(300-150), (200-150)/(300-150), ...]
                       = [0.0, 0.333, 0.667, 1.0, ...]
```

### Transformação Inverse (Inferência)

```
X_raw[f] = X_normalized[f] * (X_max[f] - X_min[f]) + X_min[f]

Exemplo:
y_pred_norm = 0.75  # Previsão normalizada
y_pred_real = 0.75 * (300 - 150) + 150 = 262.5  # Preço real
```

**Arquivo Scaler:**
```pickle
MinMaxScaler(
  feature_range=(0, 1),
  n_features_in_=34,
  data_min_=[150.0, 140.0, ..., 0.0],     # Mínimo de cada feature
  data_max_=[300.0, 310.0, ..., 1.0],     # Máximo de cada feature
  data_range_=[150.0, 170.0, ..., 1.0]    # Intervalo
)
```

---

## 🎯 Otimização e Treinamento

### Algoritmo Adam (Optimizer)

```
Atualização de pesos em cada batch:

1. Computar gradiente: g_t = ∇L(θ)
   L = Loss (MSE neste caso)
   θ = todos os pesos

2. Atualizar momentos (adaptativo por parâmetro):
   m_t = β₁ * m_{t-1} + (1 - β₁) * g_t        (média móvel)
   v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²       (variância móvel)
   
   Padrões: β₁ = 0.9, β₂ = 0.999

3. Viés de correção:
   m̂_t = m_t / (1 - β₁ᵗ)
   v̂_t = v_t / (1 - β₂ᵗ)

4. Atualização:
   θ = θ - learning_rate * m̂_t / (√v̂_t + ε)
   
   learning_rate = 0.0005 (neste modelo)
   ε = 10⁻⁷ (pequeno para evitar divisão por zero)
```

### Learning Rate Schedule

```
Epoch   Learning Rate   Eventos
─────────────────────────────────────
1       0.0005          Início
...
50      0.0005          Continua
60      0.0005          ReduceLROnPlateau ativado
70      0.00025         Val loss não melhora (LR reduzido em 50%)
80      0.000125        Val loss ainda não melhora
...
130     0.000125        Early Stopping: Paciência=30 atingida
        MODEL SAVED     Melhor modelo até agora preservado
```

---

## 🔍 Loss Function: Mean Squared Error (MSE)

### Cálculo

```
MSE = (1/N) * Σ(y_real - y_pred)²

Onde:
  N = número de amostras
  y_real = preço real
  y_pred = preço previsto

Exemplo:
  N = 3
  y_real = [100, 105, 110]
  y_pred = [99, 106, 108]
  
  Erros = [1, -1, 2]
  Erros² = [1, 1, 4]
  MSE = (1+1+4)/3 = 2.0
```

### Por que MSE?

- Penaliza erros grandes quadraticamente
- Diferenciável (importante para backprop)
- Apropriado para regressão contínua
- Sensível a outliers (bom para séries financeiras)

---

## 📈 Backpropagation Through Time (BPTT)

### Como LSTM aprende com 70 timesteps

```
Forward Pass (computar predições):
t=1: x₁ → lstm1 → lstm2 → dense → output₁
t=2: x₂ → lstm1 → lstm2 → dense → output₂
...
t=70: x₇₀ → lstm1 → lstm2 → dense → output₇₀
     (apenas este output é usado para loss)

Loss:
L = (y_real - output₇₀)²

Backward Pass (propagação de erro):
∂L/∂output₇₀ → ∂L/∂dense → ∂L/∂lstm2 → ∂L/∂lstm1

Dentro do LSTM:
Gradientes fluxam através de 70 timesteps:
  ∂L/∂lstm2[70] → ∂L/∂lstm2[69] → ... → ∂L/∂lstm2[1]

Cell state preserva gradientes (forget gate permite descartar antigos)
```

### Problema de Vanishing Gradient

```
Sem LSTM:
∂L/∂x₁ = ∂L/∂output * ∂output/∂hidden₇₀ * ... * ∂hidden₂/∂hidden₁
       ≈ ∂L/∂output * 0.8^69  (se ∂hidden/∂hidden ≈ 0.8)
       ≈ 0 (muito pequeno para aprender)

Com LSTM:
Informação viaja via cell state (soma, não multiplicação):
∂c_t/∂c_{t-1} = 1 (se forget gate ≈ 1)
Permite gradientes fluxarem para trás 70 passos!
```

---

## 💾 Regularização L2

### O Que É

```
Loss Total = MSE + λ * (soma de todos os pesos²)

Onde:
  MSE = erro de previsão
  λ = 0.001 (força de regularização)
  Pesos = todos os W em todas as camadas

Exemplo com 2 pesos:
  w₁ = 0.5, w₂ = 2.0
  Penalty = 0.001 * (0.5² + 2.0²) = 0.001 * 4.25 = 0.00425
  
  Total Loss = MSE + 0.00425
```

### Efeito

```
SEM L2 (λ = 0):
  • Modelo pode ter pesos muito grandes
  • Altamente sensível a pequenas mudanças de entrada
  • Overfitting

COM L2 (λ = 0.001):
  • Pesos penalizados (prefere valores pequenos)
  • Mais suave (robustez)
  • Melhor generalização
  
λ = 0.001 é uma escolha moderada:
  • Não muito grande (não prejudica aprendizado)
  • Não muito pequeno (ainda regulariza efetivamente)
```

---

## 🔄 Data Augmentation Implícita

### Sequências Sobrepostas

```
Dados brutos (1000 dias):
┌─────────────────────────────────────────────────────────┐
│ t1  t2  t3  t4  t5  t6  ... t1000                       │
└─────────────────────────────────────────────────────────┘

Com Sequence Length = 70:

Sequência 1: [t1:t70] → predict t71
Sequência 2: [t2:t71] → predict t72
Sequência 3: [t3:t72] → predict t73
...
Sequência 931: [t931:t1000] → predict t1001

Resultado:
  • 931 sequências de 1000 dados
  • Sobreposição: 69/70 dias compartilhados
  • Aumento efetivo de dados
  • Diferentes "visões" do mesmo padrão
```

---

## 🎲 Dropout Detalhado

### Mecanismo

```
Durante TREINAMENTO (dropout ativo):

Layer de 32 neurônios com dropout 0.14:

Sem dropout:    [0.5, 0.3, 0.8, 0.2, ..., 0.6]
                                                     
Com dropout:    [0.5,  ×,  0.8,  ×,  ..., 0.6]
                     dropped

Máscara aleatória a cada batch!

Durante INFERÊNCIA:
Sem dropout: [0.5, 0.3, 0.8, 0.2, ..., 0.6]
             (todas ativações usadas, mas escaladas por 1/(1-dropout))

Propósito:
- Simula "ensemble" de modelos reduzidos
- Força aprendizado redundante
- Evita co-adaptação de neurônios
```

### Por Que 28%?

```
Dropout Rate vs Performance:

0% dropout:    ├─ Risco alto de overfitting
5% dropout:    ├─ Pouca regularização
14% dropout:   ├─ Suave
28% dropout:   ├─ **ESCOLHIDO** (bom balance)
40% dropout:   ├─ Começa a prejudicar
50% dropout:   ├─ Muito dropout
70% dropout:   └─ Praticamente desabilita o modelo
```

---

## 📊 Métricas de Avaliação - Fórmulas Detalhadas

### R² Score (Coeficiente de Determinação)

```
        SS_res        Σ(y_real - y_pred)²
R² = 1 - ─────── = 1 - ──────────────────
        SS_tot        Σ(y_real - y_mean)²

Interpretação:
  R² = 1.0    → Previsão perfeita
  R² = 0.9054 → Explica 90.54% da variância (EXCELENTE)
  R² = 0.5    → Modelo medíocre
  R² < 0      → Pior que usar média (muito ruim)

Exemplo:
  y_real = [100, 105, 110]
  y_pred = [99, 106, 108]
  y_mean = 105
  
  SS_res = (100-99)² + (105-106)² + (110-108)² = 1 + 1 + 4 = 6
  SS_tot = (100-105)² + (105-105)² + (110-105)² = 25 + 0 + 25 = 50
  
  R² = 1 - 6/50 = 0.88 (excelente)
```

### MAE (Mean Absolute Error)

```
         1  n
MAE = ─── Σ |y_real - y_pred|
        n i=1

Interpretação direta (em dólares):
  MAE = $16.48 → Em média, modelo erra por ±$16.48
  
  • Se preço médio TSLA ≈ $250:
    Erro percentual ≈ 16.48 / 250 ≈ 6.6%
```

### RMSE (Root Mean Squared Error)

```
         √(1/n * Σ(y_real - y_pred)²)

Penaliza erros maiores:
  RMSE > MAE sempre (penalização quadrática)
  
Exemplo:
  Erros: [1, 1, 100]
  MAE = (1+1+100)/3 = 34
  RMSE = √((1+1+10000)/3) = √3334 = 57.7
  
  RMSE muito maior que MAE → Erros outliers significativos
```

### MAPE (Mean Absolute Percentage Error)

```
           1  n  |y_real - y_pred|
MAPE = ─── Σ  ─────────────────
        n i=1   |y_real|

MAPE = 4.77% → Em média, erro relativo é 4.77%

Vantagem: Escala-independente
  • MAE de $10 em preço de $100 é "melhor" que $10 em preço de $50
  • MAPE detecta isso
```

---

## ⏱️ Timeline de Treinamento

```
Época    Loss      Val Loss   LR        Evento
─────────────────────────────────────────────────
1        0.0850    0.0920     0.0005    Início
10       0.0420    0.0650     0.0005    Convergência rápida
20       0.0180    0.0450     0.0005    Validação melhorando
30       0.0120    0.0380     0.0005    
40       0.0095    0.0350     0.0005    
50       0.0082    0.0340     0.0005    Validação estabilizando
60       0.0078    0.0338     0.0005    Early stop "patience" começa
70       0.0075    0.0340     0.00025   ReduceLROnPlateau: LR reduzido
80       0.0072    0.0341     0.00025   
90       0.0070    0.0340     0.00025   
100      0.0069    0.0340     0.00025   
110      0.0068    0.0341     0.000125  ReduceLROnPlateau: LR reduzido
120      0.0068    0.0342     0.000125  
130      0.0067    0.0343     0.000125  ❌ EARLY STOP
           ↑                            Melhor modelo: Época 50
        SAVE BEST                      Paciência expirou (30 épocas)
```

---

## 🚨 Possíveis Problemas e Soluções

### Problema 1: Divergência (Loss → ∞)

```
Causa: Learning rate muito alto
Solução:
  • Reduzir learning_rate para 0.0001
  • Verificar normalização das features
  • Cliper gradientes (max_norm=1.0)
```

### Problema 2: Plateau (Loss não muda)

```
Causa: Learning rate muito baixo ou modelo saturado
Solução:
  • Aumentar learning_rate
  • Aumentar LSTM units
  • Adicionar mais features relevantes
```

### Problema 3: Overfitting Severo

```
Sinais:
  Train loss: 0.001
  Val loss: 0.1 (100x maior!)

Soluções:
  • Aumentar dropout (de 0.28 para 0.4)
  • Aumentar L2 (de 0.001 para 0.01)
  • Reduzir LSTM units
  • Mais dados
```

### Problema 4: Previsões Sempre Iguais

```
Causa: Modelo "colapsa" para previsão média
Solução:
  • Verificar se saída é denormalizada corretamente
  • Verificar features (podem estar constantes)
  • Aumentar variance das features com escala
```

---

## 🔬 Validação Estatística

### Teste de Normalidade dos Resíduos

```
Resíduos = y_real - y_pred = [e1, e2, ..., en]

Teste Shapiro-Wilk:
  H0: Resíduos são normalmente distribuídos
  H1: Não são normais
  
  p-value > 0.05 → Aceitar H0 (resíduos normais ✓)
  p-value < 0.05 → Rejeitar H0 (resíduos não-normais ✗)
  
Importância:
  • Resíduos normais → Intervalos de confiança válidos
  • Permite cálculo de confiança com método-t
```

### Teste de Autocorrelação

```
Teste Durbin-Watson:
  DW = Σ(et - e{t-1})² / Σ(et)²
  
  DW ≈ 2: Sem autocorrelação ✓
  DW << 2: Autocorrelação positiva (erros tendem a mesma direção)
  DW >> 2: Autocorrelação negativa
  
Importância:
  • Se autocorrelação: modelo não capturou dependências
  • Indica features faltando
```

---

**Documentação Técnica Completa - Pronto para Arquitetura!** 🎓
