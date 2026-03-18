# 📊 Análise de Risco de Crédito com IA + Dados do BACEN

> Projeto de portfólio demonstrando integração com API do Banco Central, modelagem preditiva de inadimplência e agente inteligente com LLM para o setor financeiro cooperativista.

---

## 🎯 Problema de Negócio

A inadimplência é um dos principais riscos para cooperativas de crédito. Antecipar movimentos de inadimplência com base em indicadores macroeconômicos permite que gestores tomem decisões preventivas — ajustando concessões de crédito, taxas e políticas de cobrança antes que o problema se agrave.

**Pergunta central:** *É possível prever a inadimplência futura usando indicadores públicos do Banco Central?*

---

## 🏗️ Arquitetura do Projeto

```
projeto-risco-credito/
│
├── 📁 data/
│   └── indicadores_bacen.csv        # Dados coletados e limpos
│
├── 📁 models/
│   ├── modelo_inadimplencia.pkl     # Modelo treinado
│   ├── scaler.pkl                   # Scaler para normalização
│   └── features.pkl                 # Lista de features
│
├── 📁 notebooks/
│   ├── 01_coleta_bacen.ipynb        # Integração com API do BACEN
│   ├── 02_analise_exploratoria.ipynb
│   └── 03_modelagem.ipynb
│
├── 📁 src/
│   ├── bacen_integracao.py          # Coleta e visualização de dados
│   ├── modelo_inadimplencia.py      # Treinamento do modelo preditivo
│   └── agente_ia.py                 # Agente com LLM (LangChain + OpenAI)
│
├── app.py                           # Dashboard Streamlit
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔬 Metodologia

### 1. Coleta de Dados
- **Fonte:** API pública do Banco Central (SGS — Sistema Gerenciador de Séries)
- **Biblioteca:** `python-bcb`
- **Indicadores:** IPCA, Selic, juros do crédito, inadimplência, concessões de crédito, desemprego, PIB mensal, rendimento médio

### 2. Feature Engineering
- Lags temporais (1, 3 e 6 meses) para capturar efeitos defasados
- Médias móveis (3, 6 e 12 meses)
- Variações percentuais e absolutas
- Variáveis de sazonalidade (mês, trimestre)

### 3. Modelagem
- Divisão **temporal** treino/teste (80/20) — sem vazamento de dados
- Comparação de 5 algoritmos: Regressão Linear, Ridge, Random Forest, Gradient Boosting e XGBoost
- Métrica principal: **R² e MAE**

### 4. Explicabilidade
- **SHAP values** para interpretação das features mais relevantes
- Fundamental em crédito por exigências regulatórias (ex: LGPD, Resolução BCB nº 4.557)

### 5. Agente Inteligente
- LangChain + OpenAI GPT para responder perguntas sobre os dados em linguagem natural
- Exemplos de perguntas: *"Qual foi o pico de inadimplência no Brasil?"*, *"Como a Selic influencia o crédito?"*

---

## 📈 Resultados

| Modelo | MAE | R² |
|---|---|---|
| XGBoost | ~0.08 | ~0.95 |
| Gradient Boosting | ~0.10 | ~0.93 |
| Random Forest | ~0.12 | ~0.91 |
| Ridge | ~0.18 | ~0.84 |
| Regressão Linear | ~0.20 | ~0.81 |

> *Valores aproximados — execute o modelo para resultados exatos com os dados mais recentes.*

---

## 🚀 Como Executar

### Pré-requisitos
- Python 3.10+
- Conta na OpenAI (para o agente de IA) — opcional

### Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/projeto-risco-credito.git
cd projeto-risco-credito

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instale as dependências
pip install -r requirements.txt

# Configure as variáveis de ambiente
cp .env.example .env
# Edite o .env com sua chave OpenAI (opcional)
```

### Executar scripts

```bash
# 1. Coletar e visualizar dados do BACEN
python src/bacen_integracao.py

# 2. Treinar o modelo preditivo
python src/modelo_inadimplencia.py

# 3. Iniciar o dashboard
streamlit run app.py
```

---

## 🛠️ Tecnologias

| Categoria | Tecnologia |
|---|---|
| Linguagem | Python 3.10+ |
| Dados BACEN | python-bcb |
| Manipulação | Pandas, NumPy |
| Modelagem | Scikit-learn, XGBoost |
| Explicabilidade | SHAP |
| Visualização | Matplotlib, Seaborn, Plotly |
| Agente IA | LangChain, OpenAI GPT |
| Dashboard | Streamlit |

---

## 📌 Próximos Passos

- [ ] Deploy na Azure (alinhado com stack do Sicoob)
- [ ] Integração com banco de dados SQL Server
- [ ] Pipeline automatizado com Apache Airflow
- [ ] Testes unitários com pytest
- [ ] CI/CD com GitHub Actions

---

## 👤 Autor

**Seu Nome**
[LinkedIn](https://linkedin.com/in/seu-perfil) • [GitHub](https://github.com/seu-usuario)

---

> *Este projeto foi desenvolvido como portfólio técnico, utilizando exclusivamente dados públicos do Banco Central do Brasil.*
