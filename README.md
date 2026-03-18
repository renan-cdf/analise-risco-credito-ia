# 📊 Análise de Risco de Crédito com IA + Dados do BACEN

> Projeto de portfólio demonstrando integração com API do Banco Central, modelagem preditiva de inadimplência e agente inteligente com LLM para o setor financeiro.

🌐 **[Acesse o dashboard online](https://iariscocredito.streamlit.app)**

---

## 🎯 Problema de Negócio

A inadimplência é um dos principais riscos para instituições financeiras. Antecipar movimentos de inadimplência com base em indicadores macroeconômicos permite que gestores tomem decisões preventivas — ajustando concessões de crédito, taxas e políticas de cobrança antes que o problema se agrave.

**Pergunta central:** *É possível prever a inadimplência futura usando indicadores públicos do Banco Central?*

---

## 🏗️ Estrutura do Projeto

```
analise-risco-credito-ia/
│
├── bacen.py              # Coleta e visualização de dados do BACEN
├── modelo.py             # Treinamento e comparação de modelos preditivos
├── agente.py             # Agente de IA via terminal (Groq + Llama 3)
├── app.py                # Dashboard interativo (Streamlit)
├── requirements.txt      # Dependências do projeto
└── README.md
```

---

## 🔬 Metodologia

### 1. Coleta de Dados
- **Fonte:** API pública do Banco Central — SGS (Sistema Gerenciador de Séries)
- **Biblioteca:** `python-bcb`
- **Indicadores:** IPCA, Selic Meta, Juros PF e Inadimplência total
- **Período:** 2015 → hoje (atualizado automaticamente a cada execução)

### 2. Feature Engineering
- Lags temporais (1 e 3 meses) para capturar efeitos defasados
- Médias móveis (3, 6 e 12 meses)
- Variações mensais e trimestrais (momentum)

### 3. Modelagem
- Divisão **temporal** treino/teste (80/20) — sem vazamento de dados
- Comparação de 5 algoritmos: Regressão Linear, Ridge, Random Forest, Gradient Boosting e XGBoost
- Métricas: **R², MAE e RMSE**

### 4. Agente Inteligente
- LLM Llama 3.3 70B via Groq para responder perguntas sobre os dados em linguagem natural
- Integrado ao modelo preditivo para responder perguntas de previsão em tempo real

---

## 📈 Resultados

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| 🏆 Random Forest | 0.2187 | 0.2741 | 0.5557 |
| XGBoost | 0.2190 | 0.2742 | 0.5555 |
| Gradient Boosting | 0.2210 | 0.2750 | 0.5520 |
| Ridge | 0.2530 | 0.3470 | 0.2860 |
| Regressão Linear | 0.2630 | 0.3530 | 0.2640 |

> O modelo explica **55,6% da variação da inadimplência** usando apenas indicadores públicos do BACEN.

---

## 🚀 Como Usar

### Pré-requisitos
- Python 3.10+
- Chave gratuita do Groq: [console.groq.com](https://console.groq.com)

### 1. Clone o repositório
```bash
git clone https://github.com/renan-cdf/analise-risco-credito-ia.git
cd analise-risco-credito-ia
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Configure a chave do Groq
Crie um arquivo `.env` na pasta do projeto:
```
GROQ_API_KEY=sua-chave-aqui
```

### 4. Gere os dados e treine o modelo
```bash
python bacen.py
python modelo.py
```

### 5. Rode o dashboard
```bash
streamlit run app.py
```
O navegador abrirá automaticamente em `http://localhost:8501`

### (Opcional) Rode o agente no terminal
```bash
python agente.py
```

---

## 🖥️ Dashboard

O dashboard possui 3 telas:

- **📊 Dashboard** — gráficos interativos dos indicadores do BACEN com correlações
- **🤖 Agente IA** — chat em linguagem natural sobre os dados, powered by Groq + Llama 3
- **🔮 Previsão** — previsão de inadimplência para o próximo mês com o modelo treinado

---

## 🛠️ Tecnologias

| Categoria | Tecnologia |
|---|---|
| Linguagem | Python 3.10+ |
| Dados BACEN | python-bcb |
| Manipulação | Pandas, NumPy |
| Modelagem | Scikit-learn, XGBoost |
| Visualização | Plotly |
| Agente IA | Groq API + Llama 3.3 70B |
| Dashboard | Streamlit |

---

## 📌 Próximos Passos

- [ ] Deploy na Azure
- [ ] Integração com banco de dados SQL Server
- [ ] Pipeline automatizado com Apache Airflow
- [ ] Testes unitários com pytest
- [ ] CI/CD com GitHub Actions

---

## 👤 Autor

**Renan Fonseca**  
[LinkedIn](https://linkedin.com/in/seu-perfil) • [GitHub](https://github.com/renan-cdf)

---

> *Este projeto foi desenvolvido como portfólio técnico, utilizando exclusivamente dados públicos do Banco Central do Brasil.*
