# =============================================================================
# PROJETO: Agente de IA com Groq (API Gratuita)
# Objetivo: Responder perguntas sobre indicadores econômicos em linguagem natural
# Autor: Projeto Portfólio - Analista de Dados com IA | Sicoob
# =============================================================================

# Instalação:
# pip install groq python-dotenv

# -----------------------------------------------------------------------------
# 1. IMPORTAÇÕES
# -----------------------------------------------------------------------------
import os
import joblib
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("✅ Bibliotecas carregadas!")

# -----------------------------------------------------------------------------
# 2. CARREGAR DADOS E MODELO
# -----------------------------------------------------------------------------
print("📂 Carregando dados e modelo...")

df_bruto    = pd.read_csv(os.path.join(BASE_DIR, "indicadores_bacen_bruto.csv"),
                           index_col=0, parse_dates=True)
df_features = pd.read_csv(os.path.join(BASE_DIR, "features_bacen.csv"),
                           index_col=0, parse_dates=True)

modelo   = joblib.load(os.path.join(BASE_DIR, "modelo_inadimplencia.pkl"))
scaler   = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))

print("✅ Dados e modelo carregados!")

# -----------------------------------------------------------------------------
# 3. CONTEXTO DOS DADOS PARA O AGENTE
# -----------------------------------------------------------------------------
def gerar_contexto(df: pd.DataFrame) -> str:
    """Gera um resumo dos dados para enviar ao agente como contexto."""
    stats      = df.describe().round(3).to_string()
    ultimos    = df.tail(6).round(3).to_string()
    correlacao = df.corr().round(3).to_string()

    return f"""
Você é um analista de dados especialista em mercado financeiro cooperativista,
com foco em risco de crédito e indicadores macroeconômicos do Brasil.

Você tem acesso aos seguintes dados do Banco Central do Brasil (BACEN):

## Indicadores disponíveis:
- IPCA (%): Inflação mensal oficial
- Inadimplência (%): Taxa de inadimplência total do sistema financeiro
- Juros PF (% a.a.): Taxa média de juros para pessoa física
- Selic Meta (% a.a.): Taxa básica de juros da economia

## Período dos dados:
De {df.index[0].strftime('%B/%Y')} até {df.index[-1].strftime('%B/%Y')} ({len(df)} meses)

## Estatísticas descritivas:
{stats}

## Últimos 6 meses:
{ultimos}

## Correlação entre indicadores:
{correlacao}

## Sobre o modelo preditivo:
- Algoritmo: Random Forest Regressor
- Target: Inadimplência (%)
- R²: 0.556 (explica 55.6% da variação da inadimplência)
- MAE: 0.219 pontos percentuais
- Features utilizadas: {len(features)} variáveis (lags, médias móveis, variações)

Responda sempre em português, de forma clara e objetiva.
Quando citar números, contextualize com o cenário econômico brasileiro.
Se a pergunta não estiver relacionada aos dados disponíveis, informe educadamente.
"""


def fazer_previsao() -> str:
    """Gera previsão de inadimplência com o modelo treinado."""
    ultima_linha = df_features[features].iloc[[-1]]
    X_scaled     = scaler.transform(ultima_linha)
    previsao     = modelo.predict(X_scaled)[0]
    ultimo_real  = df_bruto['Inadimplência (%)'].iloc[-1]

    return (f"Com base no modelo Random Forest treinado com dados do BACEN, "
            f"a inadimplência prevista para o próximo período é de "
            f"{previsao:.2f}%. "
            f"O último valor real registrado foi {ultimo_real:.2f}%.")


# -----------------------------------------------------------------------------
# 4. AGENTE GROQ
# -----------------------------------------------------------------------------
class AgenteCredito:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ Chave GROQ_API_KEY não encontrada!\n"
                "   Crie um arquivo .env na pasta do projeto com:\n"
                "   GROQ_API_KEY=sua-chave-aqui\n"
                "   Obtenha gratuitamente em: https://console.groq.com"
            )
        self.client    = Groq(api_key=api_key)
        self.contexto  = gerar_contexto(df_bruto)
        self.historico = []
        print("✅ Agente Groq inicializado!")

    def perguntar(self, pergunta: str) -> str:
        """Envia uma pergunta ao agente e retorna a resposta."""

        # Detecta pedido de previsão e injeta resultado do modelo
        palavras_previsao = ["preveja", "previsão", "prever", "próximo", "futuro", "vai ser"]
        if any(p in pergunta.lower() for p in palavras_previsao):
            previsao_info = fazer_previsao()
            pergunta = f"{pergunta}\n\n[Resultado do modelo preditivo]: {previsao_info}"

        # Adiciona ao histórico
        self.historico.append({"role": "user", "content": pergunta})

        # Chama a API Groq
        resposta = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": self.contexto},
                *self.historico,
            ],
            max_tokens=1024,
            temperature=0.3,
        )

        texto = resposta.choices[0].message.content

        # Salva resposta no histórico (memória da conversa)
        self.historico.append({"role": "assistant", "content": texto})

        return texto

    def resetar(self):
        """Limpa o histórico da conversa."""
        self.historico = []
        print("🔄 Histórico resetado!")


# -----------------------------------------------------------------------------
# 5. INTERFACE DE CHAT NO TERMINAL
# -----------------------------------------------------------------------------
def main():
    print("\n" + "="*60)
    print("🤖 AGENTE DE ANÁLISE DE CRÉDITO — powered by Groq + Llama 3")
    print("="*60)
    print("Faça perguntas sobre os indicadores econômicos do BACEN.")
    print("Comandos especiais:")
    print("  'sair'   → encerra o agente")
    print("  'reset'  → limpa o histórico da conversa")
    print("  'dados'  → mostra os últimos 6 meses de dados")
    print("="*60)

    agente = AgenteCredito()

    exemplos = [
        "Qual foi o período de maior inadimplência nos dados?",
        "Como a Selic se comportou durante a pandemia (2020)?",
        "Existe correlação entre juros e inadimplência?",
        "Preveja a inadimplência para o próximo mês.",
        "Quais indicadores mais influenciam a inadimplência?",
        "Faça um resumo do cenário econômico atual.",
    ]
    print("\n💡 Exemplos de perguntas:")
    for i, ex in enumerate(exemplos, 1):
        print(f"   {i}. {ex}")
    print()

    while True:
        try:
            pergunta = input("Você: ").strip()

            if not pergunta:
                continue
            elif pergunta.lower() == "sair":
                print("👋 Encerrando agente. Até logo!")
                break
            elif pergunta.lower() == "reset":
                agente.resetar()
                continue
            elif pergunta.lower() == "dados":
                print("\n📊 Últimos 6 meses:")
                print(df_bruto.tail(6).round(3).to_string())
                print()
                continue

            print("\n🤖 Agente: ", end="", flush=True)
            resposta = agente.perguntar(pergunta)
            print(resposta)
            print()

        except KeyboardInterrupt:
            print("\n\n👋 Encerrando agente. Até logo!")
            break
        except Exception as e:
            print(f"\n❌ Erro: {e}\n")


if __name__ == "__main__":
    main()
