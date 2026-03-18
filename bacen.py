# =============================================================================
# PROJETO: Integração com API do Banco Central (BACEN)
# Objetivo: Coletar indicadores econômicos para análise de risco de crédito
# Autor: Projeto Portfólio - Analista de Dados com IA | Sicoob
# =============================================================================

# -----------------------------------------------------------------------------
# 1. IMPORTAÇÕES
# -----------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bcb import sgs
from datetime import datetime

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Bibliotecas carregadas com sucesso!")

# -----------------------------------------------------------------------------
# 2. COLETA DE INDICADORES ECONÔMICOS VIA SGS (BACEN)
# -----------------------------------------------------------------------------
DATA_INICIO = "2015-03-18"
DATA_FIM    = datetime.today().strftime("%Y-%m-%d")

print(f"📡 Coletando dados do BACEN de {DATA_INICIO} até {DATA_FIM}...")

indicadores = sgs.get(
    {
        "IPCA (%)":            433,    # Inflação mensal
        "Inadimplência (%)":   21082,  # Inadimplência total
        "Juros PF (% a.a.)":   20714,  # Taxa de juros pessoa física
        "Selic Meta (% a.a.)": 4189,   # Taxa Selic meta (mensal)
    },
    start=DATA_INICIO,
    end=DATA_FIM,
)

print(f"✅ Dados coletados! Shape: {indicadores.shape}")
print(indicadores.head())

# -----------------------------------------------------------------------------
# 3. LIMPEZA E TRATAMENTO DOS DADOS
# -----------------------------------------------------------------------------
print("\n🔧 Tratando dados...")

indicadores_clean = indicadores.ffill().bfill()

print(indicadores_clean.info())
print(f"\nValores nulos:\n{indicadores_clean.isnull().sum()}")
print(f"\n📈 Estatísticas descritivas:\n{indicadores_clean.describe().round(2)}")

# -----------------------------------------------------------------------------
# 4. ANÁLISE EXPLORATÓRIA
# -----------------------------------------------------------------------------

# 4.1 — Evolução dos indicadores
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Indicadores Econômicos BACEN (2015–Hoje)', fontsize=16, fontweight='bold')

cores   = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
colunas = indicadores_clean.columns.tolist()

for ax, coluna, cor in zip(axes.flatten(), colunas, cores):
    ax.plot(indicadores_clean.index, indicadores_clean[coluna], color=cor, linewidth=1.5)
    ax.set_title(coluna, fontsize=12, fontweight='bold')
    ax.set_xlabel('Data')
    ax.set_ylabel('%')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('indicadores_economicos.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ indicadores_economicos.png salvo")

# 4.2 — Mapa de correlação
plt.figure(figsize=(8, 6))
sns.heatmap(
    indicadores_clean.corr(),
    annot=True, fmt='.2f', cmap='RdYlGn',
    center=0, square=True, linewidths=0.5
)
plt.title('Correlação entre Indicadores Econômicos', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlacao_indicadores.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ correlacao_indicadores.png salvo")

# 4.3 — Inadimplência vs Selic (eixo duplo)
fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.set_xlabel('Data')
ax1.set_ylabel('Inadimplência (%)', color='#E74C3C')
ax1.plot(indicadores_clean.index, indicadores_clean['Inadimplência (%)'],
         color='#E74C3C', label='Inadimplência', linewidth=2)
ax1.tick_params(axis='y', labelcolor='#E74C3C')

ax2 = ax1.twinx()
ax2.set_ylabel('Selic Meta (% a.a.)', color='#3498DB')
ax2.plot(indicadores_clean.index, indicadores_clean['Selic Meta (% a.a.)'],
         color='#3498DB', label='Selic Meta', linewidth=2, linestyle='--')
ax2.tick_params(axis='y', labelcolor='#3498DB')

fig.suptitle('Inadimplência vs Taxa Selic', fontsize=14, fontweight='bold')
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))
plt.tight_layout()
plt.savefig('inadimplencia_vs_selic.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ inadimplencia_vs_selic.png salvo")

# -----------------------------------------------------------------------------
# 5. FEATURE ENGINEERING PARA MODELO DE CRÉDITO
# -----------------------------------------------------------------------------
print("\n⚙️ Criando features para modelo de crédito...")

df_features = indicadores_clean.copy()

# Variações mensais e trimestrais (momentum)
for col in indicadores_clean.columns:
    df_features[f'{col}_var_1m'] = df_features[col].pct_change(1) * 100
    df_features[f'{col}_var_3m'] = df_features[col].pct_change(3) * 100

# Médias móveis (tendência)
for col in indicadores_clean.columns:
    df_features[f'{col}_mm3']  = df_features[col].rolling(3).mean()
    df_features[f'{col}_mm6']  = df_features[col].rolling(6).mean()
    df_features[f'{col}_mm12'] = df_features[col].rolling(12).mean()

# Lags (efeito defasado dos indicadores)
for col in indicadores_clean.columns:
    df_features[f'{col}_lag1'] = df_features[col].shift(1)
    df_features[f'{col}_lag3'] = df_features[col].shift(3)

df_features = df_features.dropna()

print(f"✅ Features criadas! Shape final: {df_features.shape}")

# -----------------------------------------------------------------------------
# 6. EXPORTAÇÃO
# -----------------------------------------------------------------------------
indicadores_clean.to_csv('indicadores_bacen_bruto.csv')
df_features.to_csv('features_bacen.csv')
print("✅ Arquivos CSV exportados!")

# -----------------------------------------------------------------------------
# 7. RESUMO FINAL
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("📋 RESUMO DA INTEGRAÇÃO BACEN")
print("="*60)
print(f"  Período           : {DATA_INICIO} → {DATA_FIM}")
print(f"  Indicadores       : {len(indicadores_clean.columns)}")
print(f"  Registros brutos  : {len(indicadores_clean)}")
print(f"  Features geradas  : {len(df_features.columns)}")
print(f"  Registros modelo  : {len(df_features)}")
print("\n  Arquivos gerados:")
print("    - indicadores_economicos.png")
print("    - correlacao_indicadores.png")
print("    - inadimplencia_vs_selic.png")
print("    - indicadores_bacen_bruto.csv")
print("    - features_bacen.csv")
print("\n🚀 Próximo passo: usar 'features_bacen.csv' no modelo de crédito!")
print("="*60)