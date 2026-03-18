# =============================================================================
# PROJETO: Modelo Preditivo de Inadimplência
# Objetivo: Prever inadimplência futura com indicadores do BACEN
# Autor: Projeto Portfólio - Analista de Dados com IA | Sicoob
# =============================================================================

# -----------------------------------------------------------------------------
# 1. IMPORTAÇÕES
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os
warnings.filterwarnings("ignore")

from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

plt.style.use('seaborn-v0_8-darkgrid')
print("✅ Bibliotecas carregadas com sucesso!")

# -----------------------------------------------------------------------------
# 2. CARREGAR DADOS
# -----------------------------------------------------------------------------
print("\n📂 Carregando features_bacen.csv...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "features_bacen.csv"), index_col=0, parse_dates=True)

print(f"✅ Dados carregados! Shape: {df.shape}")
print(df.head())

# -----------------------------------------------------------------------------
# 3. DEFINIR TARGET E FEATURES
# -----------------------------------------------------------------------------
TARGET = "Inadimplência (%)"

# Remove colunas derivadas do próprio target para evitar vazamento de dados
colunas_target = [c for c in df.columns if "Inadimplência" in c and c != TARGET]
df_model = df.drop(columns=colunas_target)

FEATURES = [c for c in df_model.columns if c != TARGET]
X = df_model[FEATURES]
y = df_model[TARGET]

print(f"\n🎯 Target  : {TARGET}")
print(f"📊 Features: {len(FEATURES)}")
print(f"📅 Período : {df.index[0].date()} → {df.index[-1].date()}")

# -----------------------------------------------------------------------------
# 4. DIVISÃO TEMPORAL TREINO / TESTE
# -----------------------------------------------------------------------------
# Em séries temporais NUNCA use divisão aleatória — isso causa vazamento!
split_idx = int(len(df_model) * 0.8)

X_train = X.iloc[:split_idx]
X_test  = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]

# Normalização
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n📊 Divisão treino/teste:")
print(f"  Treino : {X_train.index[0].date()} → {X_train.index[-1].date()} ({len(X_train)} meses)")
print(f"  Teste  : {X_test.index[0].date()}  → {X_test.index[-1].date()}  ({len(X_test)} meses)")

# -----------------------------------------------------------------------------
# 5. TREINAMENTO E COMPARAÇÃO DE MODELOS
# -----------------------------------------------------------------------------
print("\n🤖 Treinando modelos...")

modelos = {
    "Regressão Linear":  LinearRegression(),
    "Ridge":             Ridge(alpha=1.0),
    "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    "XGBoost":           XGBRegressor(n_estimators=200, random_state=42, verbosity=0),
}

resultados = []
previsoes  = {}

for nome, modelo in modelos.items():
    modelo.fit(X_train_sc, y_train)
    y_pred = modelo.predict(X_test_sc)
    previsoes[nome] = y_pred

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    resultados.append({"Modelo": nome, "MAE": mae, "RMSE": rmse, "R²": r2})
    print(f"  {nome:<25} MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

df_resultados = pd.DataFrame(resultados).sort_values("R²", ascending=False)
melhor_nome   = df_resultados.iloc[0]["Modelo"]
melhor_modelo = modelos[melhor_nome]
y_pred_best   = previsoes[melhor_nome]

print(f"\n🏆 Melhor modelo: {melhor_nome}")

# -----------------------------------------------------------------------------
# 6. VISUALIZAÇÕES
# -----------------------------------------------------------------------------

# 6.1 — Comparativo de modelos
plt.figure(figsize=(10, 5))
cores_bar = ['#2ECC71' if i == 0 else '#95A5A6' for i in range(len(df_resultados))]
bars = plt.bar(df_resultados["Modelo"], df_resultados["R²"], color=cores_bar)
plt.title("Comparativo de Modelos — R²", fontsize=14, fontweight='bold')
plt.ylabel("R² (quanto maior, melhor)")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
for bar, val in zip(bars, df_resultados["R²"]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{val:.3f}", ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig("modelo_comparativo.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ modelo_comparativo.png salvo")

# 6.2 — Real vs Previsto (melhor modelo)
plt.figure(figsize=(14, 5))
plt.plot(y_test.index, y_test.values,  label="Real",    color="#E74C3C", linewidth=2)
plt.plot(y_test.index, y_pred_best,    label="Previsto", color="#3498DB", linewidth=2, linestyle="--")
plt.fill_between(y_test.index, y_test.values, y_pred_best, alpha=0.15, color="#3498DB")
plt.title(f"Inadimplência: Real vs Previsto — {melhor_nome}", fontsize=14, fontweight='bold')
plt.ylabel("Inadimplência (%)")
plt.xlabel("Data")
plt.legend()
plt.tight_layout()
plt.savefig("modelo_previsao.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ modelo_previsao.png salvo")

# 6.3 — Importância das features (top 15)
if hasattr(melhor_modelo, "feature_importances_"):
    importancias = pd.Series(melhor_modelo.feature_importances_, index=FEATURES)
    top15 = importancias.nlargest(15).sort_values()

    plt.figure(figsize=(10, 6))
    top15.plot(kind="barh", color="#3498DB")
    plt.title(f"Top 15 Features Mais Importantes — {melhor_nome}", fontsize=13, fontweight='bold')
    plt.xlabel("Importância")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ feature_importance.png salvo")

# 6.4 — Todos os modelos: Real vs Previsto
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle("Comparativo Visual — Todos os Modelos", fontsize=15, fontweight='bold')

for ax, (nome, y_pred) in zip(axes.flatten(), previsoes.items()):
    r2 = r2_score(y_test, y_pred)
    ax.plot(y_test.index, y_test.values, label="Real",    color="#E74C3C", linewidth=1.8)
    ax.plot(y_test.index, y_pred,        label="Previsto", color="#3498DB", linewidth=1.8, linestyle="--")
    ax.set_title(f"{nome}  (R²={r2:.3f})", fontsize=11, fontweight='bold')
    ax.set_ylabel("Inadimplência (%)")
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=30)

# Esconde o último subplot se sobrar
if len(previsoes) < 6:
    axes.flatten()[-1].set_visible(False)

plt.tight_layout()
plt.savefig("todos_modelos.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ todos_modelos.png salvo")

# -----------------------------------------------------------------------------
# 7. SALVAR MODELO
# -----------------------------------------------------------------------------
joblib.dump(melhor_modelo, "modelo_inadimplencia.pkl")
joblib.dump(scaler,        "scaler.pkl")
joblib.dump(FEATURES,      "features.pkl")

print("\n💾 Modelo salvo: modelo_inadimplencia.pkl")
print("💾 Scaler salvo: scaler.pkl")
print("💾 Features     : features.pkl")

# -----------------------------------------------------------------------------
# 8. RESUMO FINAL
# -----------------------------------------------------------------------------
melhor = df_resultados.iloc[0]
print("\n" + "="*60)
print("✅ MODELO TREINADO COM SUCESSO!")
print("="*60)
print(f"  Melhor modelo : {melhor['Modelo']}")
print(f"  MAE           : {melhor['MAE']:.4f} pontos percentuais")
print(f"  RMSE          : {melhor['RMSE']:.4f}")
print(f"  R²            : {melhor['R²']:.4f}")
print(f"  Features      : {len(FEATURES)}")
print(f"\n  Arquivos gerados:")
print("    - modelo_comparativo.png")
print("    - modelo_previsao.png")
print("    - feature_importance.png")
print("    - todos_modelos.png")
print("    - modelo_inadimplencia.pkl")
print("    - scaler.pkl")
print("    - features.pkl")
print("="*60)
print("\n🚀 Próximo passo: agente de IA com LangChain!")