import pandas as pd
import warnings
import matplotlib.pyplot as plt
import shap
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from xgboost import XGBRegressor
# Ignorar avisos irrelevantes do pandas
warnings.filterwarnings('ignore')
print("Iniciando Pipeline KDD (Previsão de Prazo de Resolução)...")

# ==========================================
# ETAPA 1 + 2: CARREGAMENTO DO PARQUET
# ==========================================
print("[1/3] Carregando dados pré-processados (Parquet)...")
df_kdd = pd.read_parquet('data/df_ml2.parquet', engine='pyarrow')
print(f"  -> {len(df_kdd):,} registros carregados com sucesso.")

# ==========================================
# ETAPA 3: TRANSFORMAÇÃO (Encoding)
# ==========================================
print("[2/3] Etapa de Transformação (Encoding)...")
encoder_bairro_ml2 = LabelEncoder()
encoder_servico_ml2 = LabelEncoder()

df_kdd['Bairro_ID'] = encoder_bairro_ml2.fit_transform(df_kdd['BAIRRO'])
df_kdd['Servico_ID'] = encoder_servico_ml2.fit_transform(df_kdd['GRUPOSERVICO_DESCRICAO'])

# ==========================================
# ETAPA 4: FEATURE ENGINEERING E DATA MINING
# ==========================================
print("[3/3] Engenharia de Atributos e Treinamento...")

df_kdd['DATA_DEMANDA'] = pd.to_datetime(df_kdd['DATA_DEMANDA'], errors='coerce')
df_kdd = df_kdd.dropna(subset=['DATA_DEMANDA'])

df_kdd['Ano'] = df_kdd['DATA_DEMANDA'].dt.year
df_kdd['Mes'] = df_kdd['DATA_DEMANDA'].dt.month
df_kdd['Dia_do_Mes'] = df_kdd['DATA_DEMANDA'].dt.day
df_kdd['Dia_da_Semana'] = df_kdd['DATA_DEMANDA'].dt.dayofweek
df_kdd['Trimestre'] = df_kdd['DATA_DEMANDA'].dt.quarter
df_kdd['Semana_do_Ano'] = df_kdd['DATA_DEMANDA'].dt.isocalendar().week.astype(int)
df_kdd['Estacao_Chuva'] = df_kdd['Mes'].apply(lambda x: 1 if x in [4, 5, 6, 7, 8] else 0)

# 9 Variáveis Robustas!
X = df_kdd[['Ano', 'Mes', 'Dia_do_Mes', 'Dia_da_Semana', 'Trimestre', 'Semana_do_Ano', 'Estacao_Chuva', 'Bairro_ID', 'Servico_ID']].rename(
    columns={'Dia_da_Semana': 'Dia da Semana', 'Dia_do_Mes': 'Dia do Mês', 'Semana_do_Ano': 'Semana do Ano', 'Estacao_Chuva': 'Temporada de Chuva', 'Bairro_ID': 'Bairro', 'Servico_ID': 'Serviço'}
)
y = df_kdd['Dias_Resolucao']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f" -> Ensinando as IAs com {len(X_train)} registros e {X.shape[1]} variáveis...\n")

# Modelos concorrentes, preservando a calibração original do seu Random Forest
modelos_regressao = {
    'Random Forest Regressor': RandomForestRegressor(
        n_estimators=150, 
        max_depth=15, 
        min_samples_split=15, 
        random_state=42, 
        n_jobs=-1
    ),
    'Árvore de Decisão Regressora': DecisionTreeRegressor(random_state=42),
    'Regressão Linear': LinearRegression(),
    'XGBoost Regressor': XGBRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
}

modelos_treinados = {}

for nome, modelo in modelos_regressao.items():
    modelo.fit(X_train, y_train)
    modelos_treinados[nome] = modelo
    
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print("=" * 60)
    print(f"AVALIAÇÃO REGRESSÃO: {nome.upper()}")
    print("=" * 60)
    print(f"[TREINO - 80%] MAE: {mae_train:.2f} dias | R²: {r2_train:.4f}")
    print(f"[TESTE - 20%]  MAE: {mae_test:.2f} dias | R²: {r2_test:.4f}\n")

# ==========================================
# ETAPA 5: EXPORTAÇÃO DO MODELO VENCEDOR
# ==========================================
print("Salvando o Random Forest Regressor e os encoders...")
modelo_vencedor_ml2 = modelos_treinados['Random Forest Regressor']
joblib.dump(modelo_vencedor_ml2, 'modelo_prazo_ml2.pkl', compress=3) 
joblib.dump(encoder_bairro_ml2, 'encoder_bairro_ml2.pkl', compress=3)
joblib.dump(encoder_servico_ml2, 'encoder_servico_ml2.pkl', compress=3)

# ==========================================
# ETAPA 6: EXPLICABILIDADE CIENTÍFICA COM SHAP (REGRESSÃO)
# ==========================================
print("\n[Iniciando exportação dos gráficos SHAP para o Dashboard...]")
os.makedirs('dashboard/assets', exist_ok=True)

# Amostragem estatística controlada (SHAP) para evitar travamento de CPU
X_shap = X_test.sample(n=250, random_state=42)
X_bg = X_train.sample(n=100, random_state=42)

for nome, modelo in modelos_treinados.items():
    print(f" -> Processando SHAP e exportando imagens para: {nome}...")
    
    if isinstance(modelo, (RandomForestRegressor, DecisionTreeRegressor, XGBRegressor)):
        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer(X_shap)
    elif isinstance(modelo, LinearRegression):
        masker = shap.maskers.Independent(data=X_bg)
        explainer = shap.LinearExplainer(modelo, masker)
        shap_values = explainer(X_shap)

    # Tratamento de string para evitar erros de encode no sistema de arquivos
    nome_slug = nome.lower().replace(" ", "_").replace("á", "a").replace("ã", "a").replace("í", "i").replace("ó", "o")

    # 1. Exportação do Gráfico SHAP de Barras
    plt.figure(figsize=(7, 4))
    shap.plots.bar(shap_values, show=False)
    plt.title(f"SHAP Importância de Atributos: {nome}", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(f'dashboard/assets/ML2/shap_{nome_slug}_bar.png', dpi=150)
    plt.close()

    # 2. Exportação do Gráfico SHAP Enxame de Abelhas (Beeswarm)
    plt.figure(figsize=(7, 4))
    shap.plots.beeswarm(shap_values, show=False)
    plt.title(f"SHAP Enxame de Abelhas (Impacto): {nome}", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(f'dashboard/assets/ML2/shap_{nome_slug}_beeswarm.png', dpi=150)
    plt.close()

print("\nSUCESSO! Todos os gráficos SHAP do ML2 foram consolidados na pasta 'dashboard/assets/ML2'.")