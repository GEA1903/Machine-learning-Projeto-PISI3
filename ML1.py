import pandas as pd
import warnings
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from xgboost import XGBClassifier

# Ignorar avisos irrelevantes do pandas
warnings.filterwarnings('ignore')
print("Iniciando Pipeline KDD para o Projeto REPORT!...")

# ==========================================
# ETAPA 1 + 2: CARREGAMENTO DO PARQUET
# ==========================================
print("[1/3] Carregando dados pré-processados (Parquet)...")
df_kdd = pd.read_parquet('data/df_ml1.parquet', engine='pyarrow')
print(f"  -> {len(df_kdd):,} registros carregados com sucesso.")

# ==========================================
# ETAPA 3: TRANSFORMAÇÃO (Encoding)
# ==========================================
print("[2/3] Etapa de Transformação (Encoding)...")
encoder_bairro = LabelEncoder()
encoder_servico = LabelEncoder()

df_kdd['Bairro_ID'] = encoder_bairro.fit_transform(df_kdd['BAIRRO'])
df_kdd['Servico_ID'] = encoder_servico.fit_transform(df_kdd['GRUPOSERVICO_DESCRICAO'])

# ==========================================
# ETAPA 4: DATA MINING E AVALIAÇÃO DETALHADA
# ==========================================
print("[3/3] Etapa de Data Mining (Treinando as IAs)...")

# Renomeando as colunas para que o SHAP as leia com a formatação bonita nos gráficos
X = df_kdd[['Mes', 'Dia_Semana_Num', 'Bairro_ID', 'Servico_ID']].rename(
    columns={'Dia_Semana_Num': 'Dia da Semana', 'Bairro_ID': 'Bairro', 'Servico_ID': 'Serviço'}
)
y = df_kdd['Alvo_Gargalo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f" -> Ensinando as IAs com {len(X_train)} registros...\n")

modelos = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Árvore de Decisão': DecisionTreeClassifier(random_state=42),
    'Regressão Logística': LogisticRegression(random_state=42, max_iter=500),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', n_jobs=-1)
}
modelos_treinados = {}

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    modelos_treinados[nome] = modelo 
    
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    print("=" * 60)
    print(f"AVALIAÇÃO DETALHADA: {nome.upper()}")
    print("=" * 60)
    print(f"\n[DESEMPENHO NO TREINO - 80%]")
    print(f"Acurácia: {acc_train * 100:.2f}%")
    print(classification_report(y_train, y_pred_train, target_names=['Fluxo Normal (0)', 'Gargalo (1)']))
    
    print(f"\n[DESEMPENHO NO TESTE - 20%]")
    print(f"Acurácia: {acc_test * 100:.2f}%")
    print(classification_report(y_test, y_pred_test, target_names=['Fluxo Normal (0)', 'Gargalo (1)']))
    print("\n")

# ==========================================
# ETAPA 5: EXPORTAÇÃO DO XGBOOST PARA PRODUÇÃO
# ==========================================
print("Salvando o XGBoost e os tradutores para o backend do App...")
modelo_xgb = modelos_treinados['XGBoost']
joblib.dump(modelo_xgb, 'modelo_report_xgb.pkl')
joblib.dump(encoder_bairro, 'encoder_bairro_ml1.pkl')
joblib.dump(encoder_servico, 'encoder_servico_ml1.pkl')

# ==========================================
# ETAPA 6: EXPLICABILIDADE CIENTÍFICA COM SHAP (OS 3 ALGORITMOS)
# ==========================================
print("\n[Iniciando exportação dos gráficos SHAP para o Dashboard...]")
import os

os.makedirs('dashboard/assets', exist_ok=True)

X_shap = X_test.sample(n=250, random_state=42)
X_bg = X_train.sample(n=100, random_state=42) 

for nome, modelo in modelos_treinados.items():
    print(f" -> Processando SHAP e exportando imagens para: {nome}...")
    
    if isinstance(modelo, (RandomForestClassifier, DecisionTreeClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(modelo)
        shap_values_raw = explainer(X_shap)
        shap_values_raw = explainer(X_shap)
        if len(shap_values_raw.shape) == 3:
            shap_values = shap_values_raw[:, :, 1]
        else:
            shap_values = shap_values_raw
            
    elif isinstance(modelo, LogisticRegression):
        masker = shap.maskers.Independent(data=X_bg)
        explainer = shap.LinearExplainer(modelo, masker)
        shap_values = explainer(X_shap)

    # 2. Remove acentos e espaços para evitar bugs nos URLs de imagens
    nome_slug = nome.lower().replace(" ", "_").replace("á", "a").replace("ã", "a").replace("í", "i").replace("ó", "o")

    # Gráfico SHAP de Barras
    plt.figure(figsize=(7, 4))
    shap.plots.bar(shap_values, show=False)
    plt.title(f"SHAP Importância de Atributos: {nome}", fontsize=12, pad=15)
    plt.tight_layout()
    # Salva diretamente na subpasta do dashboard
    plt.savefig(f'dashboard/assets/ML1/shap_{nome_slug}_bar.png', dpi=150)
    plt.close() 

    # Gráfico SHAP Enxame de Abelhas
    plt.figure(figsize=(7, 4))
    shap.plots.beeswarm(shap_values, show=False)
    plt.title(f"SHAP Enxame de Abelhas (Impacto): {nome}", fontsize=12, pad=15)
    plt.tight_layout()
    # Salva diretamente na subpasta do dashboard
    plt.savefig(f'dashboard/assets/ML1/shap_{nome_slug}_beeswarm.png', dpi=150)
    plt.close()

print("\nSUCESSO! Todos os gráficos SHAP foram consolidados na pasta 'dashboard/assets/ML1'.")