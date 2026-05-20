import pandas as pd
import warnings
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib
import plotly.express as px

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
print("Dados transformados com sucesso! Prontos para a Mineração (Data Mining).")

# ==========================================
# ETAPA 4: DATA MINING E AVALIAÇÃO DETALHADA DE TODOS
# ==========================================
print("[3/3] Etapa de Data Mining e Avaliação (Treinando as IAs)...")
X = df_kdd[['Mes', 'Dia_Semana_Num', 'Bairro_ID', 'Servico_ID']]
y = df_kdd['Alvo_Gargalo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f" -> Ensinando as IAs com {len(X_train)} registros...\n")

modelos = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Árvore de Decisão': DecisionTreeClassifier(random_state=42),
    'Regressão Logística': LogisticRegression(random_state=42, max_iter=500)
}

resultados = []
modelos_treinados = {}

# Iteração automatizada para treinamento e extração de métricas de TODOS os algoritmos
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    modelos_treinados[nome] = modelo 
    
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    
    # Coleta de métricas simples para o Ranking
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    
    resultados.append({
        'Algoritmo': nome,
        'Acurácia (Treino)': round(acc_train * 100, 2),
        'Acurácia (Teste)': round(acc_test * 100, 2),
        'F1-Score (Teste)': round(f1_test * 100, 2)
    })
    
    # =======================================================
    # NOVIDADE: Imprimir Matriz Treino/Teste para CADA modelo
    # =======================================================
    print("=" * 60)
    print(f"AVALIAÇÃO DETALHADA: {nome.upper()}")
    print("=" * 60)
    print(f"\n[DESEMPENHO NO TREINO - 80%]")
    print(f"Acurácia: {acc_train * 100:.2f}%")
    print(classification_report(y_train, y_pred_train, target_names=['Fluxo Normal (0)', 'Gargalo/Atraso (1)']))
    
    print(f"\n[DESEMPENHO NO TESTE - 20%]")
    print(f"Acurácia: {acc_test * 100:.2f}%")
    print(classification_report(y_test, y_pred_test, target_names=['Fluxo Normal (0)', 'Gargalo/Atraso (1)']))
    print("\n")

# ==========================================
# ETAPA 5: SELEÇÃO E EXPORTAÇÃO DO VENCEDOR
# ==========================================
'''modelo_vencedor = modelos_treinados['Random Forest']
vencedor_nome = 'Random Forest'

print("\nSalvando o modelo e os tradutores para o backend do App...")
joblib.dump(modelo_vencedor, 'modelo_report_rf.pkl')
joblib.dump(encoder_bairro, 'encoder_bairro.pkl')
joblib.dump(encoder_servico, 'encoder_servico.pkl')
print("SUCESSO! Ciclo KDD concluído.\n")'''

# ==========================================
# FEATURE IMPORTANCE (Vencedor)
# ==========================================

importancias_modelos = {}
features = ['Mês', 'Dia da Semana', 'Bairro', 'Serviço']

for nome, modelo in modelos.items():
    print(f" -> Treinando e extraindo coeficientes: {nome}...")
    modelo.fit(X_train, y_train)
    
    # Extração de pesos dependendo da natureza do modelo científico
    if hasattr(modelo, 'feature_importances_'):
        pesos = modelo.feature_importances_ * 100
    elif hasattr(modelo, 'coef_'):
        # Regressão Logística: Magnitude absoluta dos coeficientes normalizada para %
        coef_absolutos = np.abs(modelo.coef_[0])
        pesos = (coef_absolutos / np.sum(coef_absolutos)) * 100
    else:
        pesos = np.zeros(len(features))
        
    df_imp = pd.DataFrame({'Atributo': features, 'Importância (%)': pesos})
    importancias_modelos[nome] = df_imp.sort_values(by='Importância (%)', ascending=False)

# ==========================================
# ETAPA 5: GERAÇÃO DOS GRÁFICOS DE FEATURE IMPORTANCE (DOS 3)
# ==========================================
print("\n[5/5] Exibindo Gráficos de Importância de Variáveis...")

# Paleta Okabe-Ito acessível
OKABE_ITO = ['#0072B2', '#E69F00', '#009E73', '#D55E00']

for nome, df_imp in importancias_modelos.items():
    fig = px.bar(
        df_imp, 
        x='Importância (%)', 
        y='Atributo', 
        orientation='h',
        title=f'Análise de Atributos (Caixa Preta): {nome}',
        labels={'Importância (%)': 'Peso/Grau de Influência (%)', 'Atributo': 'Variável'},
        template='plotly_white',
        color='Atributo',
        color_discrete_sequence=OKABE_ITO
    )
    fig.update_traces(
        textposition='outside', 
        texttemplate='<b>%{x:.1f}%</b>', 
        textfont_size=13
    )
    fig.update_layout(
        xaxis_ticksuffix='%', 
        margin=dict(r=80, l=20, t=50, b=20),
        height=380,
        width=850,
        showlegend=False
    )
    # Abre o gráfico interativo no navegador
    fig.show()

print("\nCiclo concluído. Os 3 gráficos de influência foram gerados.")