import pandas as pd
import os
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import plotly.express as px

# Ignorar avisos irrelevantes do pandas
warnings.filterwarnings('ignore')
print("Iniciando Pipeline KDD para o Projeto REPORT!...")

# ==========================================
# ETAPA 1: SELEÇÃO DE DADOS
# ==========================================
print("[1/5] Etapa de Seleção...")
pasta_dados = 'data/'
arquivos_csv = [f for f in os.listdir(pasta_dados) if f.endswith('.csv')]
dfs = []

for arquivo in arquivos_csv:
    caminho = os.path.join(pasta_dados, arquivo)
    df_temp = pd.read_csv(caminho, sep=';', encoding='latin1', on_bad_lines='skip', low_memory=False)
    dfs.append(df_temp)

df_all = pd.concat(dfs, ignore_index=True)

# Seleciona apenas as colunas que têm potencial preditivo para o nosso problema
colunas_alvo = ['DATA_DEMANDA', 'BAIRRO', 'GRUPOSERVICO_DESCRICAO', 'SITUACAO']
df_kdd = df_all[colunas_alvo].copy()

# ==========================================
# ETAPA 2: PRÉ-PROCESSAMENTO E LIMPEZA
# ==========================================
print("[2/5] Etapa de Pré-processamento...")
# Remoção de Nulos Críticos
df_kdd['DATA_DEMANDA'] = pd.to_datetime(df_kdd['DATA_DEMANDA'], format='mixed', errors='coerce')
df_kdd.dropna(subset=['DATA_DEMANDA', 'BAIRRO', 'GRUPOSERVICO_DESCRICAO', 'SITUACAO'], inplace=True)

# Padronização de Texto e Remoção de Ruídos (Regex)
df_kdd['SITUACAO'] = df_kdd['SITUACAO'].astype(str).str.strip().str.upper()
df_kdd['SITUACAO'] = df_kdd['SITUACAO'].str.replace(r'.*EXECU.*', 'EXECUCAO', regex=True)
df_kdd['SITUACAO'] = df_kdd['SITUACAO'].str.replace(r'.*FISCALIZA.*', 'FISCALIZACAO', regex=True)
df_kdd['SITUACAO'] = df_kdd['SITUACAO'].str.replace(r'.*PREPARA.*', 'PREPARACAO', regex=True)
df_kdd['SITUACAO'] = df_kdd['SITUACAO'].str.replace(r'.*PENDEN.*', 'PENDENTE', regex=True)

df_kdd['GRUPOSERVICO_DESCRICAO'] = df_kdd['GRUPOSERVICO_DESCRICAO'].astype(str).str.strip().str.upper()
df_kdd['GRUPOSERVICO_DESCRICAO'] = df_kdd['GRUPOSERVICO_DESCRICAO'].str.replace(r'.*ARBORIZA.*', 'ARBORIZACAO', regex=True)
df_kdd['GRUPOSERVICO_DESCRICAO'] = df_kdd['GRUPOSERVICO_DESCRICAO'].str.replace(r'.*PAVIMENTA.*', 'PAVIMENTACAO', regex=True)
df_kdd['GRUPOSERVICO_DESCRICAO'] = df_kdd['GRUPOSERVICO_DESCRICAO'].str.replace(r'.*ILUMINA.*', 'ILUMINACAO PUBLICA', regex=True)

df_kdd['BAIRRO'] = df_kdd['BAIRRO'].astype(str).str.strip().str.upper()

# ==========================================
# ETAPA 3: TRANSFORMAÇÃO (Feature Engineering)
# ==========================================
print("[3/5] Etapa de Transformação...")
# Extração de Features Temporais (O tempo importa muito para a previsão)
df_kdd['Mes'] = df_kdd['DATA_DEMANDA'].dt.month
df_kdd['Dia_Semana_Num'] = df_kdd['DATA_DEMANDA'].dt.dayofweek # Retorna 0 (Seg) a 6 (Dom)

# Simplificando a Variável Alvo
# Ensinar a IA a prever se um chamado vai dar problema (1) ou se flui bem (0)
status_gargalo = ['PENDENTE', 'PREPARACAO', 'CADASTRADA']
df_kdd['Alvo_Gargalo'] = df_kdd['SITUACAO'].apply(lambda x: 1 if x in status_gargalo else 0)

# Encoding (Traduzindo Bairro e Serviço para Matemática)
encoder_bairro = LabelEncoder()
encoder_servico = LabelEncoder()

df_kdd['Bairro_ID'] = encoder_bairro.fit_transform(df_kdd['BAIRRO'])
df_kdd['Servico_ID'] = encoder_servico.fit_transform(df_kdd['GRUPOSERVICO_DESCRICAO'])
print("Dados transformados com sucesso! Prontos para a Mineração (Data Mining).")

# ==========================================
# ETAPA 4: DATA MINING (Mineração de Dados)
# ==========================================
print("[4/5] Etapa de Data Mining (Treinando a IA)...")
# Definir as "Pistas" (X) e a "Resposta" (y)
X = df_kdd[['Mes', 'Dia_Semana_Num', 'Bairro_ID', 'Servico_ID']]
y = df_kdd['Alvo_Gargalo']

# Divisão de Treino e Teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f" -> Ensinando a IA com {len(X_train)} registros...")

# Inicializando e Treinando o Modelo 
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
modelo_rf.fit(X_train, y_train)

# ==========================================
# ETAPA 5: AVALIAÇÃO E INTERPRETAÇÃ
# ==========================================
print("[5/5] Etapa de Avaliação e Interpretação...")

# Prova do Treino 
y_pred_train = modelo_rf.predict(X_train)
print("\n=== DESEMPENHO NOS DADOS DE TREINO (80%) ===")
print(f"Acurácia: {accuracy_score(y_train, y_pred_train) * 100:.2f}%")
print(classification_report(y_train, y_pred_train, target_names=['Fluxo Normal (0)', 'Gargalo/Atraso (1)']))

# Prova Final 
y_pred_test = modelo_rf.predict(X_test)
print("\n=== DESEMPENHO NOS DADOS DE TESTE (20%) - MUNDO REAL ===")
print(f"Acurácia: {accuracy_score(y_test, y_pred_test) * 100:.2f}%")
print(classification_report(y_test, y_pred_test, target_names=['Fluxo Normal (0)', 'Gargalo/Atraso (1)']))

# 3. Empacotando o "Cérebro" para uso no aplicativo REPORT!
print("\nSalvando o modelo e os tradutores para o backend do App...")
# O joblib salva a IA treinada em um arquivo físico no computador
joblib.dump(modelo_rf, 'modelo_report_rf.pkl')
joblib.dump(encoder_bairro, 'encoder_bairro.pkl')
joblib.dump(encoder_servico, 'encoder_servico.pkl')
print("SUCESSO! Ciclo KDD concluído. O motor preditivo está pronto para produção.")

# ==========================================
# FEATURE IMPORTANCE
# ==========================================
print("\nAnalisando o que causa os gargalos no Recife...")
# Extraindo a importância do classificador
importancias = modelo_rf.feature_importances_
features = ['Mês', 'Dia da Semana', 'Bairro', 'Serviço']

# Criando o DataFrame para o gráfico
df_imp_ml1 = pd.DataFrame({'Atributo': features, 'Importância (%)': importancias * 100})
df_imp_ml1 = df_imp_ml1.sort_values(by='Importância (%)', ascending=True)

fig_imp_ml1 = px.bar(
    df_imp_ml1, 
    x='Importância (%)', 
    y='Atributo', 
    orientation='h',
    title='O que mais influencia a criação de um Gargalo?',
    labels={'Importância (%)': 'Peso na Decisão da IA (%)', 'Atributo': 'Variável'},
    template='plotly_white',
    color='Importância (%)',
    color_continuous_scale='Viridis'
)
fig_imp_ml1.update_traces(
    textposition='outside', 
    texttemplate='<b>%{x:.1f}%</b>', 
    textfont_size=15
)
fig_imp_ml1.update_layout(
    xaxis_ticksuffix='%', 
    margin=dict(r=80, l=20, t=50, b=20),
    height=400,
    width=900
)
fig_imp_ml1.show()