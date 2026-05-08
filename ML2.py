import pandas as pd
import os
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import plotly.express as px

# Ignorar avisos irrelevantes do pandas
warnings.filterwarnings('ignore')
print("Iniciando Pipeline KDD (Previsão de Prazo de Resolução)...")
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
colunas_alvo = ['DATA_DEMANDA', 'DATA_ULT_SITUACAO', 'BAIRRO', 'GRUPOSERVICO_DESCRICAO', 'SITUACAO']
df_kdd = df_all[colunas_alvo].copy()

# ==========================================
# ETAPA 2: PRÉ-PROCESSAMENTO E LIMPEZA
# ==========================================
print("[2/5] Etapa de Pré-processamento...")
# Tratamento de Datas
df_kdd['DATA_DEMANDA'] = pd.to_datetime(df_kdd['DATA_DEMANDA'], format='mixed', errors='coerce')
df_kdd['DATA_ULT_SITUACAO'] = pd.to_datetime(df_kdd['DATA_ULT_SITUACAO'], format='mixed', errors='coerce')
df_kdd.dropna(subset=['DATA_DEMANDA', 'DATA_ULT_SITUACAO', 'BAIRRO', 'GRUPOSERVICO_DESCRICAO'], inplace=True)

# Padronização de Texto
df_kdd['SITUACAO'] = df_kdd['SITUACAO'].astype(str).str.strip().str.upper()
df_kdd['BAIRRO'] = df_kdd['BAIRRO'].astype(str).str.strip().str.upper()
df_kdd['GRUPOSERVICO_DESCRICAO'] = df_kdd['GRUPOSERVICO_DESCRICAO'].astype(str).str.strip().str.upper()
df_kdd['GRUPOSERVICO_DESCRICAO'] = df_kdd['GRUPOSERVICO_DESCRICAO'].str.replace(r'.*ARBORIZA.*', 'ARBORIZACAO', regex=True)
df_kdd['GRUPOSERVICO_DESCRICAO'] = df_kdd['GRUPOSERVICO_DESCRICAO'].str.replace(r'.*PAVIMENTA.*', 'PAVIMENTACAO', regex=True)
df_kdd['GRUPOSERVICO_DESCRICAO'] = df_kdd['GRUPOSERVICO_DESCRICAO'].str.replace(r'.*ILUMINA.*', 'ILUMINACAO PUBLICA', regex=True)

# Só manter os casos "ATENDIDA" para a IA aprender quanto tempo leva um serviço concluído
df_kdd = df_kdd[df_kdd['SITUACAO'] == 'ATENDIDA']

# ==========================================
# ETAPA 3: TRANSFORMAÇÃO (Feature Engineering)
# ==========================================
print("[3/5] Etapa de Transformação (Ajuste Fino)...")
# Criando a Variável Alvo
df_kdd['Dias_Resolucao'] = (df_kdd['DATA_ULT_SITUACAO'] - df_kdd['DATA_DEMANDA']).dt.days

# CAÇA AOS OUTLIERS 
limite_dias = df_kdd['Dias_Resolucao'].quantile(0.90) 
df_kdd = df_kdd[(df_kdd['Dias_Resolucao'] >= 0) & (df_kdd['Dias_Resolucao'] <= limite_dias)]

print(f" -> Filtro aplicado: Focando em problemas resolvidos em até {limite_dias:.0f} dias.")

# Extraindo Variáveis de Tempo
df_kdd['Mes'] = df_kdd['DATA_DEMANDA'].dt.month
df_kdd['Ano'] = df_kdd['DATA_DEMANDA'].dt.year 
df_kdd['Dia_Semana'] = df_kdd['DATA_DEMANDA'].dt.dayofweek 

# Encoding
encoder_bairro_ml2 = LabelEncoder()
encoder_servico_ml2 = LabelEncoder()
df_kdd['Bairro_ID'] = encoder_bairro_ml2.fit_transform(df_kdd['BAIRRO'])
df_kdd['Servico_ID'] = encoder_servico_ml2.fit_transform(df_kdd['GRUPOSERVICO_DESCRICAO'])

# ==========================================
# ETAPA 4: DATA MINING (Random Forest Calibrado)
# ==========================================
print("[4/5] Etapa de Data Mining (Treinando a IA com amarras)...")
X = df_kdd[['Mes', 'Ano', 'Dia_Semana', 'Bairro_ID', 'Servico_ID']]
y = df_kdd['Dias_Resolucao']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TREINAMENTO COM HIPERPARÂMETROS
# OBS: Se atentem a esses parametros, não modifiquem sem entender o motivo.
# max_depth=15: Impede que a IA crie árvores infinitas e decore o passado.
# min_samples_split=15: A IA só cria uma regra se achar pelo menos 15 casos parecidos.
modelo_regressao = RandomForestRegressor(
    n_estimators=150, 
    max_depth=15, 
    min_samples_split=15, 
    random_state=42, 
    n_jobs=-1
)
modelo_regressao.fit(X_train, y_train)
# ==========================================
# ETAPA 5: AVALIAÇÃO E INTERPRETAÇÃO
# ==========================================
print("[5/5] Etapa de Avaliação e Interpretação...")
y_pred = modelo_regressao.predict(X_test)

# Avaliação com MAE (Erro Médio Absoluto)
erro_medio = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMARGEM DE ERRO DO MODELO: +/- {erro_medio:.2f} dias")
print(f"Isso significa que a previsão que o funcionário verá costuma errar para mais ou para menos por apenas {erro_medio:.1f} dias na média.")

# Salvando o modelo e os encoders do ML2
joblib.dump(modelo_regressao, 'modelo_prazo_ml2.pkl')
joblib.dump(encoder_bairro_ml2, 'encoder_bairro_ml2.pkl')
joblib.dump(encoder_servico_ml2, 'encoder_servico_ml2.pkl')
print("\nSUCESSO! ML2 Concluído. Motor de Estimativa de Prazos pronto.")

# ==========================================
# FEATURE IMPORTANCE
# ==========================================
print("\nGerando gráfico de Feature Importance...")
# Extraindo os pesos matemáticos do algoritmo
pesos = modelo_regressao.feature_importances_
colunas = ['Mes', 'Ano', 'Dia_Semana', 'Bairro', 'Serviço']

# Criando o DataFrame
df_importancia = pd.DataFrame({'Variavel': colunas, 'Peso_Porcentagem': pesos * 100})
df_importancia = df_importancia.sort_values(by='Peso_Porcentagem', ascending=True)

# Gerando o Gráfico
fig_importancia = px.bar(
    df_importancia, 
    x='Peso_Porcentagem', 
    y='Variavel', 
    orientation='h', 
    title='Abertura da Caixa Preta: O que dita o tempo de uma obra?',
    labels={
        'Peso_Porcentagem': 'Grau de Influência (%)', 
        'Variavel': 'Características (Features)'
    },
    template='plotly_white',
    color='Peso_Porcentagem',
    color_continuous_scale='Teal' 
)
fig_importancia.update_traces(
    textposition='outside', 
    texttemplate='<b>%{x:.1f}%</b>', 
    textfont_size=15
)
fig_importancia.update_layout(
    xaxis_ticksuffix='%',
    margin=dict(r=80, l=20, t=50, b=20),
    height=400,  
    width=900
)
fig_importancia.show()