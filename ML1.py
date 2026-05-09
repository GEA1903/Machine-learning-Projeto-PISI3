import pandas as pd
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
# ETAPA 1 + 2: CARREGAMENTO DO PARQUET
# ==========================================
# O Parquet já contém os dados limpos e pré-processados.
# Rodamos o preprocessamento.py uma única vez para gerar esse arquivo;
# agora apenas o lemos — muito mais rápido que reprocessar os CSVs.
print("[1/3] Carregando dados pré-processados (Parquet)...")
df_kdd = pd.read_parquet('data/df_ml1.parquet', engine='pyarrow')
print(f"  -> {len(df_kdd):,} registros carregados com sucesso.")

# ==========================================
# ETAPA 3: TRANSFORMAÇÃO (Encoding)
# ==========================================
# O LabelEncoder converte texto em números (ex: 'BOA VIAGEM' → 42).
# Ele precisa ser ajustado (fit) aqui, nos mesmos dados do treino,
# e salvo com joblib para que o app use a mesma "tabela de tradução".
print("[2/3] Etapa de Transformação (Encoding)...")
encoder_bairro = LabelEncoder()
encoder_servico = LabelEncoder()

df_kdd['Bairro_ID'] = encoder_bairro.fit_transform(df_kdd['BAIRRO'])
df_kdd['Servico_ID'] = encoder_servico.fit_transform(df_kdd['GRUPOSERVICO_DESCRICAO'])
print("Dados transformados com sucesso! Prontos para a Mineração (Data Mining).")

# ==========================================
# ETAPA 4: DATA MINING (Mineração de Dados)
# ==========================================
print("[3/3] Etapa de Data Mining e Avaliação (Treinando a IA)...")
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
df_imp_ml1 = df_imp_ml1.sort_values(by='Importância (%)', ascending=False)

# Paleta Okabe-Ito: criada especificamente para daltonismo
# Cada barra recebe uma cor única e completamente distinta
OKABE_ITO = ['#0072B2', '#E69F00', '#009E73', '#D55E00']

fig_imp_ml1 = px.bar(
    df_imp_ml1, 
    x='Importância (%)', 
    y='Atributo', 
    orientation='h',
    title='O que mais influencia a criação de um Gargalo?',
    labels={'Importância (%)': 'Peso na Decisão da IA (%)', 'Atributo': 'Variável'},
    template='plotly_white',
    color='Atributo',                         # Cor por categoria (cada barra = cor única)
    color_discrete_sequence=OKABE_ITO         # Paleta Okabe-Ito segura para daltônicos
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
    width=900,
    showlegend=False                           # Esconde a legenda (redundante com os rótulos do eixo Y)
)
fig_imp_ml1.show()