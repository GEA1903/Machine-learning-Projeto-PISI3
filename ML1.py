import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Algoritmos concorrentes importados para viabilizar a análise comparativa de performance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib
import plotly.express as px

# Ignorar avisos irrelevantes do pandas para assegurar a clareza dos logs no terminal
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
# ETAPA 4: DATA MINING (Mineração de Dados e Análise Comparativa)
# ==========================================
print("[3/3] Etapa de Data Mining e Avaliação (Treinando as IAs)...")
# Definir as "Pistas" (X) e a "Resposta" (y)
X = df_kdd[['Mes', 'Dia_Semana_Num', 'Bairro_ID', 'Servico_ID']]
y = df_kdd['Alvo_Gargalo']

# Divisão de Treino e Teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f" -> Ensinando as IAs com {len(X_train)} registros...")

# ============================================================================
# DIRETRIZ CIENTÍFICA (PP2): COMPETIÇÃO MULTIALGORÍTMICA DE CLASSIFICAÇÃO
# Para mitigar vieses de seleção e conferir robustez metodológica à triagem de
# urgências, estabeleceu-se um ambiente competitivo entre três famílias distintas
# de classificadores. A avaliação final pauta-se no F1-Score Ponderado, métrica 
# estatística ideal para conjuntos de dados que apresentam desbalanceamento natural.
# ============================================================================
modelos = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Árvore de Decisão': DecisionTreeClassifier(random_state=42),
    'Regressão Logística': LogisticRegression(random_state=42, max_iter=500)
}

resultados = []
modelos_treinados = {}

# Iteração automatizada para treinamento, predição e extração de métricas de generalização
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    modelos_treinados[nome] = modelo 
    
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    
    resultados.append({
        'Algoritmo': nome,
        'Acurácia (Treino)': round(acc_train * 100, 2),
        'Acurácia (Teste)': round(acc_test * 100, 2),
        'F1-Score (Teste)': round(f1_test * 100, 2)
    })

# Consolidação da matriz de desempenho para fins de auditoria e inserção no artigo científico
df_comparacao = pd.DataFrame(resultados).sort_values(by='F1-Score (Teste)', ascending=False)
print("\n=== TABELA DE COMPARAÇÃO DE ALGORITMOS ===")
print(df_comparacao.to_markdown(index=False))

# Seleção programática do modelo ótimo para exportação e integração com o backend
vencedor_nome = df_comparacao.iloc[0]['Algoritmo']
modelo_vencedor = modelos_treinados[vencedor_nome]
print(f"\n🏆 O modelo escolhido foi: {vencedor_nome} (Melhor F1-Score)")

# ==========================================
# ETAPA 5: AVALIAÇÃO E INTERPRETAÇÃO DO VENCEDOR
# ==========================================
print("\n[5/5] Etapa de Avaliação Detalhada do Modelo Vencedor...")

# Prova do Treino 
y_pred_train_vencedor = modelo_vencedor.predict(X_train)
print("\n=== DESEMPENHO NOS DADOS DE TREINO (80%) ===")
print(f"Acurácia: {accuracy_score(y_train, y_pred_train_vencedor) * 100:.2f}%")
print(classification_report(y_train, y_pred_train_vencedor, target_names=['Fluxo Normal (0)', 'Gargalo/Atraso (1)']))

# Prova Final 
y_pred_test_vencedor = modelo_vencedor.predict(X_test)
print("\n=== DESEMPENHO NOS DADOS DE TESTE (20%) - MUNDO REAL ===")
print(f"Acurácia: {accuracy_score(y_test, y_pred_test_vencedor) * 100:.2f}%")
print(classification_report(y_test, y_pred_test_vencedor, target_names=['Fluxo Normal (0)', 'Gargalo/Atraso (1)']))

# 3. Empacotando o "Cérebro" para uso no aplicativo REPORT!
print("\nSalvando o modelo e os tradutores para o backend do App...")
# O joblib salva a IA treinada em um arquivo físico no computador
joblib.dump(modelo_vencedor, 'modelo_report_rf.pkl')
joblib.dump(encoder_bairro, 'encoder_bairro.pkl')
joblib.dump(encoder_servico, 'encoder_servico.pkl')
print("SUCESSO! Ciclo KDD concluído. O motor preditivo está pronto para produção.")

# ==========================================
# FEATURE IMPORTANCE
# ==========================================
print("\nAnalisando o que causa os gargalos no Recife...")

# Contingência arquitetural: classificadores baseados em distância ou regressões lineares puras 
# não possuem o atributo 'feature_importances_'. Caso o vencedor mude, isola-se o Random Forest 
# para garantir a geração contínua do gráfico de importância relativa.
if hasattr(modelo_vencedor, 'feature_importances_'):
    importancias = modelo_vencedor.feature_importances_
else:
    importancias = modelos_treinados['Random Forest'].feature_importances_

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