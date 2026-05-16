import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
# Modelos concorrentes inseridos para estruturar a validação cruzada do erro preditivo
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import plotly.express as px

# Ignorar avisos irrelevantes do pandas
warnings.filterwarnings('ignore')
print("Iniciando Pipeline KDD (Previsão de Prazo de Resolução)...")

# ==========================================
# ETAPA 1 + 2: CARREGAMENTO DO PARQUET
# ==========================================
# O arquivo df_ml2.parquet já contém:
#   - Dados limpos (datas convertidas, nulos removidos, texto padronizado)
#   - Apenas chamados com SITUACAO == 'ATENDIDA'
#   - Outliers de prazo já filtrados (acima do percentil 90 removidos)
#   - Features temporais já calculadas (Mes, Ano, Dia_Semana)
# Basta carregar e ir direto ao Encoding e ao Treinamento.
print("[1/3] Carregando dados pré-processados (Parquet)...")
df_kdd = pd.read_parquet('data/df_ml2.parquet', engine='pyarrow')
print(f"  -> {len(df_kdd):,} registros carregados com sucesso.")

# ==========================================
# ETAPA 3: TRANSFORMAÇÃO (Encoding)
# ==========================================
# O LabelEncoder converte texto em números para os algoritmos processarem.
# Precisam ser treinados nos mesmos dados e salvos para produção.
print("[2/3] Etapa de Transformação (Encoding)...")
encoder_bairro_ml2 = LabelEncoder()
encoder_servico_ml2 = LabelEncoder()

df_kdd['Bairro_ID'] = encoder_bairro_ml2.fit_transform(df_kdd['BAIRRO'])
df_kdd['Servico_ID'] = encoder_servico_ml2.fit_transform(df_kdd['GRUPOSERVICO_DESCRICAO'])

# ==========================================
# ETAPA 4: DATA MINING (Mineração de Dados e Competição de Modelos)
# ==========================================
print("[3/3] Etapa de Data Mining (Treinando os modelos de Regressão)...")
X = df_kdd[['Mes', 'Ano', 'Dia_Semana_Num', 'Bairro_ID', 'Servico_ID']]
y = df_kdd['Prazo_Resolucao_Dias']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DIRETRIZ CIENTÍFICA (PP3): AVALIAÇÃO COMPARATIVA BASEADA NO ERRO MÉDIO (MAE)
# Para estabelecer uma previsão de prazos confiável no REPORT!, estruturou-se 
# uma árvore de decisão regressora e um modelo linear contra a Random Forest. 
# O critério de otimização prioriza o Menor Erro Médio Absoluto (MAE), indicando
# a margem real de desvio em dias que a aplicação apresentará ao cidadão.
modelos_regressao = {
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Árvore de Decisão Regressora': DecisionTreeRegressor(random_state=42),
    'Regressão Linear': LinearRegression()
}

resultados_mae = []
modelos_treinados = {}

# Loop de treinamento e mensuração do desvio absoluto
for nome, modelo in modelos_regressao.items():
    modelo.fit(X_train, y_train)
    modelos_treinados[nome] = modelo
    
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    resultados_mae.append({
        'Algoritmo': nome,
        'MAE (Treino - Dias)': round(mae_train, 2),
        'MAE (Teste - Dias)': round(mae_test, 2)
    })

# ==========================================
# ETAPA 5: AVALIAÇÃO E INTERPRETAÇÃO DOS RESULTADOS
# ==========================================
print("\n[5/5] Etapa de Avaliação e Interpretação...")

# Ordenação da matriz de erros: algoritmos com menores taxas de erro ocupam o topo
df_comparacao_ml2 = pd.DataFrame(resultados_mae).sort_values(by='MAE (Teste - Dias)')
print("\n=== TABELA DE COMPARAÇÃO DE ALGORITMOS (ML2) ===")
print(df_comparacao_ml2.to_markdown(index=False))

vencedor_ml2_nome = df_comparacao_ml2.iloc[0]['Algoritmo']
modelo_vencedor_ml2 = modelos_treinados[vencedor_ml2_nome]
print(f"\n🏆 O modelo vencedor foi: {vencedor_ml2_nome} (Menor Erro Médio Absoluto).")

# Mantendo as métricas clássicas originais para o vencedor
print("\n=== AVALIAÇÃO DETALHADA DO VENCEDOR ===")
y_pred_test_vencedor = modelo_vencedor_ml2.predict(X_test)
print(f"Erro Médio Absoluto (Mundo Real): Erramos o prazo em +/- {mean_absolute_error(y_test, y_pred_test_vencedor):.2f} dias")
print(f"R² Score (Explicabilidade): {r2_score(y_test, y_pred_test_vencedor):.4f}")

# Salvando a instância serializada
print("\nSalvando o modelo vencedor para produção...")
joblib.dump(modelo_vencedor_ml2, 'modelo_report_regressor.pkl')

# ==========================================
# FEATURE IMPORTANCE (Abertura da Caixa Preta)
# ==========================================
print("\nGerando gráfico de importância das variáveis...")

# Salvaguarda algorítmica: Regressões lineares não computam relevância por 'feature_importances_'. 
# Se houver alternância no modelo vencedor, o estimador baseado em florestas de decisão é isolado 
# exclusivamente para a extração do gráfico estrutural de pesos.
if hasattr(modelo_vencedor_ml2, 'feature_importances_'):
    pesos = modelo_vencedor_ml2.feature_importances_
else:
    pesos = modelos_treinados['Random Forest Regressor'].feature_importances_

colunas = ['Mes', 'Ano', 'Dia_Semana', 'Bairro', 'Serviço']

# Criando o DataFrame
df_importancia = pd.DataFrame({'Variavel': colunas, 'Peso_Porcentagem': pesos * 100})
df_importancia = df_importancia.sort_values(by='Peso_Porcentagem', ascending=False)

# Paleta Okabe-Ito: criada especificamente para daltonismo
OKABE_ITO = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7']

# Gerando o Gráfico original
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
    color='Variavel',                          # Cor por categoria (cada barra = cor única)
    color_discrete_sequence=OKABE_ITO          # Paleta Okabe-Ito segura para daltônicos
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
    width=900,
    showlegend=False
)
fig_importancia.show()