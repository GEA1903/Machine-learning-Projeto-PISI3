import pandas as pd
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
# ETAPA 1 + 2 + 3 (parcial): CARREGAMENTO DO PARQUET
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
# ENCODING (parte da antiga Etapa 3)
# ==========================================
# Mantém aqui porque os encoders fazem parte do modelo:
# precisam ser treinados nos mesmos dados e salvos para produção.
print("[2/3] Etapa de Encoding...")
encoder_bairro_ml2 = LabelEncoder()
encoder_servico_ml2 = LabelEncoder()
df_kdd['Bairro_ID'] = encoder_bairro_ml2.fit_transform(df_kdd['BAIRRO'])
df_kdd['Servico_ID'] = encoder_servico_ml2.fit_transform(df_kdd['GRUPOSERVICO_DESCRICAO'])

# ==========================================
# ETAPA 4: DATA MINING (Random Forest Calibrado)
# ==========================================
print("[3/3] Etapa de Data Mining (Treinando a IA com amarras)...")
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
df_importancia = df_importancia.sort_values(by='Peso_Porcentagem', ascending=False)

# Paleta Okabe-Ito: criada especificamente para daltonismo
# Cada barra recebe uma cor única e completamente distinta
OKABE_ITO = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7']

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
    showlegend=False                           # Esconde a legenda (redundante com os rótulos do eixo Y)
)
fig_importancia.show()