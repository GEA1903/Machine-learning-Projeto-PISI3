import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Carregar o dataset da EMLURB 
df = pd.read_csv('emlurb_156_servicos.csv', sep=';', encoding='utf-8')


# No dataset da EMLURB, as colunas importantes costumam ser:
# 'servico', 'bairro', 'data_demanda', 'hora_demanda'

# Convertendo para datetime para a IA entender o tempo
df['data_demanda'] = pd.to_datetime(df['data_demanda'])
df['mes'] = df['data_demanda'].dt.month
df['dia_semana'] = df['data_demanda'].dt.dayofweek
df['hora'] = pd.to_datetime(df['hora_demanda']).dt.hour


# Vamos ensinar a IA a prever o 'servico' baseado no 'bairro' e 'tempo'
# Dica: Como 'bairro' é texto, precisamos converter para números (LabelEncoding)
from sklearn.preprocessing import LabelEncoder
le_bairro = LabelEncoder()
df['bairro_id'] = le_bairro.fit_transform(df['bairro'].astype(str))

# Definindo entrada (X) e saída (y)
X = df[['bairro_id', 'mes', 'dia_semana', 'hora']]
y = df['servico']

# Treinando o modelo
modelo_recife = RandomForestClassifier(n_estimators=100)
modelo_recife.fit(X, y)

# Salvar modelo e o conversor de bairros
joblib.dump(modelo_recife, 'ia_emlurb_recife.pkl')
joblib.dump(le_bairro, 'conversor_bairros.pkl')

print("IA treinada com os dados reais do Recife!")