"""
preprocessamento.py
--------------------
PROPÓSITO: Ler os CSVs brutos UMA única vez, limpar os dados e salvar
           o resultado em arquivos .parquet eficientes.

QUANDO RODAR: Apenas na primeira vez, ou quando os arquivos CSV mudarem.
              Depois disso, o ML1.py e ML2.py carregam os .parquet
              diretamente — muito mais rápido.

SAÍDAS:
  data/df_ml1.parquet  →  dados prontos para o classificador (ML1)
  data/df_ml2.parquet  →  dados prontos para o regressor    (ML2)
"""

import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================
def padronizar_texto(serie: pd.Series) -> pd.Series:
    """
    Padroniza uma coluna de texto: remove espaços extras, converte
    para maiúsculas e normaliza as categorias de serviço mais comuns.
    Centraliza a lógica que antes estava duplicada para df_ml1 e df_ml2.
    """
    serie = serie.astype(str).str.strip().str.upper()
    serie = serie.str.replace(r'.*ARBORIZA.*',  'ARBORIZACAO',       regex=True)
    serie = serie.str.replace(r'.*PAVIMENTA.*', 'PAVIMENTACAO',      regex=True)
    serie = serie.str.replace(r'.*ILUMINA.*',   'ILUMINACAO PUBLICA', regex=True)
    return serie

# ==========================================
# ETAPA 1: SELEÇÃO — lendo os CSVs brutos
# ==========================================
# Por que isso é lento? Porque CSV é texto puro:
# o pandas precisa ler caractere por caractere, inferir tipos
# de cada coluna e decodificar o encoding latin1.
print("[1/3] Lendo arquivos CSV brutos...")
pasta_dados = 'data/'
arquivos_csv = [f for f in os.listdir(pasta_dados) if f.endswith('.csv')]

dfs = []
for arquivo in arquivos_csv:
    caminho = os.path.join(pasta_dados, arquivo)
    df_temp = pd.read_csv(
        caminho,
        sep=';',
        encoding='latin1',
        on_bad_lines='skip',
        low_memory=False
    )
    dfs.append(df_temp)
    print(f"  -> Lido: {arquivo} ({len(df_temp):,} linhas)")

df_all = pd.concat(dfs, ignore_index=True)
print(f"  -> Total unificado: {len(df_all):,} linhas\n")

# ==========================================
# ETAPA 2A: PRÉ-PROCESSAMENTO PARA O ML1
# (Classificação de Gargalos)
# ==========================================
print("[2/3] Preparando dados para o ML1 (Classificador de Gargalo)...")

colunas_ml1 = ['DATA_DEMANDA', 'BAIRRO', 'GRUPOSERVICO_DESCRICAO', 'SITUACAO']
df_ml1 = df_all[colunas_ml1].copy()

# Conversão de datas e remoção de nulos críticos
df_ml1['DATA_DEMANDA'] = pd.to_datetime(df_ml1['DATA_DEMANDA'], format='mixed', errors='coerce')
df_ml1.dropna(subset=['DATA_DEMANDA', 'BAIRRO', 'GRUPOSERVICO_DESCRICAO', 'SITUACAO'], inplace=True)

# Padronização de texto com Regex
df_ml1['SITUACAO'] = df_ml1['SITUACAO'].astype(str).str.strip().str.upper()
df_ml1['SITUACAO'] = df_ml1['SITUACAO'].str.replace(r'.*EXECU.*',    'EXECUCAO',     regex=True)
df_ml1['SITUACAO'] = df_ml1['SITUACAO'].str.replace(r'.*FISCALIZA.*','FISCALIZACAO', regex=True)
df_ml1['SITUACAO'] = df_ml1['SITUACAO'].str.replace(r'.*PREPARA.*',  'PREPARACAO',   regex=True)
df_ml1['SITUACAO'] = df_ml1['SITUACAO'].str.replace(r'.*PENDEN.*',   'PENDENTE',     regex=True)

# Reutilizando a função auxiliar — evita duplicação de lógica com o df_ml2
df_ml1['GRUPOSERVICO_DESCRICAO'] = padronizar_texto(df_ml1['GRUPOSERVICO_DESCRICAO'])
df_ml1['BAIRRO'] = df_ml1['BAIRRO'].astype(str).str.strip().str.upper()

# Feature Engineering (fazemos aqui para não repetir em cada run do ML1)
df_ml1['Mes']           = df_ml1['DATA_DEMANDA'].dt.month
df_ml1['Dia_Semana_Num']= df_ml1['DATA_DEMANDA'].dt.dayofweek

# Variável alvo binária: 1 = gargalo, 0 = fluxo normal
status_gargalo = ['PENDENTE', 'PREPARACAO', 'CADASTRADA']
df_ml1['Alvo_Gargalo'] = df_ml1['SITUACAO'].apply(lambda x: 1 if x in status_gargalo else 0)

# Salvando em Parquet
# Por que Parquet é rápido? Porque armazena dados em formato BINÁRIO e COLUNAR:
# - Binário = sem parsing de texto, leitura direta pela CPU
# - Colunar = se você pede só a coluna 'Bairro_ID', ele lê apenas ela no disco
saida_ml1 = os.path.join(pasta_dados, 'df_ml1.parquet')
df_ml1.to_parquet(saida_ml1, index=False, engine='pyarrow')
print(f"  -> Salvo: {saida_ml1} ({len(df_ml1):,} linhas)\n")

# ==========================================
# ETAPA 2B: PRÉ-PROCESSAMENTO PARA O ML2
# (Regressão de Prazo de Resolução)
# ==========================================
print("[3/3] Preparando dados para o ML2 (Regressor de Prazo)...")

colunas_ml2 = ['DATA_DEMANDA', 'DATA_ULT_SITUACAO', 'BAIRRO', 'GRUPOSERVICO_DESCRICAO', 'SITUACAO']
df_ml2 = df_all[colunas_ml2].copy()

# Conversão de datas e remoção de nulos
df_ml2['DATA_DEMANDA']       = pd.to_datetime(df_ml2['DATA_DEMANDA'],       format='mixed', errors='coerce')
df_ml2['DATA_ULT_SITUACAO']  = pd.to_datetime(df_ml2['DATA_ULT_SITUACAO'], format='mixed', errors='coerce')
df_ml2.dropna(subset=['DATA_DEMANDA', 'DATA_ULT_SITUACAO', 'BAIRRO', 'GRUPOSERVICO_DESCRICAO'], inplace=True)

# Padronização de texto — reutilizando a função auxiliar
df_ml2['SITUACAO'] = df_ml2['SITUACAO'].astype(str).str.strip().str.upper()
df_ml2['BAIRRO']   = df_ml2['BAIRRO'].astype(str).str.strip().str.upper()
df_ml2['GRUPOSERVICO_DESCRICAO'] = padronizar_texto(df_ml2['GRUPOSERVICO_DESCRICAO'])

# Apenas chamados concluídos (para aprender quanto tempo uma conclusão leva)
df_ml2 = df_ml2[df_ml2['SITUACAO'] == 'ATENDIDA']

# Feature Engineering: variável alvo e features temporais
df_ml2['Dias_Resolucao'] = (df_ml2['DATA_ULT_SITUACAO'] - df_ml2['DATA_DEMANDA']).dt.days

# Remoção de outliers extremos (acima do percentil 90)
# Fazemos isso aqui para que o ML2 já carregue um dataset "limpo"
limite_dias = df_ml2['Dias_Resolucao'].quantile(0.90)
df_ml2 = df_ml2[(df_ml2['Dias_Resolucao'] >= 0) & (df_ml2['Dias_Resolucao'] <= limite_dias)]
print(f"  -> Filtro de outliers: mantendo até {limite_dias:.0f} dias de resolução.")

df_ml2['Mes']       = df_ml2['DATA_DEMANDA'].dt.month
df_ml2['Ano']       = df_ml2['DATA_DEMANDA'].dt.year
df_ml2['Dia_Semana']= df_ml2['DATA_DEMANDA'].dt.dayofweek

# Salvando em Parquet
saida_ml2 = os.path.join(pasta_dados, 'df_ml2.parquet')
df_ml2.to_parquet(saida_ml2, index=False, engine='pyarrow')
print(f"  -> Salvo: {saida_ml2} ({len(df_ml2):,} linhas)\n")

print("=" * 50)
print("PRÉ-PROCESSAMENTO CONCLUÍDO!")
print("Agora execute ML1.py e ML2.py normalmente.")
print("=" * 50)
