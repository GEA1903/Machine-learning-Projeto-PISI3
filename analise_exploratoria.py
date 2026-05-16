#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import sys
import unicodedata
import plotly.express as px
import warnings

# ============================================================
# PALETA OKABE-ITO — Padrão científico para acessibilidade a
# daltônicos (adotada pela revista Nature). 8 cores com máximo
# contraste entre si para todos os tipos de daltonismo.
# ============================================================
OKABE_ITO = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#56B4E9', '#CC79A7', '#F0E442', '#000000']

# Ignorar avisos irrelevantes do pandas
warnings.filterwarnings('ignore')

# Define a pasta local onde estão os seus CSVs
pasta_dados = 'data/'
dfs = []

print("Iniciando a leitura dos datasets locais...")

# Lista todos os arquivos terminados em .csv dentro da pasta 'data'
arquivos_csv = [f for f in os.listdir(pasta_dados) if f.endswith('.csv')]

for arquivo in arquivos_csv:
    caminho_completo = os.path.join(pasta_dados, arquivo)
    print(f"Carregando {arquivo}...")
    try:
        # Lê o arquivo direto do seu HD (D:) usando as mesmas regras de formatação
        df = pd.read_csv(caminho_completo, sep=';', encoding='latin1', on_bad_lines='skip', low_memory=False)
        dfs.append(df)
    except Exception as e:
        print(f" -> Erro ao ler o arquivo {arquivo}: {e}")

# Verificação de segurança (Parada se a pasta estiver vazia ou com erro)
if not dfs:
    print("ERRO CRÍTICO: Nenhum dataset foi carregado. Verifique se os arquivos estão na pasta 'data'. Execução interrompida.")
    sys.exit()

# Concatenação de todos os anos
df_all = pd.concat(dfs, ignore_index=True)
print(f"\nLeitura concluída com sucesso! Volume total da base: {len(df_all)} registros.")


# In[2]:


print("Iniciando rotina de limpeza e engenharia de dados...")

# 1. Tratamento de Datas
df_all['DATA_DEMANDA'] = pd.to_datetime(df_all['DATA_DEMANDA'], format='mixed', errors='coerce')
df_all['DATA_ULT_SITUACAO'] = pd.to_datetime(df_all['DATA_ULT_SITUACAO'], format='mixed', errors='coerce')

# 2. Remoção Estrita de Nulos
reg_antes = len(df_all)
df_all.dropna(subset=['DATA_DEMANDA'], inplace=True)
print(f"Linhas descartadas por falta de DATA_DEMANDA: {reg_antes - len(df_all)}")

# 3. Engenharia de Variáveis (Features) para EDA
df_all['Ano_Mes'] = df_all['DATA_DEMANDA'].dt.to_period('M').astype(str)
df_all['Mes'] = df_all['DATA_DEMANDA'].dt.month
# Extraindo o nome do dia da semana
df_all['Dia_Semana'] = df_all['DATA_DEMANDA'].dt.day_name()

# 4. Padronização de Strings (Tirar espaços invisíveis e deixar tudo maiúsculo)
df_all['BAIRRO'] = df_all['BAIRRO'].astype(str).str.strip().str.upper()
df_all['GRUPOSERVICO_DESCRICAO'] = df_all['GRUPOSERVICO_DESCRICAO'].astype(str).str.strip().str.upper()


# In[3]:


print("Iniciando agrupamento avançado de categorias e situações")

# --- Tratamento da coluna SITUACAO ---
df_all['SITUACAO'] = df_all['SITUACAO'].str.strip()
df_all['SITUACAO'] = df_all['SITUACAO'].str.replace(r'.*EXECU.*', 'EXECUCAO', regex=True)
df_all['SITUACAO'] = df_all['SITUACAO'].str.replace(r'.*FISCALIZA.*', 'FISCALIZACAO', regex=True)
df_all['SITUACAO'] = df_all['SITUACAO'].str.replace(r'.*PREPARA.*', 'PREPARACAO', regex=True)
df_all['SITUACAO'] = df_all['SITUACAO'].str.replace(r'.*PENDEN.*', 'PENDENTE', regex=True)

# --- Tratamento da coluna GRUPOSERVICO_DESCRICAO ---
df_all['GRUPOSERVICO_DESCRICAO'] = df_all['GRUPOSERVICO_DESCRICAO'].str.strip()
df_all['GRUPOSERVICO_DESCRICAO'] = df_all['GRUPOSERVICO_DESCRICAO'].str.replace(r'.*ARBORIZA.*', 'ARBORIZACAO', regex=True)
df_all['GRUPOSERVICO_DESCRICAO'] = df_all['GRUPOSERVICO_DESCRICAO'].str.replace(r'.*PAVIMENTA.*', 'PAVIMENTACAO', regex=True)
df_all['GRUPOSERVICO_DESCRICAO'] = df_all['GRUPOSERVICO_DESCRICAO'].str.replace(r'.*ILUMINA.*', 'ILUMINACAO PUBLICA', regex=True)

# --- Normalização de SERVICO_DESCRICAO e LOGRADOURO ---
# Remove acentos para unificar variantes com encoding diferente:
# 'MANUTENÇÃO EM LÂMPADA', 'MANUTENCAO EM LAMPADA', 'MANUTENO EM LMPADA'
# todas viram 'MANUTENCAO EM LAMPADA' e são contadas juntas.
def remover_acentos(texto):
    if not isinstance(texto, str):
        return texto
    texto = texto.strip().upper()
    nfkd = unicodedata.normalize('NFD', texto)
    return ''.join(c for c in nfkd if unicodedata.category(c) != 'Mn')

df_all['SERVICO_DESCRICAO'] = df_all['SERVICO_DESCRICAO'].astype(str).apply(remover_acentos)
df_all['LOGRADOURO']        = df_all['LOGRADOURO'].astype(str).apply(remover_acentos)

print("Base pronta e padronizada para os gráficos!")
print("Limpeza finalizada. A base está pronta para a Análise Exploratória.")


# In[4]:


df_all['DATA_DEMANDA'] = pd.to_datetime(df_all['DATA_DEMANDA'], errors='coerce')

df_filtrado = df_all[(df_all['DATA_DEMANDA'] >= '2020-01-01') & (df_all['DATA_DEMANDA'] <= '2025-12-31')].copy()

vol_mensal = df_filtrado['DATA_DEMANDA'].dt.to_period('M').value_counts().sort_index().reset_index()
vol_mensal.columns = ['Ano', 'Volume de Denúncias']

# SOLUÇÃO DEFINITIVA: Deleta exatamente a última linha (o mês incompleto que causa a queda)
vol_mensal = vol_mensal.iloc[:-1]

vol_mensal['Ano'] = vol_mensal['Ano'].dt.to_timestamp()

fig1 = px.line(
    vol_mensal, 
    x='Ano', 
    y='Volume de Denúncias', 
    title='Evolução do Volume Total de Denúncias',
    markers=True
)

fig1.update_layout(
    yaxis=dict(
        tickmode='array',
        tickvals=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000],
        ticktext=['1k', '2k', '3k', '4k', '5k', '6k', '7k', '8k', '9k', '10k', '11k', '12k', '13k', '14k', '15k'],
        rangemode='tozero',
        title='Volume'
    ),
    xaxis=dict(
        dtick="M12",         
        tickformat="%Y",     
        ticklabelmode="period", 
        title=None,          
        showgrid=False       
    )
)

fig1.show()


# In[5]:


top5_cat = df_all['GRUPOSERVICO_DESCRICAO'].value_counts().head(5).index
df_top5_cat = df_all[df_all['GRUPOSERVICO_DESCRICAO'].isin(top5_cat)]

vol_mes_cat = df_top5_cat.groupby(['Mes', 'GRUPOSERVICO_DESCRICAO']).size().reset_index(name='Volume')

fig2 = px.bar(vol_mes_cat, x='Mes', y='Volume', color='GRUPOSERVICO_DESCRICAO',
              barmode='group', title='2. Sazonalidade das Categorias (Top 5 Serviços por Mês)',
              labels={'Mes': 'Mês do Ano (1 a 12)', 'GRUPOSERVICO_DESCRICAO': 'Categoria'},
              color_discrete_sequence=OKABE_ITO)  # Okabe-Ito: seguro para daltônicos
fig2.show()


# In[6]:


heatmap_data = df_all.groupby(['Dia_Semana', 'Mes']).size().reset_index(name='Volume')

# Ordem dos dias para ficar visualmente correto no eixo Y do Plotly
ordem_dias = ['Sunday', 'Saturday', 'Friday', 'Thursday', 'Wednesday', 'Tuesday', 'Monday']

fig3 = px.density_heatmap(
    heatmap_data, 
    x='Mes', 
    y='Dia_Semana', 
    z='Volume',
    title='2. Padrão de Acessos: Dias da Semana vs Meses do Ano',
    category_orders={'Dia_Semana': ordem_dias},
    color_continuous_scale='Cividis',  # Cividis: máximo contraste para daltônicos em escalas contínuas
    labels={'Mes': 'Mês do Ano', 'Dia_Semana': 'Dia da Semana'}
)
fig3.show()


# In[7]:


# Top 10 Bairros (Barras Horizontais)
top10_bairros = df_all['BAIRRO'].value_counts().head(10).reset_index()
top10_bairros.columns = ['Bairro', 'Volume']

fig4 = px.bar(top10_bairros, x='Volume', y='Bairro', orientation='h',
              title='3. Ranking de Volume: Top 10 Bairros Críticos',
              color='Bairro',                      # Cor por rótulo: cada barra com cor única
              color_discrete_sequence=OKABE_ITO)   # Okabe-Ito: seguro para daltônicos
fig4.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
fig4.show()

# Top 20 Bairros (Treemap)
top20_bairros = df_all['BAIRRO'].value_counts().head(20).reset_index()
top20_bairros.columns = ['Bairro', 'Volume']

fig5 = px.treemap(top20_bairros, path=['Bairro'], values='Volume',
                  title='5. Representatividade Visual Espacial: Top 20 Bairros',
                  color='Bairro',                      # Cor por rótulo: cada tile com cor única
                  color_discrete_sequence=OKABE_ITO)   # Okabe-Ito: seguro para daltônicos
fig5.show()


# In[8]:


top5_bairros_nomes = df_all['BAIRRO'].value_counts().head(5).index
df_bairros_criticos = df_all[df_all['BAIRRO'].isin(top5_bairros_nomes)]

# Calcular a proporção percentual
df_prop = df_bairros_criticos.groupby(['BAIRRO', 'SITUACAO']).size().reset_index(name='Contagem')
df_prop['Porcentagem'] = df_prop.groupby('BAIRRO')['Contagem'].transform(lambda x: x / x.sum() * 100)

fig6 = px.bar(
    df_prop, 
    x='BAIRRO', 
    y='Porcentagem', 
    color='SITUACAO',
    title='4. Proporção de Resoluções nos 5 Bairros Críticos',
    labels={'Porcentagem': 'Porcentagem (%)', 'BAIRRO': 'Bairro'},
    barmode='stack',
    color_discrete_sequence=OKABE_ITO  # Okabe-Ito: seguro para daltônicos
)
fig6.show()


# In[9]:


top10_vias = df_all['LOGRADOURO'].value_counts().head(10).reset_index()
top10_vias.columns = ['Logradouro', 'Quantidade_Defeitos']


fig_vias = px.bar(
    top10_vias,
    x='Quantidade_Defeitos',
    y='Logradouro',
    orientation='h',
    title='5. Top 10 Vias Mais Críticas (Maior Número de Defeitos)',
    labels={'Quantidade_Defeitos': 'Nº de Ocorrências', 'Logradouro': 'Via/Logradouro'},
    color='Logradouro',                    # Cor por rótulo: cada barra com cor única
    color_discrete_sequence=OKABE_ITO      # Okabe-Ito: seguro para daltônicos
)
fig_vias.update_layout(
    yaxis={'categoryorder': 'total ascending', 'tickangle': 0},
    xaxis={'tickangle': 0},
    yaxis_title=None,   # Remove o título rotacionado do eixo Y
    showlegend=False,
    height=500
)
fig_vias.show()


# In[10]:


# O código do gráfico
status_balanco = df_all['SITUACAO'].value_counts().reset_index()
status_balanco.columns = ['Situacao', 'Total']

fig8 = px.pie(
    status_balanco, 
    values='Total', 
    names='Situacao', 
    hole=0.5, 
    title='8. Balanço de Eficiência Pública (Gráfico de Rosca)',
    color_discrete_sequence=OKABE_ITO,  # Okabe-Ito: seguro para daltônicos
    labels={'Situacao': 'Status', 'Total': 'Chamados'}
)

fig8.update_traces(textinfo='percent+label', pull=[0.1 if c == 'PENDENTE' else 0 for c in status_balanco['Situacao']])
fig8.show()


# In[11]:


# 1. Preparação dos dados: Isolar o maior grupo e contar sub-serviços
maior_grupo = df_all['GRUPOSERVICO_DESCRICAO'].value_counts().idxmax()
df_detalhe = df_all[df_all['GRUPOSERVICO_DESCRICAO'] == maior_grupo]
top10_servicos = df_detalhe['SERVICO_DESCRICAO'].value_counts().head(10).reset_index()
top10_servicos.columns = ['Serviço', 'Volume']

# 2. Gráfico de Barras Horizontais com gradiente de cor
fig9 = px.bar(
    top10_servicos,
    x='Volume',
    y='Serviço',
    orientation='h',
    title=f'9. Detalhamento Crítico: Principais Queixas em "{maior_grupo}"',
    color='Serviço',                       # Cor por rótulo: cada barra com cor única
    color_discrete_sequence=OKABE_ITO,     # Okabe-Ito: seguro para daltônicos
    labels={'Volume': 'Qtd. Ocorrências', 'Serviço': 'Subcategoria de Serviço'}
)
fig9.update_layout(
    yaxis={'categoryorder': 'total ascending', 'tickangle': 0},
    xaxis={'tickangle': 0},
    yaxis_title=None,   # Remove o título rotacionado do eixo Y
    showlegend=False,
    height=500
)
fig9.show()



# In[12]:


# 1. Preparação: Filtrar os Top 10 Bairros E as Top 10 Categorias
top_bairros = df_all['BAIRRO'].value_counts().head(10).index
top_categorias = df_all['GRUPOSERVICO_DESCRICAO'].value_counts().head(10).index

df_identidade = df_all[
    (df_all['BAIRRO'].isin(top_bairros)) & 
    (df_all['GRUPOSERVICO_DESCRICAO'].isin(top_categorias))
]

# 2. Criar matriz de cruzamento
matriz_identidade = pd.crosstab(df_identidade['BAIRRO'], df_identidade['GRUPOSERVICO_DESCRICAO'])

# 3. Gráfico de Calor (Heatmap) Ajustado
fig10 = px.imshow(
    matriz_identidade,
    labels=dict(x="Categoria do Problema", y="Bairro", color="Volume"),
    title='10. Identidade Urbana: Bairros vs. Principais Categorias',
    color_continuous_scale='Oranges',  # Monocromático: máximo contraste sem ambiguidade de cor
    text_auto=',.0f',                  # Mostra o valor numérico dentro de cada célula
    aspect="auto",
    height=600
)

# Arrumando a bagunça: inclina as palavras e deixa no eixo de baixo para não bater no título
fig10.update_layout(
    xaxis_tickangle=-45,
    margin=dict(b=120) 
)

fig10.show()


# In[13]:


# GRÁFICO 1: A Dor da Cidade
# 1. Pegar os 5 serviços mais problemáticos
top5_cats = df_all['GRUPOSERVICO_DESCRICAO'].value_counts().head(5).index
df_top5 = df_all[df_all['GRUPOSERVICO_DESCRICAO'].isin(top5_cats)]

# 2. Agrupar por Mês para ver a evolução
vol_temporal = df_top5.groupby(['Mes', 'GRUPOSERVICO_DESCRICAO']).size().reset_index(name='Volume')

# 3. Gráfico de Linha
fig_story1 = px.line(
    vol_temporal, 
    x='Mes', 
    y='Volume', 
    color='GRUPOSERVICO_DESCRICAO',
    markers=True,
    title='Ato 1: O Ciclo de Vida dos Maiores Problemas (Top 5 Categorias)',
    labels={'Mes': 'Mês do Ano', 'Volume': 'Volume de Solicitações'},
    color_discrete_sequence=OKABE_ITO  # Okabe-Ito: seguro para daltônicos
)
fig_story1.show()


# In[14]:


# GRÁFICO 2: O Mapa dos Gargalos (Baseado no Top 5)
# 1. Filtrar apenas problemas não resolvidos do Top 5
status_gargalo = ['PENDENTE', 'PREPARACAO', 'CADASTRADA']
df_gargalos = df_top5[df_top5['SITUACAO'].isin(status_gargalo)]

# 2. Contar quais bairros têm as maiores filas de espera
fila_bairros = df_gargalos['BAIRRO'].value_counts().head(10).reset_index()
fila_bairros.columns = ['Bairro', 'Volume Não Resolvido']

# 3. Gráfico de Barras
fig_story2 = px.bar(
    fila_bairros,
    x='Volume Não Resolvido',
    y='Bairro',
    orientation='h',
    color='Bairro',                        # Cor por rótulo: cada barra com cor única
    color_discrete_sequence=OKABE_ITO,     # Okabe-Ito: seguro para daltônicos
    title='Ato 2: Onde a Fila Trava? (Top 10 Bairros com Mais Pendências no Top 5)'
)
fig_story2.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
fig_story2.show()


# In[15]:


# GRÁFICO 3: O Pior Gargalo nos Bairros em Crise (Top 5 do ato 2)
# 1. Identificar os 5 bairros com maior volume de casos não resolvidos
top5_bairros_criticos = fila_bairros.sort_values(by='Volume Não Resolvido', ascending=False).head(5)['Bairro']
# 2. Filtrar nossa base do Top 5 APENAS para esses bairros em situação crítica
df_foco = df_top5[df_top5['BAIRRO'].isin(top5_bairros_criticos)]
# 3. Calcular a Taxa de Ineficiência (%) cruzando Bairro vs Serviço
matriz_foco = df_foco.groupby(['BAIRRO', 'GRUPOSERVICO_DESCRICAO']).agg(
    Total_Chamados=('SITUACAO', 'count'),
    Nao_Resolvidos=('SITUACAO', lambda x: x.isin(status_gargalo).sum())
).reset_index()

# Faz a matemática da porcentagem
matriz_foco['Taxa_Ineficiencia_%'] = (matriz_foco['Nao_Resolvidos'] / matriz_foco['Total_Chamados']) * 100
# 4.Gráfico: Barras Agrupadas
fig_story3_novo = px.bar(
    matriz_foco,
    x='BAIRRO',
    y='Taxa_Ineficiencia_%',
    color='GRUPOSERVICO_DESCRICAO',
    barmode='group',
    text_auto='.1f', # Coloca o valor exato em cima de cada barra
    title='Ato 3: Qual serviço é mais negligenciado nos bairros em crise?',
    labels={
        'BAIRRO': 'Bairros Críticos (TOP 5)', 
        'Taxa_Ineficiencia_%': 'Taxa de Ineficiência (%)',
        'GRUPOSERVICO_DESCRICAO': 'Categoria'
    },
    color_discrete_sequence=OKABE_ITO  # Okabe-Ito: seguro para daltônicos
)
fig_story3_novo.update_layout(yaxis_ticksuffix='%', height=600)
fig_story3_novo.update_traces(textposition='outside', texttemplate='%{y:.1f}', textfont_size=25)
fig_story3_novo.show()


# In[16]:


resolvidos = df_all[df_all['SITUACAO'] == 'ATENDIDA'].copy()
resolvidos['DATA_DEMANDA'] = pd.to_datetime(resolvidos['DATA_DEMANDA'], errors='coerce')
resolvidos['DATA_ULT_SITUACAO'] = pd.to_datetime(resolvidos['DATA_ULT_SITUACAO'], errors='coerce')
resolvidos = resolvidos.dropna(subset=['DATA_DEMANDA', 'DATA_ULT_SITUACAO'])

resolvidos['TEMPO_DIAS'] = (resolvidos['DATA_ULT_SITUACAO'] - resolvidos['DATA_DEMANDA']).dt.days
resolvidos = resolvidos[resolvidos['TEMPO_DIAS'] >= 0]

resolvidos = resolvidos[resolvidos['GRUPOSERVICO_DESCRICAO'] != 'DENUNCIAS']

tempo_medio = resolvidos.groupby('GRUPOSERVICO_DESCRICAO')['TEMPO_DIAS'].mean().reset_index()
tempo_medio['TEMPO_DIAS'] = tempo_medio['TEMPO_DIAS'].round(0)

top10_lentos = tempo_medio.sort_values(by='TEMPO_DIAS', ascending=False).head(10)

fig_tempo = px.bar(
    top10_lentos,
    x='TEMPO_DIAS',
    y='GRUPOSERVICO_DESCRICAO',
    orientation='h',
    title='Tempo Médio de Resolução da Prefeitura',
    text='TEMPO_DIAS',
    color='GRUPOSERVICO_DESCRICAO',        # Cor por rótulo: cada barra com cor única
    color_discrete_sequence=OKABE_ITO,     # Okabe-Ito: seguro para daltônicos
    labels={'TEMPO_DIAS': '', 'GRUPOSERVICO_DESCRICAO': ''}
)
fig_tempo.update_layout(
    yaxis={'categoryorder':'total ascending'},
    height=500,
    margin=dict(r=120),
    showlegend=False,
    xaxis=dict(showgrid=True, showticklabels=True, title=None, dtick=200),
    yaxis_title=None
)
fig_tempo.update_traces(
    textposition='outside',
    texttemplate='<b>%{text} dias</b>',
    cliponaxis=False
)
fig_tempo.show()


# Aplicação de Cluster e Random Forest

# In[18]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

print("Preparando dados OTIMIZADOS para Clusterização...")

# 1. ENGENHARIA DE FEATURES
status_gargalo = ['PENDENTE', 'PREPARACAO', 'CADASTRADA']
df_bairros_cluster = df_all.groupby('BAIRRO').agg(
    Volume_Total=('SITUACAO', 'count'),
    Nao_Resolvidos=('SITUACAO', lambda x: x.isin(status_gargalo).sum())
).reset_index()

df_bairros_cluster['Taxa_Ineficiencia_%'] = (df_bairros_cluster['Nao_Resolvidos'] / df_bairros_cluster['Volume_Total']) * 100

# OTIMIZAÇÃO 1: Corte estatístico mais rigoroso (Foca apenas em bairros com demanda real e constante)
df_bairros_cluster = df_bairros_cluster[df_bairros_cluster['Volume_Total'] > 500].dropna()

# 2. PADRONIZAÇÃO 
X = df_bairros_cluster[['Volume_Total', 'Taxa_Ineficiencia_%']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. TREINANDO O MODELO K-MEANS
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_bairros_cluster['Cluster_ID'] = kmeans.fit_predict(X_scaled)

# 4. NOMEANDO OS CLUSTERS
df_bairros_cluster['Perfil_do_Bairro'] = df_bairros_cluster['Cluster_ID'].astype(str)
df_bairros_cluster['Perfil_do_Bairro'] = 'Grupo ' + df_bairros_cluster['Perfil_do_Bairro']

# 5. VISUALIZAÇÃO OTIMIZADA
fig_cluster = px.scatter(
    df_bairros_cluster, 
    x='Volume_Total', 
    y='Taxa_Ineficiencia_%', 
    color='Perfil_do_Bairro',
    size='Volume_Total', # OTIMIZAÇÃO 2: O tamanho da bolha agora representa o impacto na cidade
    hover_name='BAIRRO',
    title='Clusterização K-Means: Perfil de Crise dos Bairros (Otimizado)',
    labels={
        'Volume_Total': 'Volume Total de Ocorrências', 
        'Taxa_Ineficiencia_%': 'Taxa de Ineficiência (%)'
    },
    template='plotly_white',
    color_discrete_sequence=OKABE_ITO,  # Okabe-Ito: seguro para daltônicos
    size_max=35  # Controla o tamanho limite da maior bolha
)

# OTIMIZAÇÃO 3: Linhas de média da cidade
media_vol = df_bairros_cluster['Volume_Total'].mean()
media_inef = df_bairros_cluster['Taxa_Ineficiencia_%'].mean()
fig_cluster.add_vline(x=media_vol, line_dash="dash", line_color="red", opacity=0.3)
fig_cluster.add_hline(y=media_inef, line_dash="dash", line_color="red", opacity=0.3)

fig_cluster.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
fig_cluster.show()


from sklearn.cluster import AgglomerativeClustering


# 6. CLUSTERIZAÇÃO ALTERNATIVA (AGLOMERATIVO HIERÁRQUICO)

# CONTRAPROVA CIENTÍFICA (PP1): IMPLEMENTAÇÃO DE AGRUPAMENTO COMPLEMENTAR
# Para certificar que as "zonas de crise urbana" descobertas pelo K-Means não 
# são distorções heurísticas do algoritmo, a base georreferenciada é submetida 
# a um classificador Aglomerativo Hierárquico. A convergência espacial entre 
# dois modelos matemáticos independentes valida cientificamente os agrupamentos.
# ============================================================
print("\nGerando Clusterização Hierárquica (Modelo Alternativo)...")
aglo = AgglomerativeClustering(n_clusters=4)
df_bairros_cluster['Cluster_Hierarquico'] = aglo.fit_predict(X_scaled)

# Mapeamento taxonômico das partições geradas
df_bairros_cluster['Perfil_Aglo'] = df_bairros_cluster['Cluster_Hierarquico'].astype(str)
df_bairros_cluster['Perfil_Aglo'] = 'Camada ' + df_bairros_cluster['Perfil_Aglo']

# Definição de paleta cromática exclusiva para diferenciação visual nos relatórios
PALETA_AGLO = ['#E69F00', '#56B4E9', '#009E73', '#CC79A7']

# GERAÇÃO DO MAPA DE DISPERSÃO HIERÁRQUICO OTIMIZADO
fig_cluster_2 = px.scatter(
    df_bairros_cluster, 
    x='Volume_Total', 
    y='Taxa_Ineficiencia_%', 
    color='Perfil_Aglo',
    size='Volume_Total', # A proporcionalidade volumétrica das coordenadas é preservada
    hover_name='BAIRRO',
    title='Clusterização Hierárquica: Perfil de Crise dos Bairros (Comparativo)',
    labels={
        'Volume_Total': 'Volume Total de Ocorrências', 
        'Taxa_Ineficiencia_%': 'Taxa de Ineficiência (%)'
    },
    template='plotly_white',
    color_discrete_sequence=PALETA_AGLO, 
    size_max=35
)

# Inclusão dos balizadores cartesianos baseados na média paramétrica da cidade
fig_cluster_2.add_vline(x=media_vol, line_dash="dash", line_color="grey", opacity=0.3)
fig_cluster_2.add_hline(y=media_inef, line_dash="dash", line_color="grey", opacity=0.3)

fig_cluster_2.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
fig_cluster_2.show()
