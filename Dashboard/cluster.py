import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

def render_cluster(df_geral=None):
    if df_geral is None:
        return dbc.Container([html.H3("Aguardando carregamento dos dados...")])

    status_gargalo = ['PENDENTE', 'PREPARACAO', 'CADASTRADA']
    df_bairros = df_geral.groupby('BAIRRO').agg(
        Volume_Total=('SITUACAO', 'count'),
        Nao_Resolvidos=('SITUACAO', lambda x: x.isin(status_gargalo).sum())
    ).reset_index()

    df_bairros['Taxa_Ineficiencia_%'] = (df_bairros['Nao_Resolvidos'] / df_bairros['Volume_Total']) * 100
    df_bairros = df_bairros[df_bairros['Volume_Total'] > 500].dropna()

    X = df_bairros[['Volume_Total', 'Taxa_Ineficiencia_%']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_bairros['Cluster_KMeans'] = 'Grupo ' + kmeans.fit_predict(X_scaled).astype(str)

    aglo = AgglomerativeClustering(n_clusters=4)
    df_bairros['Cluster_Hierarquico'] = 'Camada ' + aglo.fit_predict(X_scaled).astype(str)

    dbscan = DBSCAN(eps=0.4, min_samples=4)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    # Mapeia o -1 para "Ruído/Outlier" e o resto para "Zona X"
    df_bairros['Cluster_DBSCAN'] = [
        'Anomalia (Outlier)' if label == -1 else f'Zona Adjacente {label}' 
        for label in dbscan_labels
    ]

    media_vol = df_bairros['Volume_Total'].mean()
    media_inef = df_bairros['Taxa_Ineficiencia_%'].mean()

    fig_kmeans = px.scatter(
        df_bairros, x='Volume_Total', y='Taxa_Ineficiencia_%', color='Cluster_KMeans',
        size='Volume_Total', hover_name='BAIRRO',
        title='1. Algoritmo K-Means: Padrões de Crise Urbano', template='plotly_white',
        color_discrete_sequence=['#0072B2', '#D55E00', '#009E73', '#E69F00'], size_max=35,
        labels={'Volume_Total': 'Volume de Chamados', 'Taxa_Ineficiencia_%': 'Ineficiência (%)'}
    )
    fig_kmeans.add_vline(x=media_vol, line_dash="dash", line_color="red", opacity=0.3)
    fig_kmeans.add_hline(y=media_inef, line_dash="dash", line_color="red", opacity=0.3)
    fig_kmeans.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))

    fig_aglo = px.scatter(
        df_bairros, x='Volume_Total', y='Taxa_Ineficiencia_%', color='Cluster_Hierarquico',
        size='Volume_Total', hover_name='BAIRRO',
        title='2. Algoritmo Hierárquico: Contraprova Espacial', template='plotly_white',
        color_discrete_sequence=['#E69F00', '#56B4E9', '#009E73', '#CC79A7'], size_max=35,
        labels={'Volume_Total': 'Volume de Chamados', 'Taxa_Ineficiencia_%': 'Ineficiência (%)'}
    )
    fig_aglo.add_vline(x=media_vol, line_dash="dash", line_color="grey", opacity=0.3)
    fig_aglo.add_hline(y=media_inef, line_dash="dash", line_color="grey", opacity=0.3)
    fig_aglo.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))

    # Gráfico 3: DBSCAN 
    fig_dbscan = px.scatter(
        df_bairros, x='Volume_Total', y='Taxa_Ineficiencia_%', color='Cluster_DBSCAN',
        size='Volume_Total', hover_name='BAIRRO',
        title='3. Algoritmo DBSCAN: Densidade e Isolamento de Anomalias', template='plotly_white',
        # Definindo uma paleta onde a anomalia ganha destaque (ex: Vermelho/Preto para anomalias se preferir)
        color_discrete_sequence=['#009E73', '#56B4E9', '#D55E00', '#CC79A7', '#000000'], size_max=35,
        labels={'Volume_Total': 'Volume de Chamados', 'Taxa_Ineficiencia_%': 'Ineficiência (%)'}
    )
    fig_dbscan.add_vline(x=media_vol, line_dash="dash", line_color="grey", opacity=0.3)
    fig_dbscan.add_hline(y=media_inef, line_dash="dash", line_color="grey", opacity=0.3)
    fig_dbscan.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))

    return dbc.Container([
        html.H3("Clusterização Geográfica de Zonas Críticas", className="text-primary my-4 fw-bold"),
        html.P("Análise comparativa de aprendizado não supervisionado para identificação de demandas sistêmicas estruturais.", className="text-secondary mb-4"),

        # ================= PRIMEIRA LINHA: K-MEANS =================
        dbc.Row([
            # Coluna do Gráfico (Esquerda)
            dbc.Col([
                dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_kmeans)), className="shadow-sm border-0 h-100")
            ], md=6, className="mb-4"),

            # Coluna do Texto (Direita)
            dbc.Col([
                html.H4("1. Algoritmo K-Means: Padrões de Crise Urbano", className="text-dark fw-bold mb-3"),

                html.H6("Descrição dos Grupos", className="text-primary fw-bold mb-2"),
                html.Ul([
                    html.Li([html.Strong("Grupo 0 (Verde - Canto Superior Esquerdo) — Os 'Esquecidos':"), " Têm baixo volume de queixas (abaixo de 6k), mas a taxa de ineficiência é muito alta (acima de 21%)."]),
                    html.Li([html.Strong("Grupo 1 (Azul - Canto Inferior Esquerdo) — Os 'Tranquilos':"), " Estão abaixo da média em volume e abaixo da média em ineficiência. São bairros que demandam pouco e com tempo de resolução aceitável."]),
                    html.Li([html.Strong("Grupo 2 (Laranja - Centro/Superior Direito) — A 'Zona de Risco Crônico':"), " Bairros grandes que sofrem com alto volume de denúncias E taxa de abandono acima da média."]),
                    html.Li([html.Strong("Grupo 3 (Amarelo) — Os 'Gigantes Prioritários':"), " Bolhas maiores com volumes colossais de denúncias (perto de 35k), mas onde a ineficiência cai para a média ou abaixo dela."]),
                ], className="text-muted mb-3", style={"fontSize": "14px"}),

                html.H6("Análise Final", className="text-primary fw-bold mb-2"),
                html.Div(
                    "Diagnóstico Operacional: O K-Means funciona traçando distâncias matemáticas diretas, gerando divisões bem geométricas. Ele escancara uma desigualdade de atenção clara: os bairros com volumes massivos de queixas (Grupo 3) mantêm a ineficiência sob controle — por maior pressão política ou comercial —, enquanto uma massa de bairros periféricos de baixo volume (Grupo 0) fica invisível na fila de prioridades, acumulando as piores taxas de atraso.",
                    className="border-start border-4 border-primary ps-3 bg-light p-3 rounded text-muted",
                    style={"fontSize": "14px", "fontStyle": "italic"}
                )
            ], md=6, className="mb-4 d-flex flex-column justify-content-center")
        ], className="align-items-stretch mb-4"),

        # ================= SEGUNDA LINHA: HIERÁRQUICO =================
        dbc.Row([
            # Coluna do Gráfico (Esquerda)
            dbc.Col([
                dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_aglo)), className="shadow-sm border-0 h-100")
            ], md=6, className="mb-4"),

            # Coluna do Texto (Direita)
            dbc.Col([
                html.H4("2. Algoritmo Hierárquico: Contraprova Espacial", className="text-dark fw-bold mb-3"),

                html.H6("Descrição dos Grupos", className="text-success fw-bold mb-2"),
                html.Ul([
                    html.Li([html.Strong("Camada 1 (Rosa - Canto Superior Esquerdo) — O 'Gargalo Isolado':"), " Isola perfeitamente a massa de baixo volume que sofre com ineficiência crítica (acima de 21%)."]),
                    html.Li([html.Strong("Camada 0 (Amarelo - Canto Inferior Esquerdo) — A 'Base Estável':"), " Concentra os bairros com baixa demanda e resposta rápida do município. É o ecossistema ideal de funcionamento."]),
                    html.Li([html.Strong("Camada 3 (Verde - Centro) — A 'Fricção Intermediária':"), " Bairros de volume médio (entre 6k e 13k) com ineficiência flutuando alto. Mostra onde o sistema começa a perder o fôlego."]),
                    html.Li([html.Strong("Camada 2 (Azul Claro) — Os 'Extremos de Volume':"), " Engloba toda a metade direita (acima de 14k chamados). O algoritmo entende que o volume extremo os torna uma categoria totalmente à parte."]),
                ], className="text-muted mb-3", style={"fontSize": "14px"}),

                html.H6("Análise Final", className="text-success fw-bold mb-2"),
                html.Div(
                    "Diagnóstico Operacional: Por construir os grupos de baixo para cima, o modelo hierárquico serviu como uma excelente contraprova. Ele validou os 'Esquecidos' e os 'Tranquilos', mas trouxe um insight valioso: a alta demanda quebra o padrão de agrupamento comum. Quando um bairro estoura a barreira dos 15k chamados, ele gera uma anomalia de escala tão grande que o algoritmo o isola, operando sob uma lógica logística totalmente à parte do resto da cidade.",
                    className="border-start border-4 border-success ps-3 bg-light p-3 rounded text-muted",
                    style={"fontSize": "14px", "fontStyle": "italic"}
                )
            ], md=6, className="mb-4 d-flex flex-column justify-content-center")
        ], className="align-items-stretch mb-4"),

        # ================= TERCEIRA LINHA: DBSCAN =================
        dbc.Row([
            # Coluna do Gráfico (Esquerda)
            dbc.Col([
                dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_dbscan)), className="shadow-sm border-0 h-100")
            ], md=6, className="mb-4"),

            # Coluna do Texto (Direita)
            dbc.Col([
                html.H4("3. Algoritmo DBSCAN: Densidade e Isolamento de Anomalias", className="text-dark fw-bold mb-3"),

                html.H6("Descrição dos Grupos", className="text-info fw-bold mb-2"),
                html.Ul([
                    html.Li([html.Strong("Zona Adjacente 0 (Verde) — A 'Realidade Urbana Densa':"), " Maior grupo do estudo (0 a 15k chamados). Mostra que, para a imensa maioria da cidade, as variações de ineficiência (8% a 35%) fazem parte de uma mesma 'massa' contínua."]),
                    html.Li([html.Strong("Anomalia / Outlier (Azul Claro) — Os 'Casos Atípicos':"), " Pontos que o algoritmo não agrupou por densidade. Inclui bairros com eficiência surpreendentemente boa para o volume e todos os pontos acima de 15k chamados."]),
                ], className="text-muted mb-3", style={"fontSize": "14px"}),

                html.H6("Análise Final", className="text-info fw-bold mb-2"),
                html.Div(
                    "Diagnóstico Operacional: O DBSCAN é o algoritmo mais realista para auditoria pública porque busca densidade real sem forçar grupos artificiais. A análise revela que os bairros de altíssimo volume não são apenas um grupo diferente, são anomalias estatísticas (Outliers). O grande aprendizado é que a prefeitura não pode usar a mesma régua para a massa verde e para as anomalias azuis; estas últimas exigem forças-tarefa e contratos customizados.",
                    className="border-start border-4 border-info ps-3 bg-light p-3 rounded text-muted",
                    style={"fontSize": "14px", "fontStyle": "italic"}
                )
            ], md=6, className="mb-4 d-flex flex-column justify-content-center")
        ], className="align-items-stretch mb-4")

], fluid=True)