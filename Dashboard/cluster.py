import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

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

    return dbc.Container([
        html.H3("Clusterização Geográfica de Zonas Críticas", className="text-primary my-4 fw-bold"),
        html.P("Análise comparativa de aprendizado não supervisionado para identificação de demandas sistêmicas estruturais.", className="text-secondary mb-4"),
        dbc.Row([
            dbc.Col([dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_kmeans)), className="shadow-sm border-0")], md=6, className="mb-4"),
            dbc.Col([dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_aglo)), className="shadow-sm border-0")], md=6, className="mb-4")
        ])
    ], fluid=True)