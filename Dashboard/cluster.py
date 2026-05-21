import dash_bootstrap_components as dbc
from dash import html, dcc
import pandas as pd
import plotly.express as px
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def render_cluster():
    # ==========================================
    # 1. CARREGAMENTO DOS DADOS (PARQUET)
    # ==========================================
    CAMINHO_PARQUET = 'data/df_ml1.parquet'
    
    if not os.path.exists(CAMINHO_PARQUET):
        return dbc.Container([html.H3("Arquivo de dados não encontrado. Verifique a pasta data/")])
        
    df_all = pd.read_parquet(CAMINHO_PARQUET, engine='pyarrow')
    
    # ==========================================
    # 2. ENGENHARIA DE FEATURES
    # ==========================================
    status_gargalo = ['PENDENTE', 'PREPARACAO', 'CADASTRADA']
    df_bairros_cluster = df_all.groupby('BAIRRO').agg(
        Volume_Total=('SITUACAO', 'count'),
        Nao_Resolvidos=('SITUACAO', lambda x: x.isin(status_gargalo).sum())
    ).reset_index()

    df_bairros_cluster['Taxa_Ineficiencia_%'] = (df_bairros_cluster['Nao_Resolvidos'] / df_bairros_cluster['Volume_Total']) * 100

    # OTIMIZAÇÃO 1: Corte estatístico rigoroso (> 500 ocorrências)
    df_bairros_cluster = df_bairros_cluster[df_bairros_cluster['Volume_Total'] > 500].dropna()

    # ==========================================
    # 3. PADRONIZAÇÃO E TREINAMENTO K-MEANS
    # ==========================================
    X = df_bairros_cluster[['Volume_Total', 'Taxa_Ineficiencia_%']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_bairros_cluster['Cluster_ID'] = kmeans.fit_predict(X_scaled)

    # NOMEANDO OS CLUSTERS
    df_bairros_cluster['Perfil_do_Bairro'] = df_bairros_cluster['Cluster_ID'].astype(str)
    df_bairros_cluster['Perfil_do_Bairro'] = 'Grupo ' + df_bairros_cluster['Perfil_do_Bairro']

    # ==========================================
    # 4. VISUALIZAÇÃO OTIMIZADA
    # ==========================================
    fig_cluster = px.scatter(
        df_bairros_cluster, 
        x='Volume_Total', 
        y='Taxa_Ineficiencia_%', 
        color='Perfil_do_Bairro',
        size='Volume_Total', 
        hover_name='BAIRRO',
        title='Clusterização K-Means: Perfil de Crise dos Bairros (Otimizado)',
        labels={
            'Volume_Total': 'Volume Total de Ocorrências', 
            'Taxa_Ineficiencia_%': 'Taxa de Ineficiência (%)',
            'Perfil_do_Bairro': 'Classificação'
        },
        template='plotly_white',
        color_discrete_sequence=px.colors.qualitative.G10,
        size_max=35 
    )

    # OTIMIZAÇÃO 3: Linhas de média da cidade (Tracejadas em Vermelho)
    media_vol = df_bairros_cluster['Volume_Total'].mean()
    media_inef = df_bairros_cluster['Taxa_Ineficiencia_%'].mean()
    fig_cluster.add_vline(x=media_vol, line_dash="dash", line_color="red", opacity=0.4)
    fig_cluster.add_hline(y=media_inef, line_dash="dash", line_color="red", opacity=0.4)

    # Acabamento das bolhas
    fig_cluster.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    
    # Ajustes finais de layout para o Dashboard
    fig_cluster.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # ==========================================
    # 5. RENDERIZAÇÃO DA PÁGINA
    # ==========================================
    return dbc.Container([
        html.H3("Clusterização Geográfica (K-Means)", className="text-primary mt-4 fw-bold"),
        html.P("Agrupamento inteligente dos bairros baseado no volume de demandas operacionais e na taxa de ineficiência, utilizando padronização de escala (StandardScaler).", className="text-secondary mb-4"),
        
        dbc.Card([
            dbc.CardHeader(
                html.H5("DISPERSÃO DE BAIRROS - MODELO K-MEANS OTIMIZADO", className="mb-0 fw-bold text-white"), 
                className="bg-info"
            ),
            dbc.CardBody([
                dcc.Graph(figure=fig_cluster, config={'displayModeBar': False}),
                html.Hr(className="my-4"),
                html.P(
                    "💡 Interpretação: As linhas tracejadas vermelhas representam a média da cidade. "
                    "Bairros localizados no quadrante superior direito estão acima da média tanto em volume quanto em ineficiência, "
                    "exigindo intervenção imediata da gestão pública.", 
                    className="text-muted small text-center mb-0 fw-bold"
                )
            ])
        ], className="shadow border-0 mb-5")
    ], fluid=True)