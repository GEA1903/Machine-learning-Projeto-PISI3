import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import plotly.express as px
import pandas as pd

def render_rf():
    # ==========================================================
    # 1. DADOS DE PERFORMANCE DOS ALGORITMOS (Extraídos dos MLs)
    # ==========================================================
    # DataFrame do ML1 (Classificação de Gargalos - F1-Score)
    df_ml1 = pd.DataFrame({
        'Algoritmo': ['Random Forest', 'Árvore de Decisão', 'Regressão Logística'],
        'Acurácia (Treino)': [98.50, 99.10, 75.20],
        'Acurácia (Teste)': [88.20, 82.15, 74.80],
        'F1-Score (Teste)': [87.90, 81.90, 72.10]
    })
    
    # DataFrame do ML2 (Regressão de Prazos - MAE)
    df_ml2 = pd.DataFrame({
        'Algoritmo': ['Random Forest Regressor', 'Árvore de Decisão Reg.', 'Regressão Linear'],
        'MAE (Treino - Dias)': [12.5, 5.2, 45.3],
        'MAE (Teste - Dias)': [18.2, 25.4, 46.8]
    })
    
    # ==========================================================
    # 2. GERAÇÃO DOS GRÁFICOS COMPARATIVOS
    # ==========================================================
    # Gráfico de barras para comparar o F1-Score (Maior é melhor)
    fig_ml1 = px.bar(
        df_ml1, x='Algoritmo', y='F1-Score (Teste)',
        title="Desempenho na Classificação de Urgência (ML1)",
        template="plotly_white", color='Algoritmo',
        color_discrete_sequence=['#0072B2', '#D55E00', '#009E73']
    )
    fig_ml1.update_layout(showlegend=False)

    # Gráfico de barras para comparar o MAE (Menor é melhor)
    fig_ml2 = px.bar(
        df_ml2, x='Algoritmo', y=['MAE (Treino - Dias)', 'MAE (Teste - Dias)'],
        barmode='group',
        title="Erro Médio Absoluto na Previsão de Prazo (ML2)",
        template="plotly_white",
        color_discrete_sequence=['#56B4E9', '#E69F00']
    )

    # ==========================================================
    # 3. RENDERIZAÇÃO DA PÁGINA (Layout em Dash)
    # ==========================================================
    return dbc.Container([
        html.H3("Avaliação e Performance dos Modelos (Machine Learning)", className="text-primary my-4 fw-bold"),
        html.P("Comparativo analítico entre os modelos de Inteligência Artificial submetidos aos dados da EMLURB.", className="text-secondary mb-4"),
        
        # --- SEÇÃO ML1: CLASSIFICAÇÃO ---
        dbc.Row([
            dbc.Col([
                html.H5("ML1: Algoritmos de Classificação (F1-Score e Acurácia)", className="text-dark fw-bold"),
                # Tabela Interativa do ML1
                dash_table.DataTable(
                    data=df_ml1.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df_ml1.columns],
                    style_cell={'textAlign': 'center', 'padding': '10px'},
                    style_header={'backgroundColor': '#1E293B', 'color': 'white', 'fontWeight': 'bold'},
                    style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#F1F5F9'}]
                )
            ], width=6),
            
            dbc.Col([dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_ml1)), className="shadow-sm border-0")], width=6)
        ], className="mb-5 align-items-center"),

        html.Hr(className="my-5"),

        # --- SEÇÃO ML2: REGRESSÃO ---
        dbc.Row([
            dbc.Col([
                html.H5("ML2: Algoritmos de Regressão (Erro Médio Absoluto - MAE)", className="text-dark fw-bold"),
                # Tabela Interativa do ML2
                dash_table.DataTable(
                    data=df_ml2.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df_ml2.columns],
                    style_cell={'textAlign': 'center', 'padding': '10px'},
                    style_header={'backgroundColor': '#1E293B', 'color': 'white', 'fontWeight': 'bold'},
                    style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#F1F5F9'}]
                )
            ], width=6),
            
            dbc.Col([dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_ml2)), className="shadow-sm border-0")], width=6)
        ], className="mb-5 align-items-center")
        
    ], fluid=True)