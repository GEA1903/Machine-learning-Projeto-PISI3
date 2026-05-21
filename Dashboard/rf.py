import dash_bootstrap_components as dbc
from dash import html, dash_table
import pandas as pd

def gerar_tabela_terminal(df):
    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
        style_cell={
            'textAlign': 'center', 'padding': '10px', 
            'fontFamily': 'monospace', 'fontSize': '13px'
        },
        style_header={
            'backgroundColor': '#1E293B', 'color': 'white', 
            'fontWeight': 'bold', 'fontFamily': 'sans-serif'
        },
        style_data_conditional=[
            {
                'if': {'column_id': 'Classe / Métrica'},
                'fontWeight': 'bold', 'textAlign': 'left',
                'paddingLeft': '15px', 'fontFamily': 'sans-serif'
            },
            {
                'if': {'row_index': 2}, 
                'backgroundColor': '#F8F9FA', 'fontWeight': 'bold'
            }
        ],
        style_table={'overflowX': 'auto'}
    )

def render_rf():
    # --- DADOS DO RANDOM FOREST ---
    df_treino_rf = pd.DataFrame({
        'Classe / Métrica': ['Fluxo Normal (0)', 'Gargalo/Atraso (1)', 'accuracy', 'macro avg', 'weighted avg'],
        'precision': ['0.95', '0.83', '', '0.89', '0.92'],
        'recall': ['0.95', '0.81', '', '0.88', '0.92'],
        'f1-score': ['0.95', '0.82', '0.92', '0.89', '0.92'],
        'support': ['356981', '101279', '458260', '458260', '458260']
    })
    df_teste_rf = pd.DataFrame({
        'Classe / Métrica': ['Fluxo Normal (0)', 'Gargalo/Atraso (1)', 'accuracy', 'macro avg', 'weighted avg'],
        'precision': ['0.93', '0.76', '', '0.85', '0.89'],
        'recall': ['0.93', '0.75', '', '0.84', '0.89'],
        'f1-score': ['0.93', '0.76', '0.89', '0.84', '0.89'],
        'support': ['89353', '25213', '114566', '114566', '114566']
    })

    # --- DADOS DA ÁRVORE DE DECISÃO ---
    df_treino_dt = pd.DataFrame({
        'Classe / Métrica': ['Fluxo Normal (0)', 'Gargalo/Atraso (1)', 'accuracy', 'macro avg', 'weighted avg'],
        'precision': ['0.94', '0.85', '', '0.89', '0.92'],
        'recall': ['0.96', '0.79', '', '0.88', '0.92'],
        'f1-score': ['0.95', '0.82', '0.92', '0.88', '0.92'],
        'support': ['356981', '101279', '458260', '458260', '458260']
    })
    df_teste_dt = pd.DataFrame({
        'Classe / Métrica': ['Fluxo Normal (0)', 'Gargalo/Atraso (1)', 'accuracy', 'macro avg', 'weighted avg'],
        'precision': ['0.92', '0.77', '', '0.85', '0.89'],
        'recall': ['0.94', '0.72', '', '0.83', '0.89'],
        'f1-score': ['0.93', '0.74', '0.89', '0.84', '0.89'],
        'support': ['89353', '25213', '114566', '114566', '114566']
    })

    # --- DADOS DA REGRESSÃO LOGÍSTICA ---
    df_treino_lr = pd.DataFrame({
        'Classe / Métrica': ['Fluxo Normal (0)', 'Gargalo/Atraso (1)', 'accuracy', 'macro avg', 'weighted avg'],
        'precision': ['0.78', '0.00', '', '0.39', '0.61'],
        'recall': ['1.00', '0.00', '', '0.50', '0.78'],
        'f1-score': ['0.88', '0.00', '0.78', '0.44', '0.68'],
        'support': ['356981', '101279', '458260', '458260', '458260']
    })

    return dbc.Container([
        html.H3("ML1: Explicabilidade e Performance com SHAP", className="text-primary mt-4 fw-bold"),
        html.P("Auditoria metodológica cruzando os relatórios de classificação com as saídas estatísticas locais e globais do SHAP.", className="text-secondary mb-4"),
        
        # --- CARD 1: RANDOM FOREST ---
        dbc.Card([
            dbc.CardHeader(html.H5("1. RANDOM FOREST (VENCEDOR)", className="mb-0 fw-bold text-white"), className="bg-primary"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Acurácia Treino - 80%: 92.25%", className="fw-bold mb-2 small text-muted", style={'fontFamily': 'monospace'}),
                        gerar_tabela_terminal(df_treino_rf)
                    ], width=12, lg=6, className="mb-3 mb-lg-0"),
                    dbc.Col([
                        html.H6("Acurácia Teste - 20%: 89.40%", className="fw-bold mb-2 small text-muted", style={'fontFamily': 'monospace'}),
                        gerar_tabela_terminal(df_teste_rf)
                    ], width=12, lg=6),
                ]),
                html.Hr(className="my-4"),
                # Inclusão das Imagens SHAP lado a lado
                dbc.Row([
                    dbc.Col(html.Img(src='/assets/shap_random_forest_bar.png', className='img-fluid shadow-sm rounded border'), width=12, md=6, className="mb-3 mb-md-0"),
                    dbc.Col(html.Img(src='/assets/shap_random_forest_beeswarm.png', className='img-fluid shadow-sm rounded border'), width=12, md=6),
                ])
            ])
        ], className="shadow border-0 mb-5 border-start border-primary border-5"),

        # --- CARD 2: ÁRVORE DE DECISÃO ---
        dbc.Card([
            dbc.CardHeader(html.H5("2. ÁRVORE DE DECISÃO CLÁSSICA", className="mb-0 fw-bold text-white"), style={"backgroundColor": "#D55E00"}),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Acurácia Treino - 80%: 92.25%", className="fw-bold mb-2 small text-muted", style={'fontFamily': 'monospace'}),
                        gerar_tabela_terminal(df_treino_dt)
                    ], width=12, lg=6, className="mb-3 mb-lg-0"),
                    dbc.Col([
                        html.H6("Acurácia Teste - 20%: 89.11%", className="fw-bold mb-2 small text-muted", style={'fontFamily': 'monospace'}),
                        gerar_tabela_terminal(df_teste_dt)
                    ], width=12, lg=6),
                ]),
                html.Hr(className="my-4"),
                dbc.Row([
                    dbc.Col(html.Img(src='/assets/shap_arvore_de_decisao_bar.png', className='img-fluid shadow-sm rounded border'), width=12, md=6, className="mb-3 mb-md-0"),
                    dbc.Col(html.Img(src='/assets/shap_arvore_de_decisao_beeswarm.png', className='img-fluid shadow-sm rounded border'), width=12, md=6),
                ])
            ])
        ], className="shadow border-0 mb-5"),

        # --- CARD 3: REGRESSÃO LOGÍSTICA ---
        dbc.Card([
            dbc.CardHeader(html.H5("3. REGRESSÃO LOGÍSTICA", className="mb-0 fw-bold text-white"), className="bg-dark"),
            dbc.CardBody([
                html.H6("Acurácia Treino - 80%: 77.90%", className="fw-bold mb-2 small text-muted", style={'fontFamily': 'monospace'}),
                gerar_tabela_terminal(df_treino_lr),
                html.Hr(className="my-4"),
                dbc.Row([
                    dbc.Col(html.Img(src='/assets/shap_regressao_logistica_bar.png', className='img-fluid shadow-sm rounded border'), width=12, md=6, className="mb-3 mb-md-0"),
                    dbc.Col(html.Img(src='/assets/shap_regressao_logistica_beeswarm.png', className='img-fluid shadow-sm rounded border'), width=12, md=6),
                ])
            ])
        ], className="shadow border-0 mb-5")
    ], fluid=True)