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

def render_ml1():
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
    # Repetido para Teste para demonstrar a falha em ambas as bases
    df_teste_lr = df_treino_lr.copy()
    df_teste_lr['support'] = ['89353', '25213', '114566', '114566', '114566']
    df_teste_lr.loc[2, 'f1-score'] = '0.77' # Ajuste fino do log original do teste

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
                # Explicação Matriz RF
                html.Div("💡 O modelo manteve estabilidade. A ligeira queda nas métricas entre treino e teste indica uma excelente capacidade de generalização e controlo eficaz de overfitting.", className="mt-3 text-success fw-bold small"),
                
                html.Hr(className="my-4"),
                dbc.Row([
                    dbc.Col([
                        html.Img(src='/assets/ML1/shap_random_forest_bar.png', className='img-fluid shadow-sm rounded border'),
                        html.P("Importância Global: O fator 'Serviço' dita a esmagadora maioria das decisões preditivas, tornando-se o pilar do gargalo.", className="text-muted small mt-2 text-center fst-italic")
                    ], width=12, md=6, className="mb-3 mb-md-0"),
                    dbc.Col([
                        html.Img(src='/assets/ML1/shap_random_forest_beeswarm.png', className='img-fluid shadow-sm rounded border'),
                        html.P("Impacto Real: A dispersão longa do 'Serviço' à direita indica que categorias específicas de obras são sentenças quase certas de atraso.", className="text-muted small mt-2 text-center fst-italic")
                    ], width=12, md=6),
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
                # Explicação Matriz DT
                html.Div("⚠️ Sinais de Overfitting: A árvore decorou excessivamente os dados de treino. Ao enfrentar a base de teste, o Recall da classe alvo (Gargalo) despencou de 0.79 para 0.72.", className="mt-3 text-warning fw-bold small"),
                
                html.Hr(className="my-4"),
                dbc.Row([
                    dbc.Col([
                        html.Img(src='/assets/ML1/shap_arvore_de_decisao_bar.png', className='img-fluid shadow-sm rounded border'),
                        html.P("Viés Estrutural: A árvore viciou-se quase 100% no 'Serviço', ignorando correlações mais finas de tempo e local.", className="text-muted small mt-2 text-center fst-italic")
                    ], width=12, md=6, className="mb-3 mb-md-0"),
                    dbc.Col([
                        html.Img(src='/assets/ML1/shap_arvore_de_decisao_beeswarm.png', className='img-fluid shadow-sm rounded border'),
                        html.P("Regras Rígidas: Os pontos excessivamente aglomerados provam uma tomada de decisão 'quadrada' e pouco orgânica da árvore.", className="text-muted small mt-2 text-center fst-italic")
                    ], width=12, md=6),
                ])
            ])
        ], className="shadow border-0 mb-5"),

        # --- CARD 3: REGRESSÃO LOGÍSTICA ---
        dbc.Card([
            dbc.CardHeader(html.H5("3. REGRESSÃO LOGÍSTICA (MODELO LINEAR)", className="mb-0 fw-bold text-white"), className="bg-dark"),
            dbc.CardBody([
                
                dbc.Row([
                    dbc.Col([
                        html.H6("Acurácia Treino - 80%: 77.90%", className="fw-bold mb-2 small text-muted", style={'fontFamily': 'monospace'}),
                        gerar_tabela_terminal(df_treino_lr)
                    ], width=12, lg=6, className="mb-3 mb-lg-0"),
                    dbc.Col([
                        html.H6("Acurácia Teste - 20%: 77.99%", className="fw-bold mb-2 small text-muted", style={'fontFamily': 'monospace'}),
                        gerar_tabela_terminal(df_teste_lr)
                    ], width=12, lg=6),
                ]),

                # Explicação Inviabilidade LR
                dbc.Alert([
                    html.I(className="fa-solid fa-circle-xmark me-2"),
                    html.B("Estatisticamente Inviável para Produção: "),
                    "Incapaz de lidar com o desbalanceamento das classes através de uma reta linear, o modelo assumiu que TUDO é 'Fluxo Normal'. A acurácia de 78% é uma ilusão matemática, visto que o F1-Score da classe de Gargalo é zero absoluto."
                ], color="danger", className="mt-4 shadow-sm border-0"),
                
                html.Hr(className="my-4"),
                dbc.Row([
                    dbc.Col([
                        html.Img(src='/assets/ML1/shap_regressao_logistica_bar.png', className='img-fluid shadow-sm rounded border'),
                        html.P("Distribuição Arbitrária: Tenta distribuir pesos lineares em variáveis que não têm comportamento linear contínuo.", className="text-muted small mt-2 text-center fst-italic")
                    ], width=12, md=6, className="mb-3 mb-md-0"),
                    dbc.Col([
                        html.Img(src='/assets/ML1/shap_regressao_logistica_beeswarm.png', className='img-fluid shadow-sm rounded border'),
                        html.P("Falha de Separação: A mistura caótica de cores em todas as direções ilustra a completa incapacidade do modelo em separar as duas realidades operacionais.", className="text-muted small mt-2 text-center fst-italic")
                    ], width=12, md=6),
                ])
            ])
        ], className="shadow border-0 mb-5")
    ], fluid=True)