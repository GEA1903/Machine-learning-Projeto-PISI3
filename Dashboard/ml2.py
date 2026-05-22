import dash_bootstrap_components as dbc
from dash import html, dash_table
import pandas as pd

def gerar_tabela_regressao(df):
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
                'if': {'column_id': 'Métrica'},
                'fontWeight': 'bold', 'textAlign': 'left',
                'paddingLeft': '15px', 'fontFamily': 'sans-serif'
            }
        ],
        style_table={'overflowX': 'auto'}
    )

def render_ml2():
    # --- 1. DADOS RANDOM FOREST REGRESSOR (DADOS REAIS) ---
    df_rf = pd.DataFrame({
        'Métrica': ['Erro Médio Absoluto (MAE)', 'Coeficiente de Determinação (R²)'],
        'Base de Treino (80%)': ['4.67 dias', '0.3993'],
        'Base de Teste (20%)': ['4.87 dias', '0.3318']
    })

    # --- 2. DADOS ÁRVORE DE DECISÃO REGRESSORA (DADOS REAIS) ---
    df_dt = pd.DataFrame({
        'Métrica': ['Erro Médio Absoluto (MAE)', 'Coeficiente de Determinação (R²)'],
        'Base de Treino (80%)': ['3.10 dias', '0.6404'],
        'Base de Teste (20%)': ['5.10 dias', '0.1481']
    })

    # --- 3. DADOS REGRESSÃO LINEAR (DADOS REAIS) ---
    df_lr = pd.DataFrame({
        'Métrica': ['Erro Médio Absoluto (MAE)', 'Coeficiente de Determinação (R²)'],
        'Base de Treino (80%)': ['6.77 dias', '0.0253'],
        'Base de Teste (20%)': ['6.78 dias', '0.0253'] 
    })

    return dbc.Container([
        html.H3("ML2: Previsão de Tempo de Resolução (Regressão)", className="text-primary mt-4 fw-bold"),
        html.P("Auditoria metodológica dos modelos matemáticos e árvores regressoras configuradas para prever os prazos (em dias) das obras da EMLURB.", className="text-secondary mb-4"),
        
        # --- CARD 1: RANDOM FOREST REGRESSOR ---
        dbc.Card([
            dbc.CardHeader(html.H5("1. RANDOM FOREST REGRESSOR (VENCEDOR)", className="mb-0 fw-bold text-white"), className="bg-primary"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(gerar_tabela_regressao(df_rf), width=12, lg=8, className="mx-auto")
                ]),
                
                # Explicação Matriz RF
                html.Div("💡 Estabilidade Preditiva: O erro manteve-se constante (cerca de 4.8 dias) entre o treino e o teste. As amarras de profundidade aplicadas evitaram que o modelo decorasse os dados, garantindo capacidade real de generalização.", className="mt-3 text-success fw-bold small text-center"),
                
                html.Hr(className="my-4"),
                dbc.Row([
                    dbc.Col([
                        html.Img(src='/assets/ML2/shap_random_forest_regressor_bar.png', className='img-fluid shadow-sm rounded border'),
                        html.P("Importância Global: A natureza do 'Serviço' solicitado é o fator definitivo que puxa a estimativa de dias para mais ou para menos.", className="text-muted small mt-2 text-center fst-italic")
                    ], width=12, md=6, className="mb-3 mb-md-0"),
                    dbc.Col([
                        html.Img(src='/assets/ML2/shap_random_forest_regressor_beeswarm.png', className='img-fluid shadow-sm rounded border'),
                        html.P("Impacto Direto: A vasta dispersão horizontal do 'Serviço' comprova que a IA conseguiu capturar as nuances temporais de diferentes tipos de obras.", className="text-muted small mt-2 text-center fst-italic")
                    ], width=12, md=6),
                ])
            ])
        ], className="shadow border-0 mb-5 border-start border-primary border-5"),

        # --- CARD 2: ÁRVORE DE DECISÃO REGRESSORA ---
        dbc.Card([
            dbc.CardHeader(html.H5("2. ÁRVORE DE DECISÃO REGRESSORA", className="mb-0 fw-bold text-white"), style={"backgroundColor": "#D55E00"}),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(gerar_tabela_regressao(df_dt), width=12, lg=8, className="mx-auto")
                ]),
                
                # Explicação Matriz DT
                html.Div("⚠️ Overfitting Severo: O modelo decorou o passado (margem de erro de apenas 3.1 dias no treino), mas a sua performance desabou drasticamente no mundo real (teste), tornando-se não confiável.", className="mt-3 text-warning fw-bold small text-center"),
                
                html.Hr(className="my-4"),
                dbc.Row([
                    dbc.Col([
                        html.Img(src='/assets/ML2/shap_arvore_de_decisao_regressora_bar.png', className='img-fluid shadow-sm rounded border'),
                        html.P("Viés de Memorização: A árvore baseou quase toda a sua lógica de prazo exclusivamente no Serviço, negligenciando Bairro e Mês.", className="text-muted small mt-2 text-center fst-italic")
                    ], width=12, md=6, className="mb-3 mb-md-0"),
                    dbc.Col([
                        html.Img(src='/assets/ML2/shap_arvore_de_decisao_regressora_beeswarm.png', className='img-fluid shadow-sm rounded border'),
                        html.P("Falta de Nuance: O comportamento aglomerado dos pontos evidencia que a árvore criou regras absolutas, incapaz de entender a variação orgânica do tempo.", className="text-muted small mt-2 text-center fst-italic")
                    ], width=12, md=6),
                ])
            ])
        ], className="shadow border-0 mb-5"),

        # --- CARD 3: REGRESSÃO LINEAR ---
        dbc.Card([
            dbc.CardHeader(html.H5("3. REGRESSÃO LINEAR CLÁSSICA", className="mb-0 fw-bold text-white"), className="bg-dark"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(gerar_tabela_regressao(df_lr), width=12, lg=8, className="mx-auto")
                ]),

                # Explicação Inviabilidade LR
                dbc.Alert([
                    html.I(className="fa-solid fa-circle-xmark me-2"),
                    html.B("Estatisticamente Inviável (Underfitting): "),
                    "O modelo possui uma margem de erro altíssima (errosa quase uma semana inteira nos prazos) e um R² próximo de zero absoluto. Problemas urbanos não seguem regras lineares retas, inviabilizando modelos matemáticos clássicos para esta tarefa."
                ], color="danger", className="mt-4 shadow-sm border-0 text-center"),
                
                html.Hr(className="my-4"),
                dbc.Row([
                    dbc.Col([
                        html.Img(src='/assets/ML2/shap_regressao_linear_bar.png', className='img-fluid shadow-sm rounded border'),
                        html.P("Limitação Matemática: Tenta traçar relações lineares diretas para variáveis categóricas (como Bairro e Serviço), o que causa ruído estatístico.", className="text-muted small mt-2 text-center fst-italic")
                    ], width=12, md=6, className="mb-3 mb-md-0"),
                    dbc.Col([
                        html.Img(src='/assets/ML2/shap_regressao_linear_beeswarm.png', className='img-fluid shadow-sm rounded border'),
                        html.P("Correlação Inexistente: O modelo fracassa em identificar como valores específicos aumentam ou diminuem o prazo de resolução da obra.", className="text-muted small mt-2 text-center fst-italic")
                    ], width=12, md=6),
                ])
            ])
        ], className="shadow border-0 mb-5")
    ], fluid=True)