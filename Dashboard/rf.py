import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import pandas as pd
import plotly.express as px

# Função auxiliar para manter a padronização e evitar redundância
def gerar_tabela_terminal(df):
    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
        style_cell={
            'textAlign': 'center', 'padding': '12px', 
            'fontFamily': 'monospace', 'fontSize': '14px'
        },
        style_header={
            'backgroundColor': '#1E293B', 'color': 'white', 
            'fontWeight': 'bold', 'fontFamily': 'sans-serif'
        },
        style_data_conditional=[
            {
                'if': {'column_id': 'Classe / Métrica'},
                'fontWeight': 'bold', 'textAlign': 'left',
                'paddingLeft': '20px', 'fontFamily': 'sans-serif'
            },
            {
                'if': {'row_index': 2}, 
                'backgroundColor': '#F8F9FA', 'fontWeight': 'bold'
            }
        ],
        style_table={'overflowX': 'auto'}
    )

def render_rf():
    # ==========================================
    # 1. DADOS DO RANDOM FOREST (O VENCEDOR)
    # ==========================================
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

    df_imp_rf = pd.DataFrame({
        'Variável': ['Serviço', 'Bairro', 'Mês', 'Dia da Semana'],
        'Peso/Grau de Influência (%)': [76.1, 15.1, 4.5, 4.3]
    })

    fig_rf = px.bar(
        df_imp_rf, x='Peso/Grau de Influência (%)', y='Variável', orientation='h',
        title='Análise de Atributos (Caixa Preta): Random Forest',
        template='plotly_white', color='Variável',
        color_discrete_sequence=['#0072B2', '#E69F00', '#009E73', '#D55E00'] 
    )
    fig_rf.update_traces(textposition='outside', texttemplate='<b>%{x:.1f}%</b>', textfont_size=13)
    fig_rf.update_layout(
        xaxis=dict(range=[0, 100], ticksuffix='%'), 
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(r=80, l=20, t=50, b=20), height=380, showlegend=False
    )

    # ==========================================
    # 2. DADOS DA ÁRVORE DE DECISÃO
    # ==========================================
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

    df_imp_dt = pd.DataFrame({
        'Variável': ['Serviço', 'Bairro', 'Mês', 'Dia da Semana'],
        'Peso/Grau de Influência (%)': [79.1, 10.1, 5.6, 5.2]
    })

    fig_dt = px.bar(
        df_imp_dt, x='Peso/Grau de Influência (%)', y='Variável', orientation='h',
        title='Análise de Atributos (Caixa Preta): Árvore de Decisão',
        template='plotly_white', color='Variável',
        color_discrete_sequence=['#0072B2', '#E69F00', '#009E73', '#D55E00'] 
    )
    fig_dt.update_traces(textposition='outside', texttemplate='<b>%{x:.1f}%</b>', textfont_size=13)
    fig_dt.update_layout(
        xaxis=dict(range=[0, 100], ticksuffix='%'), 
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(r=80, l=20, t=50, b=20), height=380, showlegend=False
    )

    # ==========================================
    # 3. DADOS DA REGRESSÃO LOGÍSTICA
    # ==========================================
    df_treino_lr = pd.DataFrame({
        'Classe / Métrica': ['Fluxo Normal (0)', 'Gargalo/Atraso (1)', 'accuracy', 'macro avg', 'weighted avg'],
        'precision': ['0.78', '0.00', '', '0.39', '0.61'],
        'recall': ['1.00', '0.00', '', '0.50', '0.78'],
        'f1-score': ['0.88', '0.00', '0.78', '0.44', '0.68'],
        'support': ['356981', '101279', '458260', '458260', '458260']
    })

    df_imp_lr = pd.DataFrame({
        'Variável': ['Mês', 'Serviço', 'Dia da Semana', 'Bairro'],
        'Peso/Grau de Influência (%)': [37.9, 37.2, 24.4, 0.6]
    })

    fig_lr = px.bar(
        df_imp_lr, x='Peso/Grau de Influência (%)', y='Variável', orientation='h',
        title='Análise de Atributos (Caixa Preta): Regressão Logística',
        template='plotly_white', color='Variável',
        color_discrete_sequence=['#0072B2', '#E69F00', '#009E73', '#D55E00'] 
    )
    fig_lr.update_traces(textposition='outside', texttemplate='<b>%{x:.1f}%</b>', textfont_size=13)
    fig_lr.update_layout(
        xaxis=dict(range=[0, 50], ticksuffix='%'), 
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(r=80, l=20, t=50, b=20), height=380, showlegend=False
    )

    # ==========================================
    # 4. LAYOUT FINAL DA PÁGINA
    # ==========================================
    return dbc.Container([
        html.H3("Avaliação de Desempenho - Modelos Preditivos", className="text-primary mt-4 fw-bold"),
        html.P("Exibição detalhada dos relatórios de classificação estruturados diretamente a partir dos logs de execução do Pipeline KDD.", className="text-secondary mb-4"),
        
        # --- BANNER DE VENCEDOR ---
        dbc.Alert([
            html.I(className="fa-solid fa-trophy me-2"),
            html.B("Modelo Selecionado: Random Forest. "),
            "Apresentou a melhor capacidade de generalização no mundo real (Base de Teste) com o maior F1-Score na classe de Gargalos, equilibrando o desbalanceamento das classes e controlando o overfitting gerado pela Árvore de Decisão Simples."
        ], color="success", className="shadow-sm rounded-pill mb-5 border-0"),

        # --- CARD 1: RANDOM FOREST (VENCEDOR) ---
        dbc.Card([
            dbc.CardHeader(
                html.H5("1. RANDOM FOREST (VENCEDOR) - RELATÓRIOS COMPARATIVOS", className="mb-0 fw-bold text-white"), 
                className="bg-primary"
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Acurácia (Treino - 80%): 92.25%", className="fw-bold mb-3 text-dark", style={'fontFamily': 'monospace', 'fontSize': '15px'}),
                        gerar_tabela_terminal(df_treino_rf)
                    ], width=12, lg=6, className="mb-4 mb-lg-0 border-end-lg"),
                    
                    dbc.Col([
                        html.H6("Acurácia (Teste - 20%): 89.40%", className="fw-bold mb-3 text-dark", style={'fontFamily': 'monospace', 'fontSize': '15px'}),
                        gerar_tabela_terminal(df_teste_rf)
                    ], width=12, lg=6)
                ]),
                
                html.Hr(className="my-5"),
                dcc.Graph(figure=fig_rf, config={'displayModeBar': False})
            ])
        ], className="shadow border-0 mb-5 border-start border-primary border-5"),

        # --- CARD 2: ÁRVORE DE DECISÃO ---
        dbc.Card([
            dbc.CardHeader(
                html.H5("2. ÁRVORE DE DECISÃO CLÁSSICA - RELATÓRIOS COMPARATIVOS", className="mb-0 fw-bold text-white"), 
                style={"backgroundColor": "#D55E00"} 
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Acurácia (Treino - 80%): 92.25%", className="fw-bold mb-3 text-dark", style={'fontFamily': 'monospace', 'fontSize': '15px'}),
                        gerar_tabela_terminal(df_treino_dt)
                    ], width=12, lg=6, className="mb-4 mb-lg-0 border-end-lg"),
                    
                    dbc.Col([
                        html.H6("Acurácia (Teste - 20%): 89.11%", className="fw-bold mb-3 text-dark", style={'fontFamily': 'monospace', 'fontSize': '15px'}),
                        gerar_tabela_terminal(df_teste_dt)
                    ], width=12, lg=6)
                ]),
                
                html.Hr(className="my-5"),
                dcc.Graph(figure=fig_dt, config={'displayModeBar': False})
            ])
        ], className="shadow border-0 mb-5"),

        # --- CARD 3: REGRESSÃO LOGÍSTICA ---
        dbc.Card([
            dbc.CardHeader(
                html.H5("3. REGRESSÃO LOGÍSTICA - [DESEMPENHO NO TREINO - 80%]", className="mb-0 fw-bold text-white"), 
                className="bg-dark"
            ),
            dbc.CardBody([
                html.H6("Acurácia: 77.90%", className="fw-bold mb-3 text-dark", style={'fontFamily': 'monospace', 'fontSize': '16px'}),
                gerar_tabela_terminal(df_treino_lr),
                html.Hr(className="my-5"),
                dcc.Graph(figure=fig_lr, config={'displayModeBar': False})
            ])
        ], className="shadow border-0 mb-5")

    ], fluid=True)