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
    # --- DADOS DAS MATRIZES ---
    df_treino_dt = pd.DataFrame({
        'Classe / Métrica': ['Fluxo Normal (0)', 'Gargalo (1)', 'accuracy', 'macro avg', 'weighted avg'],
        'precision': ['0.94', '0.85', '', '0.89', '0.92'], 'recall': ['0.96', '0.79', '', '0.88', '0.92'],
        'f1-score': ['0.95', '0.82', '0.92', '0.88', '0.92'], 'support': ['356981', '101279', '458260', '458260', '458260']
    })
    df_teste_dt = pd.DataFrame({
        'Classe / Métrica': ['Fluxo Normal (0)', 'Gargalo (1)', 'accuracy', 'macro avg', 'weighted avg'],
        'precision': ['0.92', '0.77', '', '0.85', '0.89'], 'recall': ['0.94', '0.72', '', '0.83', '0.89'],
        'f1-score': ['0.93', '0.74', '0.89', '0.84', '0.89'], 'support': ['89353', '25213', '114566', '114566', '114566']
    })
    df_treino_rf = pd.DataFrame({
        'Classe / Métrica': ['Fluxo Normal (0)', 'Gargalo (1)', 'accuracy', 'macro avg', 'weighted avg'],
        'precision': ['0.95', '0.83', '', '0.89', '0.92'], 'recall': ['0.95', '0.81', '', '0.88', '0.92'],
        'f1-score': ['0.95', '0.82', '0.92', '0.89', '0.92'], 'support': ['356981', '101279', '458260', '458260', '458260']
    })
    df_teste_rf = pd.DataFrame({
        'Classe / Métrica': ['Fluxo Normal (0)', 'Gargalo (1)', 'accuracy', 'macro avg', 'weighted avg'],
        'precision': ['0.93', '0.76', '', '0.85', '0.89'], 'recall': ['0.93', '0.75', '', '0.84', '0.89'],
        'f1-score': ['0.93', '0.76', '0.89', '0.84', '0.89'], 'support': ['89353', '25213', '114566', '114566', '114566']
    })
    df_treino_xgb = pd.DataFrame({
        'Classe / Métrica': ['Fluxo Normal (0)', 'Gargalo (1)', 'accuracy', 'macro avg', 'weighted avg'],
        'precision': ['0.94', '0.78', '', '0.86', '0.90'], 'recall': ['0.94', '0.78', '', '0.86', '0.90'],
        'f1-score': ['0.94', '0.78', '0.90', '0.86', '0.90'], 'support': ['356981', '101279', '458260', '458260', '458260']
    })
    df_teste_xgb = pd.DataFrame({
        'Classe / Métrica': ['Fluxo Normal (0)', 'Gargalo (1)', 'accuracy', 'macro avg', 'weighted avg'],
        'precision': ['0.94', '0.77', '', '0.86', '0.90'], 'recall': ['0.94', '0.78', '', '0.86', '0.90'],
        'f1-score': ['0.94', '0.78', '0.90', '0.86', '0.90'], 'support': ['89353', '25213', '114566', '114566', '114566']
    })
    df_treino_lr = pd.DataFrame({
        'Classe / Métrica': ['Fluxo Normal (0)', 'Gargalo (1)', 'accuracy', 'macro avg', 'weighted avg'],
        'precision': ['0.78', '0.00', '', '0.39', '0.61'], 'recall': ['1.00', '0.00', '', '0.50', '0.78'],
        'f1-score': ['0.88', '0.00', '0.78', '0.44', '0.68'], 'support': ['356981', '101279', '458260', '458260', '458260']
    })
    df_teste_lr = df_treino_lr.copy()
    df_teste_lr['support'] = ['89353', '25213', '114566', '114566', '114566']
    df_teste_lr.loc[2, 'f1-score'] = '0.77'

    # --- DATAFRAME DE COMPARAÇÃO FINAL DE TESTE ---
    df_comparacao = pd.DataFrame({
        'Métrica (Base de Teste)': ['Acurácia Geral', 'Precisão (Classe Gargalo)', 'Recall (Classe Gargalo)', 'F1-Score (Classe Gargalo)'],
        'Árvore de Decisão': ['89.11%', '**0.77**', '0.72', '0.74'],
        'Random Forest': ['89.40%', '0.76', '0.75', '0.76'],
        'XGBoost': ['**90.17%**', '**0.77**', '**0.78**', '**0.78**'],
        'Regressão Logística': ['77.99%', '0.00', '0.00', '0.00']
    })

    return dbc.Container([
        html.H3("ML1: Auditoria Evolutiva de Algoritmos (Classificação)", className="text-primary mt-4 fw-bold"),
        html.P("Análise comparativa detalhada do pipeline de modelagem preditiva aplicado aos gargalos operacionais.", className="text-secondary mb-4"),
        
        # ==========================================
        # NOVO: PROPÓSITO DO MODELO (ML1)
        # ==========================================
        dbc.Card([
            dbc.CardHeader(html.H5("Justificativa e Propósito do Modelo de Classificação", className="mb-0 fw-bold text-white"), className="bg-primary"),
            dbc.CardBody([
                html.P([
                    html.Strong("Por que este modelo existe? "),
                    "O classificador preditivo foi concebido para empoderar o cidadão e otimizar a triagem da infraestrutura urbana. ",
                    "No exato momento em que um usuário realiza uma denúncia no aplicativo, a Inteligência Artificial processa as variáveis contextuais para fornecer um retorno imediato na tela, ",
                    "informando de forma transparente se aquela ocorrência específica constitui um ", html.Strong("problema crônico de zeladoria (gargalo estrutural sistêmico)"),
                    " ou se é um evento isolado de fluxo normal. Isso estabelece um canal transparente de prestação de contas e alinha as expectativas da população em tempo real."
                ], className="mb-0 text-dark small")
            ])
        ], className="shadow border-0 mb-4 border-start border-primary border-5"),

        # ==========================================
        # SEÇÃO: GLOSSÁRIO DE ÍNDICES MÉRICOS
        # ==========================================
        dbc.Card([
            dbc.CardHeader(html.H6("Glossário Metodológico de Avaliação Científica", className="mb-0 fw-bold")),
            dbc.CardBody([
                html.Ul([
                    html.Li([html.Strong("Acurácia: "), "Mede a porcentagem total de acertos do modelo entre todas as previsões feitas na base de dados."]),
                    html.Li([html.Strong("Precisão: "), "Indica a proporção de gargalos previstos corretamente em relação ao total de alarmes emitidos pela IA."]),
                    html.Li([html.Strong("Recall (Sensibilidade): "), "Mede a capacidade do modelo de capturar os gargalos reais existentes na malha urbana da cidade."]),
                    html.Li([html.Strong("F1-Score: "), "Constitui a média harmônica ponderada entre Precisão e Recall, sendo o principal índice de equilíbrio global."]),
                    html.Li([html.Strong("Suporte: "), "Indica o montante absoluto de registros reais avaliados para cada classe durante a validação estatística."]),
                ], className="mb-0 small text-muted")
            ])
        ], className="shadow-sm border-0 mb-5 bg-light"),

        # --- 1. ÁRVORE DE DECISÃO ---
        dbc.Card([
            dbc.CardHeader(html.H5("1. ÁRVORE DE DECISÃO CLÁSSICA (BASELINE)", className="mb-0 fw-bold text-white"), style={"backgroundColor": "#D55E00"}),
            dbc.CardBody([
                html.P("O algoritmo mapeia decisões como um fluxograma de regras lógicas consecutivas (Se/Então), dividindo os dados sequencialmente com base em atributos temporais e espaciais até alcançar a classificação final de cada chamado urbano.", className="text-muted small fst-italic mb-4"),
                dbc.Row([
                    dbc.Col([html.H6("Acurácia Treino: 92.25%", className="fw-bold mb-2 small text-muted", style={'fontFamily': 'monospace'}), gerar_tabela_terminal(df_treino_dt)], width=12, lg=6, className="mb-3 mb-lg-0"),
                    dbc.Col([html.H6("Acurácia Teste: 89.11%", className="fw-bold mb-2 small text-muted", style={'fontFamily': 'monospace'}), gerar_tabela_terminal(df_teste_dt)], width=12, lg=6),
                ]),
                html.Div("⚠️ Overfitting Evidente: O modelo tendeu a memorizar excessivamente a estrutura interna da base de treino. Ao deparar-se com novos dados da base de teste, a capacidade de rastreamento do gargalo (Recall) decaiu severamente de 0.79 para 0.72.", className="mt-3 text-warning fw-bold small"),
                html.Hr(className="my-4"),
                dbc.Row([
                    dbc.Col([html.Img(src='/assets/ML1/shap_arvore_de_decisao_bar.png', className='img-fluid shadow-sm rounded border')], width=12, md=6, className="mb-3 mb-md-0"),
                    dbc.Col([html.Img(src='/assets/ML1/shap_arvore_de_decisao_beeswarm.png', className='img-fluid shadow-sm rounded border')], width=12, md=6),
                ])
            ])
        ], className="shadow border-0 mb-5"),

        # --- 2. RANDOM FOREST ---
        dbc.Card([
            dbc.CardHeader(html.H5("2. RANDOM FOREST (EVOLUÇÃO POR BAGGING)", className="mb-0 fw-bold text-white"), className="bg-primary"),
            dbc.CardBody([
                html.P("Estrutura um comitê combinando centenas de Árvores de Decisão independentes criadas sob amostragem aleatória de dados e variáveis (Bagging), definindo o diagnóstico preditivo final através de uma votação de maioria absoluta.", className="text-muted small fst-italic mb-4"),
                dbc.Row([
                    dbc.Col([html.H6("Acurácia Treino: 92.25%", className="fw-bold mb-2 small text-muted", style={'fontFamily': 'monospace'}), gerar_tabela_terminal(df_treino_rf)], width=12, lg=6, className="mb-3 mb-lg-0"),
                    dbc.Col([html.H6("Acurácia Teste: 89.40%", className="fw-bold mb-2 small text-muted", style={'fontFamily': 'monospace'}), gerar_tabela_terminal(df_teste_rf)], width=12, lg=6),
                ]),
                html.Div("💡 Estabilização Operacional: A criação do conjunto paralelo de árvores mitigou a instabilidade inerente à árvore simples, promovendo um ganho de consistência e elevando o F1-Score do gargalo para 0.76 na base de testes.", className="mt-3 text-primary fw-bold small"),
                html.Hr(className="my-4"),
                dbc.Row([
                    dbc.Col([html.Img(src='/assets/ML1/shap_random_forest_bar.png', className='img-fluid shadow-sm rounded border')], width=12, md=6, className="mb-3 mb-md-0"),
                    dbc.Col([html.Img(src='/assets/ML1/shap_random_forest_beeswarm.png', className='img-fluid shadow-sm rounded border')], width=12, md=6),
                ])
            ])
        ], className="shadow border-0 mb-5"),

        # --- 3. XGBOOST ---
        dbc.Card([
            dbc.CardHeader(html.H5("3. XGBOOST CLASSIFIER (O CAMPEÃO POR BOOSTING)", className="mb-0 fw-bold text-white"), className="bg-success"),
            dbc.CardBody([
                html.P("Algoritmo de aprendizado sequencial avançado (Boosting) em que cada árvore é construída consecutivamente com foco explícito em corrigir os resíduos matemáticos e erros cometidos pelas árvores anteriores.", className="text-muted small fst-italic mb-4"),
                dbc.Row([
                    dbc.Col([html.H6("Acurácia Treino: 90.34%", className="fw-bold mb-2 small text-muted", style={'fontFamily': 'monospace'}), gerar_tabela_terminal(df_treino_xgb)], width=12, lg=6, className="mb-3 mb-lg-0"),
                    dbc.Col([html.H6("Acurácia Teste: 90.17%", className="fw-bold mb-2 small text-muted", style={'fontFamily': 'monospace'}), gerar_tabela_terminal(df_teste_xgb)], width=12, lg=6),
                ]),
                html.Div("🏆 Generalização Preditiva Superior: Ao regularizar os gradientes sequenciais, o XGBoost abriu mão de superajustar a base de treino em prol de máxima estabilidade. Apresentou a menor variação do estudo (apenas 0.17% de queda entre treino e teste), consolidando os melhores índices práticos.", className="mt-3 text-success fw-bold small"),
                html.Hr(className="my-4"),
                dbc.Row([
                    dbc.Col([html.Img(src='/assets/ML1/shap_xgboost_bar.png', className='img-fluid shadow-sm rounded border'), html.P("Importância Consolidada: O Serviço se mantém como pilar absoluto, mas as amarras do gradiente equalizaram as micro-decisões do modelo.", className="text-muted small mt-2 text-center fst-italic")], width=12, md=6, className="mb-3 mb-md-0"),
                    dbc.Col([html.Img(src='/assets/ML1/shap_xgboost_beeswarm.png', className='img-fluid shadow-sm rounded border'), html.P("Impacto Elegante: A dispersão do Serviço é mais controlada e menos errática que a do Random Forest, atestando a qualidade matemática da previsão.", className="text-muted small mt-2 text-center fst-italic")], width=12, md=6),
                ])
            ])
        ], className="shadow border-0 mb-5 border-start border-success border-5"),

        # ==========================================
        # SEÇÃO: TABELA COMPARATIVA E JUSTIFICATIVA
        # ==========================================
        dbc.Card([
            dbc.CardHeader(html.H5("Matriz de Auditoria Comparativa (Modelos em Teste)", className="mb-0 fw-bold text-white"), className="bg-primary"),
            dbc.CardBody([
                html.P("Comparativo direto do desempenho prático dos 4 classificadores puramente sobre a base de testes (dados inéditos). Os maiores valores em cada linha encontram-se destacados.", className="text-muted small mb-4"),
                dash_table.DataTable(
                    data=df_comparacao.to_dict('records'),
                    columns=[{"name": i, "id": i, "presentation": "markdown"} for i in df_comparacao.columns],
                    style_cell={'textAlign': 'center', 'padding': '12px', 'fontFamily': 'monospace', 'fontSize': '14px'},
                    style_header={'backgroundColor': '#1E293B', 'color': 'white', 'fontWeight': 'bold', 'fontFamily': 'sans-serif'},
                    style_data_conditional=[{'if': {'column_id': 'Métrica (Base de Teste)'}, 'fontWeight': 'bold', 'textAlign': 'left', 'paddingLeft': '15px', 'fontFamily': 'sans-serif'}],
                    style_table={'overflowX': 'auto'}
                ),
                html.Div([
                    html.Strong("Fundamentação Científica da Escolha do Modelo: ", className="text-dark d-block mb-2 fs-5"),
                    "A análise quantitativa unificada na matriz de auditoria estabelece o ", html.Strong("XGBoost"), " como a escolha ideal para o motor preditivo do ecossistema. Ele conquistou soberania estatística ao cravar a maior Acurácia Geral (**90.17%**), o maior Recall (**0.78**) e o maior F1-Score (**0.78**) da classe crítica de gargalos, empatando no topo da Precisão teórica. Esse comportamento numérico comprova a eficácia das penalizações matemáticas por regularização embutidas em sua arquitetura de boosting, blindando o algoritmo contra as flutuações e ruídos temporais da zeladoria municipal."
                ], className="mt-4 p-4 bg-white rounded border border-primary border-3 small text-muted")
            ])
        ], className="shadow border-0 mb-5"),

        # --- 4. REGRESSÃO LOGÍSTICA ---
        dbc.Card([
            dbc.CardHeader(html.H5("4. REGRESSÃO LOGÍSTICA (FALHA CONSTRUTIVA)", className="mb-0 fw-bold text-white"), className="bg-dark"),
            dbc.CardBody([
                html.P("Modelo matemático estatístico clássico que utiliza uma função sigmoide para computar a probabilidade linear de ocorrência de uma classe binária através da atribuição de pesos fixos para cada variável independente.", className="text-muted small fst-italic mb-4"),
                dbc.Row([
                    dbc.Col([html.H6("Acurácia Treino: 77.90%", className="fw-bold mb-2 small text-muted", style={'fontFamily': 'monospace'}), gerar_tabela_terminal(df_treino_lr)], width=12, lg=6, className="mb-3 mb-lg-0"),
                    dbc.Col([html.H6("Acurácia Teste: 77.99%", className="fw-bold mb-2 small text-muted", style={'fontFamily': 'monospace'}), gerar_tabela_terminal(df_teste_lr)], width=12, lg=6),
                ]),
                html.Hr(className="my-4"),
                dbc.Row([
                    dbc.Col([html.Img(src='/assets/ML1/shap_regressao_logistica_bar.png', className='img-fluid shadow-sm rounded border'), html.P("Importância Artificial: Atribuição linear e subestimada de pesos às variáveis do sistema.", className="text-muted small mt-2 text-center fst-italic")], width=12, md=6, className="mb-3 mb-md-0"),
                    dbc.Col([
                        html.Img(src='/assets/ML1/shap_regressao_logistica_beeswarm.png', className='img-fluid shadow-sm rounded border'),
                        html.P("Sobreposição Caótica: Pontos dispersos sem nenhuma separação lógica de cores.", className="text-muted small mt-2 text-center fst-italic")
                    ], width=12, md=6),
                ]),
                
                dbc.Alert([
                    html.H5("Diagnóstico de Inviabilidade Metodológica (Modelo Linear):", className="fw-bold mb-3 text-info"),
                    html.P([
                        html.Strong("Evidência de Colapso nos Gráficos SHAP: "),
                        "O gráfico de barras do SHAP escancara a incapacidade do classificador linear ao atribuir um peso médio inexpressivo de apenas +0.12 ao Serviço e quase zero ao Bairro. A comprovação geométrica definitiva da quebra do algoritmo está no gráfico Enxame de Abelhas (Beeswarm): as variáveis falham em segregar os chamados, resultando em uma nuvem caótica onde pontos azuis (valores baixos) e vermelhos (valores altos) acumulam-se em sobreposição total sobre o eixo neutro. Em algoritmos maduros, as cores se separam nitidamente para os extremos laterais. Essa mistura caótica atesta visualmente que o modelo é cego para correlações de infraestrutura."
                    ], className="small mb-3"),
                    html.P([
                        html.Strong("Por que a reta estatística falha no contexto da Cidade? "),
                        "Demandas de zeladoria urbana não possuem um comportamento linear simples. Um gargalo operacional não surge da mera soma isolada das variáveis, mas sim de complexas interações multidimensionais (condições logísticas de determinados serviços operando sob sobrecarga sazonal em locais específicos). Como algoritmos lineares tentam traçar uma linha reta fixa para cindir as classes, eles colapsam diante de distribuições complexas e bases desbalanceadas. Para reduzir seu erro global, a Regressão Logística tomou a rota matemática mais conservadora: classificou absolutamente todas as entradas como classe 0 (Fluxo Normal). Isso gera uma ilusória acurácia de 77.99% (que é exatamente a proporção natural da maioria da base), mas crava zero absoluto em Precision, Recall e F1-Score para a classe alvo de Gargalos, inviabilizando completamente sua aplicação prática."
                    ], className="small mb-0")
                ], color="info", className="mt-4 p-4 shadow-sm border-0 border-start border-info border-5")
            ])
        ], className="shadow border-0 mb-5")
    ], fluid=True)