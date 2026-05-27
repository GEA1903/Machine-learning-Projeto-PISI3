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
    # --- 1. DADOS ÁRVORE DE DECISÃO REGRESSORA ---
    df_treino_dt = pd.DataFrame({
        'Métrica': ['Erro Médio Absoluto (MAE)', 'Coeficiente de Determinação (R²)'],
        'Base de Treino (80%)': ['1.71 dias', '0.8152'],
        'Base de Teste (20%)': ['5.11 dias', '0.0212']
    })

    # --- 2. DADOS RANDOM FOREST REGRESSOR ---
    df_rf = pd.DataFrame({
        'Métrica': ['Erro Médio Absoluto (MAE)', 'Coeficiente de Determinação (R²)'],
        'Base de Treino (80%)': ['4.50 dias', '0.4344'],
        'Base de Teste (20%)': ['4.78 dias', '0.3489']
    })

    # --- 3. DADOS XGBOOST REGRESSOR ---
    df_xgb = pd.DataFrame({
        'Métrica': ['Erro Médio Absoluto (MAE)', 'Coeficiente de Determinação (R²)'],
        'Base de Treino (80%)': ['1.95 dias', '0.8107'],
        'Base de Teste (20%)': ['4.79 dias', '0.2346']
    })

    # --- 4. DADOS REGRESSÃO LINEAR ---
    df_lr = pd.DataFrame({
        'Métrica': ['Erro Médio Absoluto (MAE)', 'Coeficiente de Determinação (R²)'],
        'Base de Treino (80%)': ['6.76 dias', '0.0287'],
        'Base de Teste (20%)': ['6.72 dias', '0.0273'] 
    })

    # --- DATAFRAME DE COMPARAÇÃO FINAL DE TESTE ---
    df_comparacao = pd.DataFrame({
        'Métrica (Base de Teste)': ['Erro Médio Absoluto (MAE)', 'Coeficiente de Determinação (R²)'],
        'Árvore de Decisão': ['5.11 dias', '0.0212'],
        'Random Forest': ['**4.78 dias**', '**0.3489**'],
        'XGBoost': ['4.79 dias', '0.2346'],
        'Regressão Linear': ['6.72 dias', '0.0273']
    })

    return dbc.Container([
        html.H3("ML2: Auditoria Evolutiva de Algoritmos (Regressão)", className="text-primary mt-4 fw-bold"),
        html.P("Análise comparativa detalhada do pipeline de modelagem preditiva aplicado à estimativa de prazos de resolução.", className="text-secondary mb-4"),
        
        # ==========================================
        # PROPÓSITO DO MODELO (ML2)
        # ==========================================
        dbc.Card([
            dbc.CardHeader(html.H5("Justificativa e Propósito do Modelo de Regressão", className="mb-0 fw-bold text-white"), className="bg-primary"),
            dbc.CardBody([
                html.P([
                    html.Strong("Por que este modelo existe? "),
                    "O motor preditivo de regressão foi desenvolvido para institucionalizar a governança interna e blindar a tomada de decisão logística. ",
                    "Ao automatizar matematicamente a projeção do tempo de atendimento, o modelo remove por completo a responsabilidade técnica e o peso ético das costas do funcionário público encarregado de fixar prazos manualmente. ",
                    "Deixar essa atribuição a critérios humanos subjetivos abre margem para erros graves, induzindo ou ao desperdício severo de recursos operacionais por subestimativa, ",
                    "ou à imposição de prazos inviáveis que geram falsas expectativas e acarretam estresse extremo e desgaste psicológico nas equipes de operários em campo. ",
                    "A inteligência preditiva protege o corpo técnico e otimiza o erário público por meio de uma régua justa baseada puramente em histórico estatístico e padrões climáticos sazonais."
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
                    html.Li([html.Strong("Erro Médio Absoluto (MAE): "), "Mede a distância média real (em dias) entre as previsões calculadas pela IA e os prazos reais em que as obras foram atendidas."]),
                    html.Li([html.Strong("Coeficiente de Determinação (R²): "), "Indica a porcentagem de variação dos prazos que o modelo consegue explicar e prever (quanto mais próximo de 1.00, mais precisa é a IA)."]),
                ], className="mb-0 small text-muted")
            ])
        ], className="shadow-sm border-0 mb-5 bg-light"),

        # --- 1. ÁRVORE DE DECISÃO REGRESSORA ---
        dbc.Card([
            dbc.CardHeader(html.H5("1. ÁRVORE DE DECISÃO REGRESSORA (BASELINE)", className="mb-0 fw-bold text-white"), style={"backgroundColor": "#D55E00"}),
            dbc.CardBody([
                html.P("Divide a base de dados em ramificações baseadas em perguntas lógicas sobre as características das obras, calculando a média dos prazos de resolução dos registros contidos em cada folha final para emitir a estimativa temporal.", className="text-muted small fst-italic mb-4"),
                dbc.Row([dbc.Col(gerar_tabela_regressao(df_treino_dt), width=12, lg=8, className="mx-auto")]),
                html.Div("Sinais de Memorização Frágil: Com a inclusão das variáveis meteorológicas e de calendário (Feature Engineering), o algoritmo apresentou um sobreajuste extremo no treino (decorando dados para atingir R² de 0.8152). Entretanto, desmoronou drasticamente ao enfrentar dados inéditos na validação, cravando o pior R² (0.0212) entre os modelos baseados em árvores e elevando seu erro para 5.11 dias.", className="mt-3 text-warning fw-bold small text-center"),
                html.Hr(className="my-4"),
                dbc.Row([
                    dbc.Col([html.Img(src='/assets/ML2/shap_arvore_de_decisao_regressora_bar.png', className='img-fluid shadow-sm rounded border')], width=12, md=6, className="mb-3 mb-md-0"),
                    dbc.Col([html.Img(src='/assets/ML2/shap_arvore_de_decisao_regressora_beeswarm.png', className='img-fluid shadow-sm rounded border')], width=12, md=6),
                ])
            ])
        ], className="shadow border-0 mb-5"),

        # --- 2. RANDOM FOREST REGRESSOR ---
        dbc.Card([
            dbc.CardHeader(html.H5("2. RANDOM FOREST REGRESSOR (O VENCEDOR PELA ESTABILIDADE)", className="mb-0 fw-bold text-white"), className="bg-primary"),
            dbc.CardBody([
                html.P("Combina centenas de árvores de regressão construídas de forma paralela e independente (Bagging), mitigando distorções individuais ao consolidar o prazo estimado final através da média matemática de todo o comitê.", className="text-muted small fst-italic mb-4"),
                dbc.Row([dbc.Col(gerar_tabela_regressao(df_rf), width=12, lg=8, className="mx-auto")]),
                html.Div("Soberania em Generalização: Graças à técnica de votação coletiva (Bagging) e às amarras estruturais aplicadas, o modelo absorveu perfeitamente a densidade das novas variáveis climáticas e conteve o ruído analítico. Ele resistiu ao impulso de superajustar a base de treino e manteve a maior estabilidade operacional no teste, garantindo o menor Erro Absoluto do estudo (4.78 dias) e o melhor R² validado (0.3489).", className="mt-3 text-primary fw-bold small text-center"),
                html.Hr(className="my-4"),
                dbc.Row([
                    dbc.Col([html.Img(src='/assets/ML2/shap_random_forest_regressor_bar.png', className='img-fluid shadow-sm rounded border')], width=12, md=6, className="mb-3 mb-md-0"),
                    dbc.Col([html.Img(src='/assets/ML2/shap_random_forest_regressor_beeswarm.png', className='img-fluid shadow-sm rounded border')], width=12, md=6),
                ])
            ])
        ], className="shadow border-0 mb-5 border-start border-primary border-5"),

        # --- 3. XGBOOST REGRESSOR ---
        dbc.Card([
            dbc.CardHeader(html.H5("3. XGBOOST REGRESSOR (A ARMADILHA DO BOOSTING)", className="mb-0 fw-bold text-white"), className="bg-success"),
            dbc.CardBody([
                html.P("O algoritmo constrói árvores regressoras de forma sequencial, fazendo com que cada novo modelo aprenda e foque explicitamente em corrigir os resíduos numéricos e erros cometidos pelas estruturas predecessoras.", className="text-muted small fst-italic mb-4"),
                dbc.Row([dbc.Col(gerar_tabela_regressao(df_xgb), width=12, lg=8, className="mx-auto")]),
                html.Div("Ajuste Sequencial Agressivo: Embora seja o estado da arte em tarefas de classificação, o método de correção obsessiva de erros provou-se uma armadilha para prever prazos voláteis. Ao analisar a nova densidade de variáveis temporais, o XGBoost decorou agressivamente os padrões do treino (R² inflado de 0.8107). Quando testado com dados do mundo real, sua capacidade preditiva não se sustentou, registrando queda abrupta (R² de 0.2346 e erro de 4.79 dias).", className="mt-3 text-success fw-bold small text-center"),
                html.Hr(className="my-4"),
                dbc.Row([
                    dbc.Col([html.Img(src='/assets/ML2/shap_xgboost_regressor_bar.png', className='img-fluid shadow-sm rounded border')], width=12, md=6, className="mb-3 mb-md-0"),
                    dbc.Col([html.Img(src='/assets/ML2/shap_xgboost_regressor_beeswarm.png', className='img-fluid shadow-sm rounded border')], width=12, md=6),
                ])
            ])
        ], className="shadow border-0 mb-5"),

        # ==========================================
        # SEÇÃO: TABELA COMPARATIVA E JUSTIFICATIVA
        # ==========================================
        dbc.Card([
            dbc.CardHeader(html.H5("Matriz de Auditoria Comparativa (Modelos em Teste)", className="mb-0 fw-bold text-white"), className="bg-primary"),
            dbc.CardBody([
                html.P("Comparativo direto do desempenho prático dos 4 regressores puramente sobre a base de testes (dados inéditos). Os melhores índices encontram-se destacados em negrito (menor MAE e maior R²).", className="text-muted small mb-4"),
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
                    "A análise matemática unificada na tabela de auditoria estabelece o ", html.Strong("Random Forest Regressor"), " como a arquitetura ideal para o motor de estimativas do aplicativo. Enquanto o algoritmo avançado do XGBoost caiu na armadilha técnica de superajustar os parâmetros climáticos e temporais inseridos no treino, o Random Forest conteve o excesso preditivo pela suavização de médias (Bagging). Ele obteve de forma sólida o menor Erro Médio Absoluto do estudo (**4.78 dias**) e o maior Coeficiente de Determinação validado (**0.3489**). Fica assim cientificamente atestado que a robustez do consenso múltiplo supera abordagens de gradiente isoladas em bases de grande ruído operacional urbano."
                ], className="mt-4 p-4 bg-white rounded border border-primary border-3 small text-muted")
            ])
        ], className="shadow border-0 mb-5"),

        # --- 4. REGRESSÃO LINEAR ---
        dbc.Card([
            dbc.CardHeader(html.H5("4. REGRESSÃO LINEAR CLÁSSICA (FALHA CONSTRUTIVA)", className="mb-0 fw-bold text-white"), className="bg-dark"),
            dbc.CardBody([
                html.P("Modelo estatístico clássico que tenta ajustar uma reta matemática contínua ponderando coeficientes ponderados fixos para prever o tempo de resolução como uma combination linear direta das variáveis de entrada.", className="text-muted small fst-italic mb-4"),
                dbc.Row([dbc.Col(gerar_tabela_regressao(df_lr), width=12, lg=8, className="mx-auto")]),
                html.Hr(className="my-4"),
                dbc.Row([
                    dbc.Col([html.Img(src='/assets/ML2/shap_regressao_linear_bar.png', className='img-fluid shadow-sm rounded border'), html.P("Importância Ruidosa: Diferente das árvores, a reta falha em dar peso absoluto ao Serviço, superestimando o Ano e o Bairro.", className="text-muted small mt-2 text-center fst-italic")], width=12, md=6, className="mb-3 mb-md-0"),
                    dbc.Col([html.Img(src='/assets/ML2/shap_regressao_linear_beeswarm.png', className='img-fluid shadow-sm rounded border'), html.P("Ausência de Direção Lógica: A total aglomeração de pontos mostra que a fórmula linear não consegue traduzir os dados em dias reais.", className="text-muted small mt-2 text-center fst-italic")], width=12, md=6),
                ]),
                
                dbc.Alert([
                    html.H5("Diagnóstico de Inviabilidade Metodológica (Modelo Linear):", className="fw-bold mb-3 text-info"),
                    html.P([
                        html.Strong("Evidência de Colapso nos Gráficos SHAP: "),
                        "A inconsistência estatística fica explícita no gráfico de importância global do SHAP, onde o modelo falha em reconhecer a dominância absoluta da variável 'Serviço' sobre o tempo de execução. O colapso definitivo se consolida no gráfico Enxame de Abelhas (Beeswarm): os pontos de impacto acumulam-se em um bloco vertical caótico e amorfo sobre a linha central zero. Não há expansão horizontal nem segregação de cores, o que comprova visualmente que alterar as variáveis contextuais (como Estação de Chuvas ou Bairro) não impulsiona tração matemática ou direcionamento analítico consistente."
                    ], className="small mb-3"),
                    html.P([
                        html.Strong("Por que a reta estatística falha no contexto da Zeladoria Urbana? "),
                        "Os prazos operacionais não se comportam de maneira estritamente linear contínua. O tempo para sanar uma demanda complexa de infraestrutura versus uma simples manutenção de praça não obedece a um fator multiplicador constante escalável (Y = ax + b). Interações conjuntas entre as estações pluviais em dados bairros geram atrasos não-lineares. Diante da incapacidade intrínseca de adaptar-se a essas distorções orgânicas de calendário e logística, o modelo linear regrediu num subajuste generalizado (Underfitting). Esse fracasso consubstancia-se no erro preditivo grosseiro de quase uma semana civil (**6.72 dias**) e num nível de correlação global ínfimo (**0.0273**), o qual atesta a inépcia em explicar as oscilações sistêmicas."
                    ], className="small mb-0")
                ], color="info", className="mt-4 p-4 shadow-sm border-0 border-start border-info border-5")
            ])
        ], className="shadow border-0 mb-5")
    ], fluid=True)