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
        'Base de Treino (80%)': ['3.10 dias', '0.6404'],
        'Base de Teste (20%)': ['5.10 dias', '0.1481']
    })

    # --- 2. DADOS RANDOM FOREST REGRESSOR ---
    df_rf = pd.DataFrame({
        'Métrica': ['Erro Médio Absoluto (MAE)', 'Coeficiente de Determinação (R²)'],
        'Base de Treino (80%)': ['4.67 dias', '0.3993'],
        'Base de Teste (20%)': ['4.87 dias', '0.3318']
    })

    # --- 3. DADOS XGBOOST REGRESSOR ---
    df_xgb = pd.DataFrame({
        'Métrica': ['Erro Médio Absoluto (MAE)', 'Coeficiente de Determinação (R²)'],
        'Base de Treino (80%)': ['3.17 dias', '0.6388'],
        'Base de Teste (20%)': ['4.97 dias', '0.2309']
    })

    # --- 4. DADOS REGRESSÃO LINEAR ---
    df_lr = pd.DataFrame({
        'Métrica': ['Erro Médio Absoluto (MAE)', 'Coeficiente de Determinação (R²)'],
        'Base de Treino (80%)': ['6.77 dias', '0.0253'],
        'Base de Teste (20%)': ['6.78 dias', '0.0253'] 
    })

    # --- DATAFRAME DE COMPARAÇÃO FINAL DE TESTE ---
    df_comparacao = pd.DataFrame({
        'Métrica (Base de Teste)': ['Erro Médio Absoluto (MAE)', 'Coeficiente de Determinação (R²)'],
        'Árvore de Decisão': ['5.10 dias', '0.1481'],
        'Random Forest': ['**4.87 dias**', '**0.3318**'],
        'XGBoost': ['4.97 dias', '0.2309'],
        'Regressão Linear': ['6.78 dias', '0.0253']
    })

    return dbc.Container([
        html.H3("ML2: Auditoria Evolutiva de Algoritmos (Regressão)", className="text-primary mt-4 fw-bold"),
        html.P("Análise comparativa detalhada do pipeline de modelagem preditiva aplicado à estimativa de prazos de resolução.", className="text-secondary mb-4"),
        
        # ==========================================
        # NOVO: PROPÓSITO DO MODELO (ML2)
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
                    "A inteligência preditiva protege o corpo técnico e otimiza o erário público por meio de uma régua justa baseada puramente em histórico estatístico."
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
                html.Div("Aviso — Sinais de Memorização Frágil: O algoritmo apresentou alto poder de ajuste inicial no treino, mas desmoronou drasticamente ao enfrentar dados inéditos na validação, cravando o pior R² (0.1481) entre os modelos baseados em árvores.", className="mt-3 text-warning fw-bold small text-center"),
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
                html.Div("Nota — Soberania em Generalização: Graças ao voto por média e às amarras estruturais aplicadas, o modelo barrou o ruído das flutuações urbanas, mantendo o menor desvio entre treino e teste e consolidando-se como o modelo mais seguro para produção.", className="mt-3 text-primary fw-bold small text-center"),
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
                html.Div("Aviso — Ajuste Sequencial Agressivo: Embora seja o estado da arte na classificação, o método de correção de erros do Boosting mostrou-se sensível demais à volatilidade dos prazos da EMLURB, gerando um sobreajuste que inflou o treino mas perdeu eficácia no teste.", className="mt-3 text-success fw-bold small text-center"),
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
                    "A análise matemática unificada na tabela de auditoria estabelece o ", html.Strong("Random Forest Regressor"), " como a arquitetura ideal para o motor de estimativas do aplicativo. Enquanto algoritmos complexos como o XGBoost caíram na armadilha de superajustar o treino, o Random Forest obteve o menor Erro Médio Absoluto do estudo (**4.87 dias**) e o maior Coeficiente de Determinação (**0.3318**) na base de testes. Isso prova cientificamente que, para prever prazos contínuos e voláteis do município, a inteligência coletiva por voto médio (Bagging) apresenta maior robustez contra ruídos do que a correção agressiva por gradiente (Boosting)."
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
                        "A inconsistência estatística fica explícita no gráfico de importância global do SHAP, onde o modelo falha em reconhecer a dominância absoluta da variável 'Serviço' sobre o tempo de execução. O colapso definitivo se consolida no gráfico Enxame de Abelhas (Beeswarm): os pontos de impacto acumulam-se em um bloco vertical caótico e amorfo sobre a linha central zero. Não há expansão horizontal nem segregação de cores, o que comprova visualmente que alterar os valores de Bairro ou Serviço não causa nenhuma tração matemática ou direcionamento lógico na fórmula deste algoritmo."
                    ], className="small mb-3"),
                    html.P([
                        html.Strong("Por que a reta estatística falha no contexto da Zeladoria Urbana? "),
                        "Os prazos operacionais para a resolução de problemas na cidade não se comportam de maneira linear contínua. O tempo necessário para solucionar uma demanda de 'Calçadas' em relação a uma troca de 'Luminárias' não é um fator constante que pode ser escalado linearmente em uma equação de primeiro grau (Y = ax + b). Prazos urbanos dependem de interações contextuais complexas, gargalos de maquinário e sazonalidade de chuvas. Como a Regressão Linear força uma reta rígida sobre os dados e é incapaz de capturar interações entre variáveis não lineares, o modelo entra em subajuste extremo (Underfitting). O resultado prático é um erro inaceitável de quase uma semana inteira (**6.78 dias**) e um R² próximo de zero absoluto (**0.0253**), provando que o modelo falha em explicar 98% da variação real do sistema."
                    ], className="small mb-0")
                ], color="info", className="mt-4 p-4 shadow-sm border-0 border-start border-info border-5")
            ])
        ], className="shadow border-0 mb-5")
    ], fluid=True)