import dash_bootstrap_components as dbc
from dash import html

def render_home(total_registros):
    return dbc.Container([
        # Cabeçalho de Boas-vindas
        dbc.Row([
            dbc.Col([
                html.H1("Bem-vindo ao REPORT!", className="display-4 text-primary fw-bold mb-3 mt-4"),
                html.P(f"Sistema de Inteligência Urbana processando {total_registros} registros operacionais da EMLURB.", 
                       className="lead text-secondary mb-4"),
            ], width=12, className="text-center")
        ]),

        # Linha dos 3 Quadros Flutuantes
        dbc.Row([
            # Card 1: EDA
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div(html.I(className="fa-solid fa-chart-line fa-3x text-primary mb-3"), className="text-center"),
                        html.H4("Análise Exploratória", className="card-title fw-bold text-center"),
                        html.P(
                            "Explore o ciclo de vida dos problemas em Recife através dos '3 Atos'. "
                            "Identifique sazonalidade, bairros críticos e taxas de ineficiência.",
                            className="card-text text-muted", style={'min-height': '80px'}
                        ),
                        dbc.Button("Acessar Gráficos", href="/eda", color="primary", className="w-100 rounded-pill mt-3"),
                        dbc.Button("Ver Correlações", href="/correlacao", color="primary", className="w-100 rounded-pill mt-3")
                    ])
                ], className="h-100 shadow border-0 hover-shadow")
            ], width=12, md=4, className="mb-4"),

            # Card 2: ML (Random Forest)
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div(html.I(className="fa-solid fa-robot fa-3x text-success mb-3"), className="text-center"),
                        html.H4("Inteligência Artificial", className="card-title fw-bold text-center"),
                        html.P(
                            "Visualize a performance dos modelos de Random Forest. Previsão de gargalos "
                            "operacionais e estimativa de prazos de resolução com precisão matemática.",
                            className="card-text text-muted", style={'min-height': '80px'}
                        ),
                        dbc.Button("Machine Learning 1", href="/ml1", color="success", className="w-100 rounded-pill mt-3"),
                        dbc.Button("Machine Learning 2", href="/ml2", color="success", className="w-100 rounded-pill mt-3")
                        
                    ])
                ], className="h-100 shadow border-0 hover-shadow")
            ], width=12, md=4, className="mb-4"),

            # Card 3: Clusterização
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div(html.I(className="fa-solid fa-circle-nodes fa-3x text-info mb-3"), className="text-center"),
                        html.H4("Divisão Geográfica", className="card-title fw-bold text-center"),
                        html.P(
                            "Entenda a divisão de Recife por perfis de crise. O algoritmo K-Means agrupa "
                            "bairros com comportamentos de demanda similares.",
                            className="card-text text-muted", style={'min-height': '80px'}
                        ),
                        dbc.Button("Ver Clusters", href="/cluster", color="info", className="w-100 rounded-pill mt-3")
                    ])
                ], className="h-100 shadow border-0 hover-shadow")
            ], width=12, md=4, className="mb-4"),
        ], className="mt-2"),

        # Footer / Nota técnica
        dbc.Row([
            dbc.Col([
                dbc.Alert(
                    "Dica: Utilize o filtro de anos na página EDA para analisar períodos específicos de gestão.",
                    color="light", className="text-center small text-muted mt-4 border-0"
                )
            ], width=12)
        ])
    ], fluid=True, className="py-5")