import dash_bootstrap_components as dbc
from dash import dcc, html

def render_eda(anos_reais):
    return dbc.Container([
        # Container do Filtro Temporal Moderno (Botões Arredondados)
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(
                        html.H5("Filtro Temporal dos 3 Atos:", className="m-0 fw-bold text-secondary"), 
                        width="auto", 
                        className="d-flex align-items-center"
                    ),
                    dbc.Col([
                        html.Div(
                            [
                                dbc.Button(
                                    str(ano),
                                    id={'type': 'btn-ano', 'index': ano},
                                    n_clicks=1, # Começa ativado
                                    color="primary",
                                    className="rounded-pill me-2 px-4 fw-bold shadow-sm",
                                    style={'transition': 'all 0.2s ease'}
                                ) for ano in anos_reais
                            ],
                            className="d-flex flex-wrap"
                        )
                    ]),
                    dbc.Col(
                        dbc.Button("Resetar Gráficos", id="reset-eda", color="secondary", outline=True, size="sm", className="rounded-pill"), 
                        width="auto",
                        className="d-flex align-items-center"
                    )
                ], className="align-items-center")
            ])
        ], className="mb-4 shadow-sm border-0 bg-light"),

        # Os 3 Atos empilhados verticalmente
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id='ato-1'))], className="mb-4 shadow border-0"), width=12),
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id='ato-2'))], className="mb-4 shadow border-0"), width=12),
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id='ato-3'))], className="mb-4 shadow border-0"), width=12),
        ])
    ], fluid=True)