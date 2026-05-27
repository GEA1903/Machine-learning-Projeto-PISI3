import dash_bootstrap_components as dbc
from dash import dcc, html
import pandas as pd

def render_simulador(df_geral):
    if not df_geral.empty:
        bairros_opcoes = [{'label': str(b), 'value': str(b)} for b in sorted(df_geral['BAIRRO'].dropna().unique())]
        servicos_opcoes = [{'label': str(s), 'value': str(s)} for s in sorted(df_geral['GRUPOSERVICO_DESCRICAO'].dropna().unique())]
    else:
        bairros_opcoes, servicos_opcoes = [], []

    meses_opcoes = [
        {'label': 'Janeiro', 'value': 1}, {'label': 'Fevereiro', 'value': 2},
        {'label': 'Março', 'value': 3}, {'label': 'Abril', 'value': 4},
        {'label': 'Maio', 'value': 5}, {'label': 'Junho', 'value': 6},
        {'label': 'Julho', 'value': 7}, {'label': 'Agosto', 'value': 8},
        {'label': 'Setembro', 'value': 9}, {'label': 'Outubro', 'value': 10},
        {'label': 'Novembro', 'value': 11}, {'label': 'Dezembro', 'value': 12}
    ]

    dias_semana_opcoes = [
        {'label': 'Domingo', 'value': 0}, {'label': 'Segunda-feira', 'value': 1},
        {'label': 'Terça-feira', 'value': 2}, {'label': 'Quarta-feira', 'value': 3},
        {'label': 'Quinta-feira', 'value': 4}, {'label': 'Sexta-feira', 'value': 5},
        {'label': 'Sábado', 'value': 6}
    ]

    return dbc.Container([
        html.H3("Laboratório de IA: Simulador em Produção", className="text-primary mt-4 fw-bold"),
        html.P("Insira os parâmetros de uma ocorrência hipotética para acionar os motores preditivos em tempo real.", className="text-secondary mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Formulário de Denúncia", className="mb-0 fw-bold text-white"), className="bg-dark"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Bairro da Ocorrência:", className="fw-bold text-primary small"),
                                dcc.Dropdown(id='sim-bairro', options=bairros_opcoes, placeholder="Ex: BOA VIAGEM", className="mb-3 shadow-sm")
                            ], md=6),
                            dbc.Col([
                                html.Label("Serviço Solicitado:", className="fw-bold text-success small"),
                                dcc.Dropdown(id='sim-servico', options=servicos_opcoes, placeholder="Ex: LUMINÁRIAS", className="mb-3 shadow-sm")
                            ], md=6),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Mês do Ano:", className="fw-bold text-secondary small"),
                                dcc.Dropdown(id='sim-mes', options=meses_opcoes, placeholder="Mês", className="mb-3 shadow-sm")
                            ], md=4),
                            dbc.Col([
                                html.Label("Dia da Semana:", className="fw-bold text-secondary small"),
                                dcc.Dropdown(id='sim-dia-semana', options=dias_semana_opcoes, placeholder="Dia", className="mb-3 shadow-sm")
                            ], md=4),
                            dbc.Col([
                                html.Label("Ano Comercial:", className="fw-bold text-secondary small"),
                                dbc.Input(id='sim-ano', type='number', value=2024, min=2020, max=2030, className="mb-3 shadow-sm")
                            ], md=4),
                        ]),
                        html.Div(
                            dbc.Button(
                                [html.I(className="fa-solid fa-paper-plane me-2"), "Enviar Denúncia e Gerar Diagnóstico"], 
                                id="btn-simular", color="primary", size="lg", className="w-100 fw-bold shadow mt-3 rounded-pill"
                            )
                        )
                    ])
                ], className="shadow border-0 h-100 border-start border-dark border-5")
            ], md=12, lg=7, className="mb-4"),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Relatório de Diagnóstico", className="mb-0 fw-bold text-white"), className="bg-info"),
                    dbc.CardBody([
                        html.Div(id="resultado-simulacao", children=[
                            html.Div([
                                html.I(className="fa-solid fa-shield-halved fa-4x text-muted mb-3 opacity-25"),
                                html.H5("Aguardando Envio...", className="text-muted fw-bold"),
                                html.P("Preencha os dados e clique no botão para simular a resposta imediata da Inteligência Artificial.", className="text-muted small")
                            ], className="text-center py-5")
                        ])
                    ], className="bg-light") 
                ], className="shadow border-0 h-100 border-start border-info border-5")
            ], md=12, lg=5, className="mb-4")
        ])
    ], fluid=True)