import dash_bootstrap_components as dbc
from dash import html

def render_rf():
    return dbc.Container([
        html.H3("Performance dos Modelos (Random Forest)", className="text-primary my-4"),
        html.P("Esta página será populada com as tabelas de treino/teste e o erro médio absoluto (MAE).")
    ], fluid=True)