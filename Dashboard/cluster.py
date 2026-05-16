import dash_bootstrap_components as dbc
from dash import html, dcc

def render_cluster():
    return dbc.Container([
        html.H3("Clusterização Geográfica (K-Means)", className="text-primary my-4"),
        html.P("Esta página conterá o gráfico de dispersão com o perfil de crise dos bairros.")
    ], fluid=True)