import dash
from dash import dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import os

# ==========================================
# 1. CARREGAMENTO E LIMPEZA DINÂMICA
# ==========================================
CAMINHO_PARQUET = 'data/df_ml1.parquet'

if os.path.exists(CAMINHO_PARQUET):
    df_geral = pd.read_parquet(CAMINHO_PARQUET, engine='pyarrow')
    
    # Tratamento da data
    df_geral['DATA_DEMANDA'] = pd.to_datetime(df_geral['DATA_DEMANDA'], errors='coerce')
    df_geral['Ano'] = df_geral['DATA_DEMANDA'].dt.year
    
    # FILTRO DE SEGURANÇA: Mantendo a realidade do seu dataset (ex: de 2020 a 2026)
    # Isso corta fora os "2007" vazados por erro de digitação na base da prefeitura
    df_geral = df_geral[(df_geral['Ano'] >= 2020) & (df_geral['Ano'] <= 2026)]
    
    # A MÁGICA ESTÁ AQUI: Convertendo numpy.int64 para int NATIVO do Python
    # Isso garante que o Dash consiga deixar as caixinhas marcadas por padrão!
    anos_reais = sorted([int(x) for x in df_geral['Ano'].dropna().unique()])
    
    # Formatando o número gigante para o padrão brasileiro (pontos no milhar)
    total_registros = f"{len(df_geral):,}".replace(",", ".")
else:
    print("ERRO: Arquivo Parquet não encontrado em data/")
    df_geral = pd.DataFrame(columns=['Ano', 'BAIRRO', 'GRUPOSERVICO_DESCRICAO', 'SITUACAO'])
    anos_reais = []
    total_registros = "0"

# (...) O Bloco 2 de Layout permanece o mesmo (...)

# ==========================================
# 2. LAYOUT E SIDEBAR
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)

SIDEBAR_STYLE = {
    "position": "fixed", "top": 0, "left": 0, "bottom": 0,
    "width": "18rem", "padding": "2rem 1rem", "backgroundColor": "#f8f9fa",
}
CONTENT_STYLE = {"marginLeft": "20rem", "marginRight": "2rem", "padding": "2rem 1rem"}

sidebar = html.Div([
    html.H2("REPORT!", className="display-6 text-primary fw-bold"),
    html.Hr(),
    html.Label("Filtro Global (Anos):", className="fw-bold"),
    dcc.Checklist(
        id='filtro-anos',
        options=[{'label': f' {ano}', 'value': ano} for ano in anos_reais],
        value=anos_reais,
        labelStyle={'display': 'block', 'margin-bottom': '5px'},
        inputStyle={"margin-right": "10px"}
    ),
    html.Hr(),
    dbc.Nav([
        dbc.NavLink("Página Inicial", href="/", active="exact"),
        dbc.NavLink("EDA (Os 3 Atos)", href="/eda", active="exact"),
        dbc.NavLink("Random Forest", href="/rf", active="exact"),
        dbc.NavLink("Clusterização", href="/cluster", active="exact"),
    ], vertical=True, pills=True),
], style=SIDEBAR_STYLE)

content = html.Div(id="page-content", style=CONTENT_STYLE)
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

# ==========================================
# 3. PÁGINAS (FRONTEND)
# ==========================================

def render_home():
    return html.Div([
        html.H3("Conceito da Base de Dados", className="text-primary"),
        dcc.Markdown(f'''
        Atualmente o sistema está processando **{total_registros}** registros oficiais da EMLURB.
        
        **Estrutura de Dados:**
        Os dados foram convertidos de CSV para **Parquet** para otimizar a leitura. 
        Este formato permite que os filtros de anos e bairros sejam aplicados de forma instantânea,
        reduzindo o consumo de memória RAM em comparação ao processamento de texto bruto.
        ''')
    ])

def render_eda():
    return html.Div([
        html.H3("EDA: Os 3 Atos Correlacionados"),
        dbc.Row([dbc.Col(dcc.Graph(id='ato-1'), width=12)]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='ato-2'), width=6),
            dbc.Col(dcc.Graph(id='ato-3'), width=6)
        ]),
        dbc.Button("Resetar Filtros de Gráfico", id="reset-eda", color="secondary", className="mt-3")
    ])

def render_rf():
    return html.Div([
        html.H3("Resultados: Random Forest"),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("ML1: Classificador"),
                dbc.CardBody([html.H4("Acurácia: 89%"), html.P("Treino: 91% | Teste: 89%")])
            ]), width=6),
            dbc.Col(dbc.Card([
                dbc.CardHeader("ML2: Regressor"),
                dbc.CardBody([html.H4("MAE: 4.8 dias"), html.P("Erro Médio Absoluto na previsão de prazos.")])
            ]), width=6),
        ])
    ])

def render_cluster():
    return html.Div([
        html.H3("Clusterização Geográfica"),
        dcc.Graph(id='graph-cluster')
    ])

# ==========================================
# 4. CALLBACKS (LÓGICA INTERATIVA)
# ==========================================

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def navigate(path):
    if path == "/eda": return render_eda()
    if path == "/rf": return render_rf()
    if path == "/cluster": return render_cluster()
    return render_home()

@app.callback(
    [Output('ato-1', 'figure'), Output('ato-2', 'figure'), Output('ato-3', 'figure')],
    [Input('filtro-anos', 'value'), Input('ato-1', 'clickData'), Input('ato-2', 'clickData'), Input('reset-eda', 'n_clicks')]
)
def update_eda_graphs(anos, click1, click2, n_reset):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Reset de cliques se o botão for apertado
    if triggered_id == 'reset-eda':
        click1 = click2 = None

    dff = df_geral[df_geral['Ano'].isin(anos)]
    
    # Ato 1: Geral
    fig1 = px.bar(dff.groupby('GRUPOSERVICO_DESCRICAO').size().reset_index(name='Qtd'), 
                  x='GRUPOSERVICO_DESCRICAO', y='Qtd', title="Ato 1: Volume por Serviço", color_discrete_sequence=['#0072B2'])

    # Ato 2: Filtrado pelo Ato 1
    if click1:
        servico = click1['points'][0]['x']
        dff = dff[dff['GRUPOSERVICO_DESCRICAO'] == servico]
    fig2 = px.bar(dff.groupby('BAIRRO').size().reset_index(name='Qtd').sort_values('Qtd', ascending=False).head(15), 
                  x='BAIRRO', y='Qtd', title="Ato 2: Bairros Mais Afetados", color_discrete_sequence=['#E69F00'])

    # Ato 3: Filtrado pelo Ato 2
    if click2:
        bairro = click2['points'][0]['x']
        dff = dff[dff['BAIRRO'] == bairro]
    fig3 = px.pie(dff, names='SITUACAO', title="Ato 3: Status das Demandas", hole=.3)

    return fig1, fig2, fig3

@app.callback(Output('graph-cluster', 'figure'), [Input('filtro-anos', 'value')])
def update_cluster(anos):
    dff = df_geral[df_geral['Ano'].isin(anos)]
    df_c = dff.groupby('BAIRRO').size().reset_index(name='Ocorrências')
    return px.scatter(df_c, x='BAIRRO', y='Ocorrências', size='Ocorrências', color='BAIRRO', title="Cluster de Bairros")

if __name__ == "__main__":
    app.run(debug=True, port=8055)