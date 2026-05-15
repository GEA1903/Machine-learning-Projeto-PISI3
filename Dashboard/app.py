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
    df_geral['DATA_DEMANDA'] = pd.to_datetime(df_geral['DATA_DEMANDA'], errors='coerce')
    df_geral['Ano'] = df_geral['DATA_DEMANDA'].dt.year
    # Filtro de segurança para a base real
    df_geral = df_geral[(df_geral['Ano'] >= 2020) & (df_geral['Ano'] <= 2026)]
    anos_reais = sorted([int(x) for x in df_geral['Ano'].dropna().unique()])
    total_registros = f"{len(df_geral):,}".replace(",", ".")
else:
    df_geral = pd.DataFrame(columns=['Ano', 'BAIRRO', 'GRUPOSERVICO_DESCRICAO', 'SITUACAO', 'Mes'])
    anos_reais = []
    total_registros = "0"

# ==========================================
# 2. LAYOUT GLOBAL (NAVBAR RESPONSIVA)
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Página Inicial", href="/")),
        dbc.NavItem(dbc.NavLink("EDA (Os 3 Atos)", href="/eda")),
        dbc.NavItem(dbc.NavLink("Random Forest", href="/rf")),
        dbc.NavItem(dbc.NavLink("Clusterização", href="/cluster")),
    ],
    brand="REPORT!",
    brand_href="/",
    color="primary",
    dark=True,
    fluid=True,
    className="mb-4 shadow"
)

# Container principal onde as páginas serão renderizadas
content = dbc.Container(id="page-content", fluid=True)

app.layout = html.Div([dcc.Location(id="url"), navbar, content])

# ==========================================
# 3. CONTEÚDO DAS PÁGINAS
# ==========================================

def render_home():
    return dbc.Row([
        dbc.Col([
            html.H3("Conceito da Base de Dados", className="text-primary"),
            html.P(f"Processando {total_registros} registros oficiais."),
            dcc.Markdown('''
            Os dados foram convertidos para **Parquet** para garantir que este dashboard 
            seja fluido mesmo com milhões de linhas.
            ''')
        ])
    ])

def render_eda():
    return dbc.Container([
        # Container do Filtro de Ano (Fixado acima dos Atos)
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(html.H5("Filtro Temporal dos 3 Atos:", className="m-0"), width="auto", className="d-flex align-items-center"),
                    dbc.Col([
                        dcc.Checklist(
                            id='filtro-anos',
                            options=[{'label': f' {ano}', 'value': ano} for ano in anos_reais],
                            value=anos_reais,
                            labelStyle={'display': 'inline-block', 'margin-right': '20px', 'font-weight': 'bold'},
                            inputStyle={"margin-right": "8px"}
                        ),
                    ]),
                    dbc.Col(dbc.Button("Resetar Gráficos", id="reset-eda", color="secondary", outline=True, size="sm"), width="auto")
                ])
            ])
        ], className="mb-4 shadow-sm border-primary"),

        # Os 3 Atos empilhados
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id='ato-1'))], className="mb-4 shadow-sm"), width=12),
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id='ato-2'))], className="mb-4 shadow-sm"), width=12),
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id='ato-3'))], className="mb-4 shadow-sm"), width=12),
        ])
    ], fluid=True)

# (Placeholder para as outras páginas)
def render_rf(): return html.H3("Random Forest")
def render_cluster(): return html.H3("Clusterização")

# ==========================================
# 4. CALLBACKS
# ==========================================

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def navigate(path):
    if path == "/eda": return render_eda()
    if path == "/rf": return render_rf()
    if path == "/cluster": return render_cluster()
    return render_home()

@app.callback(
    [Output('ato-1', 'figure'), Output('ato-2', 'figure'), Output('ato-3', 'figure')],
    [Input('filtro-anos', 'value'), 
     Input('ato-1', 'clickData'), 
     Input('ato-2', 'clickData'), 
     Input('reset-eda', 'n_clicks')]
)
def update_eda_graphs(anos, click1, click2, n_reset):
    if not anos: return [px.bar(title="Selecione um ano")] * 3
    
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    if triggered_id == 'reset-eda':
        click1 = click2 = None

    dff = df_geral[df_geral['Ano'].isin(anos)]
    OKABE_ITO = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#56B4E9', '#CC79A7']
    
    # ATO 1
    top5_cats = dff['GRUPOSERVICO_DESCRICAO'].value_counts().head(5).index
    df_top5 = dff[dff['GRUPOSERVICO_DESCRICAO'].isin(top5_cats)]
    vol_temporal = df_top5.groupby(['Mes', 'GRUPOSERVICO_DESCRICAO']).size().reset_index(name='Volume')
    
    fig1 = px.line(vol_temporal, x='Mes', y='Volume', color='GRUPOSERVICO_DESCRICAO', markers=True,
                  title='Ato 1: O Ciclo de Vida dos Maiores Problemas', template='plotly_white',
                  color_discrete_sequence=OKABE_ITO, custom_data=['GRUPOSERVICO_DESCRICAO'])
    fig1.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1), height=400)

    # ATO 2
    servico = click1['points'][0]['customdata'][0] if click1 else None
    df_a2 = df_top5[df_top5['GRUPOSERVICO_DESCRICAO'] == servico] if servico else df_top5
    status_g = ['PENDENTE', 'PREPARACAO', 'CADASTRADA']
    df_g = df_a2[df_a2['SITUACAO'].isin(status_g)]
    fila = df_g['BAIRRO'].value_counts().head(10).reset_index(name='Qtd')
    
    fig2 = px.bar(fila, x='Qtd', y='BAIRRO', orientation='h', color='Qtd', color_continuous_scale='Oranges',
                  title=f"Ato 2: Onde a Fila Trava? ({servico or 'Geral'})", template='plotly_white')
    fig2.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)

    # ATO 3
    bairros_f = [click2['points'][0]['y']] if click2 else fila.head(5)['BAIRRO'].tolist()
    df_f = df_a2[df_a2['BAIRRO'].isin(bairros_f)]
    
    if df_f.empty:
        fig3 = px.bar(title="Sem dados para este filtro")
    else:
        m = df_f.groupby(['BAIRRO', 'GRUPOSERVICO_DESCRICAO']).agg(
            T=('SITUACAO', 'count'), N=('SITUACAO', lambda x: x.isin(status_g).sum())).reset_index()
        m['Taxa'] = (m['N'] / m['T']) * 100
        fig3 = px.bar(m, x='BAIRRO', y='Taxa', color='GRUPOSERVICO_DESCRICAO', barmode='group', text_auto='.1f',
                     title="Ato 3: Taxa de Ineficiência (%)", color_discrete_sequence=OKABE_ITO, template='plotly_white')
        fig3.update_layout(yaxis_ticksuffix='%', height=500)
        fig3.update_traces(textposition='outside', textfont_size=11)

    return fig1, fig2, fig3

if __name__ == "__main__":
    app.run(debug=True, port=8055)