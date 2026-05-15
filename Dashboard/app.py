import dash
from dash import dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import os
from dash import ALL

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
app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.FLATLY, 
        "https://use.fontawesome.com/releases/v6.5.1/css/all.css" # Adicione este link!
    ], 
    suppress_callback_exceptions=True
)

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
    return dbc.Container([
        # Cabeçalho de Boas-vindas
        dbc.Row([
            dbc.Col([
                html.H1("Bem-vindo ao REPORT!", className="display-4 text-primary fw-bold mb-3"),
                html.P(f"Sistema de Inteligência Urbana processando {total_registros} registros operacionais da EMLURB.", 
                       className="lead text-secondary mb-5"),
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
                        dbc.Button("Acessar Gráficos", href="/eda", color="primary", className="w-100 rounded-pill mt-3")
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
                        dbc.Button("Ver Performance", href="/rf", color="success", className="w-100 rounded-pill mt-3")
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

        # Footer ou Nota técnica
        dbc.Row([
            dbc.Col([
                dbc.Alert(
                    "Dica: Utilize o filtro de anos na página EDA para analisar períodos específicos de gestão.",
                    color="light", className="text-center small text-muted mt-4 border-0"
                )
            ], width=12)
        ])
    ], fluid=True, className="py-5")

def render_eda():
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
                    # Linha horizontal contendo os botões lado a lado
                    dbc.Col([
                        html.Div(
                            [
                                dbc.Button(
                                    str(ano),
                                    id={'type': 'btn-ano', 'index': ano},
                                    n_clicks=1, # Começa ativado (1 clique)
                                    color="primary", # Cor quando ativado
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
    [Output('ato-1', 'figure'), Output('ato-2', 'figure'), Output('ato-3', 'figure'),
     Output({'type': 'btn-ano', 'index': ALL}, 'color'),
     Output({'type': 'btn-ano', 'index': ALL}, 'outline')],
    [Input({'type': 'btn-ano', 'index': ALL}, 'n_clicks'), 
     Input('ato-1', 'clickData'), 
     Input('ato-2', 'clickData'), 
     Input('reset-eda', 'n_clicks')]
)
def update_eda_graphs(botoes_clicks, click1, click2, n_reset):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Lógica para descobrir quais anos estão ativos baseado nos cliques (ímpar = ativo, par = inativo)
    # Iniciamos com n_clicks=1 (ativo) para todos
    anos_selecionados = []
    cores_botoes = []
    outlines_botoes = []
    
    for idx, qtd_cliques in enumerate(botoes_clicks):
        ano_atual = anos_reais[idx]
        # Se os cliques forem ímpares, o botão está ATIVADO
        if (qtd_cliques % 2) != 0:
            anos_selecionados.append(ano_atual)
            cores_botoes.append("primary")   # Preenchido com a cor principal do tema
            outlines_botoes.append(False)   # Sem contorno transparente
        else:
            cores_botoes.append("secondary") # Cor neutra fosca para o modo desativado
            outlines_botoes.append(True)     # Estilo outline para denotar inatividade

    # Trava caso o usuário desative absolutamente todos os botões
    if not anos_selecionados:
        fig_aviso = px.bar(title="Por favor, selecione pelo menos um ano para renderizar os gráficos.")
        return fig_aviso, fig_aviso, fig_aviso, cores_botoes, outlines_botoes

    # Tratamento do botão de resetar filtros internos dos gráficos
    if triggered_id == 'reset-eda':
        click1 = click2 = None

    # Filtragem no DataFrame baseado nos botões ativos
    dff = df_geral[df_geral['Ano'].isin(anos_selecionados)]
    OKABE_ITO = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#56B4E9', '#CC79A7']
    
    # --- ATO 1 ---
    top5_cats = dff['GRUPOSERVICO_DESCRICAO'].value_counts().head(5).index
    df_top5 = dff[dff['GRUPOSERVICO_DESCRICAO'].isin(top5_cats)]
    vol_temporal = df_top5.groupby(['Mes', 'GRUPOSERVICO_DESCRICAO']).size().reset_index(name='Volume')
    
    fig1 = px.line(vol_temporal, x='Mes', y='Volume', color='GRUPOSERVICO_DESCRICAO', markers=True,
                  title='Ato 1: O Ciclo de Vida dos Maiores Problemas (Top 5 Categorias)', template='plotly_white',
                  color_discrete_sequence=OKABE_ITO, custom_data=['GRUPOSERVICO_DESCRICAO'])
    fig1.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1), height=420)

    # --- ATO 2 ---
    servico = click1['points'][0]['customdata'][0] if click1 else None
    df_a2 = df_top5[df_top5['GRUPOSERVICO_DESCRICAO'] == servico] if servico else df_top5
    status_g = ['PENDENTE', 'PREPARACAO', 'CADASTRADA']
    df_g = df_a2[df_a2['SITUACAO'].isin(status_g)]
    fila = df_g['BAIRRO'].value_counts().head(10).reset_index(name='Qtd')
    
    fig2 = px.bar(fila, x='Qtd', y='BAIRRO', orientation='h', color='Qtd', color_continuous_scale='Oranges',
                  title=f"Ato 2: Onde a Fila Trava? ({servico or 'Geral'})", template='plotly_white')
    fig2.update_layout(yaxis={'categoryorder':'total ascending'}, height=520)

    # --- ATO 3 ---
    bairros_f = [click2['points'][0]['y']] if click2 else fila.head(5)['BAIRRO'].tolist()
    df_f = df_a2[df_a2['BAIRRO'].isin(bairros_f)]
    
    if df_f.empty:
        fig3 = px.bar(title="Sem dados para gerar a taxa de ineficiência neste cenário")
    else:
        m = df_f.groupby(['BAIRRO', 'GRUPOSERVICO_DESCRICAO']).agg(
            T=('SITUACAO', 'count'), N=('SITUACAO', lambda x: x.isin(status_g).sum())).reset_index()
        m['Taxa'] = (m['N'] / m['T']) * 100
        fig3 = px.bar(m, x='BAIRRO', y='Taxa', color='GRUPOSERVICO_DESCRICAO', barmode='group', text_auto='.1f',
                     title="Ato 3: Qual serviço é mais negligenciado nos bairros em crise? (Taxa de Ineficiência)", 
                     color_discrete_sequence=OKABE_ITO, template='plotly_white')
        fig3.update_layout(yaxis_ticksuffix='%', height=520)
        fig3.update_traces(textposition='outside', textfont_size=11)

    # Retornamos os 3 gráficos normais + as cores e configurações visuais dinâmicas dos botões
    return fig1, fig2, fig3, cores_botoes, outlines_botoes

if __name__ == "__main__":
    app.run(debug=True, port=8055)