import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import os

# IMPORTAÇÃO DOS NOSSOS MÓDULOS DE PÁGINAS
from home import render_home
from eda import render_eda
from correlacao import render_correlacao
from ml1 import render_ml1
from ml2 import render_ml2
from cluster import render_cluster

# ==========================================
# 1. CARREGAMENTO E LIMPEZA DINÂMICA
# ==========================================
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
CAMINHO_PARQUET = os.path.join(DIRETORIO_ATUAL, '../data/df_ml1.parquet')

if os.path.exists(CAMINHO_PARQUET):
    df_geral = pd.read_parquet(CAMINHO_PARQUET, engine='pyarrow')
    
    if 'GRUPOSERVICO_DESCRICAO' in df_geral.columns:
        df_geral['GRUPOSERVICO_DESCRICAO'] = (
            df_geral['GRUPOSERVICO_DESCRICAO']
            .str.replace(r'CALÃ.ADAS', 'CALÇADAS', regex=True)
            .str.replace(r'PRAÃ.AS', 'PRAÇAS', regex=True)
            .str.replace(r'LUMINÃ.RIAS', 'LUMINÁRIAS', regex=True)
            .str.replace(r'AÃ.Ã.O', 'AÇÃO', regex=True)
        )

    df_geral['DATA_DEMANDA'] = pd.to_datetime(df_geral['DATA_DEMANDA'], errors='coerce')
    df_geral['Ano'] = df_geral['DATA_DEMANDA'].dt.year
    df_geral['Mes'] = df_geral['DATA_DEMANDA'].dt.month
    
    df_geral = df_geral[(df_geral['Ano'] >= 2020) & (df_geral['Ano'] <= 2026)]
    anos_reais = sorted([int(x) for x in df_geral['Ano'].dropna().unique()])
    total_registros = f"{len(df_geral):,}".replace(",", ".")
else:
    df_geral = pd.DataFrame(columns=['Ano', 'BAIRRO', 'GRUPOSERVICO_DESCRICAO', 'SITUACAO', 'Mes'])
    anos_reais = []
    total_registros = "0"

# ==========================================
# 2. CONFIGURAÇÃO DO APP E NAVBAR
# ==========================================
app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.FLATLY, 
        "https://use.fontawesome.com/releases/v6.5.1/css/all.css"
    ], 
    suppress_callback_exceptions=True
)
app.title = "REPORT! Dashboard"

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Página Inicial", href="/")),
        dbc.DropdownMenu([
            dbc.DropdownMenuItem("Os 3 Atos", href="/eda"),
            dbc.DropdownMenuItem("Correlações", href="/correlacao")
        ], label="EDA", nav=True),

        dbc.DropdownMenu([
            dbc.DropdownMenuItem("Machine Learning 1", href="/ml1"),
            dbc.DropdownMenuItem("Machine Learning 2", href="/ml2")
        ], label="Machine Learning", nav=True),
        dbc.NavItem(dbc.NavLink("Clusterização", href="/cluster")),
    ],
    brand="REPORT!",
    brand_href="/",
    color="primary",
    dark=True,
    fluid=True,
    className="mb-4 shadow"
)

content = dbc.Container(id="page-content", fluid=True)
app.layout = html.Div([dcc.Location(id="url"), navbar, content])

# ==========================================
# 3. ROTEADOR DE PÁGINAS (CALLBACK)
# ==========================================
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def navigate(path):
    if path == "/eda": return render_eda(anos_reais, df_geral)
    if path == "/correlacao": return render_correlacao(df_geral)
    if path == "/ml1": return render_ml1()
    if path == "/ml2": return render_ml2()
    if path == "/cluster": return render_cluster(df_geral)
    return render_home(total_registros)

# ==========================================
# 4. CALLBACK DO PIPELINE EDA (OS 3 ATOS)
# ==========================================
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
    anos_selecionados = []
    cores_botoes = []
    outlines_botoes = []
    
    for idx, qtd_cliques in enumerate(botoes_clicks):
        ano_atual = anos_reais[idx]
        if (qtd_cliques % 2) != 0:
            anos_selecionados.append(ano_atual)
            cores_botoes.append("primary")
            outlines_botoes.append(False)
        else:
            cores_botoes.append("secondary")
            outlines_botoes.append(True)

    if not anos_selecionados:
        fig_aviso = px.bar(title="Selecione pelo menos um ano para renderizar os gráficos.")
        return fig_aviso, fig_aviso, fig_aviso, cores_botoes, outlines_botoes

    if triggered_id == 'reset-eda':
        click1 = click2 = None

    dff = df_geral[df_geral['Ano'].isin(anos_selecionados)]
    OKABE_ITO = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#56B4E9', '#CC79A7']
    
    top5_cats = dff['GRUPOSERVICO_DESCRICAO'].value_counts().head(5).index
    df_top5 = dff[dff['GRUPOSERVICO_DESCRICAO'].isin(top5_cats)]
    vol_temporal = df_top5.groupby(['Mes', 'GRUPOSERVICO_DESCRICAO']).size().reset_index(name='Volume')
    
    fig1 = px.line(vol_temporal, x='Mes', y='Volume', color='GRUPOSERVICO_DESCRICAO', markers=True,
                  title='Ato 1: O Ciclo de Vida dos Maiores Problemas (Top 5 Categorias)', template='plotly_white',
                  color_discrete_sequence=OKABE_ITO, custom_data=['GRUPOSERVICO_DESCRICAO'])
    fig1.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1), height=420)

    servico = click1['points'][0]['customdata'][0] if click1 else None
    df_a2 = df_top5[df_top5['GRUPOSERVICO_DESCRICAO'] == servico] if servico else df_top5
    status_g = ['PENDENTE', 'PREPARACAO', 'CADASTRADA']
    df_g = df_a2[df_a2['SITUACAO'].isin(status_g)]
    fila = df_g['BAIRRO'].value_counts().head(10).reset_index(name='Qtd')
    
    fig2 = px.bar(fila, x='Qtd', y='BAIRRO', orientation='h', color='BAIRRO',
                  color_discrete_sequence=OKABE_ITO,
                  title=f"Ato 2: Onde a Fila Trava? ({servico or 'Geral'})", template='plotly_white')
    fig2.update_layout(yaxis={'categoryorder':'total ascending'}, height=520, showlegend=False)

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

    return fig1, fig2, fig3, cores_botoes, outlines_botoes

# ==========================================
# 5. LÓGICA DO PAINEL CUSTOMIZÁVEL
# ==========================================

def abreviar_nome(nome):
    """Regra Nova: Nome simples (3 letras). Nome composto (1ª Letra . 1ª Letra)."""
    if not nome or nome == "LIMITE": return ""
    palavras = str(nome).replace("-", " ").split()
    if len(palavras) == 1:
        return palavras[0][:3].upper()
    else:
        # Pega a primeira letra da primeira palavra e a primeira da segunda
        return palavras[0][0].upper() + "." + palavras[1][0].upper()

@app.callback(
    Output('bairro-badges', 'children'),
    Input('bairro-autocomplete', 'value')
)
def render_bairro_badges(selecionados):
    if not selecionados: return []
    return [
        dbc.Badge(abreviar_nome(b), color="primary", className="px-3 py-2 rounded-pill shadow-sm fs-6") 
        for b in selecionados if b != "LIMITE"
    ]

@app.callback(
    Output('servico-badges', 'children'),
    Input('servico-autocomplete', 'value')
)
def render_servico_badges(selecionados):
    if not selecionados: return []
    return [
        dbc.Badge(abreviar_nome(s), color="success", className="px-3 py-2 rounded-pill shadow-sm fs-6") 
        for s in selecionados
    ]

@app.callback(
    Output('servico-autocomplete', 'disabled'),
    Input('modo-servico', 'value')
)
def toggle_caixa_servicos(modo):
    return modo == 'todos'

@app.callback(
    Output("bairro-autocomplete", "options"),
    Input("bairro-autocomplete", "search_value"),
    State("bairro-autocomplete", "value")
)
def update_bairro_autocomplete(search_value, selecionados):
    if not search_value: search_value = ""
    
    opcoes_atuais = [{'label': str(b), 'value': str(b)} for b in selecionados if b != "LIMITE"] if selecionados else []
    
    if selecionados and len(selecionados) >= 5:
        return opcoes_atuais + [{"label": "⚠️ Limite de 5 bairros atingido", "value": "LIMITE", "disabled": True}]
        
    # Sorted aplica ordem alfabética na base de bairros
    todos_bairros = sorted(df_geral['BAIRRO'].dropna().unique())
    buscas_encontradas = [{'label': str(b), 'value': str(b)} for b in todos_bairros if search_value.upper() in str(b).upper()]
    
    valores_atuais = [opt['value'] for opt in opcoes_atuais]
    resultado_final = opcoes_atuais + [b for b in buscas_encontradas if b['value'] not in valores_atuais]
    
    # Aumentado para 100 resultados renderizados (como não há mais limite de 3 letras, a lista será completa)
    return resultado_final[:100] 

@app.callback(
    Output("servico-autocomplete", "options"),
    Input("servico-autocomplete", "search_value"),
    State("servico-autocomplete", "value")
)
def update_servico_autocomplete(search_value, selecionados):
    if not search_value: search_value = ""
    
    opcoes_atuais = [{'label': str(s), 'value': str(s)} for s in selecionados] if selecionados else []
    
    # Sorted aplica ordem alfabética na base de serviços
    todos_servicos = sorted(df_geral['GRUPOSERVICO_DESCRICAO'].dropna().unique())
    buscas_encontradas = [{'label': str(s), 'value': str(s)} for s in todos_servicos if search_value.upper() in str(s).upper()]
    
    valores_atuais = [opt['value'] for opt in opcoes_atuais]
    resultado_final = opcoes_atuais + [s for s in buscas_encontradas if s['value'] not in valores_atuais]
    return resultado_final[:100] 

@app.callback(
    Output('custom-top-chart', 'figure'),
    [Input({'type': 'btn-ano', 'index': ALL}, 'n_clicks'),
     Input('bairro-autocomplete', 'value'),
     Input('servico-autocomplete', 'value'),
     Input('modo-servico', 'value'),
     Input('metrica-analise', 'value')]
)
def render_grafico_personalizado(botoes_clicks, bairros_selecionados, servicos_selecionados, modo_servico, metrica):
    anos_selecionados = [anos_reais[idx] for idx, qtd in enumerate(botoes_clicks) if (qtd % 2) != 0]

    if not bairros_selecionados or "LIMITE" in bairros_selecionados:
        bairros_selecionados = [b for b in (bairros_selecionados or []) if b != "LIMITE"]
        if not bairros_selecionados:
            return px.bar(title="Aguardando seleção... Escolha ao menos um bairro para gerar a análise.")

    df_filtrado = df_geral[(df_geral['Ano'].isin(anos_selecionados)) & (df_geral['BAIRRO'].isin(bairros_selecionados))]

    if modo_servico == 'manual':
        if not servicos_selecionados: return px.bar(title="Selecione pelo menos um tipo de serviço.")
        df_filtrado = df_filtrado[df_filtrado['GRUPOSERVICO_DESCRICAO'].isin(servicos_selecionados)]
    elif modo_servico == 'exceto':
        if servicos_selecionados: df_filtrado = df_filtrado[~df_filtrado['GRUPOSERVICO_DESCRICAO'].isin(servicos_selecionados)]

    if df_filtrado.empty: return px.bar(title="Nenhum dado encontrado.")

    # Engenharia de Dados 
    status_gargalo = ['PENDENTE', 'PREPARACAO', 'CADASTRADA']
    df_agrupado = df_filtrado.groupby(['Mes', 'BAIRRO']).agg(
        Total=('SITUACAO', 'count'),
        Pendentes=('SITUACAO', lambda x: x.isin(status_gargalo).sum()),
        Resolvidas=('SITUACAO', lambda x: (~x.isin(status_gargalo)).sum())
    ).reset_index()

    df_agrupado['Perc_Pendentes'] = (df_agrupado['Pendentes'] / df_agrupado['Total']) * 100
    df_agrupado['Perc_Resolvidas'] = (df_agrupado['Resolvidas'] / df_agrupado['Total']) * 100
    df_agrupado['ISO'] = df_agrupado['Pendentes'] / df_agrupado['Resolvidas'].replace(0, 1)

    OKABE_ITO = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7']
    
    if metrica == 'total':
        eixo_y = 'Total'
        titulo = "Volume Absoluto de Ocorrências no Período"
        y_title = "Quantidade Absoluta"
        hover_template = '<b>%{fullData.name}</b><br>Mês: %{x}<br>Volume: %{y} queixas<extra></extra>'
        custom_data = None
    elif metrica == 'resolvidas':
        eixo_y = 'Resolvidas'
        titulo = "Eficácia: Obras e Serviços Resolvidos"
        y_title = "Demandas Resolvidas"
        custom_data = ['Total', 'Perc_Resolvidas']
        hover_template = '<b>%{fullData.name}</b><br>Mês: %{x}<br>Resolvidas: %{y}<br>Total: %{customdata[0]}<br>Sucesso: %{customdata[1]:.1f}%<extra></extra>'
    elif metrica == 'pendentes':
        eixo_y = 'Pendentes'
        titulo = "O Gargalo Operacional: Serviços Pendentes/Atrasados"
        y_title = "Demandas na Fila"
        custom_data = ['Total', 'Perc_Pendentes']
        hover_template = '<b>%{fullData.name}</b><br>Mês: %{x}<br>Pendentes: %{y}<br>Total: %{customdata[0]}<br>Taxa Crítica: %{customdata[1]:.1f}%<extra></extra>'
    elif metrica == 'iso':
        eixo_y = 'ISO'
        titulo = "Termômetro (ISO): Relação Problemas vs. Soluções"
        y_title = "Índice ( > 1.0 = Perigo )"
        hover_template = '<b>%{fullData.name}</b><br>Mês: %{x}<br>ISO: %{y:.2f}<br>(Nota: Acima de 1.0 a fila acumulou)<extra></extra>'
        custom_data = None

    fig = px.line(
        df_agrupado, x='Mes', y=eixo_y, color='BAIRRO', markers=True,
        title=titulo, template='plotly_white', custom_data=custom_data,
        color_discrete_sequence=OKABE_ITO
    )
    
    if metrica == 'iso':
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", opacity=0.5, annotation_text="Limite de Colapso (>1.0)")

    fig.update_traces(hovertemplate=hover_template)
    
    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=1, dtick=1, title="Mês do Ano"),
        yaxis_title=y_title, height=380, margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title="")
    )
    
    return fig

# ==========================================
# 6. EXECUÇÃO DO SERVIDOR
# ==========================================
if __name__ == "__main__":
    app.run(debug=True, port=8055)