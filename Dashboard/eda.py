import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.express as px
import pandas as pd

# ============================================================
# PALETA OKABE-ITO — Padrão científico para acessibilidade a
# daltônicos (adotada pela revista Nature). 8 cores com máximo
# contraste entre si para todos os tipos de daltonismo.
# ============================================================
OKABE_ITO = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#56B4E9', '#CC79A7', '#F0E442', '#000000']

# Mapeamento de Dia_Semana_Num → nome do dia (para quando o parquet não tem Dia_Semana textual)
_DIAS_NUM = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

def render_eda(anos_reais, df_all=None):
    # ─── PROTEÇÃO 1: Verifica se o DataFrame foi fornecido ───
    if df_all is None or (isinstance(df_all, pd.DataFrame) and df_all.empty):
        return dbc.Container([
            dbc.Alert([
                html.H4("Dados não encontrados!", className="alert-heading fw-bold"),
                html.P("O DataFrame 'df_all' está vazio ou não foi carregado corretamente no arquivo principal (app.py/index.py)."),
                html.Hr(),
                html.P("Verifique se o caminho do seu arquivo Parquet ou CSV está correto e se a leitura dos dados foi bem-sucedida.", className="mb-0 small")
            ], color="danger", className="mt-5 shadow rounded-3")
        ])

    # ─── PROTEÇÃO 2: Evita quebras por estrutura ou colunas ausentes ───
    try:
        df_all = df_all.copy()
        
        # Garante a existência da coluna essencial para o funcionamento dos gráficos
        if 'DATA_DEMANDA' not in df_all.columns:
            raise KeyError("A coluna essencial 'DATA_DEMANDA' não foi encontrada no conjunto de dados informado.")
            
        df_all['DATA_DEMANDA'] = pd.to_datetime(df_all['DATA_DEMANDA'], errors='coerce')

        # Garante que Mes exists
        if 'Mes' not in df_all.columns:
            df_all['Mes'] = df_all['DATA_DEMANDA'].dt.month

        # Garante que Dia_Semana (texto) existe
        if 'Dia_Semana' not in df_all.columns:
            if 'Dia_Semana_Num' in df_all.columns:
                df_all['Dia_Semana'] = df_all['Dia_Semana_Num'].map(_DIAS_NUM)
            else:
                df_all['Dia_Semana'] = df_all['DATA_DEMANDA'].dt.day_name()

    except Exception as e:
        # Se qualquer conversão acima falhar, captura o erro interno e impede o travamento do Dash
        return dbc.Container([
            dbc.Alert([
                html.H4("Erro de Processamento de Dados", className="alert-heading fw-bold"),
                html.P(f"Ocorreu uma falha ao estruturar as colunas para os gráficos:"),
                html.Code(str(e), className="d-block my-2 p-2 bg-dark text-warning rounded text-start"),
                html.P("Verifique se as colunas do seu arquivo coincidem com os nomes mapeados pelo script de análise.", className="mb-0 small")
            ], color="warning", className="mt-5 shadow rounded-3")
        ])

    # Criando uma versão customizada e mais escura da paleta Oranges (remove os tons muito claros)
    PALETA_LARANJA_ESCURA = px.colors.sequential.Oranges[3:]

    # ──────────────────────────────────────────
    # FIG 1 — Evolução do Volume Total de Denúncias
    # ──────────────────────────────────────────
    df_filtrado = df_all[(df_all['DATA_DEMANDA'] >= '2020-01-01') & (df_all['DATA_DEMANDA'] <= '2025-12-31')].copy()
    vol_mensal = df_filtrado['DATA_DEMANDA'].dt.to_period('M').value_counts().sort_index().reset_index()
    vol_mensal.columns = ['Ano', 'Volume de Denúncias']

    if len(vol_mensal) > 0:
        vol_mensal = vol_mensal.iloc[:-1]

    vol_mensal['Ano'] = vol_mensal['Ano'].dt.to_timestamp()

    fig1 = px.line(
        vol_mensal, x='Ano', y='Volume de Denúncias',
        title='1. Evolução do Volume Total de Denúncias', markers=True, template='plotly_white'
    )
    fig1.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000],
            ticktext=['1k', '2k', '3k', '4k', '5k', '6k', '7k', '8k', '9k', '10k', '11k', '12k', '13k', '14k', '15k'],
            rangemode='tozero', title='Volume'
        ),
        xaxis=dict(dtick="M12", tickformat="%Y", ticklabelmode="period", title=None, showgrid=False)
    )

    # ──────────────────────────────────────────
    # FIG 2 — Sazonalidade das Categorias (Top 5 Serviços por Mês)
    # ──────────────────────────────────────────
    top5_cat = df_all['GRUPOSERVICO_DESCRICAO'].value_counts().head(5).index
    df_top5_cat = df_all[df_all['GRUPOSERVICO_DESCRICAO'].isin(top5_cat)]
    vol_mes_cat = df_top5_cat.groupby(['Mes', 'GRUPOSERVICO_DESCRICAO']).size().reset_index(name='Volume')

    fig2 = px.bar(
        vol_mes_cat, x='Mes', y='Volume', color='GRUPOSERVICO_DESCRICAO',
        barmode='group', title='2. Sazonalidade das Categorias (Top 5 Serviços por Mês)',
        labels={'Mes': 'Mês do Ano (1 a 12)', 'GRUPOSERVICO_DESCRICAO': 'Categoria'},
        template='plotly_white', color_discrete_sequence=OKABE_ITO
    )

    # ──────────────────────────────────────────
    # FIG 3 — Heatmap: Dias da Semana vs Meses do Ano
    # ──────────────────────────────────────────
    heatmap_data = df_all.groupby(['Dia_Semana', 'Mes']).size().reset_index(name='Volume')
    ordem_dias = ['Sunday', 'Saturday', 'Friday', 'Thursday', 'Wednesday', 'Tuesday', 'Monday']

    fig3 = px.density_heatmap(
        heatmap_data, x='Mes', y='Dia_Semana', z='Volume',
        title='3. Padrão de Acessos: Dias da Semana vs Meses do Ano',
        category_orders={'Dia_Semana': ordem_dias},
        color_continuous_scale='Cividis',
        labels={'Mes': 'Mês do Ano', 'Dia_Semana': 'Dia da Semana'},
        template='plotly_white'
    )

    # ──────────────────────────────────────────
    # FIG 4 — Top 10 Bairros Críticos (CORRIGIDO: Legenda começa do 0 + Barras Escuras)
    # ──────────────────────────────────────────
    top10_bairros = df_all['BAIRRO'].value_counts().head(10).reset_index()
    top10_bairros.columns = ['Bairro', 'Volume']

    fig4 = px.bar(
        top10_bairros, x='Volume', y='Bairro', orientation='h',
        title='4. Ranking de Volume: Top 10 Bairros Críticos',
        color='Volume', color_continuous_scale=PALETA_LARANJA_ESCURA,
        template='plotly_white'
    )
    fig4.update_layout(
        yaxis={'categoryorder': 'total ascending'}, 
        showlegend=False, 
        coloraxis_showscale=True,
        coloraxis_colorbar=dict(
            title="Volume",
            thickness=15,
            len=0.8
        )
    )

    # ──────────────────────────────────────────
    # FIG 5 — Top 20 Bairros (Treemap)
    # ──────────────────────────────────────────
    top20_bairros = df_all['BAIRRO'].value_counts().head(20).reset_index()
    top20_bairros.columns = ['Bairro', 'Volume']

    fig5 = px.treemap(
        top20_bairros, path=['Bairro'], values='Volume',
        title='5. Representatividade Visual Espacial: Top 20 Bairros',
        color='Volume', color_continuous_scale='Oranges'
    )
    fig5.update_layout(coloraxis_showscale=False)

    # ──────────────────────────────────────────
    # FIG 6 — Proporção de Resoluções nos 5 Bairros Críticos
    # ──────────────────────────────────────────
    top5_bairros_nomes = df_all['BAIRRO'].value_counts().head(5).index
    df_bairros_criticos = df_all[df_all['BAIRRO'].isin(top5_bairros_nomes)]
    df_prop = df_bairros_criticos.groupby(['BAIRRO', 'SITUACAO']).size().reset_index(name='Contagem')
    df_prop['Porcentagem'] = df_prop.groupby('BAIRRO')['Contagem'].transform(lambda x: x / x.sum() * 100)

    fig6 = px.bar(
        df_prop, x='BAIRRO', y='Porcentagem', color='SITUACAO',
        title='6. Proporção de Resoluções nos 5 Bairros Críticos',
        labels={'Porcentagem': 'Porcentagem (%)', 'BAIRRO': 'Bairro'},
        barmode='stack', color_discrete_sequence=OKABE_ITO, template='plotly_white'
    )

    # ──────────────────────────────────────────
    # FIG 7 — Top 10 Vias Mais Críticas (CORRIGIDO: Sem valores negativos + Barras Escuras)
    # ──────────────────────────────────────────
    if 'LOGRADOURO' in df_all.columns:
        top10_vias = df_all['LOGRADOURO'].value_counts().head(10).reset_index()
        top10_vias.columns = ['Logradouro', 'Quantidade_Defeitos']

        fig_vias = px.bar(
            top10_vias, x='Quantidade_Defeitos', y='Logradouro', orientation='h',
            title='7. Top 10 Vias Mais Críticas (Maior Número de Defeitos)',
            labels={'Quantidade_Defeitos': 'Nº de Ocorrências', 'Logradouro': 'Via/Logradouro'},
            color='Quantidade_Defeitos', color_continuous_scale=PALETA_LARANJA_ESCURA,
            template='plotly_white'
        )
        fig_vias.update_layout(
            yaxis={'categoryorder': 'total ascending', 'tickangle': 0},
            xaxis={'tickangle': 0}, yaxis_title=None, showlegend=False, height=500,
            coloraxis_showscale=False
        )
    else:
        fig_vias = None

    # ──────────────────────────────────────────
    # FIG 8 — Balanço de Eficiência Pública (Rosca)
    # ──────────────────────────────────────────
    status_balanco = df_all['SITUACAO'].value_counts().reset_index()
    status_balanco.columns = ['Situacao', 'Total']

    fig8 = px.pie(
        status_balanco, values='Total', names='Situacao', hole=0.5,
        title='8. Balanço de Eficiência Pública (Gráfico de Rosca)',
        color_discrete_sequence=OKABE_ITO, labels={'Situacao': 'Status', 'Total': 'Chamados'}
    )
    fig8.update_traces(
        textinfo='percent+label',
        pull=[0.1 if c == 'PENDENTE' else 0 for c in status_balanco['Situacao']]
    )

    # ──────────────────────────────────────────
    # FIG 9 — Detalhamento Crítico (Barras em azul único)
    # ──────────────────────────────────────────
    if 'SERVICO_DESCRICAO' in df_all.columns:
        maior_grupo = df_all['GRUPOSERVICO_DESCRICAO'].value_counts().idxmax()
        df_detalhe = df_all[df_all['GRUPOSERVICO_DESCRICAO'] == maior_grupo]
        top10_servicos = df_detalhe['SERVICO_DESCRICAO'].value_counts().head(10).reset_index()
        top10_servicos.columns = ['Serviço', 'Volume']
        fig9 = px.bar(
            top10_servicos, x='Volume', y='Serviço', orientation='h',
            title=f'9. Detalhamento Crítico: Principais Queixas em "{maior_grupo}"',
            color_discrete_sequence=['#0072B2'],
            labels={'Volume': 'Qtd. Ocorrências', 'Serviço': 'Subcategoria de Serviço'},
            template='plotly_white'
        )
        fig9.update_layout(
            yaxis={'categoryorder': 'total ascending', 'tickangle': 0},
            xaxis={'tickangle': 0}, yaxis_title=None, showlegend=False, height=500,
            coloraxis_showscale=False
        )
    else:
        fig9 = None

    # ──────────────────────────────────────────
    # FIG 10 — Identidade Urbana: Heatmap Bairros vs Categorias
    # ──────────────────────────────────────────
    top_bairros = df_all['BAIRRO'].value_counts().head(10).index
    top_categorias = df_all['GRUPOSERVICO_DESCRICAO'].value_counts().head(10).index
    df_identidade = df_all[
        (df_all['BAIRRO'].isin(top_bairros)) &
        (df_all['GRUPOSERVICO_DESCRICAO'].isin(top_categorias))
    ]
    matriz_identidade = pd.crosstab(df_identidade['BAIRRO'], df_identidade['GRUPOSERVICO_DESCRICAO'])

    fig10 = px.imshow(
        matriz_identidade,
        labels=dict(x="Categoria do Problema", y="Bairro", color="Volume"),
        title='10. Identidade Urbana: Bairros vs. Principais Categorias',
        color_continuous_scale='Oranges', text_auto=',.0f', aspect="auto", height=600
    )
    fig10.update_layout(xaxis_tickangle=-45, margin=dict(b=120))

    # ──────────────────────────────────────────
    # FIG TEMPO — Gráfico 14: Tempo Médio de Resolução
    # ──────────────────────────────────────────
    if 'DATA_ULT_SITUACAO' in df_all.columns:
        resolvidos = df_all[df_all['SITUACAO'] == 'ATENDIDA'].copy()
        resolvidos['DATA_DEMANDA'] = pd.to_datetime(resolvidos['DATA_DEMANDA'], errors='coerce')
        resolvidos['DATA_ULT_SITUACAO'] = pd.to_datetime(resolvidos['DATA_ULT_SITUACAO'], errors='coerce')
        resolvidos = resolvidos.dropna(subset=['DATA_DEMANDA', 'DATA_ULT_SITUACAO'])
        resolvidos['TEMPO_DIAS'] = (resolvidos['DATA_ULT_SITUACAO'] - resolvidos['DATA_DEMANDA']).dt.days
        resolvidos = resolvidos[resolvidos['TEMPO_DIAS'] >= 0]
        if 'DENUNCIAS' in df_all['GRUPOSERVICO_DESCRICAO'].values:
            resolvidos = resolvidos[resolvidos['GRUPOSERVICO_DESCRICAO'] != 'DENUNCIAS']
        tempo_medio = resolvidos.groupby('GRUPOSERVICO_DESCRICAO')['TEMPO_DIAS'].mean().reset_index()
        tempo_medio['TEMPO_DIAS'] = tempo_medio['TEMPO_DIAS'].round(0)
        top10_lentos = tempo_medio.sort_values(by='TEMPO_DIAS', ascending=False).head(10)
        
        min_valor = top10_lentos['TEMPO_DIAS'].min()

        fig_tempo = px.bar(
            top10_lentos, x='TEMPO_DIAS', y='GRUPOSERVICO_DESCRICAO', orientation='h',
            title='14. Tempo Médio de Resolução da Prefeitura', text='TEMPO_DIAS',
            color='TEMPO_DIAS', color_continuous_scale='Oranges',
            color_continuous_midpoint=min_valor - 5,
            labels={'TEMPO_DIAS': '', 'GRUPOSERVICO_DESCRICAO': ''}, template='plotly_white'
        )
        fig_tempo.update_layout(
            yaxis={'categoryorder': 'total ascending'}, height=500,
            margin=dict(r=120), showlegend=False,
            xaxis=dict(showgrid=True, showticklabels=True, title=None, dtick=200),
            yaxis_title=None, 
            coloraxis_showscale=False
        )
        fig_tempo.update_traces(
            textposition='outside', texttemplate='<b>%{text} dias</b>', cliponaxis=False
        )
    else:
        fig_tempo = None

    # ══════════════════════════════════════════════════
    # LAYOUT DO DASHBOARD
    # ══════════════════════════════════════════════════
    return dbc.Container([

        # --- FILTRO TEMPORAL ---
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(
                        html.H5("Filtro Temporal dos 3 Atos:", className="m-0 fw-bold text-secondary"),
                        width="auto", className="d-flex align-items-center"
                    ),
                    dbc.Col([
                        html.Div(
                            [
                                dbc.Button(
                                    str(ano),
                                    id={'type': 'btn-ano', 'index': ano},
                                    n_clicks=1,
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
                        width="auto", className="d-flex align-items-center"
                    )
                ], className="align-items-center")
            ])
        ], className="mb-4 shadow-sm border-0 bg-light"),

        # ==========================================
        # GRÁFICO CUSTOMIZÁVEL (Motor de BI Avançado)
        # ==========================================
        dbc.Card([
            dbc.CardHeader(html.H5("Motor de BI: Cruzamento e Auditoria sob Demanda", className="mb-0 fw-bold text-white"), className="bg-info"),
            dbc.CardBody([
                dbc.Row([
                    # Container 1: BAIRRO
                    dbc.Col([
                        html.Div([
                            html.Label("Bairro", className="fw-bold text-primary mb-2 fs-5"),
                            dcc.Dropdown(
                                id='bairro-autocomplete', multi=True, searchable=True,
                                placeholder="Selecionar bairros (Máx 5)...", options=[], className="mb-2"
                            ),
                            html.Div(id='bairro-badges', className="d-flex flex-wrap gap-2 mt-3")
                        ], className="shadow rounded-4 p-4 bg-white border-0 h-100")
                    ], width=12, md=4, className="mb-4 mb-md-0"),

                    # Container 2: DENÚNCIA
                    dbc.Col([
                        html.Div([
                            html.Label("Denúncia", className="fw-bold text-success mb-2 fs-5"),
                            dbc.RadioItems(
                                id='modo-servico',
                                options=[
                                    {'label': 'Manual', 'value': 'manual'},
                                    {'label': 'Todos', 'value': 'todos'},
                                    {'label': 'Exceto', 'value': 'exceto'}
                                ],
                                value='todos', inline=True,
                                className="mb-2 small fw-bold text-secondary"
                            ),
                            dcc.Dropdown(
                                id='servico-autocomplete', multi=True, searchable=True,
                                placeholder="Selecionar denúncias...", options=[], disabled=True, className="mb-2"
                            ),
                            html.Div(id='servico-badges', className="d-flex flex-wrap gap-2 mt-3")
                        ], className="shadow rounded-4 p-4 bg-white border-0 h-100")
                    ], width=12, md=4, className="mb-4 mb-md-0"),

                    # Container 3: MÉTRICA
                    dbc.Col([
                        html.Div([
                            html.Label("Métrica", className="fw-bold text-dark mb-2 fs-5"),
                            dcc.Dropdown(
                                id='metrica-analise',
                                options=[
                                    {'label': 'Volume Absoluto', 'value': 'total'},
                                    {'label': 'Resolvidas (Qtd e %)', 'value': 'resolvidas'},
                                    {'label': 'Pendentes (Gargalo)', 'value': 'pendentes'},
                                    {'label': 'Termômetro (ISO)', 'value': 'iso'}
                                ],
                                value='total', clearable=False, className="mt-4"
                            )
                        ], className="shadow rounded-4 p-4 bg-white border-0 h-100")
                    ], width=12, md=4)
                ], className="mb-5 mt-2"),

                dcc.Graph(id='custom-top-chart', config={'displayModeBar': False})
            ], className="bg-light")
        ], className="shadow border-0 mb-5 border-start border-info border-5"),

        # --- OS 3 ATOS ---
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id='ato-1'))], className="mb-4 shadow border-0"), width=12),
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id='ato-2'))], className="mb-4 shadow border-0"), width=12),
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id='ato-3'))], className="mb-4 shadow border-0"), width=12),
        ]),

        html.Hr(className="my-5"),

        html.H3("Visão Geral e Comportamento Espacial", className="text-primary mb-4 fw-bold"),
        html.P("Análise estática do montante de dados históricos da zeladoria urbana.", className="text-secondary mb-4"),

        # Linha 1: Evolução do volume
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig1))], className="mb-4 shadow border-0"), width=12),
        ]),

        # Linha 2: Sazonalidade + Heatmap Dias da Semana
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig2))], className="mb-4 shadow border-0"), width=12, md=6),
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig3))], className="mb-4 shadow border-0"), width=12, md=6),
        ]),

        # Linha 3: Top 10 Bairros + Top 20 Treemap
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig4))], className="mb-4 shadow border-0"), width=12, md=6),
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig5))], className="mb-4 shadow border-0"), width=12, md=6),
        ]),

        # Linha 4: Proporção de Resoluções nos Bairros Críticos
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig6))], className="mb-4 shadow border-0"), width=12),
        ]),

        # Linha 5: Top 10 Vias + Rosca de Eficiência
        dbc.Row([
            *(
                [dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig_vias))], className="mb-4 shadow border-0"), width=12, md=7)]
                if fig_vias is not None else []
            ),
            dbc.Col(
                dbc.Card([dbc.CardBody(dcc.Graph(figure=fig8))], className="mb-4 shadow border-0"),
                width=12, md=5 if fig_vias is not None else 12
            ),
        ]),

        # Linha 6: Detalhamento do Maior Grupo
        *(
            [dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig9))], className="mb-4 shadow border-0"), width=12),
            ])]
            if fig9 is not None else []
        ),

        # Linha 7: Heatmap Bairros vs Categorias
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig10))], className="mb-4 shadow border-0"), width=12),
        ]),

        # Linha 8: Tempo Médio de Resolução
        *(
            [dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig_tempo))], className="mb-4 shadow border-0"), width=12),
            ])]
            if fig_tempo is not None else []
        ),

    ], fluid=True)
