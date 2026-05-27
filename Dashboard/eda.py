import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.express as px
import pandas as pd
import unicodedata

# ============================================================
# PALETA OKABE-ITO — Padrão científico para acessibilidade a
# daltônicos (adotada pela revista Nature). 8 cores com máximo
# contraste entre si para todos os tipos de daltonismo.
# ============================================================
OKABE_ITO = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#56B4E9', '#CC79A7', '#F0E442', '#000000']


def remover_acentos(texto):
    if not isinstance(texto, str):
        return texto
    texto = texto.strip().upper()
    nfkd = unicodedata.normalize('NFD', texto)
    return ''.join(c for c in nfkd if unicodedata.category(c) != 'Mn')


def render_eda(anos_reais, df_all=None):
    if df_all is None:
        return dbc.Container([html.H3("Aguardando carregamento dos dados...")])

    df_all = df_all.copy()
    df_all['DATA_DEMANDA'] = pd.to_datetime(df_all['DATA_DEMANDA'], errors='coerce')

    # ------------------------------------------------------------------
    # FIG 1 — Evolução do volume total de denúncias
    # ------------------------------------------------------------------
    df_filtrado = df_all[
        (df_all['DATA_DEMANDA'] >= '2020-01-01') & (df_all['DATA_DEMANDA'] <= '2025-12-31')
    ].copy()
    vol_mensal = df_filtrado['DATA_DEMANDA'].dt.to_period('M').value_counts().sort_index().reset_index()
    vol_mensal.columns = ['Ano', 'Volume de Denúncias']
    if len(vol_mensal) > 0:
        vol_mensal = vol_mensal.iloc[:-1]
    vol_mensal['Ano'] = vol_mensal['Ano'].dt.to_timestamp()

    fig1 = px.line(
        vol_mensal, x='Ano', y='Volume de Denúncias',
        title='1. Evolução do Volume Total de Denúncias', markers=True, template='plotly_white',
        color_discrete_sequence=[OKABE_ITO[0]]
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

    # ------------------------------------------------------------------
    # FIG 2 — Sazonalidade das categorias (Top 5 Serviços por Mês)
    # ------------------------------------------------------------------
    top5_cat = df_all['GRUPOSERVICO_DESCRICAO'].value_counts().head(5).index
    df_top5_cat = df_all[df_all['GRUPOSERVICO_DESCRICAO'].isin(top5_cat)]
    vol_mes_cat = df_top5_cat.groupby(['Mes', 'GRUPOSERVICO_DESCRICAO']).size().reset_index(name='Volume')

    fig2 = px.bar(
        vol_mes_cat, x='Mes', y='Volume', color='GRUPOSERVICO_DESCRICAO',
        barmode='group', title='2. Sazonalidade das Categorias (Top 5 Serviços por Mês)',
        labels={'Mes': 'Mês do Ano (1 a 12)', 'GRUPOSERVICO_DESCRICAO': 'Categoria'},
        template='plotly_white', color_discrete_sequence=OKABE_ITO
    )

    # ------------------------------------------------------------------
    # FIG 3 — Ranking: Top 10 Bairros Críticos (barras horizontais)
    # ------------------------------------------------------------------
    top10_bairros = df_all['BAIRRO'].value_counts().head(10).reset_index()
    top10_bairros.columns = ['Bairro', 'Volume']

    fig3 = px.bar(
        top10_bairros, x='Volume', y='Bairro', orientation='h',
        title='3. Ranking de Volume: Top 10 Bairros Críticos',
        color='Bairro', color_discrete_sequence=OKABE_ITO, template='plotly_white'
    )
    fig3.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)

    # ------------------------------------------------------------------
    # FIG 4 — Treemap: Top 20 Bairros
    # ------------------------------------------------------------------
    top20_bairros = df_all['BAIRRO'].value_counts().head(20).reset_index()
    top20_bairros.columns = ['Bairro', 'Volume']

    fig4 = px.treemap(
        top20_bairros, path=['Bairro'], values='Volume',
        title='4. Representatividade Visual Espacial: Top 20 Bairros',
        color='Bairro', color_discrete_sequence=OKABE_ITO
    )

    # ------------------------------------------------------------------
    # FIG 5 — Proporção de Resoluções nos 5 Bairros Críticos
    # ------------------------------------------------------------------
    top5_bairros_nomes = df_all['BAIRRO'].value_counts().head(5).index
    df_bairros_criticos = df_all[df_all['BAIRRO'].isin(top5_bairros_nomes)]
    df_prop = df_bairros_criticos.groupby(['BAIRRO', 'SITUACAO']).size().reset_index(name='Contagem')
    df_prop['Porcentagem'] = df_prop.groupby('BAIRRO')['Contagem'].transform(lambda x: x / x.sum() * 100)

    fig5 = px.bar(
        df_prop, x='BAIRRO', y='Porcentagem', color='SITUACAO',
        title='5. Proporção de Resoluções nos 5 Bairros Críticos',
        labels={'Porcentagem': 'Porcentagem (%)', 'BAIRRO': 'Bairro'},
        barmode='stack', color_discrete_sequence=OKABE_ITO, template='plotly_white'
    )

    # ------------------------------------------------------------------
    # FIG 6 — Heatmap: Padrão de Acessos (Dias da Semana vs Meses) — só se Dia_Semana existir
    # ------------------------------------------------------------------
    fig6_heatmap_card = html.Div()
    if 'Dia_Semana' in df_all.columns:
        heatmap_data = df_all.groupby(['Dia_Semana', 'Mes']).size().reset_index(name='Volume')
        ordem_dias = ['Sunday', 'Saturday', 'Friday', 'Thursday', 'Wednesday', 'Tuesday', 'Monday']
        fig6_heat = px.density_heatmap(
            heatmap_data, x='Mes', y='Dia_Semana', z='Volume',
            title='6. Padrão de Acessos: Dias da Semana vs Meses do Ano',
            category_orders={'Dia_Semana': ordem_dias},
            color_continuous_scale='Cividis',  # Cividis: máximo contraste para daltônicos em escalas contínuas
            labels={'Mes': 'Mês do Ano', 'Dia_Semana': 'Dia da Semana'}
        )
        fig6_heatmap_card = dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig6_heat))], className="mb-4 shadow border-0"), width=12)
        ])

    # ------------------------------------------------------------------
    # FIG 6 — Top 10 Vias Mais Críticas (só se LOGRADOURO existir)
    # ------------------------------------------------------------------
    fig6_card = html.Div()
    if 'LOGRADOURO' in df_all.columns:
        df_vias = df_all.copy()
        df_vias['LOGRADOURO'] = df_vias['LOGRADOURO'].astype(str).apply(remover_acentos)
        top10_vias = df_vias['LOGRADOURO'].value_counts().head(10).reset_index()
        top10_vias.columns = ['Logradouro', 'Quantidade_Defeitos']
        fig6 = px.bar(
            top10_vias, x='Quantidade_Defeitos', y='Logradouro', orientation='h',
            title='6. Top 10 Vias Mais Críticas (Maior Número de Defeitos)',
            labels={'Quantidade_Defeitos': 'Nº de Ocorrências', 'Logradouro': 'Via/Logradouro'},
            color='Logradouro', color_discrete_sequence=OKABE_ITO, template='plotly_white'
        )
        fig6.update_layout(
            yaxis={'categoryorder': 'total ascending', 'tickangle': 0},
            yaxis_title=None, showlegend=False, height=500
        )
        fig6_card = dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig6))], className="mb-4 shadow border-0"), width=12)
        ])

    # ------------------------------------------------------------------
    # FIG 7 — Balanço de Eficiência Pública (Rosca)
    # ------------------------------------------------------------------
    status_balanco = df_all['SITUACAO'].value_counts().reset_index()
    status_balanco.columns = ['Situacao', 'Total']

    fig7 = px.pie(
        status_balanco, values='Total', names='Situacao', hole=0.5,
        title='7. Balanço de Eficiência Pública (Gráfico de Rosca)',
        color_discrete_sequence=OKABE_ITO,
        labels={'Situacao': 'Status', 'Total': 'Chamados'}
    )
    fig7.update_traces(
        textinfo='percent+label',
        pull=[0.1 if c == 'PENDENTE' else 0 for c in status_balanco['Situacao']]
    )

    # ------------------------------------------------------------------
    # FIG 8 — Detalhamento Crítico: Principais Queixas (só se SERVICO_DESCRICAO existir)
    # ------------------------------------------------------------------
    fig8_card = html.Div()
    if 'SERVICO_DESCRICAO' in df_all.columns:
        df_serv = df_all.copy()
        df_serv['SERVICO_DESCRICAO'] = df_serv['SERVICO_DESCRICAO'].astype(str).apply(remover_acentos)
        maior_grupo = df_serv['GRUPOSERVICO_DESCRICAO'].value_counts().idxmax()
        df_detalhe = df_serv[df_serv['GRUPOSERVICO_DESCRICAO'] == maior_grupo]
        top10_servicos = df_detalhe['SERVICO_DESCRICAO'].value_counts().head(10).reset_index()
        top10_servicos.columns = ['Serviço', 'Volume']
        fig8 = px.bar(
            top10_servicos, x='Volume', y='Serviço', orientation='h',
            title=f'8. Detalhamento Crítico: Principais Queixas em "{maior_grupo}"',
            color='Serviço', color_discrete_sequence=OKABE_ITO,
            labels={'Volume': 'Qtd. Ocorrências', 'Serviço': 'Subcategoria de Serviço'},
            template='plotly_white'
        )
        fig8.update_layout(
            yaxis={'categoryorder': 'total ascending', 'tickangle': 0},
            yaxis_title=None, showlegend=False, height=500
        )
        fig8_card = dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig8))], className="mb-4 shadow border-0"), width=12)
        ])

    # ------------------------------------------------------------------
    # FIG 9 — Heatmap: Identidade Urbana (Bairros vs Categorias)
    # ------------------------------------------------------------------
    top_bairros_heat = df_all['BAIRRO'].value_counts().head(10).index
    top_cat_heat = df_all['GRUPOSERVICO_DESCRICAO'].value_counts().head(10).index
    df_identidade = df_all[
        (df_all['BAIRRO'].isin(top_bairros_heat)) &
        (df_all['GRUPOSERVICO_DESCRICAO'].isin(top_cat_heat))
    ]
    matriz_identidade = pd.crosstab(df_identidade['BAIRRO'], df_identidade['GRUPOSERVICO_DESCRICAO'])

    fig9 = px.imshow(
        matriz_identidade,
        labels=dict(x="Categoria do Problema", y="Bairro", color="Volume"),
        title='9. Identidade Urbana: Bairros vs. Principais Categorias',
        color_continuous_scale='Cividis',  # Cividis: máximo contraste para daltônicos em escalas contínuas
        text_auto=',.0f', aspect="auto", height=600
    )
    fig9.update_layout(xaxis_tickangle=-45, margin=dict(b=120))

    # ------------------------------------------------------------------
    # FIG 10 — Tempo Médio de Resolução (só se DATA_ULT_SITUACAO existir)
    # ------------------------------------------------------------------
    fig10_card = html.Div()
    if 'DATA_ULT_SITUACAO' in df_all.columns:
        resolvidos = df_all[df_all['SITUACAO'] == 'ATENDIDA'].copy()
        resolvidos['DATA_ULT_SITUACAO'] = pd.to_datetime(resolvidos['DATA_ULT_SITUACAO'], errors='coerce')
        resolvidos = resolvidos.dropna(subset=['DATA_DEMANDA', 'DATA_ULT_SITUACAO'])
        resolvidos['TEMPO_DIAS'] = (resolvidos['DATA_ULT_SITUACAO'] - resolvidos['DATA_DEMANDA']).dt.days
        resolvidos = resolvidos[resolvidos['TEMPO_DIAS'] >= 0]
        resolvidos = resolvidos[resolvidos['GRUPOSERVICO_DESCRICAO'] != 'DENUNCIAS']
        if not resolvidos.empty:
            tempo_medio = resolvidos.groupby('GRUPOSERVICO_DESCRICAO')['TEMPO_DIAS'].mean().reset_index()
            tempo_medio['TEMPO_DIAS'] = tempo_medio['TEMPO_DIAS'].round(0)
            top10_lentos = tempo_medio.sort_values(by='TEMPO_DIAS', ascending=False).head(10)
            fig10 = px.bar(
                top10_lentos, x='TEMPO_DIAS', y='GRUPOSERVICO_DESCRICAO', orientation='h',
                title='10. Tempo Médio de Resolução da Prefeitura (Top 10 mais lentos)',
                text='TEMPO_DIAS', color='GRUPOSERVICO_DESCRICAO', color_discrete_sequence=OKABE_ITO,
                labels={'TEMPO_DIAS': '', 'GRUPOSERVICO_DESCRICAO': ''}, template='plotly_white'
            )
            fig10.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=500, margin=dict(r=120), showlegend=False,
                xaxis=dict(showgrid=True, showticklabels=True, title=None, dtick=200),
                yaxis_title=None
            )
            fig10.update_traces(
                textposition='outside', texttemplate='<b>%{text} dias</b>', cliponaxis=False
            )
            fig10_card = dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig10))], className="mb-4 shadow border-0"), width=12)
            ])

    # ==========================================
    # LAYOUT DA PÁGINA
    # ==========================================
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
        # GRÁFICO CUSTOMIZÁVEL (Layout em Containers Flutuantes)
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
                                id='bairro-autocomplete',
                                multi=True, searchable=True,
                                placeholder="Buscar (Máx 5)...", options=[],
                                className="mb-2"
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
                                id='servico-autocomplete',
                                multi=True, searchable=True,
                                placeholder="Buscar denúncia...", options=[],
                                disabled=True, className="mb-2"
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
        html.P(
            "Análise estática do montante de dados históricos da zeladoria urbana. "
            "Todos os gráficos utilizam a paleta Okabe-Ito, padrão científico de acessibilidade para daltônicos.",
            className="text-secondary mb-4"
        ),

        # Gráfico 1: Evolução temporal
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig1))], className="mb-4 shadow border-0"), width=12),
        ]),

        # Gráfico 2: Sazonalidade
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig2))], className="mb-4 shadow border-0"), width=12),
        ]),

        # Gráficos 3 e 4: Ranking bairros + Treemap lado a lado
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig3))], className="mb-4 shadow border-0"), width=12, md=6),
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig4))], className="mb-4 shadow border-0"), width=12, md=6),
        ]),

        # Gráfico 5: Proporção de Resoluções nos 5 Bairros Críticos
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig5))], className="mb-4 shadow border-0"), width=12),
        ]),

        # Gráfico 6: Heatmap Dias da Semana vs Meses (aparece só se a coluna existir)
        fig6_heatmap_card,

        # Gráfico 7: Top 10 Vias (aparece só se a coluna existir no dataset)
        fig6_card,

        # Gráfico 7: Rosca de eficiência pública
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig7))], className="mb-4 shadow border-0"), width=12, md=6),
            # Gráfico 8: Detalhamento crítico (aparece ao lado se a coluna existir)
            dbc.Col(fig8_card, width=12, md=6),
        ]),

        # Gráfico 9: Heatmap Identidade Urbana
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig9))], className="mb-4 shadow border-0"), width=12),
        ]),

        # Gráfico 10: Tempo Médio de Resolução (aparece só se a coluna existir no dataset)
        fig10_card,

    ], fluid=True)