import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.express as px
import pandas as pd

def render_eda(anos_reais, df_all=None):
    if df_all is None:
        return dbc.Container([html.H3("Aguardando carregamento dos dados...")])

    df_all['DATA_DEMANDA'] = pd.to_datetime(df_all['DATA_DEMANDA'], errors='coerce')

    df_filtrado = df_all[(df_all['DATA_DEMANDA'] >= '2020-01-01') & (df_all['DATA_DEMANDA'] <= '2025-12-31')].copy()
    vol_mensal = df_filtrado['DATA_DEMANDA'].dt.to_period('M').value_counts().sort_index().reset_index()
    vol_mensal.columns = ['Ano', 'Volume de Denúncias']
    
    if len(vol_mensal) > 0:
        vol_mensal = vol_mensal.iloc[:-1]
    
    vol_mensal['Ano'] = vol_mensal['Ano'].dt.to_timestamp()

    fig1 = px.line(
        vol_mensal, x='Ano', y='Volume de Denúncias', 
        title='Volume Total de Denúncias Históricas', markers=True, template='plotly_white'
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

    top5_cat = df_all['GRUPOSERVICO_DESCRICAO'].value_counts().head(5).index
    df_top5_cat = df_all[df_all['GRUPOSERVICO_DESCRICAO'].isin(top5_cat)]

    vol_mes_cat = df_top5_cat.groupby(['Mes', 'GRUPOSERVICO_DESCRICAO']).size().reset_index(name='Volume')

    fig2 = px.bar(
        vol_mes_cat, x='Mes', y='Volume', color='GRUPOSERVICO_DESCRICAO',
        barmode='group', title='2. Sazonalidade das Categorias (Top 5 Serviços por Mês)',
        labels={'Mes': 'Mês do Ano (1 a 12)', 'GRUPOSERVICO_DESCRICAO': 'Categoria'}, template='plotly_white'
    )

    top10_bairros = df_all['BAIRRO'].value_counts().head(10).reset_index()
    top10_bairros.columns = ['Bairro', 'Volume']

    fig4 = px.bar(
        top10_bairros, x='Volume', y='Bairro', orientation='h', 
        title='3. Ranking de Volume: Top 10 Bairros Críticos',
        color='Volume', color_continuous_scale='Reds', template='plotly_white'
    )
    fig4.update_layout(yaxis={'categoryorder':'total ascending'})

    top20_bairros = df_all['BAIRRO'].value_counts().head(20).reset_index()
    top20_bairros.columns = ['Bairro', 'Volume']

    fig5 = px.treemap(
        top20_bairros, path=['Bairro'], values='Volume',
        title='5. Representatividade Visual Espacial: Top 20 Bairros',
        color='Volume', color_continuous_scale='Blues'
    )

    return dbc.Container([
        
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

        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id='ato-1'))], className="mb-4 shadow border-0"), width=12),
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id='ato-2'))], className="mb-4 shadow border-0"), width=12),
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id='ato-3'))], className="mb-4 shadow border-0"), width=12),
        ]),

        html.Hr(className="my-5"),

        html.H3("Visão Geral e Comportamento Espacial", className="text-primary mb-4 fw-bold"),
        html.P("Análise estática do montante de dados históricos da zeladoria urbana.", className="text-secondary mb-4"),

        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig1))], className="mb-4 shadow border-0"), width=12),
        ]),

        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig2))], className="mb-4 shadow border-0"), width=12),
        ]),

        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig4))], className="mb-4 shadow border-0"), width=12, md=6),
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig5))], className="mb-4 shadow border-0"), width=12, md=6),
        ]),

    ], fluid=True)