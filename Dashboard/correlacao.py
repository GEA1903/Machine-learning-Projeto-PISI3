import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import pandas as pd
import numpy as np

def render_correlacao(df_geral=None):
    if df_geral is None:
        return dbc.Container([html.H3("Aguardando carregamento dos dados...")])

    # 1. ENGENHARIA DE RECURSOS / AGREGAÇÃO
    # Criando métricas numéricas por bairro para podermos correlacionar
    status_gargalo = ['PENDENTE', 'PREPARACAO', 'CADASTRADA']
    
    df_corr = df_geral.groupby('BAIRRO').agg(
        Volume_Total=('SITUACAO', 'count'),
        Nao_Resolvidos=('SITUACAO', lambda x: x.isin(status_gargalo).sum()),
        Resolvidos=('SITUACAO', lambda x: (~x.isin(status_gargalo)).sum())
    ).reset_index()

    # Criando taxas percentuais
    df_corr['Taxa_Ineficiencia_%'] = (df_corr['Nao_Resolvidos'] / df_corr['Volume_Total']) * 100
    df_corr['Taxa_Resolucao_%'] = (df_corr['Resolvidos'] / df_corr['Volume_Total']) * 100
    
    # Filtro para evitar ruído de bairros com pouquíssimos chamados
    df_corr = df_corr[df_corr['Volume_Total'] > 500].dropna()

    # Selecionando apenas as colunas numéricas para a matriz de correlação
    colunas_numericas = ['Volume_Total', 'Nao_Resolvidos', 'Resolvidos', 'Taxa_Ineficiencia_%', 'Taxa_Resolucao_%']
    matriz_corr = df_corr[colunas_numericas].corr()

    # 2. CONSTRUÇÃO DOS GRÁFICOS
    
    # Gráfico 1: Heatmap da Matriz de Correlação
    fig_heatmap = px.imshow(
        matriz_corr,
        text_auto='.2f', # Mostra o valor do coeficiente dentro do quadrado
        aspect="auto",
        color_continuous_scale='RdBu_r', # Escala do Vermelho (negativa) ao Azul (positiva)
        zmin=-1, zmax=1,                 # Limites do coeficiente de Pearson
        title="Matriz de Correlação de Pearson (Métricas por Bairro)",
        template='plotly_white',
        labels=dict(x="Variáveis", y="Variáveis", color="Coeficiente")
    )
    fig_heatmap.update_layout(margin=dict(l=40, r=40, t=60, b=40))

    # Gráfico 2: Scatter Plot de Tendência (Volume vs Ineficiência)
    # 2.1. Cria o gráfico de dispersão normal (sem o parâmetro trendline)
    fig_dispensao = px.scatter(
        df_corr, x='Volume_Total', y='Taxa_Ineficiencia_%',
        hover_name='BAIRRO',
        title='Dispersão e Linha de Tendência: Volume vs. Ineficiência',
        template='plotly_white',
        labels={'Volume_Total': 'Volume Total de Chamados', 'Taxa_Ineficiencia_%': 'Taxa de Ineficiência (%)'}
    )
    fig_dispensao.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))

    # 2.2. CALCULO MANUAL DA REGRESSÃO (A mágica sem statsmodels)
    if not df_corr.empty:
        x_vals = df_corr['Volume_Total']
        y_vals = df_corr['Taxa_Ineficiencia_%']
        
        # np.polyfit calcula os coeficientes da reta (y = mx + b)
        m, b = np.polyfit(x_vals, y_vals, 1)
        
        # Criamos apenas dois pontos (o início e o fim) para traçar a reta perfeita
        x_trend = [x_vals.min(), x_vals.max()]
        y_trend = [m * x_vals.min() + b, m * x_vals.max() + b]
        
        # Injetamos a linha vermelha diretamente no gráfico criado pelo Plotly
        fig_dispensao.add_scatter(
            x=x_trend, 
            y=y_trend, 
            mode='lines', 
            name='Tendência Linear', 
            line=dict(color='red', width=2, dash='dash'),
            showlegend=True
        )    # 3. LAYOUT DA PÁGINA
    return dbc.Container([
        html.H3("Análise de Correlação de Dados Estreitos", className="text-primary my-4 fw-bold"),
        html.P(
            "Investigação estatística das relações lineares entre a demanda volumétrica dos bairros "
            "e a capacidade operacional de resolução da EMLURB (Série Histórica 2020-2025).", 
            className="text-secondary mb-4"
        ),
        
        # Primeira Linha: Os dois gráficos principais lado a lado
        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_heatmap)), className="shadow-sm border-0")
            ], md=6, className="mb-4"),
            
            dbc.Col([
                dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_dispensao)), className="shadow-sm border-0")
            ], md=6, className="mb-4"),
        ]),

        # Segunda Linha: Análise Estatística Avançada e Insights Gerenciais
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        html.H5("Análise Crítica da Matriz de Pearson", className="mb-0 text-white fw-bold"),
                        className="bg-primary text-white"
                    ),
                    dbc.CardBody([
                        html.P([
                            html.Strong("1. Desmistificação do Volume Operacional (r = 0.13): "),
                            "A correlação extremamente baixa entre o ", html.Code("Volume_Total"), " e a ", html.Code("Taxa_Ineficiencia_%"),
                            " revela que o acúmulo absoluto de demandas não é o fator causador do atraso nas resoluções. ",
                            "Bairros com altíssimo volume conseguem manter taxas de eficiência similares ou superiores a bairros com pouca demanda."
                        ], className="text-muted mb-3"),
                        
                        html.P([
                            html.Strong("2. Correlações Volumétricas Nominais (r = 0.95 e 1.00): "),
                            "A relação quase perfeita entre os valores absolutos (", html.Code("Nao_Resolvidos"), " e ", html.Code("Resolvidos"), ") ",
                            "com o volume total é um comportamento esperado (relação matemática direta: mais chamados geram proporcionalmente mais saídas). ",
                            "Por isso, focar nas taxas percentuais traz mais valor estratégico do que avaliar números absolutos."
                        ], className="text-muted mb-3"),

                        html.P([
                            html.Strong("3. O Alerta Operacional (r = 0.35): "),
                            "Há uma correlação positiva moderada de ", html.Strong("0.35"), " entre o volume de ", html.Code("Nao_Resolvidos"), 
                            " e a ", html.Code("Taxa_Ineficiencia_%"), ". Isso indica que quando o estoque de pendências físicas acumula ",
                            "além de um limite crítico em termos nominais, a velocidade de escoamento do sistema como um todo começa a sofrer gargalos."
                        ], className="text-muted")
                    ])
                ], className="shadow border-0 h-100")
            ], md=6, className="mb-4"),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        html.H5("Interpretação da Tendência e Comportamento de Escala", className="mb-0 text-white fw-bold"),
                        className="bg-dark text-white"
                    ),
                    dbc.CardBody([
                        html.P([
                            html.Strong("Comportamento da Linha de Tendência (Tracejada Vermelha):"),
                        ], className="mb-2"),
                        
                        html.P(
                            "A linha reta possui uma inclinação ascendente sutil. Visualmente, ela valida o coeficiente de 0.13, "
                            "mostrando que o sistema da EMLURB apresenta uma resiliência de escala impressionante. "
                            "Mesmo quando um bairro salta de 5.000 chamados para mais de 30.000 (extremo direito do gráfico), "
                            "a taxa de ineficiência sofre uma flutuação marginal de poucos pontos percentuais.",
                            className="text-muted mb-3"
                        ),
                        
                        html.Div([
                            html.Strong("Direcionamento Estratégico para Gestão PÚBLICA:", className="text-dark d-block mb-1"),
                            html.Ul([
                                html.Li("Não adianta deslocar equipes focando apenas nos bairros que mais abrem chamados."),
                                html.Li("Os bairros localizados no quadrante superior esquerdo (baixo volume, mas ineficiência > 30%) sofrem de gargalos processuais crônicos que merecem auditoria de processos."),
                                html.Li("A previsibilidade do sistema permite planejar SLAs fixos independentemente do crescimento populacional do bairro.")
                            ], className="text-muted ps-3")
                        ], className="p-3 bg-light rounded border-start border-warning border-3")
                    ])
                ], className="shadow border-0 h-100")
            ], md=6, className="mb-4"),
        ])
    ], fluid=True)