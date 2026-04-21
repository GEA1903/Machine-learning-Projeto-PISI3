# Projeto-DSI------gp4
---

# 📊 Análise de Demandas Urbanas: Recife (Emlurb 156)

Este repositório contém um projeto de análise exploratória e engenharia de dados baseado nos registros de atendimento da **EMLURB** (Empresa de Manutenção e Limpeza Urbana), obtidos através do Portal de Dados Abertos da Cidade do Recife.

O objetivo é identificar gargalos na infraestrutura urbana, padrões de sazonalidade (como o impacto do inverno e do carnaval) e medir a eficiência da gestão pública na resolução de problemas por bairro e categoria.

## 🛠️ Tecnologias e Ferramentas

* **Linguagem:** Python 3.10+
* **Ambiente:** Jupyter Notebook
* **Bibliotecas Principais:**
    * `pandas`: Manipulação e limpeza de dados.
    * `plotly`: Visualizações interativas e storytelling.
    * `requests`: (Opcional) Coleta de dados via API.
    * `os` & `sys`: Gestão de arquivos locais.

---

## 🚀 Instalação e Configuração

Para garantir que todas as dependências sejam instaladas corretamente e não haja conflito de versões, recomendamos o uso do **Anaconda** ou **Miniconda**.

### 1. Instalar o Anaconda
Se você ainda não tem o Anaconda, faça o download em: [anaconda.com/download](https://www.anaconda.com/download/).

### 2. Criar o Ambiente Virtual
Abra o seu terminal (ou Anaconda Prompt) e execute:

```bash
# Cria o ambiente chamado 'recife_dados'
conda create --name recife_dados python=3.10

# Ativa o ambiente
conda activate recife_dados
```

### 3. Instalar Dependências
Com o ambiente ativado, instale as bibliotecas necessárias:

```bash
pip install pandas plotly requests jupyter
```

---

## 📂 Estrutura do Projeto

* `/data`: Pasta destinada aos arquivos CSV baixados do portal (Ex: `emlurb_2020.csv`, `emlurb_2021.csv`, etc).
* `analise_emlurb.ipynb`: Notebook principal com o código de limpeza, processamento e geração de gráficos.
* `README.md`: Documentação do projeto.

---

## 📈 O Processo de Dados

O pipeline de dados desenvolvido no notebook `ipynb` segue quatro etapas:

### 1. Carregamento e Concatenção
O sistema varre a pasta `/data`, identifica todos os arquivos `.csv` e os une em um único DataFrame global. No exemplo atual, o projeto processa mais de **570.000 registros**.

### 2. Engenharia de Dados e Limpeza
* **Datas:** Conversão de formatos mistos para objetos `datetime`.
* **Features:** Criação de colunas de Sazonalidade (Mês, Dia da Semana, Ano-Mês).
* **Padronização:** Limpeza de strings (Uppercase/Strip) e agrupamento de categorias similares via Regex (Ex: unificar "ARBORIZA", "ARBORIZACAO" e "Poda" em um único grupo).

### 3. Análise de Eficiência
O código calcula o **Tempo Médio de Resolução** subtraindo a data da demanda da data da última situação, permitindo identificar quais serviços levam centenas de dias para serem concluídos.

### 4. Storytelling Visual (Os 3 Atos)
O projeto culmina em uma narrativa visual:
* **Ato 1:** Ciclo de vida dos problemas (Sazonalidade).
* **Ato 2:** Onde a fila trava (Bairros com mais pendências).
* **Ato 3:** Taxa de Ineficiência (O que é mais negligenciado em cada área crítica).

---

## 📊 Exemplos de Visualizações Geradas

O projeto gera gráficos interativos que respondem:
1.  **Qual o impacto das chuvas de inverno na pavimentação da Várzea?**
2.  **Como a limpeza urbana se comporta no pós-Carnaval?**
3.  **Qual a taxa de ineficiência de iluminação em Boa Viagem comparada à periferia?**



---

## 📝 Como Executar

1.  Coloque seus arquivos CSV na pasta `data/`.
2.  Inicie o Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
3.  Abra o arquivo `analise_emlurb.ipynb` e execute todas as células (Run All).

---

**Autores:** Arthur Alves, Breno Jansen, Caio Carvalho, Davi Eufrásio, Gabriel Escobar e Pedro Peres
**Fonte dos Dados:** [Portal de Dados Abertos do Recife](http://dados.recife.pe.gov.br/)
