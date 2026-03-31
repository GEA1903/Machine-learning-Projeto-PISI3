import pandas as pd

def carregar_e_limpar_dados():
    try:
        # Carregando o dataset com as configurações corretas para Recife
        path = 'data/emlurb_2025.csv'
        df = pd.read_csv(path, sep=';', encoding='latin-1')

        # Limpeza Inicial: Remove linhas sem latitude ou longitude (essencial para o Mapa)
        df_limpo = df.dropna(subset=['latitude', 'longitude'])

        print(f"✅ Arquivo '{path}' carregado com sucesso!")
        print(f"📊 Total de registros: {len(df)}")
        print(f"📍 Registros com localização (válidos): {len(df_limpo)}")
        
        print("\n--- Top 5 Categorias de Incidentes em Recife ---")
        print(df_limpo['GRUPOSERVICO_DESCRICAO'].value_counts().head(5))

        return df_limpo

    except FileNotFoundError:
        print("❌ Erro: O arquivo CSV não foi encontrado na pasta 'data/'.")
    except Exception as e:
        print(f"❌ Ocorreu um erro inesperado: {e}")

if __name__ == "__main__":
    dados = carregar_e_limpar_dados()