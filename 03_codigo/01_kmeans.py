from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ============================================================
# P1 — Código-base: clusterização não supervisionada com K-Means
# ============================================================
# Altere somente os campos da seção CONFIGURAÇÃO conforme o
# arquivo e as variáveis definidos para o seu grupo.
# ============================================================

# ---------------------- CONFIGURAÇÃO -------------------------
ARQUIVO_DADOS = "01_dados_recebidos/02_cartao_credito.csv"
COLUNAS_ATRIBUTOS = ['idade', 'limite_credito', 'gasto_mensal', 
'transacoes_mes', 'parcelamentos_ativos']

# Variáveis obrigatórias para o gráfico principal do seu grupo.
EIXO_X = "limite_credito"
EIXO_Y = "gasto_mensal"
TITULO_GRAFICO = "P1 — Cartão de crédito: clusters encontrados pelo K-Means"
K_ESCOLHIDO = 4

# Mantém os resultados reprodutíveis.
RANDOM_STATE = 42
PASTA_RESULTADOS = Path("04_resultados")
PASTA_DADOS = Path("02_dados_tratados")


def carregar_e_validar_dados(caminho: str, colunas: list[str]) -> pd.DataFrame:
    """Carrega o CSV e verifica se as colunas esperadas estão presentes."""
    dados = pd.read_csv(caminho)
    colunas_ausentes = [coluna for coluna in colunas if coluna not in dados.columns]

    if colunas_ausentes:
        raise ValueError(
            "As seguintes colunas não foram encontradas no CSV: "
            f"{colunas_ausentes}"
        )

    if dados[colunas].isnull().any().any():
        raise ValueError(
            "Há valores ausentes nas colunas utilizadas. "
            "Trate os dados antes de executar a clusterização."
        )

    return dados


def avaliar_valores_k(dados_padronizados, valores_k: list[int]) -> pd.DataFrame:
    """Executa K-Means para cada k e registra inércia e silhueta."""
    resultados = []

    for k in valores_k:
        modelo = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        clusters = modelo.fit_predict(dados_padronizados)

        resultados.append(
            {
                "k": k,
                "inercia": modelo.inertia_,
                "silhueta": silhouette_score(dados_padronizados, clusters),
            }
        )

    return pd.DataFrame(resultados)


def salvar_grafico_metricas(resultados: pd.DataFrame) -> None:
    """Gera gráficos de inércia e silhueta para apoiar a escolha de k."""
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.lineplot(data=resultados, x="k", y="inercia", marker="o", ax=eixos[0])
    eixos[0].set_title("Método do cotovelo")
    eixos[0].set_xlabel("Número de clusters (k)")
    eixos[0].set_ylabel("Inércia")
    eixos[0].set_xticks(resultados["k"])

    sns.lineplot(data=resultados, x="k", y="silhueta", marker="o", ax=eixos[1])
    eixos[1].set_title("Índice de silhueta")
    eixos[1].set_xlabel("Número de clusters (k)")
    eixos[1].set_ylabel("Silhueta média")
    eixos[1].set_xticks(resultados["k"])

    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / "01_metricas_escolha_k.png", dpi=150)
    plt.show()


def treinar_modelo_final(dados_padronizados, k: int):
    """Treina o K-Means final usando o valor de k escolhido pelo grupo."""
    modelo = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    clusters = modelo.fit_predict(dados_padronizados)
    return modelo, clusters


def salvar_grafico_clusters(dados: pd.DataFrame) -> None:
    """Gera o gráfico principal de dispersão solicitado na P1."""
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=dados,
        x=EIXO_X,
        y=EIXO_Y,
        hue="cluster_kmeans",
        palette="tab10",
        alpha=0.65,
        s=35,
    )
    plt.title(TITULO_GRAFICO)
    plt.xlabel(EIXO_X)
    plt.ylabel(EIXO_Y)
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / "02_grafico_clusters.png", dpi=150)
    plt.show()

def rotular_dados_clusterizados(dados: pd.DataFrame) -> pd.DataFrame:
  mapeamento = {
    0: "perfil_gasto_consumo_baixo",
    1: "perfil_gasto_consumo_moderado",
    2: "perfil_gasto_consumo_alto",
    3: "perfil_alto_parcelamento"
  }
  dados["rotulo_grupo"] = dados["cluster_kmeans"].map(mapeamento)
  return dados

def main() -> None:
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    PASTA_DADOS.mkdir(exist_ok=True)
  
    dados = carregar_e_validar_dados(ARQUIVO_DADOS, COLUNAS_ATRIBUTOS)
    atributos = dados[COLUNAS_ATRIBUTOS].copy()

    print("\n--- Informações do dataset ---")
    print(f"Registros: {len(dados)}")
    print(f"Colunas utilizadas: {COLUNAS_ATRIBUTOS}")
    print("\n--- Estatísticas descritivas ---")
    print(atributos.describe().round(2))

    scaler = StandardScaler()
    atributos_padronizados = scaler.fit_transform(atributos)

    valores_k = [2, 3, 4, 5]
    resultados = avaliar_valores_k(atributos_padronizados, valores_k)

    print("\n--- Comparação entre valores de k ---")
    print(resultados.round(4).to_string(index=False))
    resultados.to_csv(PASTA_RESULTADOS / "metricas_escolha_k.csv", index=False)
    salvar_grafico_metricas(resultados)

    if K_ESCOLHIDO not in valores_k:
        raise ValueError("K_ESCOLHIDO deve ser um dos valores: 2, 3, 4 ou 5.")

    modelo_final, clusters = treinar_modelo_final(atributos_padronizados, K_ESCOLHIDO)
    dados["cluster_kmeans"] = clusters

    print("\n--- Quantidade de registros por cluster ---")
    print(dados["cluster_kmeans"].value_counts().sort_index())
    print("\n--- Médias dos atributos por cluster ---")
    print(dados.groupby("cluster_kmeans")[COLUNAS_ATRIBUTOS].mean().round(2))

    salvar_grafico_clusters(dados)
  
    caminho_saida = PASTA_RESULTADOS / "dados_clusterizados.csv"
    dados.to_csv(caminho_saida, index=False)
    print(f"\nArquivo criado: {caminho_saida}")
    print("\nPróximo passo: interpretar cada cluster e criar a coluna rotulo_grupo.")

    caminho_dados = PASTA_DADOS / "dados_clusterizados_rotulados.csv"
    dados_finais = rotular_dados_clusterizados(dados)
    dados_finais.to_csv(caminho_dados, index=False)

if __name__ == "__main__":
    main()
