from pathlib import Path 
import pandas as pd  
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPClassifier  
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt 


PASTA_DADOS = Path("../02_dados_tratados")
PASTA_RESULTADOS = Path("../04_resultados")

class ClassificadorPerfis:
  def __init__(self) -> None:
    self.scaler = StandardScaler()
    PASTA_DADOS.mkdir(exist_ok=True)
    PASTA_RESULTADOS.mkdir(exist_ok=True)


  def carregar_dados(self, caminho_arquivo: str) -> pd.DataFrame:
    return pd.read_csv(caminho_arquivo)
    
  def preparar_dados(self, dados: pd.DataFrame) -> None:

      colunas_atributos = [
            'idade', 
            'limite_credito', 
            'gasto_mensal', 
            'transacoes_mes', 
            'parcelamentos_ativos'
      ]

      X = dados[colunas_atributos]
      y = dados['rotulo_grupo']

      X_treino, X_teste, self.y_treino, self.y_teste = train_test_split(
            X, 
            y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y 
      )

      self.X_treino = self.scaler.fit_transform(X_treino)
      self.X_teste = self.scaler.transform(X_teste)
 
    
  def executar_experimento(self, limite_iteracoes: int) -> None:
    
    self.modelo = MLPClassifier(hidden_layer_sizes=(5,), max_iter=limite_iteracoes, random_state=42, verbose=True)
    self.modelo.fit(self.X_treino, self.y_treino)
    
    predicao = self.modelo.predict(self.X_teste)
    acuracia = accuracy_score(self.y_teste, predicao)
    print(f"Resultado com {limite_iteracoes} épocas: {acuracia * 100:.2f}% de acerto")

    plt.clf() 
    plt.plot(self.modelo.loss_curve_) 
    plt.title(f"Queda do Erro - {limite_iteracoes} Épocas")
    
    caminho_imagem = PASTA_RESULTADOS / f"curva_perda_{limite_iteracoes}.png"
    plt.savefig(caminho_imagem)
    



  def prever_novos_casos(self, dados_ineditos: pd.DataFrame) -> None:
    colunas_atributos = [
        "idade",
        "limite_credito",
        "gasto_mensal",
        "transacoes_mes",
        "parcelamentos_ativos"]

    X_novos = dados_ineditos[colunas_atributos]

    X_novos_padronizados = self.scaler.transform(X_novos)

    predicoes = self.modelo.predict(X_novos_padronizados)

    resultados = dados_ineditos.copy()
    resultados["rotulo_previsto"] = predicoes

    resultados["resultado"] = resultados.apply(
        lambda linha: "ACERTO"
        if linha["rotulo_esperado"] == linha["rotulo_previsto"]
        else "ERRO",
        axis=1
        )

    print("\n--- Previsão de Casos Inéditos ---")

    for numero, (_, linha) in enumerate(resultados.iterrows(), start=1):
        print(
            f"Cliente {numero} -> "
            f"Esperado: {linha['rotulo_esperado']} | "
            f"Previsto: {linha['rotulo_previsto']} | "
            f"{linha['resultado']}")

    caminho_saida = PASTA_RESULTADOS / "casos_de_teste.csv"
    resultados.to_csv(caminho_saida, index=False)

    print(f"\nArquivo salvo em: {caminho_saida}")

def main() -> None:
  
  motor = ClassificadorPerfis()
  
  caminho_treino = PASTA_DADOS / "dados_clusterizados_rotulados.csv"
  dados_treino = motor.carregar_dados(caminho_treino)
  motor.preparar_dados(dados_treino)
  
  print ("--- INICIANDO TREINAMENTOS ---")
  for ciclos in [50, 500, 5000]:
     motor.executar_experimento(ciclos)
    
  caminho_novos_clientes = PASTA_DADOS / "novos_clientes.csv"
  novos_clientes = motor.carregar_dados(caminho_novos_clientes)
  motor.prever_novos_casos(novos_clientes)

if __name__ == "__main__":
    main()
