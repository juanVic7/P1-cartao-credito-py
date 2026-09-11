from pathlib import Path  # Para gerenciar caminhos de arquivos e pastas
import pandas as pd  # Para carregar e manipular o CSV
from sklearn.model_selection import train_test_split  # Para separar dados de treino e teste
from sklearn.preprocessing import StandardScaler  # Para padronizar a escala das variáveis
from sklearn.neural_network import MLPClassifier  # Para criar a rede neural supervisionada
from sklearn.metrics import accuracy_score  # Para avaliar os acertos do modelo
import matplotlib.pyplot as plt  # Para gerar os gráficos de erro


PASTA_DADOS = Path("02_dados_tratados")
PASTA_RESULTADOS = Path("04_resultados")

class ClassificadorPerfis:
  def __init__(self) -> None:
    self.scaler = StandardScaler()
    PASTA_DADOS.mkdir(exist_ok=True)
    PASTA_RESULTADOS.mkdir(exist_ok=True)


  def carregar_dados(self, caminho_arquivo: str) -> pd.DataFrame:
        # Tarefa: Fazer o pd.read_csv e retornar a tabela limpa
        # o csv para ser lido está na pasta 02_dados_tratados, olhe as colunas desse arquivo
        # Exemplo de leitura de dados no PDF 3, página 14
        pass # ao terminar de fazer o método, apague esse "pass"
    
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
            test_size=0.2, # 20% teste, 80% treino
            random_state=42, # Mantém os resultados reprodutíveis
            stratify=y # Mantém a proporção de classes entre treino e teste
      )

      self.X_treino = self.scaler.fit_transform(X_treino)
      self.X_teste = self.scaler.transform(X_teste)
 
    
  def executar_experimento(self, limite_iteracoes: int) -> None:
    
    self.modelo = MLPClassifier(hidden_layer_sizes=(5,), max_iter=limite_iteracoes, random_state=42, verbose=True)
    self.modelo.fit(self.X_treino, self.y_treino)
    
    predicao = self.modelo.predict(self.X_teste)
    acuracia = accuracy_score(self.y_teste, predicao)
    print(f"Resultado com {limite_iteracoes} épocas: {acuracia * 100:.2f}% de acerto")

    plt.clf() # Limpa a tela para não misturar com o gráfico anterior
    plt.plot(self.modelo.loss_curve_) # Puxa o histórico e desenha a linha
    plt.title(f"Queda do Erro - {limite_iteracoes} Épocas")
    
    caminho_imagem = PASTA_RESULTADOS / f"curva_perda_{limite_iteracoes}.png"
    plt.savefig(caminho_imagem)
    


  
  def prever_novos_casos(self, dados_ineditos: pd.DataFrame) -> None:
        # Tarefa: Usar o self.scaler.transform nos dados novos
        # e imprimir as predições geradas pela rede
        # Consulta: Exigência de 5 casos inéditos (Roteiro da Prova, Seção de Testes)
        pass # ao terminar de fazer o método, apague esse "pass"


def main() -> None:
  pass

if __name__ == "__main__":
    main()


#É o lufe n tem jeito
# BINGO!