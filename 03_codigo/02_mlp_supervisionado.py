from pathlib import Path  # Para gerenciar caminhos de arquivos e pastas
import pandas as pd  # Para carregar e manipular o CSV
from sklearn.model_selection import train_test_split  # Para separar dados de treino e teste
from sklearn.preprocessing import StandardScaler  # Para padronizar a escala das variáveis
from sklearn.neural_network import MLPClassifier  # Para criar a rede neural supervisionada
from sklearn.metrics import accuracy_score  # Para avaliar os acertos do modelo

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
        # Tarefa: Separar X e y, fazer o train_test_split e aplicar (PDF da aula 3, pág 14 contém um exemplo)
        # o StandardScaler (fit_transform no treino, transform no teste)
        # Salvar os resultados dentro da classe (ex: self.X_treino)
        # Regra contra vazamento de dados (Roteiro da Prova, item 4.2.7)
        pass # ao terminar de fazer o método, apague esse "pass"       
    
  def executar_experimento(self, limite_iteracoes: int) -> None:
        # Tarefa: Configurar o MLPClassifier com o limite_iteracoes,
        # treinar a rede, calcular acurácias, plotar e salvar o gráfico
        # Estrutura do MotorAnalise e uso de redes neurais (Capítulos 4 e 5).
        pass # ao terminar de fazer o método, apague esse "pass"
    
  def prever_novos_casos(self, dados_ineditos: pd.DataFrame) -> None:
        # Tarefa: Usar o self.scaler.transform nos dados novos
        # e imprimir as predições geradas pela rede
        # Consulta: Exigência de 5 casos inéditos (Roteiro da Prova, Seção de Testes)
        pass # ao terminar de fazer o método, apague esse "pass"


def main() -> None:
  pass

if __name__ == "__main__":
    main()
