# P1 — Cartão de Crédito: Clusterização com K-Means e Classificação com MLP

## 1. Sobre o trabalho

Este projeto foi desenvolvido para a **P1 do Grupo 02 — Cartão de Crédito**. A atividade utiliza uma base de dados **inteiramente sintética**, criada exclusivamente para fins acadêmicos, contendo **8.000 registros** e **cinco atributos numéricos** relacionados a padrões de utilização de cartão de crédito.

O objetivo geral é **identificar padrões de utilização do cartão, consumo mensal e parcelamentos ativos** por meio de duas etapas conectadas de Machine Learning:

1. **K-Means — aprendizado não supervisionado:** descobrir agrupamentos naturais nos dados sem utilizar uma classe pronta.
2. **MLPClassifier — aprendizado supervisionado:** utilizar os grupos interpretados e rotulados após o K-Means como classes para treinar uma rede neural capaz de reproduzir essa segmentação em novos casos.

> **Importante:** o K-Means não é uma rede neural. Ele é um algoritmo de clusterização. A rede neural é utilizada somente na segunda etapa, depois da criação dos rótulos.

Também é importante destacar que a MLP **não descobre uma verdade externa sobre os clientes**. Ela aprende a reproduzir os rótulos derivados da segmentação criada pelo próprio grupo. Como os dados são sintéticos, os resultados não devem ser usados para decisões reais de crédito ou para conclusões causais sobre pessoas.

---

## 2. Objetivo do projeto

O projeto demonstra um fluxo completo de análise de dados e aprendizado de máquina, partindo de um conjunto sem rótulos e chegando a um classificador supervisionado.

Na prática, o trabalho busca:

- analisar os cinco atributos fornecidos pelo professor;
- padronizar as variáveis para trabalhar com escalas comparáveis;
- testar diferentes quantidades de clusters com `k = 2`, `3`, `4` e `5`;
- comparar **inércia** e **índice de silhueta**;
- escolher e justificar um valor final de `k`;
- interpretar os clusters encontrados;
- criar rótulos significativos para cada grupo;
- treinar uma rede neural `MLPClassifier` usando esses rótulos;
- comparar os treinamentos com `max_iter = 50`, `500` e `5000`;
- avaliar perda, quantidade real de iterações e acurácia;
- testar o melhor modelo em pelo menos cinco novos casos.

---

## 3. Dados utilizados

O arquivo original é:

```text
01_dados_recebidos/02_cartao_credito.csv
```

A base contém **8.000 registros** e não possui identificadores pessoais, classe, rótulo, diagnóstico ou resposta pronta.

As cinco variáveis obrigatórias utilizadas tanto no K-Means quanto na MLP são:

| Atributo | Tipo | Significado |
|---|---|---|
| `idade` | Numérico | Idade do cliente no cenário sintético. |
| `limite_credito` | Numérico | Limite total disponível no cartão. |
| `gasto_mensal` | Numérico | Valor aproximado gasto por mês. |
| `transacoes_mes` | Numérico | Quantidade de transações realizadas no mês. |
| `parcelamentos_ativos` | Numérico | Quantidade de compras parceladas ainda ativas. |

No gráfico principal exigido na atividade são utilizados:

- **Eixo X:** `limite_credito`
- **Eixo Y:** `gasto_mensal`
- **Cor:** `cluster_kmeans`

---

## 4. Estrutura do projeto

```text
P1-cartao-credito-py-main/
│
├── 01_dados_recebidos/
│   └── 02_cartao_credito.csv
│
├── 02_dados_tratados/
│   ├── dados_clusterizados_rotulados.csv
│   └── novos_clientes.csv
│
├── 03_codigo/
│   ├── 01_kmeans.py
│   └── 02_mlp_supervisionado.py
│
├── 04_resultados/
│   ├── metricas_escolha_k.csv
│   ├── 01_metricas_escolha_k.png
│   ├── 02_grafico_clusters.png
│   ├── dados_clusterizados.csv
│   ├── curva_perda_50.png
│   ├── curva_perda_500.png
│   ├── curva_perda_5000.png
│   └── casos_de_teste.csv
│
├── .gitignore
└── README.md
```

### `01_dados_recebidos/`

Contém o CSV original fornecido para a atividade.

### `02_dados_tratados/`

Contém o dataset após a clusterização e rotulagem e também os cinco novos casos utilizados para testar o classificador.

### `03_codigo/`

Contém os dois programas principais:

- `01_kmeans.py`: executa a etapa não supervisionada;
- `02_mlp_supervisionado.py`: executa a etapa supervisionada com rede neural.

### `04_resultados/`

Reúne as métricas, gráficos e resultados produzidos pelos scripts.

---

# 5. Etapa 1 — K-Means

O primeiro estágio do projeto utiliza o algoritmo **K-Means**, presente no arquivo:

```text
03_codigo/01_kmeans.py
```

O K-Means é um algoritmo de **aprendizado não supervisionado**. Isso significa que ele recebe os atributos dos registros, mas não recebe uma classe correta dizendo previamente a qual grupo cada cliente pertence.

Seu objetivo é separar os dados em grupos internamente semelhantes, chamados de **clusters**.

## 5.1 Carregamento e validação

A função `carregar_e_validar_dados()` lê o CSV utilizando `pandas` e verifica:

- se todas as cinco colunas obrigatórias existem;
- se há valores ausentes nas variáveis utilizadas.

Caso alguma coluna esteja ausente ou existam valores nulos, o programa interrompe a execução para evitar o treinamento com dados inconsistentes.

---

## 5.2 Padronização com StandardScaler

Antes de executar o K-Means, os cinco atributos são padronizados com:

```python
StandardScaler()
```

Essa etapa é necessária porque os atributos possuem escalas muito diferentes. Por exemplo, `limite_credito` e `gasto_mensal` podem assumir valores na casa dos milhares, enquanto `parcelamentos_ativos` possui números muito menores.

Sem padronização, as variáveis de maior escala poderiam dominar o cálculo de distância utilizado pelo K-Means.

O `StandardScaler` transforma cada atributo para uma escala comparável, baseada em sua média e desvio padrão.

---

## 5.3 Comparação dos valores de k

Conforme solicitado na atividade, foram testados:

```python
valores_k = [2, 3, 4, 5]
```

Para cada valor de `k`, o programa calcula:

### Inércia

A **inércia** representa a soma das distâncias quadráticas entre os registros e o centro do cluster ao qual pertencem.

Quanto menor a inércia, mais compactos tendem a ser os grupos. Entretanto, a inércia normalmente diminui quando `k` aumenta, por isso ela não deve ser utilizada isoladamente.

### Índice de silhueta

O **índice de silhueta** considera ao mesmo tempo a coesão dentro de cada grupo e a separação entre grupos diferentes.

Valores maiores indicam, em geral, uma separação mais clara dos clusters.

### Resultados obtidos

| k | Inércia | Silhueta |
|---:|---:|---:|
| 2 | 18836,7169 | 0,4864 |
| 3 | 12860,8944 | 0,3931 |
| 4 | 8916,8369 | 0,4278 |
| 5 | 8333,1359 | 0,3655 |

Os resultados também são salvos em:

```text
04_resultados/metricas_escolha_k.csv
04_resultados/01_metricas_escolha_k.png
```

---

## 5.4 Escolha final: k = 4

O valor final utilizado foi:

```python
K_ESCOLHIDO = 4
```

A escolha não foi feita apenas procurando o maior índice de silhueta.

Embora `k = 2` tenha apresentado a maior silhueta, `k = 4` oferece uma segmentação mais detalhada e interpretável. Entre `k = 3` e `k = 4`, a inércia diminui de aproximadamente **12.860,89 para 8.916,84**, enquanto a silhueta melhora de aproximadamente **0,3931 para 0,4278**.

Ao passar de `k = 4` para `k = 5`, a inércia apresenta uma redução bem menor, enquanto a silhueta cai para aproximadamente **0,3655**. Isso indica que a inclusão de um quinto cluster não trouxe uma melhoria proporcional na qualidade da segmentação.

Além das métricas, `k = 4` produziu grupos com quantidades razoáveis de registros e características médias suficientemente distintas para receber rótulos interpretáveis.

Por isso, o grupo adotou **quatro clusters**.

> A escolha de `k` deve ser entendida como um equilíbrio entre métricas quantitativas e interpretabilidade, e não como a busca automática pelo maior número de clusters ou pela maior silhueta isoladamente.

---

## 5.5 Treinamento do modelo final

O modelo final utiliza:

```python
KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)
```

### `n_clusters=4`

Define os quatro agrupamentos escolhidos pelo grupo.

### `random_state=42`

Permite reproduzir os resultados obtidos com a mesma base e configuração.

### `n_init=10`

Faz o K-Means testar diferentes inicializações dos centroides e selecionar uma solução adequada entre elas.

Depois do treinamento, cada registro recebe a coluna:

```text
cluster_kmeans
```

---

## 5.6 Quantidade de registros por cluster

A distribuição encontrada foi:

| Cluster | Quantidade de registros | Percentual aproximado |
|---:|---:|---:|
| 0 | 2.347 | 29,34% |
| 1 | 2.065 | 25,81% |
| 2 | 1.914 | 23,93% |
| 3 | 1.674 | 20,93% |
| **Total** | **8.000** | **100%** |

Nenhum dos quatro grupos ficou extremamente pequeno, o que também favoreceu sua interpretação.

---

## 5.7 Médias dos atributos por cluster

| Cluster | Idade | Limite de crédito | Gasto mensal | Transações/mês | Parcelamentos ativos |
|---:|---:|---:|---:|---:|---:|
| 0 | 26,72 | 2.405,03 | 1.002,63 | 2,25 | 1,08 |
| 1 | 39,58 | 8.417,46 | 3.581,09 | 7,44 | 2,40 |
| 2 | 51,97 | 16.973,40 | 7.129,14 | 14,02 | 3,01 |
| 3 | 33,14 | 4.941,61 | 1.409,87 | 3,63 | 5,31 |

Essas médias foram utilizadas para interpretar os agrupamentos.

---

## 5.8 Rotulagem dos clusters

Números de cluster como `0`, `1`, `2` e `3` não explicam o significado do grupo. Por isso, após analisar as médias e as características de cada cluster, foi criada a coluna:

```text
rotulo_grupo
```

O mapeamento utilizado foi:

| Cluster | Rótulo interpretável | Característica principal |
|---:|---|---|
| 0 | `perfil_gasto_consumo_baixo` | Menores limites, gastos e frequência de transações. |
| 1 | `perfil_gasto_consumo_moderado` | Valores intermediários de limite, gasto e frequência. |
| 2 | `perfil_gasto_consumo_alto` | Maiores limites, gastos mensais e número de transações. |
| 3 | `perfil_alto_parcelamento` | Destaque para a maior média de parcelamentos ativos. |

O dataset final dessa etapa é salvo em:

```text
02_dados_tratados/dados_clusterizados_rotulados.csv
```

Esse arquivo se torna a entrada da etapa supervisionada.

---

## 5.9 Gráfico obrigatório dos clusters

O programa gera um gráfico de dispersão utilizando exatamente os eixos definidos para o grupo:

- **X:** `limite_credito`
- **Y:** `gasto_mensal`
- **Cor:** `cluster_kmeans`

Arquivo gerado:

```text
04_resultados/02_grafico_clusters.png
```

O gráfico permite observar visualmente como os registros dos quatro clusters se distribuem em relação ao limite de crédito e ao gasto mensal.

---

# 6. Etapa 2 — Rede Neural MLP

A segunda etapa é implementada no arquivo:

```text
03_codigo/02_mlp_supervisionado.py
```

Nesta fase, o projeto deixa de ser não supervisionado e passa a trabalhar com um problema **supervisionado**, porque agora existe uma variável de saída:

```text
rotulo_grupo
```

Essa variável foi criada pelo grupo a partir dos clusters do K-Means.

A função da MLP é aprender a relação entre os cinco atributos originais e esses rótulos.

---

## 6.1 Classe `ClassificadorPerfis`

O programa organiza a lógica principal dentro da classe:

```python
ClassificadorPerfis
```

Ela centraliza:

- carregamento dos dados;
- divisão entre treino e teste;
- padronização;
- criação e treinamento da MLP;
- geração das curvas de perda;
- classificação dos novos casos.

---

## 6.2 Divisão entre treino e teste

A separação utiliza:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Isso produz aproximadamente:

- **80% dos dados para treinamento:** 6.400 registros;
- **20% dos dados para teste:** 1.600 registros.

O parâmetro `stratify=y` preserva aproximadamente a proporção das quatro classes nos conjuntos de treino e teste.

---

## 6.3 Padronização sem vazamento de dados

O `StandardScaler` da etapa supervisionada é ajustado **somente sobre os dados de treinamento**:

```python
self.X_treino = self.scaler.fit_transform(X_treino)
```

Depois, o mesmo scaler é aplicado ao conjunto de teste apenas com:

```python
self.X_teste = self.scaler.transform(X_teste)
```

Essa separação é importante para evitar que informações estatísticas do conjunto de teste sejam utilizadas durante o ajuste do pré-processamento.

---

## 6.4 Arquitetura da MLP

A rede é criada com:

```python
MLPClassifier(
    hidden_layer_sizes=(5,),
    max_iter=limite_iteracoes,
    random_state=42,
    verbose=True
)
```

### Entradas

São utilizados os mesmos cinco atributos do K-Means:

```text
idade
limite_credito
gasto_mensal
transacoes_mes
parcelamentos_ativos
```

### Camada oculta

```python
hidden_layer_sizes=(5,)
```

Essa configuração define **uma camada oculta com 5 neurônios**.

### Saída

A rede classifica cada registro em um dos quatro rótulos:

```text
perfil_gasto_consumo_baixo
perfil_gasto_consumo_moderado
perfil_gasto_consumo_alto
perfil_alto_parcelamento
```

### `verbose=True`

O parâmetro obrigatório `verbose=True` faz o treinamento exibir a evolução da função de perda a cada iteração, tornando possível acompanhar o processo de aprendizado.

### `random_state=42`

Mantém o experimento reproduzível.

---

# 7. Experimentos obrigatórios da MLP

Conforme solicitado, todos os parâmetros foram mantidos iguais e somente `max_iter` foi alterado entre os três experimentos:

```python
for ciclos in [50, 500, 5000]:
    motor.executar_experimento(ciclos)
```

É importante diferenciar **`max_iter`** da quantidade real de iterações executadas. `max_iter` é apenas o limite máximo permitido; o treinamento pode terminar antes caso o algoritmo atinja seu critério de convergência.

## 7.1 Tabela de resultados

| Experimento | `max_iter` | Iterações executadas (`n_iter_`) | Perda final (`loss_`) | Acurácia treino | Acurácia teste | Observação |
|---|---:|---:|---:|---:|---:|---|
| A | 50 | 50 | 0,076543 | 98,80% | 99,06% | Atingiu o limite de 50 iterações; ainda havia espaço para reduzir a perda. |
| B | 500 | 253 | 0,013051 | 99,78% | 99,81% | Convergiu antes do limite e apresentou melhora de perda e acurácia. |
| C | 5000 | 253 | 0,013051 | 99,78% | 99,81% | Resultado idêntico ao experimento B; aumentar o limite não trouxe benefício. |

---

## 7.2 Interpretação dos experimentos

Os resultados mostram que a função de perda diminuiu significativamente entre o experimento de 50 iterações e os experimentos com limites maiores.

O experimento A atingiu exatamente as **50 iterações permitidas**, indicando que o treinamento foi interrompido pelo limite definido em `max_iter`.

Ao aumentar o limite para 500, o modelo continuou aprendendo até **253 iterações**, quando atingiu seu critério de convergência. A perda caiu de aproximadamente **0,0765 para 0,0131**, enquanto a acurácia de teste aumentou de **99,06% para 99,81%**.

Quando `max_iter` foi aumentado para 5000, o modelo novamente terminou em **253 iterações** e apresentou exatamente a mesma perda e as mesmas acurácias do experimento com 500.

Isso demonstra que **definir mais iterações máximas não significa obrigatoriamente obter um modelo melhor**. Depois que o treinamento converge, aumentar apenas `max_iter` pode não produzir qualquer melhoria.

---

## 7.3 Melhor experimento

O **experimento B (`max_iter=500`)** pode ser considerado a melhor escolha entre os três.

Ele:

- alcançou a menor perda observada;
- obteve **99,81% de acurácia no conjunto de teste**;
- convergiu em 253 iterações;
- produziu o mesmo resultado do limite de 5000 iterações;
- evita utilizar um limite muito maior sem ganho prático.

O experimento C não é pior em resultado, mas é desnecessariamente mais permissivo, pois a rede já havia convergido muito antes do limite.

> Uma acurácia alta neste trabalho significa principalmente que a MLP conseguiu reproduzir muito bem os rótulos derivados do K-Means. Ela não representa validação de uma classificação real de risco ou perfil financeiro.

---

# 8. Curvas de perda

A MLP mantém o histórico da função de perda no atributo:

```python
loss_curve_
```

O programa gera uma curva para cada experimento:

```text
04_resultados/curva_perda_50.png
04_resultados/curva_perda_500.png
04_resultados/curva_perda_5000.png
```

Essas imagens permitem acompanhar a redução do erro ao longo do treinamento e verificar se o modelo ainda estava melhorando ou se já havia atingido estabilidade.

---

# 9. Teste com cinco novos clientes

Além da divisão tradicional entre treino e teste, o trabalho possui cinco casos inéditos no arquivo:

```text
02_dados_tratados/novos_clientes.csv
```

Cada caso possui:

- os cinco atributos de entrada;
- `rotulo_esperado`;
- posteriormente, `rotulo_previsto`;
- indicação final de `ACERTO` ou `ERRO`.

O método `prever_novos_casos()`:

1. seleciona os cinco atributos;
2. utiliza o mesmo scaler ajustado no conjunto de treino;
3. executa `predict()` com o modelo mais recente;
4. compara o rótulo previsto com o esperado;
5. salva o resultado em CSV.

O arquivo final é:

```text
04_resultados/casos_de_teste.csv
```

### Resultado dos cinco casos

| Caso | Rótulo esperado | Rótulo previsto | Resultado |
|---:|---|---|---|
| 1 | `perfil_gasto_consumo_alto` | `perfil_gasto_consumo_alto` | ACERTO |
| 2 | `perfil_gasto_consumo_moderado` | `perfil_gasto_consumo_moderado` | ACERTO |
| 3 | `perfil_gasto_consumo_alto` | `perfil_gasto_consumo_alto` | ACERTO |
| 4 | `perfil_gasto_consumo_baixo` | `perfil_gasto_consumo_baixo` | ACERTO |
| 5 | `perfil_gasto_consumo_moderado` | `perfil_gasto_consumo_moderado` | ACERTO |

O modelo acertou os **5 de 5 casos apresentados**.

> Esses cinco casos são uma verificação complementar. Eles não substituem a avaliação sobre os 20% de dados separados para teste durante o treinamento.

---

# 10. Fluxo completo da solução

```text
02_cartao_credito.csv
        │
        ▼
Validação das 5 colunas
        │
        ▼
Padronização dos atributos
        │
        ▼
Teste de k = 2, 3, 4 e 5
        │
        ▼
Inércia + Silhueta + interpretação
        │
        ▼
Escolha de k = 4
        │
        ▼
K-Means final
        │
        ▼
cluster_kmeans
        │
        ▼
Análise das médias dos clusters
        │
        ▼
Criação de rotulo_grupo
        │
        ▼
dados_clusterizados_rotulados.csv
        │
        ▼
Divisão 80% treino / 20% teste
        │
        ▼
StandardScaler ajustado no treino
        │
        ▼
MLPClassifier
        │
        ├── max_iter = 50
        ├── max_iter = 500
        └── max_iter = 5000
        │
        ▼
Comparação de perda e acurácia
        │
        ▼
Melhor configuração: max_iter = 500
        │
        ▼
Teste com 5 casos inéditos
```

---

# 11. Tecnologias utilizadas

O projeto foi desenvolvido em **Python** e utiliza:

| Biblioteca | Finalidade |
|---|---|
| `pandas` | Leitura, manipulação, agrupamento e exportação dos dados. |
| `matplotlib` | Criação das curvas de perda e suporte aos gráficos. |
| `seaborn` | Visualização das métricas e dos clusters. |
| `scikit-learn` | `StandardScaler`, `KMeans`, `silhouette_score`, `train_test_split`, `MLPClassifier` e `accuracy_score`. |
| `pathlib` | Manipulação de caminhos e diretórios. |

---

# 12. Como executar

## 12.1 Pré-requisitos

É necessário ter o Python instalado.

Instale as dependências com:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

---

## 12.2 Executar o K-Means

A partir da **raiz do projeto**, execute:

```bash
python 03_codigo/01_kmeans.py
```

Esse programa irá:

- ler o CSV original;
- validar as cinco variáveis;
- padronizar os atributos;
- testar `k = 2`, `3`, `4` e `5`;
- calcular inércia e silhueta;
- gerar o gráfico de apoio das métricas;
- executar o K-Means final com `k = 4`;
- gerar o gráfico obrigatório dos clusters;
- criar `cluster_kmeans`;
- criar `rotulo_grupo`;
- salvar o dataset rotulado.

---

## 12.3 Executar a MLP

Na implementação atual, os caminhos da MLP são relativos à pasta `03_codigo`. Portanto, execute:

```bash
cd 03_codigo
python 02_mlp_supervisionado.py
```

O programa irá:

- ler `dados_clusterizados_rotulados.csv`;
- separar treino e teste;
- ajustar o `StandardScaler` apenas no treino;
- treinar a MLP com 50, 500 e 5000 iterações máximas;
- mostrar o treinamento no terminal com `verbose=True`;
- calcular a acurácia de teste;
- salvar as três curvas de perda;
- classificar os cinco novos clientes;
- salvar `casos_de_teste.csv`.

---

# 13. Principais arquivos gerados

### `04_resultados/metricas_escolha_k.csv`

Tabela com inércia e silhueta para `k = 2`, `3`, `4` e `5`.

### `04_resultados/01_metricas_escolha_k.png`

Gráficos do método do cotovelo e do índice de silhueta.

### `04_resultados/dados_clusterizados.csv`

Dataset acrescido da coluna numérica `cluster_kmeans`.

### `04_resultados/02_grafico_clusters.png`

Gráfico obrigatório com `limite_credito` no eixo X e `gasto_mensal` no eixo Y.

### `02_dados_tratados/dados_clusterizados_rotulados.csv`

Dataset final da etapa de K-Means, contendo `cluster_kmeans` e `rotulo_grupo`.

### `04_resultados/curva_perda_50.png`

Curva de perda do experimento com `max_iter=50`.

### `04_resultados/curva_perda_500.png`

Curva de perda do experimento com `max_iter=500`.

### `04_resultados/curva_perda_5000.png`

Curva de perda do experimento com `max_iter=5000`.

### `04_resultados/casos_de_teste.csv`

Resultado dos cinco novos clientes com rótulo esperado, previsão da MLP e acerto/erro.

---

# 14. Conceitos de Machine Learning demonstrados

### Aprendizado não supervisionado

O K-Means encontra agrupamentos sem receber classes prontas.

### Clusterização

Registros com características semelhantes são associados ao mesmo grupo.

### Padronização

Transforma atributos com escalas diferentes para que possam ser comparados de forma mais equilibrada pelos algoritmos.

### Inércia e método do cotovelo

Ajudam a analisar quanto os pontos estão próximos dos centros de seus clusters e como esse valor evolui ao aumentar `k`.

### Índice de silhueta

Avalia a coesão e a separação dos grupos.

### Rotulagem interpretável

Transforma números de clusters em descrições baseadas nas características observadas nos dados.

### Aprendizado supervisionado

A MLP recebe exemplos já rotulados e aprende a prever esses rótulos.

### Separação treino/teste

Permite avaliar a rede em registros que não participaram diretamente de seu treinamento.

### Prevenção de vazamento de dados

O scaler é ajustado exclusivamente no conjunto de treinamento antes de ser aplicado ao conjunto de teste.

### Função de perda

Mostra o erro que a rede procura reduzir durante o treinamento.

### Acurácia

Representa a proporção de previsões corretas.

### Convergência

O fato de os experimentos de 500 e 5000 terminarem ambos com 253 iterações mostra que `max_iter` funciona como limite máximo, e não como obrigação de executar toda aquela quantidade.

---

# 15. Limitações e interpretação correta

Este trabalho possui algumas limitações que precisam ser reconhecidas:

1. **Os dados são sintéticos.** Os padrões encontrados pertencem à base construída para a atividade.
2. **Os clusters não representam uma verdade absoluta.** Eles são resultado das variáveis escolhidas, da padronização e da configuração do K-Means.
3. **Os rótulos foram criados pelo grupo.** Eles são interpretações das médias e características dos clusters.
4. **A MLP aprende esses rótulos.** Portanto, uma acurácia alta mostra principalmente que a rede reproduz bem a segmentação feita anteriormente.
5. **Não há conclusão causal.** O projeto identifica padrões de associação nos dados, mas não demonstra que uma variável causa outra.
6. **Os resultados não devem ser usados para decisões reais sobre pessoas ou concessão de crédito.**

---

# 16. Conclusão

O projeto cumpriu o objetivo de conectar duas abordagens diferentes de Machine Learning.

Primeiro, o **K-Means** foi utilizado para explorar uma base sem rótulos e descobrir padrões de comportamento. Foram comparados quatro valores de `k`, e o grupo escolheu `k = 4` considerando conjuntamente inércia, silhueta, distribuição dos registros e interpretabilidade.

Os quatro clusters foram analisados por meio de suas médias e receberam os rótulos `perfil_gasto_consumo_baixo`, `perfil_gasto_consumo_moderado`, `perfil_gasto_consumo_alto` e `perfil_alto_parcelamento`.

Em seguida, esses rótulos foram utilizados como variável-alvo para uma **MLPClassifier**. Os experimentos mostraram uma melhora clara ao passar de um limite de 50 para 500 iterações. O experimento de 500 convergiu em 253 iterações e atingiu aproximadamente **99,81% de acurácia no teste**. Aumentar o limite para 5000 não alterou o resultado, demonstrando que mais iterações máximas não significam necessariamente um modelo melhor.

Por fim, a rede classificou corretamente os cinco casos inéditos incluídos no projeto.

O principal aprendizado do trabalho é compreender o fluxo completo entre **descoberta de padrões**, **interpretação de clusters**, **criação de rótulos** e **treinamento de um classificador supervisionado**, sempre reconhecendo os limites da base sintética e da própria segmentação criada pelo grupo.

---

## Resumo dos resultados

| Item | Resultado |
|---|---|
| Registros | 8.000 |
| Atributos | 5 |
| Valores de `k` testados | 2, 3, 4 e 5 |
| `k` escolhido | 4 |
| Perfis criados | 4 |
| Divisão treino/teste | 80% / 20% |
| Arquitetura MLP | 1 camada oculta com 5 neurônios |
| Experimentos | `max_iter=50`, `500` e `5000` |
| Melhor configuração prática | `max_iter=500` |
| Iterações executadas no melhor experimento | 253 |
| Perda final | 0,013051 |
| Acurácia de treino | 99,78% |
| Acurácia de teste | 99,81% |
| Casos inéditos corretos | 5 de 5 |

---

**Projeto acadêmico — P1 do Grupo 02: Cartão de Crédito — K-Means e Rede Neural MLP.**
