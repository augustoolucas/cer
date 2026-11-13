# Baseline do gearbox: exploração com validação agrupada aninhada

Estudo exploratório de previsão de RUL no dataset **gearbox**, com três eventos de falha (E1, E2,
E3), seis papéis de medição por evento e duas trilhas de avaliação: previsão por série (822
endpoints) e eventos sincronizados (137 timestamps). Todo erro é reportado em RUL 0..1.

## 1. Dados e alvo

- 18 séries de medição (6 por evento), que geram **822 endpoints** de série.
- Seis papéis por evento: HSS radial (posição 1), HSS axial (2), IMS (3), LSS (4), Planetaria 0°
  (5) e Planetaria 90° (6). Quatro componentes físicos: HSS (posições 1–2), IMS (3), LSS (4) e
  Planetaria (5–6).
- Trilha de eventos: **137 timestamps sincronizados** nos 3 eventos (71 + 33 + 33), em que as seis
  medições existem com a mesma data, mesmo alvo e mesma fase; nenhuma linha é preenchida ou
  propagada.
- Alvo em 0..1; entradas apenas com histórico observado até o endpoint.

## 2. Método

Validação aninhada e agrupada com **3 folds externos** (um evento retido por vez) e **2 folds
internos** (treina em um dos eventos restantes, valida no outro). O escore interno é MAE por
`Falha` dentro de cada grupo interno, média igual entre componentes, depois média igual entre
grupos. Pré-processamento ajustado só no treino; seleção apenas por escore interno; escolhas
congeladas antes da avaliação de desempenho externo; rótulos externos não participam da
seleção nem do ajuste. Empates dentro de **0.001** (RUL) vão para menor
complexidade, depois menor RMSE, depois identificador estável.

Busca por dataset: 8 tentativas por família, até 40 adicionais para as 3 melhores (platô: mínimo
16 e 12 sem ganho ≥ **0.005**), até 12 ablações, retreino de até 16 tentativas apenas com ganho ≥
0.005, 3 sementes fixas nas 2 famílias estocásticas melhores, ensembles uniformes e limite de 8000
ajustes.

Estratégias avaliadas para a trilha de série, além do conjunto agrupado (pooled):

- **pooled anônimo**: todas as séries juntas, sem identificação de papel;
- **pooled + one-hot de medição**: idem, com a posição conhecida como metadado categórico
  (apenas posições conhecidas, nunca identidade de evento);
- **4 especialistas por componente** (HSS, IMS, LSS, Planetaria), cada um treinado apenas com suas
  medições;
- **6 especialistas por medição**, um por papel.

Cada especialista é treinado excluindo **todas** as seis medições do evento retido; especialista
ausente é falha registrada, nunca substituído por um modelo global. A trilha de eventos usa as 48
colunas de sinal (6 blocos de 8) por timestamp e uma agregação separada das seis predições de
série (média ou mediana), com a regra escolhida na validação interna.

Sondagens de estratégia: até 2 receitas por família (apenas `ridge`, `extra_trees`, `hist_gb`) em
cada estratégia, comparadas ao melhor resultado pooled já ajustado da mesma trilha; expansão além
da sonda somente com ganho interno ≥ 0.005. Nenhuma sondagem atingiu esse ganho, então nenhuma
estratégia foi expandida — mas os candidatos das sondas **permaneceram elegíveis** e entraram nos
melhores resultados por família (por exemplo, os especialistas por medição do fold E1). A escolha
do fold E3, por sua vez, vem da busca pooled principal com o metadado de medição ativado
(one-hot), escolhido na ablação interna, e não de uma sondagem.

## 3. Trilha por série: resultados

| evento | n | escolha | escore interno | MAE |
| --- | --- | --- | --- | --- |
| E1 | 426 | catboost | 0.063198 | 0.113597 |
| E2 | 198 | random_forest | 0.144370 | 0.131902 |
| E3 | 198 | hist_gb | 0.076905 | 0.114749 |

| evento | n | MAE | RMSE (por Falha) | viés |
| --- | --- | --- | --- | --- |
| E1 | 426 | 0.113597 | 0.152413 | -0.040067 |
| E2 | 198 | 0.131902 | 0.178201 | 0.120077 |
| E3 | 198 | 0.114749 | 0.160873 | -0.030218 |
| **média igual por evento** | 822 | **0.120082** | **0.163829** | **0.016597** |

Pooled sobre 822 endpoints: MAE **0.118283**, RMSE
**0.161872**, viés **0.000880**.

MAE por papel de medição (média das séries daquele papel no evento retido):

| papel | E1 | E2 | E3 | média igual por evento |
| --- | --- | --- | --- | --- |
| HSS radial | 0.109366 | 0.116475 | 0.111026 | 0.112289 |
| HSS axial | 0.130652 | 0.123676 | 0.119420 | 0.124583 |
| IMS | 0.101069 | 0.137790 | 0.122745 | 0.120534 |
| LSS | 0.095414 | 0.121165 | 0.118941 | 0.111840 |
| Planetaria 0° | 0.122275 | 0.149142 | 0.102146 | 0.124521 |
| Planetaria 90° | 0.122802 | 0.143162 | 0.114217 | 0.126727 |

Visão secundária por componente físico, com **pesos diferentes** da métrica principal (média das
posições dentro do componente, não as seis posições equiponderadas); serve apenas como complemento
e não escolhe vencedor:

| componente | E1 | E2 | E3 | média igual por evento |
| --- | --- | --- | --- | --- |
| HSS | 0.120009 | 0.120076 | 0.115223 | 0.118436 |
| IMS | 0.101069 | 0.137790 | 0.122745 | 0.120534 |
| LSS | 0.095414 | 0.121165 | 0.118941 | 0.111840 |
| Planetaria | 0.122539 | 0.146152 | 0.108181 | 0.125624 |

Referências constantes no mesmo protocolo: `mean` **0.303763** e
`median` **0.301504**, contra
**0.120082** do seletor.

Recortes por fase, proximidade do fim de vida e limites das previsões (séries):

| evento | MAE saudável | MAE pós-FPT | MAE y≤0.2 (n) | raw finitas | raw fora | pós fora |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | 0.049796 | 0.169002 | 0.290419 (24) | 426/426 | 6/426 | 0/426 |
| E2 | 0.014812 | 0.208010 | 0.332646 (30) | 198/198 | 0/198 | 0/198 |
| E3 | 0.015346 | 0.164451 | 0.187181 (36) | 198/198 | 0/198 | 0/198 |

## 4. Trilha de eventos sincronizados: resultados

48 colunas de sinal por timestamp, um alvo por evento, sem preenchimentos. Esta trilha **não** é
comparável registro a registro com a trilha por série (frames de disponibilidade diferentes).

| evento | n timestamps | escolha | escore interno | MAE |
| --- | --- | --- | --- | --- |
| E1 | 71 | ridge | 0.168155 | 0.231055 |
| E2 | 33 | ensemble (extra_trees+hist_gb+ridge, median) | 0.186562 | 0.127770 |
| E3 | 33 | ensemble (extra_trees+ridge+hist_gb, median) | 0.111171 | 0.169686 |

| evento | n | MAE | RMSE | viés |
| --- | --- | --- | --- | --- |
| E1 | 71 | 0.231055 | 0.348500 | 0.207147 |
| E2 | 33 | 0.127770 | 0.186027 | 0.107790 |
| E3 | 33 | 0.169686 | 0.228703 | -0.143621 |
| **média igual por evento** | 137 | **0.176170** | **0.254410** | **0.057106** |

Pooled sobre 137 timestamps: MAE **0.191394**, RMSE
**0.289616**, viés **0.098723**.

## 5. Trilha agregada (média/mediana das seis séries)

A regra de agregação (média ou mediana das seis predições por timestamp) foi escolhida na validação
interna e aplicada igualmente no conjunto externo:

| evento | regra | MAE | RMSE | viés |
| --- | --- | --- | --- | --- |
| E1 | median | 0.095454 | 0.126608 | -0.035684 |
| E2 | mean | 0.131747 | 0.176431 | 0.120077 |
| E3 | median | 0.109937 | 0.154043 | -0.035640 |
| **média igual por evento** | — | **0.112379** | **0.152361** | **0.016251** |

Pooled sobre 137 timestamps: MAE **0.107685**, RMSE
**0.146701**, viés **0.001845**. É o mesmo
conjunto de timestamps da seção anterior com outra entrada; as duas tabelas não devem ser fundidas
nem ranqueadas contra a tabela de 822 endpoints.

## 6. Sondagens de estratégia

| evento | referência pooled ajustada | pooled anônimo (sonda) | + one-hot | especialistas por componente | especialistas por medição |
| --- | --- | --- | --- | --- | --- |
| E1 | 0.060145 | 0.183026 | 0.183026 | 0.174101 | 0.166859 |
| E2 | 0.145183 | 0.210457 | 0.209178 | 0.196454 | 0.196329 |
| E3 | 0.076905 | 0.229845 | 0.221873 | 0.146707 | 0.142581 |

Nenhuma sonda alcançou ganho ≥ 0.005 sobre a referência ajustada, então nenhuma estratégia foi
expandida.

## 7. Estabilidade

MAEs de validação interna das 3 sementes fixas por família. “OOF médio” é o escore da candidata
com predições internas médias entre sementes, não a média dos três escores; a semente mais
favorável nunca é escolhida.

| evento | família | escores por semente (internos) | média | desvio | OOF médio da média |
| --- | --- | --- | --- | --- | --- |
| E1 | extra_trees | 0.124509, 0.125583, 0.124631 | 0.124908 | 0.000480 | 0.124918 |
| E1 | catboost | 0.060145, 0.072758, 0.062746 | 0.065216 | 0.005437 | 0.063198 |
| E2 | extra_trees | 0.174655, 0.174984, 0.172528 | 0.174056 | 0.001088 | 0.173902 |
| E2 | random_forest | 0.145183, 0.144208, 0.143878 | 0.144423 | 0.000554 | 0.144370 |
| E3 | extra_trees | 0.142581, 0.149165, 0.144681 | 0.145476 | 0.002746 | 0.145201 |
| E3 | catboost | 0.135803, 0.135456, 0.136912 | 0.136057 | 0.000621 | 0.135139 |

Famílias escolhidas: `catboost` (E1), `random_forest` (E2) e `hist_gb` (E3); nenhuma se repete.

## 8. Escores internos por família (série)

| família | E1 | E2 | E3 |
| --- | --- | --- | --- |
| mean | 0.31704 | 0.29916 | 0.30179 |
| median | 0.31012 | 0.29568 | 0.29961 |
| ridge | 0.19014 | 0.19843 | 0.16578 |
| elasticnet | 0.15389 | 0.23073 | 0.11561 |
| huber | 0.25250 | 0.39396 | 0.23541 |
| svr | 0.18434 | 0.23007 | 0.17911 |
| knn | — | 0.15336 | 0.15388 |
| extra_trees | 0.12492 | 0.17390 | 0.14520 |
| random_forest | 0.14732 | 0.14437 | 0.16585 |
| hist_gb | 0.13496 | 0.18232 | 0.07690 |
| catboost | 0.06320 | 0.18003 | 0.13514 |

## 9. Controle de tempo e cadência

Controle pré-declarado com apenas tempo de aquisição e cadência, receitas fixas e sem busca:
`mean`, `median`, `ridge` (α=1), `extra_trees` (128 árvores, profundidade livre, folha 1,
`max_features=1.0`) e `hist_gb` (100 iterações, folha 10, 15 folhas, lr 0.1, L2=1, erro absoluto,
sem parada antecipada). São cinco famílias (uma a mais que o controle do gerador), então os dois
controles não são referências idênticas.

| trilha | n | MAE | RMSE | viés |
| --- | --- | --- | --- | --- |
| série (822 endpoints) | 822 | 0.175186 | 0.228412 | -0.003670 |
| eventos (137 timestamps) | 137 | 0.182011 | 0.266630 | 0.057965 |
| agregada (137 timestamps) | 137 | 0.138397 | 0.175904 | -0.018042 |

| evento | escolha do controle | MAE (série) |
| --- | --- | --- |
| E1 | hist_gb | 0.164393 |
| E2 | extra_trees | 0.183642 |
| E3 | extra_trees | 0.177522 |

Nestes três eventos, o pipeline buscado teve MAE menor que o controle temporal na trilha por série
(0.120082 contra
0.175186). Isso descreve estes dois pipelines neste dataset;
não isola efeito de uma única mudança nem estabelece informação causal nos sensores.

## 10. Limitações

- Três eventos externos apenas; cada fold interno treina em um único evento. As diferenças entre
  famílias não são resolvidas por este desenho e não há teste de significância.
- Frames distintos: 822 endpoints de série, 137 timestamps de evento, 137 linhas agregadas. Nunca
  são agrupados nem ranqueados entre si.
- A trilha de eventos descreve operação sincronizada, não todos os timestamps de medição.
- Erros de regressão bruta não são limitados pelo intervalo do alvo; o recorte 0..1 é
  pós-processamento e é reportado separadamente das predições brutas.
- Sem MAPE, receita ótima universal ou afirmação causal.

## 11. Reprodutibilidade

- Dataset: `data/Features_gearbox_score_todos_RUL-porcentagem-data.csv`, sha256
  `332d452bbe2a1c2d4cb4a3b046e5643e386545f0db3cad323dbfd7defdbdd0e4`.
- Run principal: `results/baselines/gearbox/101e2f1fc3d406b1`; controle:
  `results/baselines/gearbox/controls/gearbox/6932403970ae9c17` (configuração preservada em
  `results/baselines/gearbox/controls/`).
- Versões: Python 3.12.3, numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, optuna 5.0.0,
  catboost 1.2.10.
- Curvas descritivas (RUL 0..1): séries por evento
  [E1](../../results/baselines/gearbox/101e2f1fc3d406b1/plots/gearbox_series_outerE1.png),
  [E2](../../results/baselines/gearbox/101e2f1fc3d406b1/plots/gearbox_series_outerE2.png),
  [E3](../../results/baselines/gearbox/101e2f1fc3d406b1/plots/gearbox_series_outerE3.png); eventos sincronizados
  [E1](../../results/baselines/gearbox/101e2f1fc3d406b1/plots/gearbox_event_outerE1.png),
  [E2](../../results/baselines/gearbox/101e2f1fc3d406b1/plots/gearbox_event_outerE2.png),
  [E3](../../results/baselines/gearbox/101e2f1fc3d406b1/plots/gearbox_event_outerE3.png).
