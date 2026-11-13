# Baseline do gerador: exploração com validação agrupada aninhada

Este documento descreve um estudo exploratório de previsão de RUL (vida útil restante) no
dataset **gerador**, comparando um seletor de modelos treinado por validação interna com
referências simples e com um controle que usa apenas tempo de aquisição e cadência. As métricas
usam a unidade do alvo (RUL em 0..1) e são sempre reportadas como erro absoluto médio (MAE),
RMSE e viés médio assinado no conjunto externo, além de escores de validação interna.

## 1. Dados e alvo

- Alvo: RUL normalizada em **0..1**, calculada a partir do histórico de cada equipamento.
- 574 endpoints (um por registro avaliado), em **4 grupos de equipamento** (máquinas A, B, C, D).
  Um grupo pode reunir mais de um componente de falha (`Falha`): A = (1, 2), B = (3), C = (4, 5),
  D = (6, 7) — 7 séries de componente ao todo, sem compartilhamento de registros entre grupos.
- Características: apenas informações já observadas até o endpoint (estatísticas de sensores,
  cadência de amostragem, tempo decorrido desde o primeiro registro observado e referências
  relativas restritas aos primeiros registros observados). A data é usada somente dessa forma
  causal; RUL, fase de falha e identificador de grupo nunca entram como entrada.

## 2. Método

Validação cruzada **aninhada e agrupada**:

- externa: leave-one-group-out nos 4 equipamentos (A, B, C, D);
- interna (para escolher receita e hiperparâmetros): leave-one-group-out nos **3 grupos
  restantes**, com o mesmo protocolo em todos os folds;
- escore interno de um candidato: MAE calculado **por componente (`Falha`)** dentro de cada grupo
  interno, média igual entre componentes, depois média igual entre os grupos internos;
- pré-processamento (imputação, normalização, constantes e recortes de quantis) é ajustado apenas
  com os identificadores de treino de cada ajuste; as features de cada endpoint usam somente
  registros até aquele endpoint;
- seleção: escores internos apenas. Empates são resolvidos por menor complexidade (dentro de
  **0.001** em RUL), depois menor RMSE interno, depois identificador estável;
- escolhas congeladas antes da avaliação de desempenho externo; rótulos externos não participam
  da seleção nem do ajuste dos modelos;
- busca por família: 8 tentativas iniciais, até 40 adicionais para as 3 melhores famílias
  (parada por platô: mínimo de 16 tentativas e 12 sem ganho ≥ **0.005** em RUL), até 12 ablações
  de um eixo (retreino de até 16 tentativas apenas se o ganho for ≥ 0.005), confirmação com 3
  sementes fixas nas 2 famílias estocásticas melhores e ensembles uniformes (média de predições
  pós-processadas), limitados a 8000 ajustes por dataset;
- 11 famílias avaliadas: `mean`, `median`, `ridge`, `elasticnet`, `huber`, `svr`, `knn`,
  `extra_trees`, `random_forest`, `hist_gb`, `catboost`; as duas referências constantes (`mean`,
  `median`) também participam da seleção interna.
- pós-processamento opcional por candidato: recorte para 0..1, média móvel exponencial
  (α ∈ {1, 0.5, 0.25}) e mínimo acumulado em ordem cronológica por componente.

Receitas escolhidas por fold:

| fold | n | componentes | escolha | escore interno | complexidade |
| --- | --- | --- | --- | --- | --- |
| A | 352 | 2 | ensemble (random_forest+svr, mean) | 0.093122 | 74 |
| B | 34 | 1 | knn | 0.150318 | 41 |
| C | 138 | 2 | knn | 0.113097 | 62 |
| D | 50 | 2 | catboost | 0.171673 | 20 |

Receitas das escolhas (o fold A é um ensemble uniforme, com os membros listados):

- **A** — `5f4f42aa23f0` (ensemble (random_forest+svr, mean)): membro svr — svr: representação summary/legacy, janela 24 h / 16 bins, grupos sensors+elapsed, rel=10; pré-processamento minmax, pesos group_equal; pós: clip01=True, EMA α=1.0, mínimo acumulado=True; membro random_forest — random_forest: representação current/legacy, janela 24 h / 32 bins, grupos sensors+elapsed, sem variância; pré-processamento signed_log1p_standard, pesos component; pós: clip01=True, EMA α=0.25, mínimo acumulado=True
- **B** — `de6dabdc5ac6` knn: representação summary/dedup, janela 6 h / 16 bins, grupos sensors+elapsed, rel=30, sem variância; pré-processamento none, pesos none; pós: clip01=True, EMA α=0.5, mínimo acumulado=False
- **C** — `3e6f45483361` knn: representação robust/legacy, janela 6 h / 64 bins, grupos sensors+elapsed, rel=10, sem variância; pré-processamento minmax, pesos none; pós: clip01=False, EMA α=0.25, mínimo acumulado=True
- **D** — `78a2b88e7626` catboost: representação current/dedup, janela 24 h / 32 bins, grupos sensors+elapsed, rel=30, sem variância; pré-processamento signed_log1p_standard, pesos none; pós: clip01=True, EMA α=1.0, mínimo acumulado=False

## 3. Resultados no conjunto externo

Métricas do seletor interno, por fold. “Média igual por grupo” atribui o mesmo peso a cada
equipamento, com MAE/RMSE/vies calculados por `Falha` antes da média; “pooled” pondera cada
registro (tabela seguinte).

| fold | n | MAE | RMSE (por Falha) | viés |
| --- | --- | --- | --- | --- |
| A | 352 | 0.307586 | 0.359060 | -0.307342 |
| B | 34 | 0.240598 | 0.266724 | 0.240598 |
| C | 138 | 0.286944 | 0.317872 | 0.286894 |
| D | 50 | 0.218289 | 0.339956 | 0.208157 |
| **média igual por grupo** | 574 | **0.263355** | **0.320903** | **0.107077** |

Pooled sobre 574 registros: MAE **0.290877**, RMSE
**0.354191**, viés **-0.087117**. As duas
agregações não são intercambiáveis: o viés macro é positivo e o pooled é negativo porque o fold A
concentra 352 dos 574 registros e tem viés negativo forte.

Recortes por fase e proximidade do fim de vida, além dos limites das previsões:

| fold | MAE saudável | MAE pós-FPT | MAE y≤0.2 (n) | raw finitas | raw fora de [0,1] | pós fora de [0,1] |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0.153744 | 0.352834 | 0.090469 (56) | 0/352 | n/a | 0/352 |
| B | — | 0.240598 | 0.387958 (10) | 34/34 | 0/34 | 0/34 |
| C | 0.000346 | 0.309335 | 0.341816 (54) | 138/138 | 0/138 | 0/138 |
| D | 0.007099 | 0.359083 | 0.843599 (2) | 50/50 | 7/50 | 0/50 |

O recorte bruto só considera valores finitos; um candidato ensemble não tem predição bruta única e
aparece como “n/a”. O pós-processamento mantém todas as 574 predições dentro de 0..1. O alvo estar
em 0..1 **não limita** o erro de um regressor bruto: o fold D tem previsões brutas fora do
intervalo antes do recorte, e erros acima de 1 continuam possíveis e não são truncados nas
métricas.

## 4. Estabilidade

Confirmação com 3 sementes fixas (mesma receita e mesmas máscaras). Os números são **MAEs de
validação interna** (não desempenho externo); “OOF médio” é o escore da candidata formada pela
média das predições internas das 3 sementes — não a média dos três escores. A semente mais
favorável nunca é escolhida.

| fold | família | escores por semente (internos) | média | desvio | OOF médio da média |
| --- | --- | --- | --- | --- | --- |
| A | random_forest | 0.098881, 0.104529, 0.103723 | 0.102378 | 0.002494 | 0.102204 |
| A | catboost | 0.154409, 0.156966, 0.142538 | 0.151304 | 0.006286 | 0.149269 |
| B | extra_trees | 0.188095, 0.187551, 0.187823 | 0.187823 | 0.000222 | 0.187745 |
| B | random_forest | 0.246890, 0.246454, 0.248015 | 0.247120 | 0.000658 | 0.247154 |
| C | extra_trees | 0.173337, 0.169743, 0.172518 | 0.171866 | 0.001538 | 0.172559 |
| C | catboost | 0.201986, 0.205101, 0.203210 | 0.203432 | 0.001281 | 0.204655 |
| D | extra_trees | 0.200515, 0.204272, 0.206244 | 0.203677 | 0.002376 | 0.203639 |
| D | catboost | 0.162710, 0.183893, 0.171946 | 0.172850 | 0.008672 | 0.171673 |

## 5. Esforço de busca

Contagens de tentativas por fold (uma tentativa pode falhar por convergência e é registrada como
falha, nunca pontuada):

| fold | tentativas | ok | falhas | triagem | ajuste fino | ablações | retreino | sementes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 118 | 108 | 10 | 56 | 49 | 11 | 0 | 2 |
| B | 119 | 109 | 10 | 59 | 48 | 10 | 0 | 2 |
| C | 110 | 100 | 10 | 58 | 40 | 10 | 0 | 2 |
| D | 114 | 107 | 7 | 57 | 28 | 11 | 16 | 2 |

## 6. Referências e famílias

Escores internos por família (RUL 0..1), escolhidos por fold:

| família | A | B | C | D |
| --- | --- | --- | --- | --- |
| mean | 0.33914 | 0.34731 | 0.29760 | 0.32240 |
| median | 0.36064 | 0.36905 | 0.30355 | 0.34247 |
| ridge | 0.13935 | 0.17320 | 0.21093 | 0.23813 |
| elasticnet | 0.18353 | 0.20299 | 0.25942 | 0.19325 |
| huber | — | 0.32618 | 0.27458 | 0.43594 |
| svr | 0.11451 | 0.23686 | 0.13983 | 0.22528 |
| knn | 0.14246 | 0.15032 | 0.11310 | 0.28871 |
| extra_trees | 0.25031 | 0.18774 | 0.17256 | 0.20364 |
| random_forest | 0.10220 | 0.24715 | 0.20285 | 0.21187 |
| hist_gb | 0.12255 | 0.20633 | 0.19586 | 0.20114 |
| catboost | 0.14927 | 0.27219 | 0.20466 | 0.17167 |

MAE externo por família, com a receita escolhida internamente para cada família e refeita nos dados
de treino do fold externo. É uma tabela descritiva: o baseline principal é o seletor interno, não a
família que parece melhor no conjunto externo.

| família | A | B | C | D |
| --- | --- | --- | --- | --- |
| mean | 0.33309 | 0.25318 | 0.36797 | 0.33943 |
| median | 0.36857 | 0.26213 | 0.38924 | 0.33776 |
| ridge | 0.20267 | 0.27770 | 0.41819 | 0.24562 |
| elasticnet | 0.43505 | 0.18295 | 0.31832 | 0.24341 |
| huber | — | 0.24252 | 0.11603 | 0.27002 |
| svr | 0.20888 | 0.22454 | 0.21375 | 0.11838 |
| knn | 0.15656 | 0.24060 | 0.28694 | 0.14068 |
| extra_trees | 0.28324 | 0.24096 | 0.35623 | 0.21504 |
| random_forest | 0.40664 | 0.21461 | 0.20313 | 0.21682 |
| hist_gb | 0.24490 | 0.26529 | 0.41831 | 0.22848 |
| catboost | 0.31383 | 0.24711 | 0.23740 | 0.21829 |

Referências constantes no mesmo protocolo (OOF, média igual por grupo): `mean`
**0.323418** e `median`
**0.339425**, contra
**0.263355** do seletor.

Configurações de referência fixas (presets históricos), avaliadas de forma descritiva sob o mesmo
protocolo interno. Ficam fora da seleção principal: não são candidatas, não entram na escolha final
nem no desempate. Não se confundem com as sondagens de estratégia, que são candidatas.

| preset | A | B | C | D |
| --- | --- | --- | --- | --- |
| mean_summary_none | 0.33914 | 0.34731 | 0.29760 | 0.32240 |
| ridge1_summary_none | 0.38771 | 0.44201 | 0.57369 | 0.62484 |
| ridge10_current_component | 0.44017 | 0.35353 | 0.55295 | 0.43025 |
| svr10_summary_none | 0.27653 | 0.32626 | 0.26506 | 0.30137 |
| svr_current_component_phase | 0.41643 | 0.46672 | 0.37911 | 0.31803 |
| et128_summary_none | 0.27852 | 0.31994 | 0.21907 | 0.35236 |
| et_current_component | 0.28371 | 0.38310 | 0.29781 | 0.39780 |

## 7. Controle de tempo e cadência

Controle pré-declarado que usa **apenas** tempo de aquisição e cadência (sem colunas de sensores),
com receitas fixas e sem busca: `mean`, `median`, `ridge` (α=1), `hist_gb` (100 iterações, folha
10, 15 folhas, lr 0.1, L2=1, erro absoluto, sem parada antecipada). Ele mede o quanto o atalho
temporal explica o alvo neste protocolo.

| fold | n | escolha | escore interno | MAE | RMSE | viés |
| --- | --- | --- | --- | --- | --- | --- |
| A | 352 | ridge | 0.153903 | 0.224286 | 0.256313 | -0.224286 |
| B | 34 | ridge | 0.221493 | 0.189883 | 0.229655 | 0.181260 |
| C | 138 | hist_gb | 0.225148 | 0.145649 | 0.173820 | 0.121050 |
| D | 50 | hist_gb | 0.238270 | 0.155272 | 0.211851 | 0.059422 |
| **média igual por grupo** | 574 | — | — | **0.178772** | **0.217910** | **0.034361** |

Famílias do controle, escore interno (RUL 0..1):

| família | A | B | C | D |
| --- | --- | --- | --- | --- |
| mean | 0.33914 | 0.34731 | 0.29760 | 0.32240 |
| median | 0.36064 | 0.36905 | 0.30355 | 0.34247 |
| ridge | 0.15390 | 0.22149 | 0.35227 | 0.27966 |
| extra_trees | — | — | — | — |
| hist_gb | 0.16401 | 0.26903 | 0.22515 | 0.23827 |

## 8. Limitações

- Quatro grupos externos apenas; diferenças de poucos milésimos não são resolvidas por este
  desenho, e nenhum teste de significância é apresentado.
- As métricas externas são out-of-fold, mas não selecionaram nada; o controle foi fixado antes de
  qualquer resultado externo.
- Nenhuma afirmação de causalidade sobre sensores, de receita ótima universal, de MAPE ou de
  superioridade geral.
- Falhas de convergência registradas não são imputadas; famílias com falhas podem ficar
  ligeiramente sub-representadas nas tabelas descritivas.
- O controle temporal usa outro conjunto de famílias e nenhuma busca; comparar os dois pipelines é
  descrever estes dois experimentos, não isolar o efeito de uma única mudança.

## 9. Reprodutibilidade

- Dataset: `data/Features_gerador_score_RUL-porcentagem-data.csv`, sha256
  `3cea9d774f19212880cee8aace0563f52003606fe27fa56a6af8d81d481be767`.
- Run principal: `results/baselines/gerador/717be61e363ea09d`; controle:
  `results/baselines/gerador/controls/gerador/4500dafde38e4de6` (configuração preservada em
  `results/baselines/gerador/controls/`).
- Versões: Python 3.12.3, numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, optuna 5.0.0,
  catboost 1.2.10.
- Curvas descritivas (RUL 0..1, um painel por `Falha`):
  [A](../../results/baselines/gerador/717be61e363ea09d/plots/gerador_outerA_by_fault.png),
  [B](../../results/baselines/gerador/717be61e363ea09d/plots/gerador_outerB_by_fault.png),
  [C](../../results/baselines/gerador/717be61e363ea09d/plots/gerador_outerC_by_fault.png),
  [D](../../results/baselines/gerador/717be61e363ea09d/plots/gerador_outerD_by_fault.png).
