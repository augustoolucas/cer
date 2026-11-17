# Estimativa de RUL em ativos CPFL: modelos tabulares de referência e replicações com Chronos-2

## Resumo

Dois conjuntos de dados de sensores (gerador e gearbox) foram usados para estimar a vida útil
restante (RUL) em fração da vida, com nested group cross-validation. Foram comparadas duas famílias
de abordagem: modelos tabulares ajustados por busca de hiperparâmetros e duas replicações
metodológicas baseadas no modelo fundacional Chronos-2, mantido congelado.

O melhor resultado geral veio do modelo tabular no gearbox: MAE de 0,120 e RMSE de 0,164 na trilha
por série (média macro, peso igual por evento), reduzido para MAE de 0,112 e RMSE de 0,152 quando
as previsões das seis medições de um mesmo evento são agregadas. No gerador, o melhor resultado foi
do controle de tempo e cadência, que usa apenas tempo e cadência de coleta, com MAE de 0,179 e RMSE
de 0,218. As replicações com Chronos-2 não superaram nenhuma dessas referências: usando
representações do modelo fundacional, o menor erro foi MAE 0,173 / RMSE 0,224 (Abdouni `fusion`,
gearbox) e MAE 0,327 / RMSE 0,379 (El-Ghoussani `mlp4`, gerador); no gerador, a ablação Abdouni
`wide` (apenas sensores correntes, sem representações) atingiu MAE 0,305 / RMSE 0,350.

## 1. Objetivo e escopo

O estudo avalia a estimativa causal da Vida Útil Restante (RUL, normalizada em fração de vida entre
0 e 1) a partir de séries históricas de sensores em dois conjuntos independentes de ativos CPFL
(**gerador** e **gearbox**), utilizando exclusivamente observações disponíveis até o instante
previsto.

O foco central é medir a generalização para um grupo inteiro não visto no treino (uma máquina no
gerador ou um evento completo de falha no gearbox) e verificar em que medida os sinais dos sensores
agregam poder preditivo além da simples passagem do tempo. Para isso, quatro níveis crescentes de
complexidade são comparados sob o mesmo protocolo de validação:

1. **Previsores constantes:** média ou mediana dos valores de treino, como piso não informativo.
2. **Controle de tempo e cadência:** modelos com receitas fixas que usam apenas tempo decorrido e
   cadência de coleta, sem leitura de sensores, para quantificar o atalho temporal.
3. **Modelos tabulares com busca de pipeline:** seleção automática interna de representação causal
   em janelas, pré-processamento, família de algoritmo e pós-processamento.
4. **Modelo fundacional congelado (`Chronos-2`) com head supervisionada:** extração de
   representações pré-treinadas acoplada a *heads* neurais treináveis, seguindo duas replicações
   metodológicas da literatura (El-Ghoussani e Abdouni) adaptadas ao protocolo CPFL.

Os conjuntos **gerador** e **gearbox** são tratados de forma totalmente independente: nenhum dado,
pré-processamento, receita ou hiperparâmetro é compartilhado ou transferido entre eles. Todos os
experimentos utilizam o mesmo alvo (`porcentagem-data`) e o mesmo esquema de *nested group
cross-validation*.

## 2. Dados

| Conjunto | Séries | Registros limpos | Grupos de avaliação | Pontos de avaliação | Sensores |
| --- | --- | --- | --- | --- | --- |
| Gerador | 7 componentes | 1.910 | 4 máquinas (A, B, C, D) | 574 | 8 |
| Gearbox | 18 medições | 2.946 | 3 eventos de falha (E1, E2, E3) | 822 | 8 |

**Gerador.** Componentes agrupados por máquina física: A = falhas 1 e 2, B = falha 3,
C = falhas 4 e 5, D = falhas 6 e 7. Componentes da mesma máquina compartilham datas e alvo,
mas têm leituras de sensores distintas; por isso o grupo, e não o componente, define a separação.

**Gearbox.** Cada evento de falha contém seis medições distribuídas em quatro componentes físicos
(HSS radial e axial, IMS, LSS, e planetária 0° e 90°). Os eventos são E1 = medições 1–6,
E2 = 7–12, E3 = 13–18. As seis medições de um evento compartilham timestamps e alvo, formando
137 timestamps sincronizados nos três eventos (71 em E1, 33 em E2 e 33 em E3).

**Pontos de avaliação.** Para cada série, os primeiros 30 registros formam o período de *warmup* e
as avaliações ocorrem com *stride* = 3 registros, incluindo sempre o ponto final. Isso resulta em
574 pontos de avaliação no gerador e 822 no gearbox (trilha por série).

**Alvo.** `porcentagem-data`: 1,0 até o FPT e decaimento linear em data até 0,0. Nenhuma outra
rotulagem foi usada nos experimentos deste relatório.

**Limpeza.** No gerador, 800 linhas eram duplicatas exatas (2.710 → 1.910). No gearbox não há
duplicatas. Linhas com mesmo instante e sensores distintos são mantidas separadas.

## 3. Protocolo de avaliação

- **Nested group cross-validation.** Cada grupo é usado uma vez como teste externo; dentro dos
  grupos restantes, um grupo é validação e os outros formam o treino. Nenhum componente do teste
  externo participa de treino ou seleção.
- **Seleção interna.** Hiperparâmetros, representação de features e pós-processamento são escolhidos
  apenas nos folds internos. As escolhas são congeladas antes da avaliação no teste externo.
- **Métricas.** MAE como métrica principal, com RMSE e viés como complemento, todas na unidade do
  alvo (0 a 1):
    - **Média macro:** métrica calculada por série (componente/medição), média igual entre séries de
      cada grupo e depois média igual entre grupos. Evita que o grupo com mais registros domine.
    - **Agregado por registro (*pooled*):** média ponderada por registro, útil para descrever o
      erro total, não para comparar grupos.
- **Referências.** Dois *baselines* sem uso de sensores servem de piso para avaliar o ganho real
  das variáveis de sensores:
    - **Previsor constante:** média ou mediana dos alvos de treino do fold.
    - **Controle de tempo e cadência:** referência pré-declarada que usa **apenas** tempo decorrido
      desde o primeiro registro observado e estatísticas de cadência de amostragem em janela fixa de
      12 h / 32 bins (padronização *z-score*, *missing indicators* e clipping em `[0, 1]`), sem
      nenhuma coluna de sensor e sem busca de hiperparâmetros. Seleciona internamente entre receitas
      fixas: no gerador, `mean`, `median`, `ridge` (α = 1) e `hist_gb` (`max_iter` = 100,
      `max_leaf_nodes` = 15, `min_samples_leaf` = 10, `lr` = 0,1, L2 = 1, `loss` =
      `absolute_error`); no gearbox, as mesmas quatro opções mais `extra_trees` (128 árvores,
      profundidade livre, `min_samples_leaf` = 1, `max_features` = 1,0).
- **Pré-processamento.** Imputação, normalização e clipping de quantis são ajustados somente no
  treino. As features usam apenas observações até o instante previsto.

## 4. Modelos tabulares

### 4.1 Método

Para cada fold externo da validação aninhada, um seletor automático busca nos folds internos a receita completa do pipeline tabular, escolhendo conjuntamente quatro componentes:

1. **Representação de features causais:** estatísticas calculadas apenas até o instante previsto (`current`, `summary` ou `robust`), largura da janela temporal e número de bins, além de tempo decorrido, cadência de amostragem e referências iniciais.
2. **Pré-processamento e pesos:** escalonamento (`none`, `minmax` ou `signed_log1p_standard`) e ponderação das amostras por grupo ou por componente.
3. **Família e hiperparâmetros:** 11 famílias avaliadas (`mean`, `median`, `ridge`, `elasticnet`, `huber`, `svr`, `knn`, `extra_trees`, `random_forest`, `hist_gb` e `catboost`), mantendo as referências constantes como candidatas.
4. **Pós-processamento causal:** clipping em `[0, 1]`, média móvel exponencial (EMA, α ∈ {1,0; 0,5; 0,25}) e *running min* em ordem cronológica por componente.

A busca em cada fold externo utiliza Optuna (`TPESampler`) em cinco etapas: triagem inicial (8 *trials* por família); ajuste fino nas 3 melhores famílias (até 40 *trials* extras, com *early stopping* após ≥ 16 *trials* e 12 sem ganho ≥ 0,005 no MAE interno); até 12 ablações de um eixo sobre a melhor receita (com retreino de até 16 *trials* quando há ganho ≥ 0,005); confirmação com 3 *seeds* fixas nas 2 melhores famílias estocásticas; e construção de *ensembles* uniformes por média ou mediana das previsões pós-processadas. A seleção adota o menor MAE macro interno (empates dentro de 0,001 favorecem menor complexidade e, em seguida, menor RMSE interno), congelando a receita antes de reajustá-la em todo o treino externo e avaliá-la no grupo retido.

No **gearbox**, o protocolo avalia três trilhas independentes:

- **Trilha por série (822 pontos):** prevê cada série de medição isoladamente. Além da busca *pooled* (com opção de *one-hot* da posição de medição), foram executadas sondagens limitadas com 4 especialistas por componente e 6 por medição (`ridge`, `extra_trees` e `hist_gb`); nenhuma sondagem atingiu ganho interno ≥ 0,005 para ser expandida, embora suas candidatas tenham permanecido elegíveis na seleção final.
- **Conjunto das seis medições (137 timestamps):** concatena as 48 colunas de sinal das seis medições disponíveis no mesmo instante e executa uma busca limitada por família, sem etapa de TPE genérico.
- **Agregação das seis previsões (137 timestamps):** combina as seis previsões geradas na trilha por série para cada timestamp sincronizado usando média ou mediana, conforme a regra escolhida na validação interna.

Esforço de busca: 1.514 ajustes no gerador e 1.326 no gearbox, com falhas de convergência registradas e não pontuadas.

### 4.2 Gerador

| Fold (teste) | n | MAE | RMSE (por componente) | Viés | Escolha interna |
| --- | --- | --- | --- | --- | --- |
| A | 352 | 0,307586 | 0,359060 | −0,307342 | Ensemble (Random Forest + SVR) |
| B | 34 | 0,240598 | 0,266724 | +0,240598 | kNN |
| C | 138 | 0,286944 | 0,317872 | +0,286894 | kNN |
| D | 50 | 0,218289 | 0,339956 | +0,208157 | CatBoost |
| **Média macro** | 574 | **0,263355** | **0,320903** | **+0,107077** | — |

Agregado por registro sobre 574 pontos: MAE 0,290877, RMSE 0,354191, viés −0,087117.

Referências no mesmo protocolo:

| Referência | MAE macro | RMSE | Viés |
| --- | --- | --- | --- |
| Previsor constante (média do treino) | 0,323418 | 0,361896 | — |
| Controle de tempo e cadência | **0,178772** | 0,217910 | +0,034361 |
| Modelo tabular | 0,263355 | 0,320903 | +0,107077 |

O controle de tempo e cadência (com escolhas internas `Ridge` em A e B; `HistGradientBoosting` em C
e D) foi melhor que o modelo tabular em todos os quatro folds. No modelo tabular, todas as 574
previsões pós-processadas ficaram em `[0, 1]` (apenas o fold D teve 7/50 previsões brutas fora
desse intervalo antes do clipping) e o erro cresce após o FPT: por exemplo, no fold A, MAE de
0,154 no período saudável contra 0,353 após o FPT.

### 4.3 Gearbox

Trilha por série (822 pontos, uma medição por vez):

| Evento | n | MAE | RMSE (por medição) | Viés | Escolha interna |
| --- | --- | --- | --- | --- | --- |
| E1 | 426 | 0,113597 | 0,152413 | −0,040067 | CatBoost |
| E2 | 198 | 0,131902 | 0,178201 | +0,120077 | Random Forest |
| E3 | 198 | 0,114749 | 0,160873 | −0,030218 | HistGradientBoosting |
| **Média macro** | 822 | **0,120082** | **0,163829** | **+0,016597** | — |

Agregado por registro sobre 822 pontos: MAE 0,118283, RMSE 0,161872, viés +0,000880.

Duas trilhas adicionais usam a estrutura dos eventos sincronizados:

| Trilha | n | MAE macro | RMSE | Viés |
| --- | --- | --- | --- | --- |
| Série (medição isolada) | 822 | 0,120082 | 0,163829 | +0,016597 |
| Conjunto das seis medições | 137 | 0,176170 | 0,254410 | +0,057106 |
| Agregação das seis previsões | 137 | **0,112379** | 0,152361 | +0,016251 |

No conjunto das seis medições, as escolhas internas foram `Ridge` (E1, MAE 0,231055) e ensembles por mediana de `ExtraTrees + HistGradientBoosting + Ridge` (E2, MAE 0,127770; E3, MAE 0,169686). Na agregação das seis previsões da trilha por série, a regra selecionada internamente foi mediana em E1 (MAE 0,095454) e E3 (MAE 0,109937), e média em E2 (MAE 0,131747).

Referências:

| Referência | MAE macro | RMSE macro |
| --- | --- | --- |
| Previsor constante (média) | 0,303763 | 0,347069 |
| Previsor constante (mediana) | 0,301504 | 0,380127 |
| Controle de tempo e cadência (série) | 0,175186 | 0,228412 |
| Modelo tabular (série) | **0,120082** | **0,163829** |

No gearbox, o modelo tabular superou o controle de tempo e cadência em 0,055 na trilha por série
(onde o controle escolheu `HistGradientBoosting` em E1 e `ExtraTrees` em E2 e E3). Nas duas
trilhas sincronizadas (137 timestamps), o controle de tempo e cadência obteve MAE macro de 0,182011
no conjunto das seis medições e 0,138397 na agregação das seis previsões. A agregação das seis
previsões do modelo tabular foi o menor erro observado em todo o estudo, mas exige as seis
medições disponíveis simultaneamente.

## 5. Replicações com Chronos-2

### 5.1 Desenho

Backbone `amazon/chronos-2` (revisão fixa, `chronos-forecasting` 2.3.2) congelado em `float32`.
Para cada ponto de avaliação, o histórico observado é convertido em grade causal de 12 h, com
interpolação limitada a 48 h e máscara para lacunas longas e prefixos sem histórico. A representação
usada é a do último *patch* de contexto observado (8 canais × 768 dimensões), excluindo *tokens*
especiais. Nenhuma observação posterior ao instante previsto entra no processamento.

Duas *heads* supervisionadas foram implementadas, seguindo os artigos de referência:

- **El-Ghoussani:** pré-processamento por clipping p1/p99 e *z-score* ajustados apenas no treino;
  média dos 8 canais da representação (768 dimensões) alimentando a *head* primária `mlp2`
  (`768 → 128 → ReLU → Dropout(0,2) → 1 → ReLU`), treinada com MSE *loss*, `Adam` (`lr` 1e-3),
  50 *epochs*, *batch* 64 e 3 *seeds*. O comprimento de contexto é escolhido internamente entre 5,
  32 e 80 passos de 12 h na primária e reutilizado nos controles de capacidade `mlp4` (mais
  profundidade) e `linear_relu`.
- **Abdouni:** usa o prefixo observado inteiro até o instante previsto (limite de 8.192 passos),
  normalizado por *z-score* sem clipping; compressão compartilhada por canal (`Linear(768 → 8) +
  GELU`, gerando 64 dimensões), concatenação com os 8 sensores correntes normalizados (72 entradas)
  e MLP `256 → 64 → 1` (GELU, Dropout 0,2 após cada *hidden layer* e saída linear), treinada com
  `AdamW` (`lr` 1e-4, `weight_decay` 1e-3), até 150 *epochs* com *early stopping* (`patience` = 15)
  sobre o RMSE de validação, 10 *seeds*. Variantes: `fusion` (representações + sensores), `deep`
  (só representações, 64 entradas) e `wide` (só sensores correntes, 8 entradas).

Adaptações em relação aos artigos originais: alvo em fração 0–1 em vez de dias; grade de 12 h em vez
de 1 h; sem regimes operacionais (não existem nos CSVs); sem NASA Score ou MAPE.

### 5.2 Gerador (574 pontos, 4 máquinas)

Nas tabelas abaixo, a primeira coluna (`MAE macro`) reporta as previsões brutas (*raw*); as colunas
seguintes (`MAE macro (clipping 0–1)`, `RMSE`, `MAE agregado` e `Viés`) referem-se às previsões
após clipping em `[0, 1]`, em média sobre as seeds.

| Modelo | Variante | MAE macro | MAE macro (clipping 0–1) | RMSE | MAE agregado | Viés |
| --- | --- | --- | --- | --- | --- | --- |
| El-Ghoussani | `mlp2` | 0,3720 | 0,3373 | 0,3945 | 0,3022 | +0,051 |
| El-Ghoussani | `mlp4` | 0,3392 | **0,3266** | 0,3794 | 0,2890 | +0,031 |
| El-Ghoussani | `linear_relu` | 0,3827 | 0,3441 | 0,4190 | 0,3305 | +0,012 |
| Abdouni | `fusion` | 0,3452 | 0,3448 | 0,3905 | 0,3276 | +0,027 |
| Abdouni | `deep` | 0,3574 | 0,3550 | 0,4045 | 0,3373 | +0,025 |
| Abdouni | `wide` | 0,3060 | 0,3046 | 0,3498 | 0,3599 | −0,068 |

Comprimentos de contexto escolhidos internamente (El-Ghoussani): A = 80, B = 80, C = 5, D = 80
passos.

### 5.3 Gearbox (822 pontos, 3 eventos)

| Modelo | Variante | MAE macro | MAE macro (clipping 0–1) | RMSE | MAE agregado | Viés |
| --- | --- | --- | --- | --- | --- | --- |
| El-Ghoussani | `mlp2` | 0,2463 | **0,2441** | 0,3006 | 0,2366 | −0,040 |
| El-Ghoussani | `mlp4` | 0,2458 | 0,2456 | 0,2947 | 0,2448 | −0,056 |
| El-Ghoussani | `linear_relu` | 0,2814 | 0,2755 | 0,3463 | 0,2544 | −0,087 |
| Abdouni | `fusion` | 0,1865 | **0,1734** | 0,2239 | 0,1725 | −0,020 |
| Abdouni | `deep` | 0,2298 | 0,2234 | 0,2824 | 0,2149 | +0,045 |
| Abdouni | `wide` | 0,2896 | 0,2841 | 0,3382 | 0,2585 | −0,148 |

Comprimentos de contexto escolhidos (El-Ghoussani): E1 = 80, E2 = 80, E3 = 5 passos.

No gearbox, o Abdouni `fusion` distribui o erro de forma equilibrada entre eventos (E1 0,171,
E2 0,184, E3 0,162). O El-Ghoussani concentra o erro em E3 (0,314).

## 6. Comparação consolidada

Métricas macro, unidade 0–1, peso igual por grupo (no gearbox, refere-se à trilha por série, com 822
pontos e 8 variáveis). Cada célula reporta MAE / RMSE:

| Conjunto | Previsor constante | Controle de tempo e cadência | Modelo tabular | El-Ghoussani | Abdouni `fusion` |
| --- | --- | --- | --- | --- | --- |
| Gerador (574, 4 máquinas) | 0,3234 / 0,3619 | **0,1788 / 0,2179** | 0,2634 / 0,3209 | 0,3266 / 0,3794 | 0,3448 / 0,3905 |
| Gearbox (822, 3 eventos) | 0,3038 / 0,3471 | 0,1752 / 0,2284 | **0,1201 / 0,1638** | 0,2441 / 0,3006 | 0,1734 / 0,2239 |

Observações:

1. Nenhuma replicação com Chronos-2 superou o modelo tabular em qualquer conjunto.
2. No gerador, todas as abordagens aprendidas ficaram acima do controle de tempo e cadência. O
   melhor modelo baseado em sensores (0,2634) teve erro 47% maior que o controle (0,1788).
3. No gearbox, o modelo tabular (0,1201) superou o controle de tempo e cadência (0,1752) em 31%.
   Isso indica que há sinal nos sensores nesse conjunto.
4. O Chronos-2 aproximou-se do controle de tempo e cadência no gearbox (0,1734 contra 0,1752), mas
   não do modelo tabular.
5. A variante `wide` do Abdouni (sem representações) foi melhor que `fusion` no gerador (0,3046
   contra 0,3448) e pior no gearbox (0,2841 contra 0,1734). O efeito das representações depende do
   conjunto.

## 7. Verificações realizadas

- **Métricas recalculadas** a partir das previsões exportadas, de forma independente, para todos os
  experimentos. Os valores coincidem com os relatórios armazenados.
- **Integridade dos dados:** hashes SHA-256 dos CSVs conferem com os usados no treino.
- **Contagens:** 574 previsões ao todo no gerador (soma dos 4 folds); 822 ao todo na trilha por
  série do gearbox (soma dos 3 eventos); 137 timestamps nas trilhas sincronizadas de evento.
- **Separação de grupos:** nenhum componente do teste externo aparece em treino ou validação;
  verificado por testes automatizados do protocolo.
- **Causalidade das features:** perturbar observações futuras não altera features nem previsões de
  instantes anteriores.
- **Retomada e cache:** execuções interrompidas retomam sem repetir ajustes já concluídos, e caches
  incompatíveis são rejeitadas em vez de reutilizadas.

## 8. Limitações

- **Poucos grupos independentes:** quatro máquinas no gerador e três eventos de falha no gearbox. As
  estimativas de generalização têm incerteza alta; diferenças abaixo de alguns centésimos não devem
  ser interpretadas como superioridade.
- **Resultados externos já observados:** as buscas descritas são exploratórias. Uma confirmação
  exigiria novos grupos, não novos ajustes nos mesmos grupos.
- **Replicação metodológica, não reprodução numérica:** os artigos usam outros dados e outras
  unidades. Os números aqui não são comparáveis aos publicados.
- **Controles e pós-processamento distintos:** os controles de tempo e cadência usam 4 famílias no
  gerador contra 5 no gearbox (por causa da inclusão de `ExtraTrees`), e as *heads* do Chronos-2
  aplicam apenas clipping em `[0, 1]`, sem EMA nem *running min*.
- **Ausência de regimes operacionais:** o artigo do Abdouni usa condições de operação (altitude,
  rotação) que não existem nos CSVs; esse ramo foi omitido.
- **Uma única rotulagem:** todos os resultados usam `porcentagem-data`. Rotulagens em dias podem
  produzir conclusões diferentes.
- **Gearbox:** os três eventos não têm documentação que comprove serem máquinas distintas; a
  separação mede generalização para outro evento de falha.

## 9. Conclusões

1. **O sinal dos sensores é aproveitável no gearbox:** modelos tabulares atingem MAE 0,120 e RMSE
   0,164, bem abaixo das referências sem sensores.
2. **No gerador, o tempo domina:** o controle baseado apenas em tempo e cadência (MAE 0,179 / RMSE
   0,218) superou todos os modelos que usam sensores.
3. **O Chronos-2 congelado não agregou valor nestes dados.** As heads supervisionadas ficaram no
   nível do previsor constante no gerador (MAE 0,327 / RMSE 0,379) e próximas do controle de tempo e
   cadência no gearbox (MAE 0,173 / RMSE 0,224), mas distantes do modelo tabular.
4. **A representação é o gargalo, não a capacidade:** aumentar a profundidade da head (`mlp2` →
   `mlp4`) não alterou o patamar de erro; a variante sem representações foi competitiva.

## 10. Próximos passos sugeridos

- Testar uma head que combine representações com features explícitas de tempo e cadência, para medir
  se o Chronos-2 contribui além da referência temporal.
- Ampliar o número de grupos independentes, única forma de reduzir a incerteza das comparações.
- Repetir a análise em rotulagens em dias, para verificar se a conclusão depende da definição do
  alvo.

## Apêndice: reprodutibilidade

| Item | Valor |
| --- | --- |
| Alvo | `porcentagem-data` (0–1) |
| Grade causal (Chronos) | 12 h; interpolação máxima 48 h |
| Modelo fundacional | `amazon/chronos-2`, revisão `29ec3766…`, `chronos-forecasting==2.3.2` |
| Seeds (El-Ghoussani) | 42, 43, 44 |
| Seeds (Abdouni) | 42 a 51 |
| Ajustes (tabulares) | 1.514 (gerador), 1.326 (gearbox) |
| Ajustes (Chronos) | 144 e 480 (gerador); 81 e 270 (gearbox) |
| CSVs | hashes SHA-256 registrados nos manifestos de cada execução |
