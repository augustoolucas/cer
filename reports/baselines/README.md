# Baselines de RUL: gerador e gearbox

Índice de dois estudos exploratórios de previsão de RUL (0..1) com protocolo comum de validação
agrupada aninhada:

- [`gerador.md`](gerador.md) — 574 endpoints em 4 grupos de equipamento (A, B, C, D).
- [`gearbox.md`](gearbox.md) — 822 endpoints de série (18 séries, 6 papéis, 3 eventos de falha) e
  um total de 137 timestamps sincronizados nos 3 eventos (71 + 33 + 33).

## Resumo

Métricas no conjunto externo, em RUL 0..1, com média igual por grupo/evento. “Seleção interna” é o
pipeline escolhido pela validação interna; “controle tempo/cadência” usa apenas tempo de aquisição
e cadência, com receitas fixas.

| estudo | trilha | seleção interna (MAE / RMSE / viés) | controle (MAE / RMSE / viés) |
| --- | --- | --- | --- |
| gerador | 4 grupos | 0.263355 / 0.320903 / 0.107077 | 0.178772 / 0.217910 / 0.034361 |
| gearbox | série (822 endpoints) | 0.120082 / 0.163829 / 0.016597 | 0.175186 / 0.228412 / -0.003670 |
| gearbox | eventos (137 timestamps) | 0.176170 / 0.254410 / 0.057106 | 0.182011 / 0.266630 / 0.057965 |
| gearbox | agregada (137 timestamps) | 0.112379 / 0.152361 / 0.016251 | 0.138397 / 0.175904 / -0.018042 |

Referências constantes (OOF, média igual por grupo/evento): gerador `mean`
0.323418 e `median` 0.339425;
gearbox `mean` 0.303763 e `median`
0.301504.

## Método comum

- Validação externa leave-one-group-out e interna leave-one-group-out nos grupos restantes;
  scalers e seleção de atributos ajustados apenas com os identificadores de treino; features de
  cada endpoint usam só registros até aquele endpoint; escolhas feitas apenas por escore interno e
  congeladas antes da avaliação de desempenho externo; rótulos externos não participam da
  seleção nem do ajuste.
- Features causais: apenas histórico observado até o endpoint; a data entra como tempo decorrido
  desde o primeiro registro observado; RUL, fase, identificador de grupo e vida útil total nunca
  são entradas.
- Escore interno e métricas externas: MAE por componente (`Falha`), média igual entre
  componentes, depois média igual entre grupos/eventos; “pooled” pondera cada registro e é
  reportado separadamente.

## Comparabilidade

Os datasets são independentes: nenhuma receita ou hiperparâmetro é transferido entre eles, e o
baseline principal é sempre o seletor interno, nunca a família que parece melhor no conjunto
externo.

Os controles temporais diferem: 4 famílias no gerador e 5 no gearbox (o `extra_trees` fixo é
necessário porque a trilha de eventos só buscou famílias especialistas). Portanto os dois
controles não são referências idênticas, e a diferença entre seleção e controle não isola uma
única causa.

No gearbox, a trilha de 137 timestamps sincronizados não é comparável registro a registro com a
trilha de 822 endpoints de série: os frames de disponibilidade são diferentes. Nenhuma tabela
ranqueia um modelo de 8 variáveis contra a entrada de 48 variáveis como se fossem o mesmo
problema.

## Limitações gerais

- Poucos grupos/eventos (4 e 3): não há teste de significância nem promessa de receita ótima
  universal.
- O alvo em 0..1 não limita o erro de regressões brutas; o recorte 0..1 é pós-processamento.
- Comparações com outros métodos só são interpretáveis com os mesmos grupos e endpoints, as mesmas
  unidades e scalers ajustados apenas no treino, referências de controle equivalentes, frames de
  disponibilidade comparáveis e nenhuma feature derivada do RUL histórico.

## Reprodutibilidade

Artefatos de cada estudo ficam em `results/baselines/<dataset>/<run>/` (manifesto, seleções,
predições OOF e curvas); as configurações de controle estão em
`results/baselines/<dataset>/controls/`. Curvas descritivas em RUL 0..1 são referenciadas ao final
de cada relatório.
