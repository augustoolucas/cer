# Baselines locais (gerador e gearbox)

Dois estudos exploratórios independentes produzidos pelo runner local `baseline_experiments.py`
(CPU, sem GPU, sem Colab). Relatórios: `gerador.md` (4 grupos de equipamento A/B/C/D, 574
endpoints) e `gearbox.md` (3 eventos E1/E2/E3, 822 endpoints provenientes de 18 séries, mais a
trilha separada de 137 timestamps sincronizados, pertencentes aos mesmos 3 eventos).

## Como reproduzir

A ordem importa: **conclua a busca principal e a busca de controle de um dataset antes de rodar
qualquer `evaluate`**, porque cada `evaluate` exige que as seleções daquele run já estejam
congeladas fold a fold (ele verifica fold por fold, não “todas as escolhas do projeto”).

```bash
# gerador
venv/bin/python baseline_experiments.py search   --dataset gerador
venv/bin/python baseline_experiments.py search   --dataset gerador \
  --config results/baselines/gerador/controls/config_cadence_elapsed_only.json \
  --output-root results/baselines/gerador/controls
venv/bin/python baseline_experiments.py evaluate --dataset gerador
venv/bin/python baseline_experiments.py evaluate --dataset gerador \
  --config results/baselines/gerador/controls/config_cadence_elapsed_only.json \
  --output-root results/baselines/gerador/controls
venv/bin/python baseline_experiments.py verify   --dataset gerador
venv/bin/python baseline_experiments.py verify   --dataset gerador \
  --config results/baselines/gerador/controls/config_cadence_elapsed_only.json \
  --output-root results/baselines/gerador/controls

# gearbox (mesmo esquema; controle com 5 famílias)
venv/bin/python baseline_experiments.py search   --dataset gearbox
venv/bin/python baseline_experiments.py search   --dataset gearbox \
  --config results/baselines/gearbox/controls/config_cadence_elapsed_only.json \
  --output-root results/baselines/gearbox/controls
venv/bin/python baseline_experiments.py evaluate --dataset gearbox
venv/bin/python baseline_experiments.py evaluate --dataset gearbox \
  --config results/baselines/gearbox/controls/config_cadence_elapsed_only.json \
  --output-root results/baselines/gearbox/controls
venv/bin/python baseline_experiments.py verify   --dataset gearbox
venv/bin/python baseline_experiments.py verify   --dataset gearbox \
  --config results/baselines/gearbox/controls/config_cadence_elapsed_only.json \
  --output-root results/baselines/gearbox/controls
```

Os dois JSONs de controle estão em diretórios **ignorados pelo git**
(`results/baselines/*/controls/`). A reprodução acima assume que esses arquivos foram
preservados localmente; eles não fazem parte do commit.

## Código, dados e pré-processamento

Commit `6e315ee` (hash composto do runner + `cer_rul.py` registrado em cada `manifest.json`).
Nenhum notebook, CSV ou parâmetro de protocolo foi alterado para produzir os relatórios. A
limpeza, as features causais e a escolha de alvo vêm de `cer_rul.py`; as features usam apenas o
histórico observado até cada endpoint.

## Comparabilidade

Os datasets são independentes: nenhum vencedor do gerador foi transferido para o gearbox, e cada
fold escolhe sua configuração por validação interna (leave-one-group-out). O “baseline principal”
é sempre o seletor interno completo, nunca a família que parece melhor no conjunto externo. As
métricas “macro” atribuem pesos iguais aos grupos/eventos; as “pooled” ponderam cada registro —
as duas aparecem separadas.

O controle de tempo/cadência (sem sensores) existe nos dois estudos, mas **não** é o mesmo
experimento: 4 famílias no gerador, 5 no gearbox (ExtraTrees fixo é necessário porque a trilha de
eventos só busca famílias especialistas).

| estudo | pipeline buscado | controle tempo/cadência |
| --- | --- | --- |
| gerador (4 grupos) | 26.335 pp | 17.877 pp |
| gearbox (3 eventos) | 12.008 pp | 17.519 pp |

## Suposições e limites

- Gerador tem 4 grupos de equipamento; gearbox tem 3 eventos. A trilha de eventos sincronizados
  só contém timestamps em que as 6 medições existem com o mesmo alvo e fase; ela é reportada
  separadamente dos 822 endpoints de série e não deve ser ranqueada contra eles.
- Features de “tempo decorrido” dependem de uma origem (“primeiro registro observado”) definida
  de forma consistente, inclusive entre execuções e curadoria do dataset; diferenças de origem
  funcionam como proxy e não devem ser lidas como efeito causal.
- Não há testes de significância, MAPE, promessa de receita ótima universal nem afirmação causal
  sobre sensores.

## Comparações futuras (Timestral e embeddings)

Para uma comparação honesta com uma abordagem temporal ou com embeddings: usar os mesmos grupos
externos e os mesmos endpoints, as mesmas unidades (pp) e scalers ajustados apenas no treino; manter
as referências de controle; comparar apenas frames com a mesma disponibilidade (uma medição vs as
6 sincronizadas); e nunca fornecer RUL histórico como feature. Divergências de MAE descrevem
pipelines nestes folds, sem inferência de superioridade geral.

## Curvas descritivas

As figuras locais (PNG) de cada estudo estão em `results/baselines/<dataset>/<run>/plots/` e são referenciadas nos relatórios `gerador.md` (§14) e `gearbox.md` (§15); elas são artefatos ignorados pelo git e derivam apenas das predições OOF já congeladas.
