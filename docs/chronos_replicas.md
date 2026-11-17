# Réplicas Metodológicas Adaptadas com Chronos-2 nos CSVs CPFL

Este documento registra o protocolo, as hipóteses explícitas e as adaptações dos dois métodos baseados em `amazon/chronos-2` (`chronos-forecasting==2.3.2`, revisão `29ec3766d36d6f73f0696f85560a422f50e8498c`) aplicados de forma independente aos datasets `gerador` (574 endpoints em 4 grupos) e `gearbox` (822 endpoints por série em 3 eventos).

> **Escopo científico**: são réplicas **metodológicas adaptadas** ao protocolo CPFL em RUL `0..1`, não reproduções numéricas dos dados privados da Nokia ou do benchmark C-MAPSS. Resultados científicos completos dependem da execução real em GPU/Colab pelo usuário.

## Arquitetura compartilhada (`chronos_rul.py` + `cer_rul.py`)

- **Dados e endpoints**: reutiliza `cer_rul.get_dataset_profile`, `load_raw_label`, `consolidate_frame` e `build_endpoint_rows` (`porcentagem-data`, `warmup=30`, `stride=3`, ponto final incluído). `gerador`: 1910 chaves limpas / 574 endpoints; `gearbox`: 2946 / 822. Nunca concatena os CSVs.
- **Validação agrupada aninhada explícita**:
  - `gerador`: outer `A(1,2), B(3), C(4,5), D(6,7)` e inner percorrendo todos os 3 grupos restantes.
  - `gearbox`: outer `E1(1..6), E2(7..12), E3(13..18)` e inner percorrendo os 2 eventos restantes.
- **Grade causal 12h**: ancorada em cada endpoint `t`, interpola linearmente apenas entre amostras `<= t` com intervalo `<= 48h`. Lacunas maiores ou prefixos anteriores à primeira observação permanecem `NaN` (mascarados, nunca zero físico). O passo final `t` é sempre observado. Prefixos > 8192 passos são recusados com erro explícito.
- **Extração Chronos-2 (`embed`)**: modelo congelado (`eval()`, `requires_grad=False`, `torch.no_grad()`, `float32`), `batch_size=64` variates, cada endpoint em seu próprio grupo (as 8 variáveis de um endpoint compartilham atenção entre si, nunca entre endpoints). Valida `use_reg_token=True`, `input_patch_size=16`, `d_model=768` e seleciona o **último patch de contexto observado** (`total_tokens - 3`, excluindo `[REG]` e o patch mascarado de saída).
- **Cache atômica e portátil**: `npz` sem pickle + `.json` de proveniência (`sensor_data_sha256`, `endpoints_sha256`, `train_ids`, estatísticas do pré-processador, `context_spec`, `model_revision`, `extractor_env`). Caches só são compartilháveis quando contexto, pré-processamento, grupos de treino e revisão coincidem. Caches extraídas em GPU no Colab podem ser consumidas no treino em CPU local (`--mode train`), sem exigir versões idênticas de Torch/CUDA entre extrator e consumidor.

## Tabela de fidelidade — Método A: El-Ghoussani et al. (`arXiv:2606.11990v3`)

| Aspecto | Artigo original | Nossa réplica adaptada | Classificação |
|---|---|---|---|
| Backbone | Chronos-2 congelado | `amazon/chronos-2` (`29ec3766...`) congelado, `float32` | Fiel |
| Pré-processamento | Clipping p1/p99 + z-score | Clipping p1/p99 + z-score ajustados só nos `train_ids` de cada inner/outer | Fiel + isolamento estrito |
| Pooling multivariado | Não especificado no texto | Média entre os 8 canais do último patch de contexto (`768` dims) | **Hipótese nossa** |
| Cabeça primária (`mlp2`) | MLP 2 camadas + ReLU | `768 -> 128 -> ReLU -> Dropout(0.2) -> 1 -> ReLU` (parâmetros reais impressos) | **Hipótese nossa** (`m=128`, `p=0.2`) |
| Controles de cabeça | — | `linear_relu` (`Linear(768->1)+ReLU`) e `mlp4`, usando o `L` da primária | Controle de capacidade |
| Contextos e passo | Grade 1h, janelas finitas | `[5, 32, 80]` passos na grade `12h`, seleção por MAE macro inner (3 seeds `42..44`) | Adaptação de cadência |
| Treino e alvo | MSE, Adam `1e-3`, `batch=64`, `50` épocas; RUL em dias (MAE ~44d) | Mesmos hiperparâmetros; alvo em fração `0..1` (`porcentagem-data`) | Adaptação de alvo |

## Tabela de fidelidade — Método B: Abdouni et al. (`PHME 4906`)

| Aspecto | Artigo original | Nossa réplica adaptada | Classificação |
|---|---|---|---|
| Contexto e escala | Prefixo inteiro; z-score (e por regime no C-MAPSS) | Prefixo observado inteiro `<= t` (grade `12h`, recusa `>8192`); z-score sem clipping, só no treino | Adaptação sem regimes |
| Regimes operacionais | KMeans + one-hot + normalização por regime (altitude/Mach/throttle) | **Omitidos** (CSVs CPFL não têm settings físicos; 6 posições do gearbox não são regimes) | Adaptação necessária |
| Compressão e Wide | `Linear(768->8)+GELU` compartilhada -> flatten `64` + sensores atuais | Igual (`d_comp=8`, flatten `64` + `8` sensores atuais normalizados = `72`, sem tempo/cadência/ID) | Fiel |
| MLP e saída | `256 -> 64 -> 1`, GELU, dropout `0.2`, saída linear | Dropout `0.2` após cada oculta (`256` e `64`), saída linear sem clamp implícito | Interpretação explicitada |
| Otimização e seeds | AdamW `1e-4`, `wd=1e-3`, `batch=1024`, `max=150`, `patience=15` (val RMSE), 10 reps | Igual (`batch=1024` vira 1 lote se `n<1024`; `batch=64` opcional declarado); seeds `42..51`; refit outer com `floor(mediana+0.5)` | Fiel |
| Ablações | Fusion vs Deep-only | `fusion` (primário), `deep` e `wide` (controle nosso), todos com early-stop próprio no inner | Fiel + controle |
| Alvo e métricas | Cap `125` ciclos + NASA Score (`13`/`10`) | Fração `0..1` sem cap, `MAE`/`RMSE`/`viés` (`macro` e `pooled`), `raw` primário + `clipped_0_1` suplementar | Adaptação ao protocolo CPFL |

## Fluxo de execução (GPU no Colab / CPU local)

1. **Smoke offline** (local ou Colab, sem Torch/pesos):
   ```bash
   python chronos_rul.py --article elghoussani --config cfg.json --mode smoke
   ```
2. **Extração GPU no Colab** (notebooks `RUL_CPFL_Chronos2_ElGhoussani_colab.ipynb` e `RUL_CPFL_Chronos2_Abdouni_colab.ipynb`):
   - Faça upload de `cer_rul.py`, `chronos_rul.py` e do CSV selecionado (`gerador` ou `gearbox`).
   - Defina `CONFIG['mode'] = 'extract'` (ou `'full'`) em runtime GPU (`device='cuda'`). A cache é salva no Drive em `cache_dir`.
3. **Treino das cabeças em CPU** (Colab ou máquina local com `.venv-chronos`):
   - Copie a pasta `cache/` gerada na GPU para o ambiente local e execute `--mode train` (nunca carrega Chronos).

## Comparabilidade com os baselines locais

- Comparar `gerador` contra o seletor interno dos baselines (`MAE = 0.263355` em `0..1`, controle temporal `0.178772`) e `gearbox` por série (`822` endpoints, `MAE = 0.120082`, controle `0.175186`).
- **Nunca** comparar a trilha de 8 variáveis (`822` endpoints) contra a agregação de 6 canais (`0.112379` em `137` timestamps sincronizados) como se tivessem a mesma disponibilidade.
- Sempre interpretar ganhos contra o controle temporal, pois máscaras e comprimentos de prefixo podem codificar idade/cadência implicitamente.
