# Candidato experimental: três ResNet50 e quatro espelhamentos

Na busca limitada a 30 minutos de GPU, o melhor pipeline de validation passou de
macro-F1 0.7300 (CNN individual) para **0.7722**, sem novo treinamento. Ele combina
as seeds 42/43/44 sem label smoothing e quatro vistas exatas de cada imagem.

O [relatório completo](../reports/experimental_v2/reviews/ensemble_tta_20260930/README.md)
contém todos os candidatos, custo, intervalos por grupos e trade-offs. O recall de
melanoma sobe de 95/167 para 97/167 acertos, mas seu F1 cai e a confusão mel→nv
aumenta. O ganho global não estabelece segurança clínica nem generalização externa.

## Usar a inferência experimental

Executar na raiz do repositório com os três runs/checkpoints restaurados e seus
hashes corretos. Nenhum peso é baixado. O construtor verifica o estudo, dataset,
configs e checkpoints e mantém os modelos em memória para chamadas seguintes.

```python
from pathlib import Path
import torch
from scripts.ensemble_inference import StudyPredictor

# Mesmas opções numéricas utilizadas no estudo e na verificação de paridade.
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

predictor = StudyPredictor(
    Path.cwd(),
    "reports/experimental_v2/tta/d4_study_20260930T020126377530Z",
    "no_smoothing_flips4",
    device="cuda",  # "cpu" também é suportado, com custo diferente
)

response = predictor.predict_bytes(Path("imagem.jpg").read_bytes())
print(response["top_prediction"])
print(response["pipeline_id"])
```

`predict_batch_bytes([bytes_a, bytes_b])` reutiliza o mesmo contrato em batch.
Entradas vazias/corrompidas levantam `ValueError` com `INVALID_INPUT`.
A resposta identifica os checkpoints, preprocessing, ordem de classes, estudo,
pipeline e contrato de inferência; a política é `none` e calibração `not_applied`.
Probabilidades não são garantias calibradas de correção.

Este carregador não altera `active_model.json`, não faz freeze/promoção e não
acessa o holdout. Sua integração futura ao gate/backend exige suporte explícito
ao conjunto de checkpoints e às vistas; o contrato de uma CNN única não basta.

## Reproduzir a avaliação de desenvolvimento

Os comandos abaixo criam novos artefatos e consomem tempo de inferência; os
resultados desta rodada já estão disponíveis. Não é necessário repeti-los para
consultar os relatórios.

```powershell
.venv\Scripts\python.exe -B -m scripts.v2_tta --reference reports/experimental_v2/selections/resnet50_v2_reference_20260929_20260929T153109921383Z --challenger reports/experimental_v2/selections/resnet50_v2_no_smoothing_20260929_20260929T201705724817Z --budget-seconds 1800
```

O script verifica exatamente as seeds planejadas, acessa só validation, declara
a família de nove pipelines antes da inferência TTA e verifica paridade da vista
identidade com os logits originais. Se o orçamento acabar, registra falha e não
produz uma seleção baseada num estudo incompleto.

O estudo é exploratório: a família foi motivada por análises anteriores da mesma
validation, e os scores dos ensembles sem TTA já tinham sido observados. Isso
está registrado nos manifestos. O bootstrap usa grupos de lesão e deltas
pareados, mas não elimina o viés da seleção adaptativa.
