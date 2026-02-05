Perfeito — vamos montar **3 “receitas” diferentes por backbone** (ResNet50 / EfficientNet-B0 / MobileNetV3-Large), mudando **LR, weight decay, loss, class imbalance e fine-tuning**. Vou te falar **exatamente o que mexer** e te dar **configs prontas** pra você só alternar e rodar.

A ideia: você para de “só trocar modelo” e passa a testar **combos que realmente mudam macro-F1**.

---

# 0) Onde vale mexer (os pontos que MAIS mudam resultado)
Além desse bloco do print, os melhores knobs são:

1) **Fine-tuning em 2 fases** (freeze → unfreeze parcial)  
2) **LR por grupo** (backbone menor LR, head maior LR)  
3) **Loss**: CE + weights + label smoothing **ou** Focal  
4) **Imbalance**: ou `class_weights` no loss **OU** `WeightedRandomSampler` (não os dois)  
5) **Augmentations** (dermato-safe)  
6) **Scheduler** (cosine/onecycle + warmup)

Se você fizer só 1 coisa: **2 fases + LR por grupo**.

---

# 1) Configs prontas por modelo (3 variações cada)
Vou te dar **3 presets por backbone**:

- **A (seguro/estável)**: CE + label smoothing + class weights  
- **B (anti-desbalance)**: Focal + class weights  
- **C (mais agressivo)**: sampler + CE (sem class_weights) + LR mais alto

> **Regra de ouro**:  
> - Se classes raras com recall baixo → tente **B** ou **C**  
> - Se overfit/gap grande val→test → tente **A** + mais regularização

---

## ✅ MobileNetV3-Large (o seu atual)
MobileNet costuma **overfitar menos**, aguenta LR um pouco maior e se beneficia de sampler/focal quando raras sofrem.

### MobileNet — Preset A (estável)
- `IMG_SIZE = 224`
- `EPOCHS = 18`
- `LABEL_SMOOTHING = 0.05`
- `LOSS_TYPE = "ce"`
- `USE_CLASS_WEIGHTS = True`
- `USE_WEIGHTED_SAMPLER = False`
- `LR_HEAD = 3e-4`
- `LR_BACKBONE = 1e-4`
- `WEIGHT_DECAY = 5e-5`
- Scheduler: cosine

### MobileNet — Preset B (foco nas raras)
- `LABEL_SMOOTHING = 0.0`
- `LOSS_TYPE = "focal"`
- `FOCAL_GAMMA = 1.5`
- `USE_CLASS_WEIGHTS = True`
- `LR_HEAD = 3e-4`
- `LR_BACKBONE = 7e-5`
- `WEIGHT_DECAY = 1e-4`
- Scheduler: cosine

### MobileNet — Preset C (sampler)
- `LOSS_TYPE = "ce"`
- `LABEL_SMOOTHING = 0.05`
- `USE_CLASS_WEIGHTS = False`
- `USE_WEIGHTED_SAMPLER = True`
- `LR_HEAD = 5e-4`
- `LR_BACKBONE = 1e-4`
- `WEIGHT_DECAY = 5e-5`

---

## ✅ ResNet50
ResNet costuma ganhar bastante com **fine-tuning parcial** e LR pequeno no backbone.

### ResNet — Preset A (estável)
- `EPOCHS = 16`
- `LABEL_SMOOTHING = 0.05`
- `LOSS_TYPE="ce"`
- `USE_CLASS_WEIGHTS=True`
- `LR_HEAD=3e-4`
- `LR_BACKBONE=3e-5`
- `WEIGHT_DECAY=1e-4`

### ResNet — Preset B (focal)
- `LABEL_SMOOTHING=0.0`
- `LOSS_TYPE="focal"`
- `FOCAL_GAMMA=2.0`
- `USE_CLASS_WEIGHTS=True`
- `LR_HEAD=3e-4`
- `LR_BACKBONE=2e-5`
- `WEIGHT_DECAY=1e-4`

### ResNet — Preset C (mais agressivo + warmup)
- `LABEL_SMOOTHING=0.05`
- `LOSS_TYPE="ce"`
- `USE_WEIGHTED_SAMPLER=True`
- `USE_CLASS_WEIGHTS=False`
- `LR_HEAD=5e-4`
- `LR_BACKBONE=5e-5`
- `WEIGHT_DECAY=5e-5`
- Warmup 1 época (se você tiver isso; senão ignore)

---

## ✅ EfficientNet-B0
EfficientNet é **sensível**: LR do backbone deve ser menor, e augmentations muito fortes podem atrapalhar.

### EfficientNet — Preset A (estável)
- `EPOCHS = 18`
- `LABEL_SMOOTHING=0.05`
- `LOSS_TYPE="ce"`
- `USE_CLASS_WEIGHTS=True`
- `LR_HEAD=3e-4`
- `LR_BACKBONE=2e-5`
- `WEIGHT_DECAY=1e-4`

### EfficientNet — Preset B (focal leve)
- `LABEL_SMOOTHING=0.0`
- `LOSS_TYPE="focal"`
- `FOCAL_GAMMA=1.5`
- `USE_CLASS_WEIGHTS=True`
- `LR_HEAD=3e-4`
- `LR_BACKBONE=1e-5`
- `WEIGHT_DECAY=1e-4`

### EfficientNet — Preset C (sampler + mais regularização)
- `LABEL_SMOOTHING=0.05`
- `LOSS_TYPE="ce"`
- `USE_WEIGHTED_SAMPLER=True`
- `USE_CLASS_WEIGHTS=False`
- `LR_HEAD=3e-4`
- `LR_BACKBONE=2e-5`
- `WEIGHT_DECAY=2e-4`

---

# 2) Mudança #1 que você PRECISA fazer: LR por grupo + 2 fases
Isso é fora do seu bloco atual. Você precisa de:

### (a) Freeze por 2 épocas (head-only)
- `FREEZE_EPOCHS = 2`

### (b) Optimizer com 2 param groups (backbone/head)
- backbone LR menor
- head LR maior

### (c) Unfreeze parcial depois
- “últimos blocos” do backbone

Se você NÃO fizer isso, trocar LR “global” não extrai o máximo do pretrained.

---

# 3) Mudança #2: augmentations “dermato-safe” (ganho real)
Se você estiver com augmentations genéricas, use esse padrão:

- flips (H/V) ✅
- crop leve ✅ (`scale 0.85–1.0`)
- jitter leve ✅ (0.05–0.15)
- blur leve ✅ (p pequeno)

Evite:
- `RandomErasing` forte
- rotações muito agressivas e zoom muito forte

Isso melhora macro-F1 sem bagunçar.

---

# 4) Como você testa isso sem virar bagunça
Crie uma variável `PRESET` e um dicionário por modelo:

- `MODEL_NAME = "mobilenet_v3_large"`
- `PRESET = "A" | "B" | "C"`

E cada run salva no `train_config.json` (você já tem).

---

# 5) Me diga onde você quer que eu aplique isso (e eu te mando as células inteiras)
Eu consigo te mandar **as células completas** prontas pra colar, mas eu preciso saber qual estilo seu NB05 está usando agora:

1) Seu modelo vem de **timm** ou de **torchvision**?  
   (por causa daquele erro de HF/timm, eu suspeito que você migrou pro torchvision)

2) Você já tem no NB05 variáveis tipo:
- `USE_CLASS_WEIGHTS`
- `USE_WEIGHTED_SAMPLER`
- `criterion` definido em alguma célula
- `optimizer` criado em uma célula específica

Se você não quiser responder, eu já assumo o padrão mais comum do NB05 (criterion/optimizer numa célula de “setup model”) e te mando:

✅ **Célula de presets (A/B/C) + setup optimizer param-groups + freeze/unfreeze + loss (CE/Focal) + sampler + scheduler**  
…tudo em um pacote colável.

Só me diz: **seu modelo hoje é `torchvision.models.mobilenet_v3_large(pretrained=True)` ou `timm.create_model(...)`?**
