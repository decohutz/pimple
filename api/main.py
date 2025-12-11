from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# Configuração básica da aplicação
# ============================================================

app = FastAPI(
    title="pimple API (mock)",
    description=(
        "API mock para o MVP do pimple. "
        "O endpoint principal é POST /analyze."
    ),
    version="0.1.0",
)

# CORS liberado para desenvolvimento (depois podemos restringir)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # em produção, restringir!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Enums de domínio (seguindo o dicionário de dados)
# ============================================================

class SexEnum(str, Enum):
    male = "male"
    female = "female"
    unknown = "unknown"


class LocalizationEnum(str, Enum):
    back = "back"
    chest = "chest"
    face = "face"
    upper_extremity = "upper extremity"
    lower_extremity = "lower extremity"
    abdomen = "abdomen"
    scalp = "scalp"
    ear = "ear"
    hand = "hand"
    foot = "foot"
    unknown = "unknown"


class RiskLevelEnum(str, Enum):
    BAIXO_RISCO = "BAIXO_RISCO"
    INCERTO = "INCERTO"
    ALTO_RISCO = "ALTO_RISCO"


# ============================================================
# Lógica mock do "modelo"
# ============================================================

def mock_predict_probability(
    sex: SexEnum,
    localization: LocalizationEnum,
    age: Optional[int],
) -> float:
    """
    Função fake e determinística para gerar probabilidade de malignidade.

    Ideia:
      - Começa em 0.30
      - Adiciona +0.20 se localização é face / scalp / back
      - Adiciona +0.05 se sexo é male
      - Adiciona +0.15 se idade >= 50

    Retorna valor entre 0 e 1.
    """
    base = 0.30

    if localization in {
        LocalizationEnum.face,
        LocalizationEnum.scalp,
        LocalizationEnum.back,
    }:
        base += 0.20

    if sex == SexEnum.male:
        base += 0.05

    if age is not None and age >= 50:
        base += 0.15

    # Garante que o valor fique entre 0 e 1
    return max(0.0, min(1.0, base))


def classify_risk(prob: float, threshold: float = 0.62) -> RiskLevelEnum:
    """
    Classifica a probabilidade em 3 faixas:

      - prob < threshold - 0.10  → BAIXO_RISCO
      - prob > threshold + 0.10  → ALTO_RISCO
      - caso contrário           → INCERTO
    """
    low_cut = threshold - 0.10
    high_cut = threshold + 0.10

    if prob < low_cut:
        return RiskLevelEnum.BAIXO_RISCO
    if prob > high_cut:
        return RiskLevelEnum.ALTO_RISCO
    return RiskLevelEnum.INCERTO


# ============================================================
# Rotas
# ============================================================

@app.get("/")
def root():
    """
    Health check simples para evitar o 404 na raiz.
    Útil para você abrir http://127.0.0.1:8000 no navegador
    e ver se o servidor está ok.
    """
    return {
        "status": "ok",
        "message": "pimple API rodando. Use POST /analyze ou /docs.",
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    sex: SexEnum = Form(...),
    localization: LocalizationEnum = Form(...),
    age: Optional[int] = Form(None),
):
    """
    Endpoint principal para análise de imagem.
    Versão mock: não usa modelo real, apenas gera probabilidade fake.

    Request: multipart/form-data com campos:
      - image: arquivo .jpg ou .png
      - sex: 'male' | 'female' | 'unknown'
      - localization: conforme LocalizationEnum
      - age: opcional (int)

    Response: ver contrato em api/spec/analysis-contract.md
    """

    # -------------------------
    # Validação do arquivo
    # -------------------------
    if image.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(
            status_code=400,
            detail="Tipo de arquivo inválido. Use JPEG ou PNG.",
        )

    try:
        content = await image.read()
        if not content:
            raise ValueError("Imagem vazia")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Não foi possível ler a imagem enviada.",
        )

    # -------------------------
    # "Predição" mock
    # -------------------------
    prob = mock_predict_probability(
        sex=sex,
        localization=localization,
        age=age,
    )

    threshold = 0.62
    risk_level = classify_risk(prob, threshold)

    analyzed_at = datetime.now(timezone.utc).isoformat()

    # -------------------------
    # Monta resposta no formato combinado
    # -------------------------
    response = {
        "prob_malignant": round(prob, 4),
        "risk_level": risk_level.value,
        "threshold": threshold,
        "echo": {
            "sex": sex.value,
            "localization": localization.value,
            "age": age,
        },
        "analyzed_at": analyzed_at,
        "disclaimer": (
            "Este aplicativo é apenas para fins educacionais e NÃO substitui "
            "avaliação médica profissional. Em caso de dúvida, consulte um dermatologista."
        ),
    }

    return response
