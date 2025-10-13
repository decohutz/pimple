# Contrato da API — /analyze (MVP)

**POST** `/analyze` (multipart/form-data)

Campos:
- `image`: arquivo (jpg/png)
- `sex`: `male|female|unknown`
- `localization`: conforme dicionário
- `age`: número opcional

Resposta 200 (JSON):
{
  "prob_malignant": 0.73,
  "risk_level": "ALTO_RISCO",
  "threshold": 0.62,
  "echo": {"sex":"male","localization":"back","age":29},
  "analyzed_at": "2025-10-13T14:00:00-03:00",
  "disclaimer": "Este app é educacional e não fornece diagnóstico."
}

Erros:
- 400 — parâmetros inválidos / imagem ilegível
- 500 — falha interna (mensagem amigável)
