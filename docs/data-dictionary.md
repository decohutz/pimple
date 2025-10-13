# Dicionário de Dados — MVP

## Entrada (questionário)
- `sex`: enum {`male`,`female`,`unknown`} — obrigatório
- `localization`: enum curto {`back`,`chest`,`face`,`upper extremity`,`lower extremity`,`neck`,`abdomen`,`scalp`,`ear`,`hand`,`foot`,`unknown`} — obrigatório
- `age`: inteiro [0..120] — opcional
- `image`: JPEG/PNG (256–2048 px) — obrigatório

## Saída (análise)
- `prob_malignant`: float [0..1]
- `risk_level`: enum {`BAIXO_RISCO`,`INCERTO`,`ALTO_RISCO`} (regra por threshold)
- `threshold`: float [0..1]
- `echo`: espelho dos metadados de entrada
- `analyzed_at`: ISO8601 (timezone local)
- `disclaimer`: string fixa
