# Visão de Arquitetura — MVP

User → App (React) → **/analyze** (API local ou mock) → retorna JSON com prob + nível

Camadas:
- **UI:** telas, histórico local, validação de formulário, mensagens
- **API (mock → baseline):** recebe imagem+metadados, responde prob
- **Notebooks (depois):** EDA, treino, export de artefatos (thresholds, etc.)

Decisões:
- Primeiro **mock** para travar contrato
- Depois trocar mock por baseline sem quebrar UI
