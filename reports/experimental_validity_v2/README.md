# Resultado da fase 2 — validade experimental

Data: 2026-09-28. Proposta ainda não ativada. Nenhum treinamento executado.

Protocolo: [experimental_protocol_v2.md](../../docs/experimental_protocol_v2.md).

## Achado principal

O suplemento oficial do ISIC 2018 associa 10.015 imagens a 7.470 lesões.
O split antigo compartilha 1.008 lesões entre conjuntos (2.407 imagens).
Há 506 lesões compartilhadas entre treino e teste. Patient_id não está disponível.

## Artefatos

- RAW adicional: `data/raw/lesions/metadata_v2/` (somente CSV oficial e proveniência).
- Auditoria completa: `data/processed/split_v2/audit_v2/`.
- Proposta canônica: `data/processed/split_v2/candidate/`.
- Esses diretórios locais continuam sob o .gitignore existente; os arquivos deste
  relatório e a tabela compacta de atribuições estão disponíveis para versionamento.
- `audit/` é uma tentativa incompleta, anterior a uma correção de resolução de path;
  `proposal_v1/` é um rascunho anterior à marcação de representantes exatos. Não são
  artefatos canônicos. Foram mantidos separados; nenhum resultado histórico foi removido.

## Distribuição base 70/15/15 (imagens)

| class | train_pool | validation | internal_holdout |
|---|---|---|---|
| mel | 779 | 167 | 167 |
| nv | 4693 | 1006 | 1006 |
| bcc | 360 | 77 | 77 |
| akiec | 229 | 49 | 49 |
| bkl | 769 | 165 | 165 |
| df | 81 | 17 | 17 |
| vasc | 100 | 21 | 21 |

A partição train_pool inclui a reserva de calibração. Não treinar nas 7.011 imagens
se a mesma reserva for usada para calibrar esse modelo.

## Distribuição efetiva (imagens)

| class | train | validation | calibration | internal_holdout |
|---|---|---|---|---|
| mel | 668 | 167 | 111 | 167 |
| nv | 4023 | 1006 | 670 | 1006 |
| bcc | 309 | 77 | 51 | 77 |
| akiec | 196 | 49 | 33 | 49 |
| bkl | 659 | 165 | 110 | 165 |
| df | 70 | 17 | 11 | 17 |
| vasc | 86 | 21 | 14 | 21 |

Totais: train 6.011, validation 1.502, calibration 1.000, internal_holdout 1.502.

## Proporção de cada classe dentro do split (%)

| class | calibration | internal_holdout | train | validation |
|---|---|---|---|---|
| mel | 11.1 | 11.12 | 11.11 | 11.12 |
| nv | 67.0 | 66.98 | 66.93 | 66.98 |
| bcc | 5.1 | 5.13 | 5.14 | 5.13 |
| akiec | 3.3 | 3.26 | 3.26 | 3.26 |
| bkl | 11.0 | 10.99 | 10.96 | 10.99 |
| df | 1.1 | 1.13 | 1.16 | 1.13 |
| vasc | 1.4 | 1.4 | 1.43 | 1.4 |

## Grupos/lesões por classe

| class | train | validation | calibration | internal_holdout |
|---|---|---|---|---|
| mel | 369 | 92 | 61 | 92 |
| nv | 3241 | 811 | 540 | 811 |
| bcc | 196 | 49 | 33 | 49 |
| akiec | 137 | 34 | 23 | 34 |
| bkl | 436 | 109 | 73 | 109 |
| df | 44 | 11 | 7 | 11 |
| vasc | 58 | 15 | 10 | 15 |

Totais: train 4.481, validation 1.121, calibration 747, internal_holdout 1.121.
Não há grupos com múltiplos lesion_id ou múltiplas classes na proposta atual.
As duas duplas de arquivos exatos ficaram no treino, resultando em 6.009 imagens
RGB únicas no treino, mantendo as 6.011 linhas e todos os arquivos originais.

## Duplicatas e candidatos

- 2 pares de bytes idênticos, ambos também com lesion_id iguais.
- 0 pares adicionais com pixels RGB iguais e bytes diferentes.
- 3.205 pares de arquivos diferentes da mesma lesão identificada oficialmente.
- 10.571 candidatos pHash D4 (Hamming <=6/63); 10.413 com lesion_id diferentes.
- 4.312 desses candidatos entre lesões diferentes cruzam splits antigos.
- Nenhum candidato entre lesões distintas satisfez RMSE RGB <=0,01.
- O menor RMSE entre lesões distintas foi aproximadamente 0,028.
- pHash isolado não prova identidade: a revisão inclui colisões com classes distintas.
- Não houve revisão manual exaustiva dos candidatos. Não se afirma ausência de
  duplicatas sob crops, transformações arbitrárias ou metadados incorretos.

Tabela completa: `data/processed/split_v2/audit_v2/duplicate_pairs.csv` (13.620 pares).
Exemplos: `review_examples.csv`, acompanhados de `qualitative_review.csv`.
Painel: `data/processed/split_v2/audit_v2/review_contact_sheet.png`.

## Teste final

Nenhum dos 10.015 registros é reivindicado como dados inéditos. O novo holdout
contém 1.068 imagens do treino antigo, 224 da validação antiga e 210 do teste antigo.
É um holdout interno reagrupado, não FINAL TEST independente. Um conjunto externo
ou prospectivo, com auditoria de overlap, continua necessário para confirmação.

## Verificações

- 28 testes sintéticos/invariantes passaram.
- Validação do candidato com rehash das imagens e máscaras passou.
- IDs, hashes RGB/arquivo e lesões não atravessam as novas partições.
- Caminhos e one-hot válidos; sete classes presentes em cada partição.
- Nenhum patient_id inventado: independência por paciente não estabelecida.
- Seeds 20260928 (externa), 20260929 (calibração).
- API, frontend, notebooks, modelos ativos e splits antigos não foram alterados.
- Gate novo é uma função pura, sem métricas de teste; ainda exige migração do NB06.
