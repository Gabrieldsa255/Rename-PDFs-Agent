# pdf-renamer-zero-error

Renomear PDFs (notas fiscais, comprovantes e extratos) com **tolerância zero a erro**.

## ✅ O que este projeto garante

- **Fail-safe**: se houver qualquer ambiguidade, o arquivo fica como **REVISAR** (não renomeia).
- **Duas etapas**:
  - **Dry-run**: analisa e gera `output/audit.csv` + `output/logs/*.json` (não altera nada).
  - **Run**: **só copia/renomeia** os arquivos **OK** para `output/renamed/`.
- **Rastreabilidade total**:
  - checksum **SHA-256** por arquivo
  - `audit.csv` (auditoria) + logs JSON por PDF
  - `registry.json` para **idempotência** (rodar 2x não bagunça)

---

## 🧾 Padrão final do nome (fixo)

**Exatamente assim (separador `__`):**

`YYYY-MM-DD__<DOC_ID|SEM-ID>__<TIPO>__<PRESTADOR>__BRL<VALOR>.pdf`

Exemplos:

- `2021-09-17__NF693__COMPROVANTE__JOSIEL__BRL125.90.pdf`
- `2024-12-01__SEM-ID__EXTRATO__BANCO_DO_BRASIL__BRL2500.00.pdf`

Regras:
- sem caracteres inválidos para Windows/macOS (`\ / : * ? " < > |`)
- sem acentos, espaços viram `_`
- se já existir, adiciona sufixo controlado: `_01`, `_02`...

> Observação importante: por padrão, `NF` exige **doc_id**. Extratos normalmente não têm doc_id e usam `SEM-ID`.

---

## 📦 Requisitos

- Python 3.10+
- Tesseract OCR (somente se houver PDFs escaneados)

### Windows (Tesseract)
1. Instale o Tesseract (UB Mannheim builds).
2. Garanta que `tesseract.exe` esteja no PATH.
3. Instale o idioma português (`por`).

### macOS (Tesseract)
```bash
brew install tesseract
brew install tesseract-lang
```

### Linux (Tesseract)
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-por
```

---

## 🧱 Estrutura recomendada de pastas

```text
meus_pdfs/
  input/         # PDFs baixados do Drive (NUNCA será alterado)
  output/        # gerado pelo script
```

---

## ⚙️ Instalação

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Opcional: copie o `config_example.yaml` para `config.yaml` e ajuste regras.

---

## 1) Dry-run (NÃO renomeia)

```bash
python rename_pdfs.py dry-run --input ./meus_pdfs/input --out ./meus_pdfs/output
```

Saídas:
- `output/audit.csv`  ✅ tabela principal (OK/REVISAR)
- `output/logs/<sha256>.json`  ✅ log detalhado por PDF

### Como revisar
Abra `audit.csv` e filtre:
- `status = REVISAR` → **não será renomeado**
- veja `reasons` e `log_json` para entender o motivo

---

## 2) Execução real (só OK)

```bash
python rename_pdfs.py run --audit ./meus_pdfs/output/audit.csv --input ./meus_pdfs/input --out ./meus_pdfs/output
```

Saídas:
- `output/renamed/` → PDFs copiados com nome final
- `output/review/` → cópias “para revisar” (opcional, ajuda triagem)
- `output/registry.json` → idempotência (não duplica trabalho em reexecuções)

---

## 🔒 Como o “zero erro” é implementado (na prática)

Um arquivo só vira **OK** se:
- todos os campos **obrigatórios** existirem **com confiança 1.0**
- e passarem validações rígidas (data válida, NF por padrões, valor total não ambíguo, prestador não é tomador)
- se houver **2+ candidatos** para qualquer campo → **REVISAR**
- se OCR tiver confiança média baixa → **REVISAR**

---

## 🧩 Calibração (recomendado)

Para chegar no “100% de precisão” no seu acervo real, o caminho correto é:
1. rodar o dry-run em 50–200 PDFs
2. pegar os `REVISAR` e ajustar regras/regex no código
3. repetir até a taxa de OK ficar alta sem falsos positivos

Se você puder, me mande 3–5 PDFs bem diferentes (ou trechos de texto extraído) e eu ajusto os regex/labels para o seu padrão real.
