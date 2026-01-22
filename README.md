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
3. Garanta o idioma português (`por`).

### macOS (Tesseract)
``bash
brew install tesseract
brew install tesseract-lang


### Linux (Tesseract)

`sudo apt-get update sudo apt-get install -y tesseract-ocr tesseract-ocr-por` 

----------

## 🧱 Estrutura recomendada de pastas

`meus_pdfs/
  input/         # PDFs baixados do Drive (NUNCA será alterado)
  output/        # gerado pelo script` 

----------

## ⚙️ Instalação (passo a passo)

### 1) Clonar / baixar o projeto

Se você já tem o projeto local, pule este passo.

`git clone <URL_DO_REPO> cd pdf-renamer-zero-error` 

### 2) Criar e ativar ambiente virtual (venv)

#### Windows (PowerShell)

`python -m venv .venv
.\.venv\Scripts\Activate.ps1` 

> Se o PowerShell bloquear, rode (uma vez):

`Set-ExecutionPolicy  -Scope CurrentUser RemoteSigned` 

#### macOS / Linux

`python3 -m venv .venv source .venv/bin/activate` 

### 3) Instalar dependências do Python

`pip install -r requirements.txt` 

### 4) (Opcional, recomendado) Verificar se o Tesseract está OK

Se você tem PDFs escaneados, isso evita dor de cabeça:

#### Windows / macOS / Linux

`tesseract --version` 

Teste se o idioma `por` está disponível:

`tesseract --list-langs` 

Você deve ver `por` na lista.

### 5) (Opcional) Configuração

Se você quiser ajustar regras, copie o exemplo:

`cp config_example.yaml config.yaml` 

No Windows (PowerShell):

`Copy-Item .\config_example.yaml .\config.yaml` 

E rode usando:

`python rename_pdfs.py --config ./config.yaml dry-run --input ./meus_pdfs/input --out ./meus_pdfs/output` 

----------

## ▶️ Como utilizar (passo a passo)

### PASSO 0 — Prepare a pasta de entrada (input)

1.  Crie a estrutura:
    

`meus_pdfs/
  input/
  output/` 

2.  Coloque seus PDFs dentro de `meus_pdfs/input/` (pode ter subpastas).
    

> ✅ O script **NUNCA altera os originais** em `input/`. Ele apenas **copia**.

----------

## 1) Dry-run (NÃO renomeia)

Rode:

`python rename_pdfs.py dry-run --input ./meus_pdfs/input --out ./meus_pdfs/output` 

Saídas:

-   `./meus_pdfs/output/audit.csv` ✅ tabela principal (OK/REVISAR)
    
-   `./meus_pdfs/output/logs/<sha256>.json` ✅ log detalhado por PDF
    

### Como revisar rapidamente (Windows PowerShell)

Mostrar colunas principais:

`Import-Csv .\meus_pdfs\output\audit.csv | Select original_path, doc_type, date_iso, doc_id, provider, status, suggested_name | Format-Table  -Auto` 

Ver apenas os REVISAR:

`Import-Csv .\meus_pdfs\output\audit.csv | Where-Object { $_.status -eq  "REVISAR" } | Select original_path, reasons, log_json | Format-Table  -Auto` 

> Regra de ouro: se estiver **REVISAR**, o sistema está te protegendo de renomear errado.

----------

## 2) Execução real (só OK)

Somente depois de revisar o `audit.csv`, rode:

`python rename_pdfs.py run --audit ./meus_pdfs/output/audit.csv --input ./meus_pdfs/input --out ./meus_pdfs/output` 

Saídas:

-   `./meus_pdfs/output/renamed/` → PDFs copiados com nome final (apenas OK)
    
-   `./meus_pdfs/output/review/` → cópias “para revisar” (triagem)
    
-   `./meus_pdfs/output/registry.json` → idempotência (não duplica em reexecuções)
    

----------

## ✅ Conferência final (sanidade)

### Ver quantos OK e REVISAR deram

Windows PowerShell:

`Import-Csv .\meus_pdfs\output\audit.csv | Group-Object status | Select Name, Count` 

### Abrir um log específico (para entender por que errou)

No `audit.csv`, pegue a coluna `log_json` e abra o arquivo `.json` indicado.

----------

## 🔒 Como o “zero erro” é implementado (na prática)

Um arquivo só vira **OK** se:

-   todos os campos **obrigatórios** existirem **com confiança 1.0**
    
-   e passarem validações rígidas (data válida, NF por padrões, valor total não ambíguo, prestador não é tomador)
    
-   se houver **2+ candidatos** para qualquer campo → **REVISAR**
    
-   se OCR tiver confiança média baixa → **REVISAR**
    

----------

## 🧩 Calibração (recomendado)

Para chegar no “100% de precisão” no seu acervo real, o caminho correto é:

1.  rodar o dry-run em 50–200 PDFs
    
2.  pegar os `REVISAR` e ajustar regras/regex no código
    
3.  repetir até a taxa de OK ficar alta sem falsos positivos
