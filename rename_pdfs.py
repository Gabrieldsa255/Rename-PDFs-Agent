#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pdf_renamer_zero_error
=====================

Fluxo:
1) dry-run: analisa e gera audit.csv + logs (NÃO renomeia)
2) run: copia/renomeia apenas linhas OK para output/renamed
   e copia REVISAR para output/review (com nome legível)

Foco de robustez (NF-e/DANFE):
- Emitente (provider) SEMPRE do bloco do emitente (ou "RECEBEMOS DE ...")
- Doc ID (nNF) pega por:
  (a) chave 44 dígitos colada
  (b) chave 44 dígitos com espaços (grupos)
  (c) "Nº" perto de DANFE/NF-e
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import re
import shutil
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pytesseract
import yaml
from dateutil import parser as date_parser
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm


# -----------------------------
# Config / Defaults
# -----------------------------

DEFAULT_CONFIG = {
    "naming": {"max_len": 180},
    "validation": {
        "min_year": 2000,
        "max_future_days": 7,
        "ocr_min_mean_conf": 90,
        "min_native_text_chars": 200,
    },
    "rules": {"date_preference": "auto"},  # auto | emission | competence
    "required_fields": {
        "NF": ["date", "provider"],  # doc_id não trava, mas entra no nome se existir
        "COMPROVANTE": ["date", "provider"],
        "EXTRATO": ["date", "provider"],
        "DOCUMENTO": ["date", "provider"],
    },
}


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class FieldPick:
    value: Optional[str]
    confidence: float
    evidence: str
    candidates: List[Dict]


@dataclass
class Extracted:
    pdf_path: Path
    sha256: str
    method: str  # native | ocr
    ocr_mean_conf: Optional[float]
    doc_type: str
    date_kind: str
    date_iso: FieldPick
    doc_id: FieldPick
    provider: FieldPick
    amount_brl: FieldPick
    status: str
    reasons: List[str]
    suggested_filename: Optional[str]


# -----------------------------
# Helpers
# -----------------------------

INVALID_CHARS = r'\/:*?"<>|'

CNPJ_RE = re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}\/?\d{4}-?\d{2}\b")
CPF_RE = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")

ADDRESS_HINTS = [
    r"\b(rua|rod|av|avenida|travessa|alameda|pra[cç]a|estrada)\b",
    r"\b(cep|bairro|municipio|munic[ií]pio|uf|fone|fax|telefone)\b",
    r"\b(n[oº]\s*\d+|\d{1,5}\s*-\s*)\b",
]

BAD_PROVIDER_PATTERNS = [
    # rótulos/áreas do DANFE
    r"identifica[cç][aã]o\s+do\s+emitente",
    r"natureza\s+da\s+opera[cç][aã]o",
    r"venda\s+para\s+consumidor\s+final",
    r"\bdanfe\b",
    r"documento\s+auxiliar",
    r"chave\s+de\s+acesso",
    r"consulta\s+de\s+autenticidade",
    r"protocolo",
    r"\bsefaz\b",
    r"\bs[ée]rie\b",
    r"\bfolha\b",
    r"\bcalculo\s+do\s+imposto\b",
    r"\bvalor\b",
    r"\btotal\b",
    r"\bdata\b",
    r"\bdestinat[aá]rio\b",
    r"\bremetente\b",
    r"\btomador\b",
    # o seu bug:
    r"inscri[cç][aã]o\s+estadual",
    r"inscri[cç][aã]o\s+municipal",
    r"\binscri[cç][aã]o\b",
    r"\binsc\.\b",
]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def strip_accents(s: str) -> str:
    s_norm = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s_norm if not unicodedata.combining(c))

def sanitize_component(s: str, *, max_len: int = 80) -> str:
    s = strip_accents(s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("&", "E")
    s = re.sub(f"[{re.escape(INVALID_CHARS)}]", " ", s)
    s = re.sub(r"[^\w\s\-\.\,\$]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s).strip("_")
    s = re.sub(r"_+", "_", s)
    s = s.strip(" .")
    return (s[:max_len] if s else "SEM_NOME")

def parse_brl_money(s: str) -> Optional[Decimal]:
    s = (s or "").strip().replace("R$", "").replace(" ", "")
    if "," in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    if d < 0:
        return None
    return d.quantize(Decimal("0.01"))

def iso_date_from_match(raw: str) -> Optional[str]:
    raw = (raw or "").strip()
    m = re.search(r"\b(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})\b", raw)
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        if len(y) == 2:
            y = "20" + y
        try:
            return dt.date(int(y), int(mo), int(d)).isoformat()
        except ValueError:
            return None
    try:
        return date_parser.parse(raw, dayfirst=True).date().isoformat()
    except Exception:
        return None

def today() -> dt.date:
    return dt.date.today()

def clean_provider_name(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\bCNPJ\b.*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\bCPF\b.*", "", s, flags=re.IGNORECASE).strip()
    s = CNPJ_RE.sub("", s).strip()
    s = CPF_RE.sub("", s).strip()
    s = re.sub(r"[:\-]+$", "", s).strip()
    s = re.sub(r"[,\.;]+$", "", s).strip()
    return s

def _is_bad_provider_candidate(s: str) -> bool:
    s0 = (s or "").strip()
    if not s0:
        return True
    s_low = s0.lower()

    for pat in BAD_PROVIDER_PATTERNS:
        if re.search(pat, s_low, flags=re.IGNORECASE):
            return True

    if CNPJ_RE.search(s0) or CPF_RE.search(s0):
        return True

    if not re.search(r"[A-Za-zÀ-ÿ]", s0):
        return True

    for pat in ADDRESS_HINTS:
        if re.search(pat, s_low, flags=re.IGNORECASE):
            return True

    # não aceitar linhas “muito rótulo” (duas palavras comuns de formulário)
    if re.fullmatch(r"[A-ZÀ-Ÿ\s_]{6,}", strip_accents(s0).upper()) and re.search(r"\b(INSCRICAO|ESTADUAL|MUNICIPAL|ENDERECO|MUNICIPIO|CEP|UF)\b", strip_accents(s0).upper()):
        return True

    return False

def _looks_like_company_name(s: str) -> bool:
    s0 = (s or "").strip()
    if len(s0) < 3:
        return False

    # sufixos comuns
    if re.search(r"\b(S\.?A\.?|LTDA\.?|EIRELI|MEI|EPP|ME)\b", s0, flags=re.IGNORECASE):
        return True

    # 2+ palavras “de verdade”
    if len(re.findall(r"[A-Za-zÀ-ÿ]{2,}", s0)) >= 2:
        return True

    # marca em caixa alta (KABUM) + opcional "S.A."
    if re.fullmatch(r"[A-ZÀ-Ÿ0-9\.\-\s]{4,}", s0) and re.search(r"[A-ZÀ-Ÿ]{4,}", s0):
        return True

    return False


# -----------------------------
# Text extraction (native + OCR fallback)
# -----------------------------

def extract_text_native(pdf_path: Path) -> str:
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return ""
    parts = []
    try:
        for page in doc:
            parts.append(page.get_text("text"))
    finally:
        doc.close()
    return "\n".join(parts)

def ocr_pdf(pdf_path: Path, dpi_scale: float = 2.0) -> Tuple[str, float]:
    doc = fitz.open(pdf_path)
    all_text = []
    confs = []
    try:
        for page in doc:
            mat = fitz.Matrix(dpi_scale, dpi_scale)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            img = ImageOps.grayscale(img)
            img = ImageOps.autocontrast(img)
            img = img.filter(ImageFilter.MedianFilter(size=3))
            img = img.filter(ImageFilter.SHARPEN)

            txt = pytesseract.image_to_string(img, lang="por")
            all_text.append(txt)

            data = pytesseract.image_to_data(img, lang="por", output_type=pytesseract.Output.DICT)
            for c in data.get("conf", []):
                try:
                    ci = float(c)
                except Exception:
                    continue
                if ci >= 0:
                    confs.append(ci)
    finally:
        doc.close()

    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return "\n".join(all_text), float(mean_conf)


# -----------------------------
# Classification and extraction
# -----------------------------

DOC_TYPE_KEYWORDS = {
    "NF": [
        r"\bnota\s+fiscal\b",
        r"\bnf[-\s]?e\b",
        r"\bnfs[-\s]?e\b",
        r"\bdanfe\b",
        r"\bsefaz\b",
        r"\bchave\s+de\s+acesso\b",
    ],
    "COMPROVANTE": [r"\bcomprovante\b", r"\brecibo\b", r"\bpix\b", r"\btransfer[êe]ncia\b", r"\bpagamento\b"],
    "EXTRATO": [r"\bextrato\b", r"\bmovimenta[cç][aã]o\b", r"\bsaldo\b", r"\bper[ií]odo\b"],
}

def classify_doc_type(text: str, filename: str) -> str:
    t = (text or "").lower()
    f = (filename or "").lower()
    score = {k: 0 for k in DOC_TYPE_KEYWORDS}
    for k, pats in DOC_TYPE_KEYWORDS.items():
        for p in pats:
            if re.search(p, t) or re.search(p, f):
                score[k] += 1
    best = max(score.items(), key=lambda kv: kv[1])
    return best[0] if best[1] >= 1 else "DOCUMENTO"


DATE_LABELS = {
    "emission": [
        r"data\s+de\s+emiss[aã]o",
        r"\bemiss[aã]o\b",
        r"data\s+da\s+emiss[aã]o",
        r"emiss[aã]o\s*:",
    ],
    "competence": [
        r"compet[êe]ncia",
        r"per[ií]odo\s+de\s+refer[êe]ncia",
        r"refer[êe]ncia",
        r"m[eê]s\s+refer[êe]ncia",
    ],
    "due": [r"vencimento", r"data\s+de\s+vencimento"],
}

def pick_date(text: str, doc_type: str, cfg: Dict, filename: str) -> Tuple[FieldPick, str, List[str]]:
    reasons = []
    t = text or ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    candidates = []

    for i, ln in enumerate(lines):
        ln_low = ln.lower()
        date_in_line = re.findall(r"\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b", ln)
        next_ln = lines[i + 1] if i + 1 < len(lines) else ""
        date_in_next = re.findall(r"\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b", next_ln)
        matches = date_in_line or date_in_next
        if not matches:
            continue

        for raw_date in matches[:2]:
            iso = iso_date_from_match(raw_date)
            if not iso:
                continue

            kind = "unknown"
            ctx_score = 0.0
            for k, pats in DATE_LABELS.items():
                for p in pats:
                    if re.search(p, ln_low):
                        ctx_score = 1.0
                        kind = "emission" if k == "emission" else ("competence" if k == "competence" else "unknown")
                        break

            evidence = ln if raw_date in ln else f"{ln} | next: {next_ln}"
            score = 1.0 if ctx_score >= 1.0 else 0.7
            candidates.append({"value": iso, "kind": kind, "score": score, "evidence": evidence})

    if not candidates:
        m = re.search(r"\b(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})\b", filename or "")
        if m:
            iso = iso_date_from_match(m.group(0))
            if iso:
                candidates.append({"value": iso, "kind": "unknown", "score": 0.6, "evidence": "filename-date"})

    min_year = int(cfg["validation"]["min_year"])
    max_future_days = int(cfg["validation"]["max_future_days"])
    now = today()

    valid = []
    for c in candidates:
        try:
            y, m, d = map(int, c["value"].split("-"))
            dd = dt.date(y, m, d)
        except Exception:
            continue
        if dd.year < min_year:
            continue
        if dd > now + dt.timedelta(days=max_future_days):
            continue
        valid.append(c)
    candidates = valid

    if not candidates:
        reasons.append("DATA: nenhuma data válida encontrada")
        return FieldPick(None, 0.0, "", []), "unknown", reasons

    pref = (cfg.get("rules", {}).get("date_preference") or "auto").lower()
    want = "emission" if (doc_type == "NF" and pref == "auto") else pref
    if want in ("emission", "competence"):
        filtered = [c for c in candidates if c["kind"] == want]
        if filtered:
            candidates = filtered

    candidates_sorted = sorted(candidates, key=lambda c: (c["score"], c["value"]), reverse=True)
    top = candidates_sorted[0]
    conf = 1.0 if top["score"] >= 1.0 else 0.8
    date_kind = top["kind"] if top["kind"] != "unknown" else ("emission" if doc_type == "NF" else "unknown")
    return FieldPick(top["value"], conf, top["evidence"], candidates_sorted), date_kind, reasons


# -----------------------------
# Doc ID (NF-e) robusto
# -----------------------------

def _extract_access_key_compact(text: str) -> Optional[str]:
    t = re.sub(r"\s+", "", text or "")
    m = re.search(r"\b(\d{44})\b", t)
    return m.group(1) if m else None

def _extract_access_key_spaced(lines: List[str]) -> Optional[str]:
    # encontra "CHAVE DE ACESSO" e junta dígitos daquela região (pode vir em grupos)
    for i, ln in enumerate(lines[:300]):
        if re.search(r"chave\s+de\s+acesso", ln, flags=re.IGNORECASE):
            chunk = " ".join(lines[i:i+5])
            digits = re.findall(r"\d+", chunk)
            joined = "".join(digits)
            if len(joined) >= 44:
                key = joined[:44]
                if key.isdigit() and len(key) == 44:
                    return key
    return None

def _nnf_from_access_key(key44: str) -> Optional[str]:
    if not key44 or len(key44) != 44 or not key44.isdigit():
        return None
    nnf = key44[25:34]  # 9 dígitos
    # mantém zeros à esquerda (como aparece no DANFE)
    return nnf if nnf.strip("0") != "" else "0"

def _extract_nnf_near_context(text: str) -> Tuple[Optional[str], str]:
    """
    Procura "Nº 026112189" dentro de uma janela perto de DANFE/NF-e.
    """
    t = (text or "").replace("\u00a0", " ")
    t = t.replace("°", "º")
    anchors = [m.start() for m in re.finditer(r"\b(?:DANFE|NF[-\s]?e|NOTA\s+FISCAL)\b", t, flags=re.IGNORECASE)]
    if not anchors:
        anchors = [0]

    pat_no = re.compile(r"\bN[ºo]\s*[:\-]?\s*(\d{5,12})\b", flags=re.IGNORECASE)

    for a in anchors[:5]:
        win = t[a:a+2000]
        m = pat_no.search(win)
        if m:
            return m.group(1), f"context_window@{a}"
    return None, ""

def pick_doc_id(text: str, doc_type: str) -> Tuple[FieldPick, List[str]]:
    reasons: List[str] = []
    if doc_type != "NF":
        return FieldPick(None, 1.0, "", []), reasons

    t = text or ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]

    # 1) chave 44d colada
    key = _extract_access_key_compact(t)
    if key:
        nnf = _nnf_from_access_key(key)
        if nnf:
            return FieldPick(f"NF-e{nnf}", 1.0, "access_key_44d_compact", [{"value": nnf, "score": 2.0}]), reasons

    # 2) chave 44d com espaços
    key2 = _extract_access_key_spaced(lines)
    if key2:
        nnf = _nnf_from_access_key(key2)
        if nnf:
            return FieldPick(f"NF-e{nnf}", 1.0, "access_key_44d_spaced", [{"value": nnf, "score": 2.0}]), reasons

    # 3) nº perto de DANFE/NF-e
    nnf3, evid3 = _extract_nnf_near_context(t)
    if nnf3:
        return FieldPick(f"NF-e{nnf3}", 0.95, evid3, [{"value": nnf3, "score": 1.6, "evidence": evid3}]), reasons

    # 4) linha com Nº + Série
    pat_line = re.compile(r"\bN[ºo]\s*[:\-]?\s*(\d{5,12})\b.*?\bS[ÉE]RIE\b", flags=re.IGNORECASE)
    for ln in lines[:300]:
        m = pat_line.search(ln.replace("°", "º"))
        if m:
            nnf4 = m.group(1)
            return FieldPick(f"NF-e{nnf4}", 0.9, f"line_no_serie: {ln}", [{"value": nnf4, "score": 1.4, "evidence": ln}]), reasons

    reasons.append("DOC_ID: não encontrado (chave 44d / nº / nº+série)")
    return FieldPick(None, 0.0, "", []), reasons


# -----------------------------
# Provider (Emitente/Prestador)
# -----------------------------

def _emitente_from_identificacao_block(text: str, lines: List[str]) -> Tuple[Optional[str], str]:
    # 1) captura direta por regex de bloco
    m = re.search(
        r"identifica[cç][aã]o\s+do\s+emitente\s*\n\s*([^\n]{3,120})",
        text or "",
        flags=re.IGNORECASE,
    )
    if m:
        cand_raw = m.group(1).strip()
        cand = clean_provider_name(cand_raw)
        if not _is_bad_provider_candidate(cand) and _looks_like_company_name(cand):
            return cand, f"regex_identificacao_emitente -> {cand_raw}"

    # 2) varre linhas a partir do rótulo
    for i, ln in enumerate(lines[:250]):
        if re.search(r"identifica[cç][aã]o\s+do\s+emitente", ln, flags=re.IGNORECASE):
            for j in range(i + 1, min(i + 25, len(lines))):
                cand_raw = lines[j].strip()
                cand = clean_provider_name(cand_raw)
                if _is_bad_provider_candidate(cand):
                    continue
                if _looks_like_company_name(cand):
                    return cand, f"identificacao_emitente_block -> {cand_raw}"
    return None, ""

def _emitente_from_recebemos_de(lines: List[str]) -> Tuple[Optional[str], str]:
    # "RECEBEMOS DE KABUM S.A. OS PRODUTOS..."
    for ln in lines[:40]:
        m = re.search(r"recebemos\s+de\s+(.+?)(?:\s+os\s+produtos|\s*$|\.)", ln, flags=re.IGNORECASE)
        if m:
            cand_raw = m.group(1).strip()
            cand = clean_provider_name(cand_raw)
            if not _is_bad_provider_candidate(cand) and _looks_like_company_name(cand):
                return cand, f"recebemos_de -> {ln}"
    return None, ""

def _emitente_from_cnpj(lines: List[str]) -> Tuple[Optional[str], str]:
    for i, ln in enumerate(lines[:350]):
        m = CNPJ_RE.search(ln)
        if not m:
            continue
        for k in range(i - 1, max(-1, i - 20), -1):
            cand_raw = lines[k].strip()
            cand = clean_provider_name(cand_raw)
            if _is_bad_provider_candidate(cand):
                continue
            if _looks_like_company_name(cand):
                return cand, f"antes_do_CNPJ({m.group(0)}) -> {cand_raw}"
    return None, ""

def pick_provider(text: str, doc_type: str) -> Tuple[FieldPick, List[str]]:
    reasons: List[str] = []
    t = text or ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]

    if doc_type == "NF":
        # prioridade: identificação do emitente -> recebemos de -> CNPJ fallback
        cand, evid = _emitente_from_identificacao_block(t, lines)
        if cand:
            return FieldPick(cand, 1.0, evid, [{"value": cand, "score": 2.0, "evidence": evid}]), reasons

        cand, evid = _emitente_from_recebemos_de(lines)
        if cand:
            return FieldPick(cand, 0.95, evid, [{"value": cand, "score": 1.8, "evidence": evid}]), reasons

        cand, evid = _emitente_from_cnpj(lines)
        if cand:
            return FieldPick(cand, 0.9, evid, [{"value": cand, "score": 1.6, "evidence": evid}]), reasons

        reasons.append("PRESTADOR/EMITENTE: não encontrei emitente no DANFE")
        return FieldPick(None, 0.0, "", []), reasons

    # Outros tipos: labels genéricos
    PROVIDER_LABELS = [
        r"prestador\s+de\s+servi[cç]os?",
        r"emitente",
        r"fornecedor",
        r"cedente",
        r"benefici[aá]rio",
        r"favorecido",
        r"raz[aã]o\s+social",
    ]
    label_re = re.compile("|".join(PROVIDER_LABELS), flags=re.IGNORECASE)

    candidates = []
    for i, ln in enumerate(lines[:350]):
        if not label_re.search(ln):
            continue

        extracted = None
        if ":" in ln:
            extracted = ln.split(":", 1)[1].strip()
        else:
            if i + 1 < len(lines):
                extracted = lines[i + 1].strip()

        extracted = clean_provider_name(extracted or "")
        if _is_bad_provider_candidate(extracted):
            continue
        if not _looks_like_company_name(extracted):
            continue

        candidates.append({"value": extracted, "score": 0.7, "evidence": ln})

    if not candidates:
        reasons.append("PRESTADOR: não encontrado")
        return FieldPick(None, 0.0, "", []), reasons

    norm_map = {}
    for c in candidates:
        norm = sanitize_component(c["value"], max_len=120).lower()
        norm_map.setdefault(norm, c)
    candidates = list(norm_map.values())

    top = sorted(candidates, key=lambda c: (c["score"], len(c["value"])), reverse=True)[0]
    return FieldPick(top["value"], 0.8, top["evidence"], candidates), reasons


# -----------------------------
# Amount (audit/log; não entra no nome)
# -----------------------------

MONEY_RE = re.compile(r"(?:R\$\s*)?(\d{1,3}(?:\.\d{3})*,\d{2}|\d+,\d{2}|\d+\.\d{2})")

def pick_amount(text: str, doc_type: str) -> Tuple[FieldPick, List[str]]:
    reasons: List[str] = []
    t = text or ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    candidates = []

    for ln in lines:
        found = MONEY_RE.findall(ln)
        if not found:
            continue
        for raw in found:
            d = parse_brl_money(raw)
            if d is None:
                continue
            candidates.append({"value": str(d), "score": 0.5, "evidence": ln})

    if not candidates:
        return FieldPick(None, 1.0, "", []), reasons

    top = sorted(candidates, key=lambda c: Decimal(c["value"]), reverse=True)[0]
    return FieldPick(top["value"], 0.6, top["evidence"], candidates), reasons


# -----------------------------
# Filename builder
# -----------------------------

def build_filename(date_iso: str, doc_id: Optional[str], doc_type: str, provider: str, cfg: Dict) -> str:
    try:
        d = dt.datetime.strptime(date_iso, "%Y-%m-%d").date()
        date_part = d.strftime("%d-%m-%Y")
    except Exception:
        date_part = date_iso

    # mantém zeros do nNF (se houver)
    nf_num = re.sub(r"\D", "", (doc_id or ""))
    nf_part = f"NF{nf_num}" if nf_num else "NFSEM-ID"

    tipo = doc_type.lower()
    if doc_id:
        dl = doc_id.lower()
        if dl.startswith("nf-e"):
            tipo = "nfe"
        elif dl.startswith("nfs-e"):
            tipo = "nfse"
        elif dl.startswith("nf"):
            tipo = "nf"

    prest = sanitize_component(provider, max_len=80)

    filename = f"{date_part}-{nf_part}-{tipo}-{prest}.pdf"
    filename = strip_accents(filename)
    filename = re.sub(f"[{re.escape(INVALID_CHARS)}]", "-", filename)
    filename = re.sub(r"\s+", "-", filename)
    filename = re.sub(r"-+", "-", filename).strip("-")

    max_len = int(cfg["naming"]["max_len"])
    return filename[:max_len]


def is_field_required(doc_type: str, field: str, cfg: Dict) -> bool:
    req = cfg.get("required_fields", {}).get(doc_type, cfg.get("required_fields", {}).get("DOCUMENTO", []))
    return field in req

def evaluate_status(ex: Extracted, cfg: Dict) -> None:
    reasons = list(ex.reasons)

    if ex.method == "ocr":
        thr = float(cfg["validation"]["ocr_min_mean_conf"])
        if ex.ocr_mean_conf is None or ex.ocr_mean_conf < thr:
            reasons.append(f"OCR: confiança média baixa ({ex.ocr_mean_conf:.1f} < {thr})")

    required_checks = [("date", ex.date_iso), ("doc_id", ex.doc_id), ("provider", ex.provider), ("amount", ex.amount_brl)]

    ok = True
    for field_name, fp in required_checks:
        if is_field_required(ex.doc_type, field_name, cfg):
            if fp.value is None:
                ok = False
                reasons.append(f"{field_name.upper()}: obrigatório e ausente")
            elif fp.confidence < 0.75:
                ok = False
                reasons.append(f"{field_name.upper()}: confiança baixa ({fp.confidence})")

    if ex.date_iso.value and ex.provider.value:
        ex.suggested_filename = build_filename(ex.date_iso.value, ex.doc_id.value, ex.doc_type, ex.provider.value, cfg)
    else:
        ex.suggested_filename = None

    ex.reasons = reasons
    ex.status = "OK" if ok else "REVISAR"


# -----------------------------
# Registry
# -----------------------------

def load_registry(out_dir: Path) -> Dict[str, Dict]:
    path = out_dir / "registry.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}

def save_registry(out_dir: Path, reg: Dict[str, Dict]) -> None:
    path = out_dir / "registry.json"
    path.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------
# Config
# -----------------------------

def load_config(config_path: Optional[Path]) -> Dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if config_path and config_path.exists():
        user_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

        def merge(a: Dict, b: Dict):
            for k, v in b.items():
                if isinstance(v, dict) and isinstance(a.get(k), dict):
                    merge(a[k], v)
                else:
                    a[k] = v

        merge(cfg, user_cfg)
    return cfg

def write_json_log(out_dir: Path, ex: Extracted, raw_text: str) -> None:
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    obj = dataclasses.asdict(ex)
    obj["pdf_path"] = str(ex.pdf_path)
    obj["raw_text_head"] = (raw_text or "")[:12000]

    path = logs_dir / f"{ex.sha256}.json"
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_unique_name(target_dir: Path, filename: str) -> str:
    p = target_dir / filename
    if not p.exists():
        return filename
    stem = p.stem
    ext = p.suffix
    for i in range(1, 200):
        cand = f"{stem}_{i:03d}{ext}"
        if not (target_dir / cand).exists():
            return cand
    raise RuntimeError("Não foi possível gerar nome único após 199 tentativas.")


# -----------------------------
# Dry-run
# -----------------------------

def dry_run(input_dir: Path, out_dir: Path, cfg: Dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_path = out_dir / "audit.csv"
    fieldnames = [
        "original_path", "sha256", "method", "ocr_mean_conf",
        "doc_type", "date_kind", "date_iso", "date_conf",
        "doc_id", "doc_id_conf", "provider", "provider_conf",
        "amount_brl", "amount_conf",
        "suggested_name", "status", "reasons", "log_json",
    ]

    pdf_files = sorted([p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"])

    with audit_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for pdf_path in tqdm(pdf_files, desc="Dry-run"):
            sha = sha256_file(pdf_path)

            raw_native = extract_text_native(pdf_path)
            raw_text = raw_native
            method = "native"
            ocr_mean = None

            if len((raw_native or "").strip()) < int(cfg["validation"]["min_native_text_chars"]):
                try:
                    raw_text, ocr_mean = ocr_pdf(pdf_path)
                    method = "ocr"
                except Exception:
                    raw_text = raw_native or ""
                    method = "native"
                    ocr_mean = None

            doc_type = classify_doc_type(raw_text, pdf_path.name)
            date_pick, date_kind, date_reasons = pick_date(raw_text, doc_type, cfg, pdf_path.name)
            docid_pick, docid_reasons = pick_doc_id(raw_text, doc_type)
            prov_pick, prov_reasons = pick_provider(raw_text, doc_type)
            amt_pick, amt_reasons = pick_amount(raw_text, doc_type)

            reasons: List[str] = []
            reasons.extend(date_reasons)
            reasons.extend(docid_reasons)
            reasons.extend(prov_reasons)
            reasons.extend(amt_reasons)

            ex = Extracted(
                pdf_path=pdf_path,
                sha256=sha,
                method=method,
                ocr_mean_conf=ocr_mean,
                doc_type=doc_type,
                date_kind=date_kind,
                date_iso=date_pick,
                doc_id=docid_pick,
                provider=prov_pick,
                amount_brl=amt_pick,
                status="REVISAR",
                reasons=reasons,
                suggested_filename=None,
            )
            evaluate_status(ex, cfg)

            write_json_log(out_dir, ex, raw_text)
            log_json = str((out_dir / "logs" / f"{sha}.json").resolve())

            w.writerow({
                "original_path": str(pdf_path.resolve()),
                "sha256": ex.sha256,
                "method": ex.method,
                "ocr_mean_conf": "" if ex.ocr_mean_conf is None else f"{ex.ocr_mean_conf:.1f}",
                "doc_type": ex.doc_type,
                "date_kind": ex.date_kind,
                "date_iso": ex.date_iso.value or "",
                "date_conf": f"{ex.date_iso.confidence:.2f}",
                "doc_id": ex.doc_id.value or "",
                "doc_id_conf": f"{ex.doc_id.confidence:.2f}",
                "provider": ex.provider.value or "",
                "provider_conf": f"{ex.provider.confidence:.2f}",
                "amount_brl": ex.amount_brl.value or "",
                "amount_conf": f"{ex.amount_brl.confidence:.2f}",
                "suggested_name": ex.suggested_filename or "",
                "status": ex.status,
                "reasons": " | ".join(ex.reasons),
                "log_json": log_json,
            })

    return audit_path


# -----------------------------
# Run
# -----------------------------

def run_from_audit(audit_csv: Path, input_dir: Path, out_dir: Path, cfg: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    renamed_dir = out_dir / "renamed"
    review_dir = out_dir / "review"
    renamed_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    reg = load_registry(out_dir)

    with audit_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in tqdm(rows, desc="Run"):
        status = (row.get("status") or "").strip().upper()
        orig_path = Path(row["original_path"])
        sha = (row.get("sha256") or "").strip()
        suggested = (row.get("suggested_name") or "").strip()

        if not orig_path.exists():
            continue
        if sha and sha256_file(orig_path) != sha:
            continue

        if status != "OK":
            safe_stem = sanitize_component(orig_path.stem, max_len=60)
            dest = review_dir / f"REVISAR__{safe_stem}__{sha[:8]}.pdf"
            if not dest.exists():
                shutil.copy2(orig_path, dest)
            continue

        if not suggested.lower().endswith(".pdf"):
            continue

        final_name = ensure_unique_name(renamed_dir, suggested)
        dest = renamed_dir / final_name

        if sha in reg and reg[sha].get("status") == "OK":
            existing = renamed_dir / (reg[sha].get("final_name") or "")
            if existing.exists():
                continue

        shutil.copy2(orig_path, dest)

        reg[sha] = {
            "status": "OK",
            "original_path": str(orig_path.resolve()),
            "final_name": final_name,
            "suggested_name": suggested,
            "method": row.get("method"),
            "ocr_mean_conf": row.get("ocr_mean_conf"),
            "doc_type": row.get("doc_type"),
            "date_kind": row.get("date_kind"),
            "date_iso": row.get("date_iso"),
            "doc_id": row.get("doc_id"),
            "provider": row.get("provider"),
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        }

    save_registry(out_dir, reg)


# -----------------------------
# CLI
# -----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="rename_pdfs.py",
        description="Renomeador de PDFs (dry-run + run) com heurísticas robustas para NF-e/DANFE.",
    )
    p.add_argument("--config", type=str, default="", help="Caminho para config.yaml (opcional).")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_dry = sub.add_parser("dry-run")
    p_dry.add_argument("--input", type=str, required=True)
    p_dry.add_argument("--out", type=str, required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--audit", type=str, required=True)
    p_run.add_argument("--input", type=str, required=True)
    p_run.add_argument("--out", type=str, required=True)

    args = p.parse_args(argv)
    cfg = load_config(Path(args.config) if args.config else None)

    if args.cmd == "dry-run":
        audit = dry_run(Path(args.input), Path(args.out), cfg)
        print(f"[OK] Dry-run concluído. Audit: {audit}")
        return 0

    if args.cmd == "run":
        run_from_audit(Path(args.audit), Path(args.input), Path(args.out), cfg)
        print("[OK] Execução real concluída. Veja output/renamed e output/review.")
        return 0

    return 2

if __name__ == "__main__":
    raise SystemExit(main())
