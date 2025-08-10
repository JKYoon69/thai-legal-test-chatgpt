# -*- coding: utf-8 -*-
"""
Thai legal parsing rules (regex + normalizers)
"""
import re

# Thai-digit → ASCII-digit map
THAI_DIGITS = str.maketrans({
    "๐": "0", "๑": "1", "๒": "2", "๓": "3", "๔": "4",
    "๕": "5", "๖": "6", "๗": "7", "๘": "8", "๙": "9",
})

NBSP = "\u00A0"
BOM = "\ufeff"

def remove_bom(text: str) -> str:
    return text.replace(BOM, "")

def normalize_spaces(text: str) -> str:
    # collapse weird spaces, ensure Windows/Mac newlines are normalized
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(NBSP, " ")
    # trim excessive empty lines (keep at most 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def normalize_thai_digits(text: str) -> str:
    return text.translate(THAI_DIGITS)

def normalize_text(text: str) -> str:
    text = remove_bom(text)
    text = normalize_spaces(text)
    text = normalize_thai_digits(text)
    return text

# -------- Regex headers --------
# number fragment
RE_NUM = r"(?P<num>\d{1,4}(?:/\d{1,3})?)"

# allow optional dot and odd spaces after number
TAIL_NUM = r"(?:\s*\.?)"

RE_HEADER = {
    # major structural units (anchored at line start)
    "ภาค": re.compile(rf"(?m)^(?P<label>ภาค)\s+{RE_NUM}{TAIL_NUM}\b"),
    "ลักษณะ": re.compile(rf"(?m)^(?P<label>ลักษณะ)\s+{RE_NUM}{TAIL_NUM}\b"),
    "หมวด": re.compile(rf"(?m)^(?P<label>หมวด)\s+{RE_NUM}{TAIL_NUM}\b"),
    "ส่วน": re.compile(rf"(?m)^(?P<label>ส่วน)\s+{RE_NUM}{TAIL_NUM}\b"),
    "บท": re.compile(rf"(?m)^(?P<label>บท)\s+{RE_NUM}{TAIL_NUM}\b"),

    # frequently appearing named sections (no number)
    "บทเฉพาะกาล": re.compile(rf"(?m)^(?P<label>บทเฉพาะกาล)\b"),
    "บทบัญญัติทั่วไป": re.compile(rf"(?m)^(?P<label>บทบัญญัติทั่วไป)\b"),

    # article / section
    # Note: tolerate extra spaces or NBSP between label and number; optional dot.
    "มาตรา": re.compile(rf"(?m)^(?P<label>มาตรา)[ \t{NBSP}]+{RE_NUM}{TAIL_NUM}\b"),
    "ข้อ": re.compile(rf"(?m)^(?P<label>ข้อ)[ \t{NBSP}]+{RE_NUM}{TAIL_NUM}\b"),
}

# order of levels (lower index = higher in hierarchy)
LEVEL_ORDER = [
    "front_matter",  # virtual
    "ภาค",
    "ลักษณะ",
    "หมวด",
    "ส่วน",
    "บท",  # also hosts บทเฉพาะกาล / บทบัญญัติทั่วไป (same level)
    "มาตรา",
    "ข้อ",
]

def label_level(label: str) -> int:
    if label in ("บทเฉพาะกาล", "บทบัญญัติทั่วไป"):
        label = "บท"
    try:
        return LEVEL_ORDER.index(label)
    except ValueError:
        return LEVEL_ORDER.index("front_matter")
