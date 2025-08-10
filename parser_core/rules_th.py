# -*- coding: utf-8 -*-
"""
Thai legal parsing rules (regex + normalizers) — article-level tree hardened
"""
import re

# Thai-digit → ASCII-digit
THAI_DIGITS = str.maketrans({
    "๐": "0", "๑": "1", "๒": "2", "๓": "3", "๔": "4",
    "๕": "5", "๖": "6", "๗": "7", "๘": "8", "๙": "9",
})

NBSP = "\u00A0"
BOM = "\ufeff"

def remove_bom(text: str) -> str:
    return text.replace(BOM, "")

def normalize_spaces(text: str) -> str:
    # normalize newlines & non-breaking spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(NBSP, " ")
    # compress >2 blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # strip trailing spaces
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text

def normalize_thai_digits(text: str) -> str:
    return text.translate(THAI_DIGITS)

def normalize_text(text: str) -> str:
    text = remove_bom(text)
    text = normalize_spaces(text)
    text = normalize_thai_digits(text)
    return text

# ---- Header regex ----------------------------------------------------------
# Allow optional trailing dot & odd spaces after number
RE_NUM = r"(?P<num>\d{1,4}(?:/\d{1,3})?)"
TAIL_NUM = r"(?:\s*\.?)"

# anchored at start-of-line; tolerate extra spaces/tabs/NBSP
_ = r"[ \t" + NBSP + r"]+"

RE_HEADER = {
    # major structural units
    "ภาค": re.compile(rf"(?m)^(?P<label>ภาค){_}{RE_NUM}{TAIL_NUM}\b"),
    "ลักษณะ": re.compile(rf"(?m)^(?P<label>ลักษณะ){_}{RE_NUM}{TAIL_NUM}\b"),
    "หมวด": re.compile(rf"(?m)^(?P<label>หมวด){_}{RE_NUM}{TAIL_NUM}\b"),
    "ส่วน": re.compile(rf"(?m)^(?P<label>ส่วน){_}{RE_NUM}{TAIL_NUM}\b"),
    "บท": re.compile(rf"(?m)^(?P<label>บท){_}{RE_NUM}{TAIL_NUM}\b"),

    # named sections without numbers
    "บทเฉพาะกาล": re.compile(rf"(?m)^(?P<label>บทเฉพาะกาล)\b"),
    "บทบัญญัติทั่วไป": re.compile(rf"(?m)^(?P<label>บทบัญญัติทั่วไป)\b"),

    # article / item
    "มาตรา": re.compile(rf"(?m)^(?P<label>มาตรา){_}{RE_NUM}{TAIL_NUM}\b"),
    "ข้อ": re.compile(rf"(?m)^(?P<label>ข้อ){_}{RE_NUM}{TAIL_NUM}\b"),
}

# hierarchy order (lower index = higher)
LEVEL_ORDER = [
    "front_matter",
    "ภาค",
    "ลักษณะ",
    "หมวด",
    "ส่วน",
    "บท",  # includes บทเฉพาะกาล / บทบัญญัติทั่วไป
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
