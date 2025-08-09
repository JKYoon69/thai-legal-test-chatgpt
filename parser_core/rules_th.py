import regex as re

THAI_DIGITS = "๐๑๒๓๔๕๖๗๘๙"
ARABIC_DIGITS = "0123456789"
THAI2ARABIC = str.maketrans({t: a for t, a in zip(THAI_DIGITS, ARABIC_DIGITS)})

def normalize_text(s: str) -> str:
    # 기본 정규화: 태국숫자→아라비아, 개행 통일, 과도한 공백 정리
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.translate(THAI2ARABIC)
    # 불필요한 탭/연속 공백 정리 (태국어에는 단어 간 공백이 적으므로 과도하게 줄이지 않음)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# 문서 유형별 계층 레벨 정의 (상위→하위)
LEVELS = {
    "code":        ["ภาค", "ลักษณะ", "หมวด", "ส่วน", "บท", "มาตรา"],
    "act":         ["หมวด", "ส่วน", "บท", "มาตรา"],
    "royal_decree":["หมวด", "ส่วน", "บท", "มาตรา"],
    "regulation":  ["หมวด", "ส่วน", "บท", "ข้อ"],
    "unknown":     ["หมวด", "ส่วน", "บท", "มาตรา", "ข้อ"],
}

# ---- 핵심: 라인 시작 anchor + 'ที่' 허용 ----
# 예: "ภาคที่ 1", "ลักษณะที่ 2", "หมวด 3", "มาตรา 12", "ข้อ 5"
RE_NUM = r"(?P<num>\d{1,4}(?:/\d{1,3})?)"
# 상위 레벨들: 'ที่'가 있을 수도/없을 수도
RE_HEADER = {
    "ภาค":      re.compile(rf"(?m)^(?P<label>ภาค)(?:ที่)?\s*{RE_NUM}\b"),
    "ลักษณะ":   re.compile(rf"(?m)^(?P<label>ลักษณะ)(?:ที่)?\s*{RE_NUM}\b"),
    "หมวด":     re.compile(rf"(?m)^(?P<label>หมวด)(?:ที่)?\s*{RE_NUM}\b"),
    "ส่วน":     re.compile(rf"(?m)^(?P<label>ส่วน)(?:ที่)?\s*{RE_NUM}\b"),
    "บท":       re.compile(rf"(?m)^(?P<label>บท)(?:ที่)?\s*{RE_NUM}\b"),
    "มาตรา":    re.compile(rf"(?m)^(?P<label>มาตรา)\s+{RE_NUM}\b"),
    "ข้อ":       re.compile(rf"(?m)^(?P<label>ข้อ)\s+{RE_NUM}\b"),
}

# 부록/말미 표시는 줄 시작에 주로 등장
RE_APPENDIX = re.compile(r"(?m)^(บัญชีท้าย|ภาคผนวก|แผนที่ท้าย)[^\n]*")
