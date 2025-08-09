import regex as re

THAI_DIGITS = "๐๑๒๓๔๕๖๗๘๙"
ARABIC_DIGITS = "0123456789"
THAI2ARABIC = str.maketrans({t: a for t, a in zip(THAI_DIGITS, ARABIC_DIGITS)})

def normalize_text(s: str) -> str:
    # 기본 정규화: 태국숫자→아라비아, 개행 통일, 과도한 공백 정리
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.translate(THAI2ARABIC)
    # '  ' -> ' ', 3줄 이상 연속 개행은 2줄로
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

# ---- 핵심: 라인 시작 anchor 기반 헤더 패턴 ----
# 문서 본문 중간의 'มาตรา 12 ตาม...' 참조를 배제하기 위해 ^로 고정
# 번호는 아라비아만(사전 변환 덕분에 태국숫자 입력도 커버)
RE_NUM = r"(?P<num>\d{1,4}(?:/\d{1,3})?)"
RE_HEADER = {
    "ภาค":      re.compile(rf"(?m)^(?P<label>ภาค)\s*{RE_NUM}\b"),
    "ลักษณะ":   re.compile(rf"(?m)^(?P<label>ลักษณะ)\s*{RE_NUM}\b"),
    "หมวด":     re.compile(rf"(?m)^(?P<label>หมวด)\s*{RE_NUM}\b"),
    "ส่วน":     re.compile(rf"(?m)^(?P<label>ส่วน)\s*{RE_NUM}\b"),
    "บท":       re.compile(rf"(?m)^(?P<label>บท)\s*{RE_NUM}\b"),
    "มาตรา":    re.compile(rf"(?m)^(?P<label>มาตรา)\s+{RE_NUM}\b"),
    "ข้อ":       re.compile(rf"(?m)^(?P<label>ข้อ)\s+{RE_NUM}\b"),
}

# 부록/말미 표시는 줄 시작에 주로 등장
RE_APPENDIX = re.compile(r"(?m)^(บัญชีท้าย|ภาคผนวก|แผนที่ท้าย)[^\n]*")
