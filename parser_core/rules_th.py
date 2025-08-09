import regex as re

THAI_DIGITS = "๐๑๒๓๔๕๖๗๘๙"
ARABIC_DIGITS = "0123456789"
THAI2ARABIC = str.maketrans({t: a for t, a in zip(THAI_DIGITS, ARABIC_DIGITS)})

def normalize_text(s: str) -> str:
    # 기본 정규화: 태국숫자→아라비아, 연속 공백/개행 정리(너무 공격적이지 않게)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.translate(THAI2ARABIC)
    # 페이지머리/여백 간단 제거—필요 시 강화
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# 문서 유형별 계층 레벨 정의 (상위→하위)
LEVELS = {
    "code":        ["ภาค", "ลักษณะ", "หมวด", "ส่วน", "บท", "มาตรา"],
    "act":         ["หมวด", "ส่วน", "บท", "มาตรา"],
    "royal_decree":["หมวด", "ส่วน", "บท", "มาตรา"],
    "regulation":  ["หมวด", "ส่วน", "บท", "ข้อ"],  # 고시/규정류는 'ข้อ' 사용
    "unknown":     ["หมวด", "ส่วน", "บท", "มาตรา", "ข้อ"],
}

# 토큰 정규식
# - 라벨(단어) + 공백 + 번호(아라비아 또는 태국숫자 변환 후 숫자) 패턴
# - 'มาตรา 12', 'ข้อ 3' 등
RE_NUM = r"(?P<num>\d{1,4}(?:/\d{1,3})?)"
RE_HEADER = {
    "ภาค": re.compile(rf"(?P<label>ภาค)\s*{RE_NUM}", re.MULTILINE),
    "ลักษณะ": re.compile(rf"(?P<label>ลักษณะ)\s*{RE_NUM}", re.MULTILINE),
    "หมวด": re.compile(rf"(?P<label>หมวด)\s*{RE_NUM}", re.MULTILINE),
    "ส่วน": re.compile(rf"(?P<label>ส่วน)\s*{RE_NUM}", re.MULTILINE),
    "บท": re.compile(rf"(?P<label>บท)\s*{RE_NUM}", re.MULTILINE),
    "มาตรา": re.compile(rf"(?P<label>มาตรา)\s*{RE_NUM}", re.MULTILINE),
    "ข้อ": re.compile(rf"(?P<label>ข้อ)\s*{RE_NUM}", re.MULTILINE),
}

# 부록/말미 표시
RE_APPENDIX = re.compile(r"(บัญชีท้าย|ภาคผนวก|แผนที่ท้าย)[^\n]*", re.MULTILINE)
