import datetime
import math
import pandas as pd
import numpy as np
import itertools
import traceback
from collections import defaultdict, Counter
import time
import requests
import json
#from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
import pytz

# ==============================================================================
# SECTION 0: 기본 데이터 정의 (Gemini 재작성)
# ==============================================================================
CHEONGAN = ["갑", "을", "병", "정", "무", "기", "경", "신", "임", "계"]
CHEONGAN_YIN_YANG = {"갑": "양", "을": "음", "병": "양", "정": "음", "무": "양", "기": "음", "경": "양", "신": "음", "임": "양", "계": "음"}
CHEONGAN_TO_HANJA = {"갑":"甲", "을":"乙", "병":"丙", "정":"丁", "무":"戊", "기":"己", "경":"庚", "신":"辛", "임":"壬", "계":"癸"}

JIJI = ["자", "축", "인", "묘", "진", "사", "오", "미", "신", "유", "술", "해"]
JIJI_TO_HANJA = {"자":"子", "축":"丑", "인":"寅", "묘":"卯", "진":"辰", "사":"巳", "오":"午", "미":"未", "신":"申", "유":"酉", "술":"戌", "해":"亥"}

JIJI_YIN_YANG_FUNCTIONAL = {
    "자": "음", "축": "음", "인": "양", "묘": "음", "진": "양", "사": "양",
    "오": "음", "미": "음", "신": "양", "유": "음", "술": "양", "해": "양"
}
JIJI_ELEMENTS_PRIMARY = {
    "인": "목", "묘": "목", "진": "토", "사": "화", "오": "화", "미": "토",
    "신": "금", "유": "금", "술": "토", "해": "수", "자": "수", "축": "토"
}

# 천간/지지와 오행 매핑
saju_element_mapping = {
    "목": ["갑", "을", "인", "묘"],
    "화": ["병", "정", "사", "오"],
    "토": ["무", "기", "진", "술", "축", "미"],
    "금": ["경", "신", "신", "유"], # 천간 辛과 지지 申 모두 '금' 오행에 해당
    "수": ["임", "계", "자", "해"]
}

# 신살 및 합충 기준 데이터
ILJI_TO_SAMHAP_GROUP = {
    "인": "인오술", "오": "인오술", "술": "인오술", "사": "사유축", "유": "사유축", "축": "사유축",
    "신": "신자진", "자": "신자진", "진": "신자진", "해": "해묘미", "묘": "해묘미", "미": "해묘미",
}
SAMHAP_TO_DOHWA = {"인오술": "묘", "사유축": "오", "신자진": "유", "해묘미": "자"}
SAMHAP_TO_HWAGAE = {"인오술": "술", "사유축": "축", "신자진": "진", "해묘미": "미"}
SAMHAP_GROUP_TO_FIRST_CHAR = {"인오술": "인", "사유축": "사", "신자진": "신", "해묘미": "해"}
SAMHAP_FIRST_CHAR_TO_YEONGMA = {"인": "신", "사": "해", "신": "인", "해": "사"}
ILGAN_TO_HONGYEOM = {
    "갑": "오", "을": "오", "병": "인", "정": "미", "무": "진", "기": "진",
    "경": "술", "신": "유", "임": "자", "계": "신"
}
CHEONGAN_HAP_PAIRS = {
    ("갑", "기"): "갑기합(토)", ("기", "갑"): "갑기합(토)", ("을", "경"): "을경합(금)", ("경", "을"): "을경합(금)",
    ("병", "신"): "병신합(수)", ("신", "병"): "병신합(수)", ("정", "임"): "정임합(목)", ("임", "정"): "정임합(목)",
    ("무", "계"): "무계합(화)", ("계", "무"): "무계합(화)",
}
JIJI_YUKHAP_PAIRS = {
    ("자", "축"): "자축육합(토)", ("축", "자"): "자축육합(토)", ("인", "해"): "인해육합(목)", ("해", "인"): "인해육합(목)",
    ("묘", "술"): "묘술육합(화)", ("술", "묘"): "묘술육합(화)", ("진", "유"): "진유육합(금)", ("유", "진"): "진유육합(금)",
    ("사", "신"): "사신육합(수)", ("신", "사"): "사신육합(수)", ("오", "미"): "오미육합", ("미", "오"): "오미육합",
}
JIJI_SAMHAP_LIST = [
    (("인", "오", "술"), "화국 삼합"), (("사", "유", "축"), "금국 삼합"),
    (("신", "자", "진"), "수국 삼합"), (("해", "묘", "미"), "목국 삼합"),
]
JIJI_BANGHAP_LIST = [
    (("인", "묘", "진"), "목국 방합"), (("사", "오", "미"), "화국 방합"),
    (("신", "유", "술"), "금국 방합"), (("해", "자", "축"), "수국 방합"),
]
JIJI_CHUNG_PAIRS = {
    ("자", "오"): "자오충", ("오", "자"): "자오충", ("축", "미"): "축미충", ("미", "축"): "축미충",
    ("인", "신"): "인신충", ("신", "인"): "인신충", ("묘", "유"): "묘유충", ("유", "묘"): "묘유충",
    ("진", "술"): "진술충", ("술", "진"): "진술충", ("사", "해"): "사해충", ("해", "사"): "사해충",
}

# 어떤 글자든 오행을 쉽게 찾기 위한 역방향 매핑
saju_reverse_element_mapping = {ch: e for e, chars in saju_element_mapping.items() for ch in chars}

# 상생/상극 관계 정의
SHENG_RELATIONS = {"목": "화", "화": "토", "토": "금", "금": "수", "수": "목"} # 내가 생하는 것 (식상)
KE_RELATIONS = {"목": "토", "화": "금", "토": "수", "금": "목", "수": "화"}   # 내가 극하는 것 (재성)
saju_sheng_in = {v: k for k, v in SHENG_RELATIONS.items()} # 나를 생하는 것 (인성)
saju_ke_in = {v: k for k, v in KE_RELATIONS.items()}     # 나를 극하는 것 (관성)

# 기타 명리학적 규칙 및 상수 데이터
HANGEUL_TO_HANJA_CHEONGAN = {h: H_HANJA for h, H_HANJA in zip(CHEONGAN, list(CHEONGAN_TO_HANJA.values()))}
HANGEUL_TO_HANJA_JIJI = {h: H_HANJA for h, H_HANJA in zip(JIJI, list(JIJI_TO_HANJA.values()))}
YANG_CHEONGAN = ["갑", "병", "무", "경", "임"]

# 절기 및 연도 계산 관련 상수
YANG_CHEONGAN = ["갑", "병", "무", "경", "임"]
PERIOD_IDX_TO_SOLAR_TERM_NAME = ["입춘", "경칩", "청명", "입하", "망종", "소서", "입추", "백로", "한로", "입동", "대설", "소한"]
MONTH_PILLAR_BORDERS = [(2, 4), (3, 6), (4, 5), (5, 6), (6, 6), (7, 7), (8, 8), (9, 8), (10, 8), (11, 7), (12, 7), (1, 6)]
SOLAR_MONTH_ORDER_TO_JIJI_IDX = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1]

# 외부 API 관련 상수
SERVICE_KEY_DECODED = "Ydis1OrP2uRyCXRimsmNAUGA2rB6UWR6bC17vMBpSN0wMGyKvpAuDCiOLCYNzykaqMi/Kz989ZDtlXqrLwlUVw=="
SOLAR_TERM_TIMES_KST = {} # 절기 API 호출 결과를 캐시하기 위한 변수

# 60갑자 리스트 생성 및 오행 매핑
yearly_heavenly_stems = CHEONGAN
yearly_earthly_branches = JIJI
valid_60_gapja_list = [(yearly_heavenly_stems[i%10], yearly_earthly_branches[i%12]) for i in range(60)]

# 천간 辛과 지지 申을 모두 포함하는 선생님의 원본 로직
saju_element_mapping = {"목":["갑","을","인","묘"],"화":["병","정","사","오"],"토":["무","기","진","술","축","미"],"금":["경","신","신","유"],"수":["임","계","자","해"]}
saju_reverse_element_mapping = {ch:e for e,chars in saju_element_mapping.items() for ch in chars}

saju_position_info = {
    "연간": {"label": "연간", "neighbors": ["월간", "연지", "월지"], "weight": 0.2},
    "연지": {"label": "연지", "neighbors": ["연간", "월간", "월지"], "weight": 0.1},
    "월간": {"label": "월간", "neighbors": ["연간", "일간", "연지", "월지", "일지"], "weight": 0.5},
    "월지": {"label": "월지", "neighbors": ["연간", "월간", "일간", "연지", "일지"], "weight": 1.0},
    "일간": {"label": "일간", "neighbors": ["월간", "시간", "월지", "일지", "시지"], "weight": 1.0},
    "일지": {"label": "일지", "neighbors": ["월간", "일간", "시간", "월지", "시지"], "weight": 0.7},
    "시간": {"label": "시간", "neighbors": ["일간", "일지", "시지"], "weight": 0.4},
    "시지": {"label": "시지", "neighbors": ["일간", "시간", "일지"], "weight": 0.4}
}

# 위치별 가중치 정보
saju_position_info = {
    "연간": {"label": "연간", "neighbors": ["월간", "연지", "월지"], "weight": 0.2},
    "연지": {"label": "연지", "neighbors": ["연간", "월간", "월지"], "weight": 0.1},
    "월간": {"label": "월간", "neighbors": ["연간", "일간", "연지", "월지", "일지"], "weight": 0.5},
    "월지": {"label": "월지", "neighbors": ["연간", "월간", "일간", "연지", "일지"], "weight": 1.0},
    "일간": {"label": "일간", "neighbors": ["월간", "시간", "월지", "일지", "시지"], "weight": 1.0},
    "일지": {"label": "일지", "neighbors": ["월간", "일간", "시간", "월지", "시지"], "weight": 0.7},
    "시간": {"label": "시간", "neighbors": ["일간", "일지", "시지"], "weight": 0.4},
    "시지": {"label": "시지", "neighbors": ["일간", "시간", "일지"], "weight": 0.4}
}

# 진술축미 지장간의 특수 가치
jinnsulchukmi_base_values = {
    "진": {"목": 6.0, "화": -2.0, "토": 10.0, "금": 2.0, "수": 7.0},
    "술": {"목": 1.0, "화": 7.0, "토": 10.0, "금": 4.0, "수": -1.0},
    "축": {"목": -2.0, "화": -3.0, "토": 10.0, "금": 7.0, "수": 10.0},
    "미": {"목": 7.0, "화": 10.0, "토": 10.0, "금": -3.0, "수": -3.0}
}

# 오행 간 상호작용 점수 테이블
COMPLEX_INTERACTION_TABLE = {
    "목": {"목": 1.5, "화": 2.0, "토": 1.0, "금": -1.5, "수": -2.0},
    "화": {"목": 2.0, "화": 1.5, "토": 1.0, "금": -2.0, "수": -1.5},
    "토": {"목": 1.0, "화": 1.0, "토": 2.0, "금": 1.0, "수": 1.0},
    "금": {"목": -1.5, "화": -2.0, "토": 1.0, "금": 1.5, "수": 2.0},
    "수": {"목": -2.0, "화": -1.5, "토": 1.0, "금": 2.0, "수": 1.5}
}

# 한난조습 점수표
HJS_SCORES_GLOBAL = {
    '자': {'한': 8, '난': 2, '조': 8, '습': 2}, '축': {'한': 9, '난': 1, '조': 3, '습': 7},
    '인': {'한': 7, '난': 3, '조': 7, '습': 3}, '묘': {'한': 4, '난': 6, '조': 3, '습': 7},
    '진': {'한': 3, '난': 7, '조': 1, '습': 9}, '사': {'한': 2, '난': 8, '조': 7, '습': 3},
    '오': {'한': 2, '난': 8, '조': 4, '습': 6}, '미': {'한': 1, '난': 9, '조': 7, '습': 3},
    '신': {'한': 3, '난': 7, '조': 4, '습': 6}, '유': {'한': 6, '난': 4, '조': 8, '습': 2},
    '술': {'한': 7, '난': 3, '조': 9, '습': 1}, '해': {'한': 7, '난': 3, '조': 3, '습': 7}
}

# ==============================================================================
# SECTION 3: 헬퍼 함수 정의 (Gemini 재작성)
# ==============================================================================

def to_LMT(dt_kst):
    """KST(한국 표준시)를 LMT(지방 평균시)로 변환합니다. (30분 차감)"""
    return dt_kst - datetime.timedelta(minutes=30)

def convert_lunar_to_solar(l_year, l_month, l_day, is_leap_month):
    """공공데이터 API를 이용해 음력을 양력으로 변환합니다."""
    api_url = "http://apis.data.go.kr/B090041/openapi/service/LrsrCldInfoService/getLunCalInfo"
    params = {
        "serviceKey": SERVICE_KEY_DECODED,
        "solYear": str(l_year),
        "lunYear": str(l_year),
        "lunMonth": str(l_month).zfill(2),
        "lunDay": str(l_day).zfill(2),
        "leapMonth": "Y" if is_leap_month else "N",
        "_type": "json"
    }

    try:
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get('response', {}).get('body', {}).get('items'):
            item = data['response']['body']['items'].get('item', {})
            if item:
                # API 응답이 단일 항목일 경우 리스트로 감싸서 처리
                items = [item] if not isinstance(item, list) else item
                first_item = items[0]
                
                sol_year = first_item.get('solYear')
                sol_month = first_item.get('solMonth')
                sol_day = first_item.get('solDay')

                if sol_year and sol_month and sol_day:
                    return datetime.date(int(sol_year), int(sol_month), int(sol_day))
        
        # API에서 유효한 날짜를 받지 못한 모든 경우
        # (디버깅을 위해 에러/정보 메시지는 원본 그대로 유지합니다)
        if data.get('response', {}).get('header', {}).get('resultCode') != '00':
            print(f"API 오류: {data['response']['header'].get('resultMsg')}")
        return None

    except requests.exceptions.RequestException as e:
        print(f"API 요청 오류: {e}")
        return None
    except json.JSONDecodeError:
        print(f"API 응답 JSON 파싱 오류. 응답: {response.text}")
        return None
    except Exception as e:
        print(f"음력->양력 변환 중 예기치 않은 오류: {e}")
        return None
    
def get_solar_terms_from_api(year, service_key):
    """지정된 연도의 24절기 정보를 공공데이터 API를 통해 가져옵니다."""
    api_url = "http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/get24DivisionsInfo"
    params = {
        "serviceKey": service_key,
        "solYear": str(year),
        "numOfRows": "30",
        "_type": "json"
    }
    
    try:
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status() # HTTP 오류 발생 시 예외 발생
        data = response.json()

        items = data.get('response', {}).get('body', {}).get('items', {}).get('item')
        if not items:
            # API 응답은 정상이나, 데이터가 없는 경우
            return {}

        # 항목이 하나일 경우 list로 만들어 일관성 유지
        if not isinstance(items, list):
            items = [items]

        solar_terms_for_year = {}
        for item in items:
            try:
                term_name = item.get('dateName')
                date_str = str(item.get('locdate'))
                time_str = str(item.get('kst', '1200')).zfill(4)

                if term_name and len(date_str) == 8:
                    dt_obj = datetime.datetime(
                        int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]),
                        int(time_str[:2]), int(time_str[2:])
                    )
                    solar_terms_for_year[term_name] = dt_obj
            except (ValueError, TypeError) as e:
                # 개별 항목 처리 중 오류 발생 시 print하고 계속 진행
                print(f"주의: {year}년 {term_name} 처리 중 오류 발생 - {e}")
                continue
        
        return solar_terms_for_year

    except requests.exceptions.RequestException as e:
        print(f"API 요청 오류: {e}")
        return None # 네트워크 오류 등 요청 자체가 실패한 경우
    except json.JSONDecodeError:
        print(f"API 응답 JSON 파싱 오류. 응답 내용: {response.text}")
        return None
    except Exception as e:
        print(f"절기 정보 API 호출 중 예기치 않은 오류: {e}")
        return None
    
def get_precise_jeolgi_datetime_lmt(astro_year, period_idx):
    """
    지정된 년도와 절기 순번(period_idx)에 해당하는 정확한 절기 시간을 LMT로 반환합니다.
    내부적으로 API 호출 결과를 캐시하여 중복 호출을 방지합니다.
    """
    # 전역 변수로 선언된 캐시와 서비스 키를 사용합니다.
    global SOLAR_TERM_TIMES_KST, SERVICE_KEY_DECODED
    
    term_name = PERIOD_IDX_TO_SOLAR_TERM_NAME[period_idx]
    calendar_year_of_term = astro_year
    term_month_approx = MONTH_PILLAR_BORDERS[period_idx][0]
    
    # 입춘(2월 4일경)보다 월이 작으면 다음 해의 절기를 가져와야 합니다.
    if term_month_approx < MONTH_PILLAR_BORDERS[0][0]:
        calendar_year_of_term = astro_year + 1

    # 캐시에 해당 년도 절기 정보가 없으면 API를 호출하여 가져옵니다.
    if calendar_year_of_term not in SOLAR_TERM_TIMES_KST:
        fetched_terms = get_solar_terms_from_api(calendar_year_of_term, SERVICE_KEY_DECODED)
        # API 호출 성공 여부와 관계없이 결과를 캐시에 저장하여 중복 호출 방지
        SOLAR_TERM_TIMES_KST[calendar_year_of_term] = fetched_terms if fetched_terms is not None else {}

    year_data = SOLAR_TERM_TIMES_KST.get(calendar_year_of_term)
    
    # 캐시에서 정확한 절기 정보를 찾아서 LMT로 변환 후 반환합니다.
    if year_data and term_name in year_data:
        term_datetime_kst = year_data[term_name]
        return to_LMT(term_datetime_kst)
    else:
        # API 호출에 실패했거나 데이터가 없는 경우, 대략적인 날짜로 대체합니다.
        border_m, border_d = MONTH_PILLAR_BORDERS[period_idx]
        return to_LMT(datetime.datetime(calendar_year_of_term, border_m, border_d, 12, 0))
    
def get_year_pillar(lmt_dt):
    """LMT를 기준으로 년주를 계산합니다. (입춘 기준)"""
    ipchun_lmt_current_cal_year = get_precise_jeolgi_datetime_lmt(lmt_dt.year, 0)
    astro_year = lmt_dt.year
    if lmt_dt < ipchun_lmt_current_cal_year:
        astro_year -= 1
    
    year_cheongan_idx = (astro_year - 4) % 10
    year_jiji_idx = (astro_year - 4) % 12
    return CHEONGAN[year_cheongan_idx], JIJI[year_jiji_idx], year_cheongan_idx, astro_year

def get_month_pillar(lmt_dt, year_cheongan_idx, astro_year):
    """LMT를 기준으로 월주를 계산합니다. (절기 기준, 월두법 적용)"""
    borders_in_cycle_lmt = [get_precise_jeolgi_datetime_lmt(astro_year, i) for i in range(12)]
    month_period_idx = 11
    for i in range(12):
        if lmt_dt < borders_in_cycle_lmt[i]:
            month_period_idx = (i - 1 + 12) % 12
            break
            
    month_cheongan_start_map = {0:2, 5:2, 1:4, 6:4, 2:6, 7:6, 3:8, 8:8, 4:0, 9:0}
    base_cheongan_idx_for_inwol = month_cheongan_start_map.get(year_cheongan_idx)
    month_cheongan_idx = (base_cheongan_idx_for_inwol + month_period_idx) % 10
    month_jiji_actual_idx = SOLAR_MONTH_ORDER_TO_JIJI_IDX[month_period_idx]
    return CHEONGAN[month_cheongan_idx], JIJI[month_jiji_actual_idx], month_cheongan_idx, month_period_idx

def get_day_pillar(lmt_dt):
    """LMT를 기준으로 일주를 계산합니다."""
    ordinal = lmt_dt.date().toordinal()
    DAY_CHEONGAN_OFFSET = 4
    DAY_JIJI_OFFSET = 2
    day_cheongan_idx = (ordinal + DAY_CHEONGAN_OFFSET) % 10
    day_jiji_idx = (ordinal + DAY_JIJI_OFFSET) % 12
    return CHEONGAN[day_cheongan_idx], JIJI[day_jiji_idx], day_cheongan_idx

def get_hour_pillar(lmt_dt, day_cheongan_idx):
    """LMT를 기준으로 시주를 계산합니다. (야자시 적용)"""
    h, m = lmt_dt.hour, lmt_dt.minute
    
    # 23시 30분 이후는 다음날의 자시로 취급되나, 지지는 '자'로 동일합니다.
    # 시간 지지 인덱스 계산
    hour_jiji_idx = 0 # 자시 기본값
    if 1.5 <= (h + m/60) < 3.5: hour_jiji_idx = 1  # 축시
    elif 3.5 <= (h + m/60) < 5.5: hour_jiji_idx = 2  # 인시
    elif 5.5 <= (h + m/60) < 7.5: hour_jiji_idx = 3  # 묘시
    elif 7.5 <= (h + m/60) < 9.5: hour_jiji_idx = 4  # 진시
    elif 9.5 <= (h + m/60) < 11.5: hour_jiji_idx = 5 # 사시
    elif 11.5 <= (h + m/60) < 13.5: hour_jiji_idx = 6 # 오시
    elif 13.5 <= (h + m/60) < 15.5: hour_jiji_idx = 7 # 미시
    elif 15.5 <= (h + m/60) < 17.5: hour_jiji_idx = 8 # 신시
    elif 17.5 <= (h + m/60) < 19.5: hour_jiji_idx = 9 # 유시
    elif 19.5 <= (h + m/60) < 21.5: hour_jiji_idx = 10# 술시
    elif 21.5 <= (h + m/60) < 23.5: hour_jiji_idx = 11# 해시

    # 야자시(23:30 이후)인 경우, 시주 천간 계산의 기준이 되는 일간을 다음날의 것으로 봅니다.
    effective_day_cheongan_idx = day_cheongan_idx
    if h >= 23 and m >= 30:
        effective_day_cheongan_idx = (day_cheongan_idx + 1) % 10

    hour_cheongan_start_map = {0:0, 5:0, 1:2, 6:2, 2:4, 7:4, 3:6, 8:6, 4:8, 9:8}
    base_cheongan_idx_for_jasi = hour_cheongan_start_map.get(effective_day_cheongan_idx)
    hour_cheongan_idx = (base_cheongan_idx_for_jasi + hour_jiji_idx) % 10

    return CHEONGAN[hour_cheongan_idx], JIJI[hour_jiji_idx]

def get_all_possible_hour_pillars(day_cheongan_idx):
    """시간을 모를 경우, 가능한 모든 시주를 반환합니다."""
    pillars = []
    hour_cheongan_start_map = {0: 0, 5: 0, 1: 2, 6: 2, 2: 4, 7: 4, 3: 6, 8: 6, 4: 8, 9: 8}
    base_cheongan_idx = hour_cheongan_start_map.get(day_cheongan_idx)
    for hour_jiji_idx in range(12):
        hour_cheongan_idx = (base_cheongan_idx + hour_jiji_idx) % 10
        hour_gan = CHEONGAN[hour_cheongan_idx]
        hour_ji = JIJI[hour_jiji_idx]
        pillars.append((hour_gan, hour_ji))
    return pillars

def get_daewoon_direction(year_gan_char, gender_str):
    """년도 천간의 음양과 성별을 바탕으로 대운의 방향(순행/역행)을 결정합니다."""
    is_yang_gan = year_gan_char in YANG_CHEONGAN
    if gender_str == "남":
        direction = "순행" if is_yang_gan else "역행"
    elif gender_str == "여":
        direction = "역행" if is_yang_gan else "순행"
    else:
        direction = "알 수 없음"
    return direction

def get_daewoon_su(birth_lmt_dt, daewoon_dir_str, astro_year, current_month_period_idx):
    """대운수(대운 시작 나이)를 계산합니다."""
    try:
        if daewoon_dir_str == "순행":
            next_jeolgi_lmt_dt = get_precise_jeolgi_datetime_lmt(astro_year, (current_month_period_idx + 1) % 12)
            time_diff = next_jeolgi_lmt_dt - birth_lmt_dt
        elif daewoon_dir_str == "역행":
            current_jeolgi_lmt_dt = get_precise_jeolgi_datetime_lmt(astro_year, current_month_period_idx)
            time_diff = birth_lmt_dt - current_jeolgi_lmt_dt
        else:
            return -1 # 방향 에러

        days_diff = time_diff.total_seconds() / (24 * 60 * 60)
        daewoon_num_float = max(0, days_diff) / 3.0
        rounded_daewoon_su = int(math.floor(daewoon_num_float + 0.5))
        
        # 대운수는 최소 1입니다.
        final_daewoon_su = 1 if rounded_daewoon_su == 0 else rounded_daewoon_su
        return final_daewoon_su
        
    except Exception as e:
        # print(f"대운수 계산 중 오류: {e}") # 디버깅 시 사용
        return -3 # 계산 중 예외 발생
    
def get_saju_element(char):
    """한글자(천간/지지)를 입력받아 해당하는 오행을 반환합니다."""
    return saju_reverse_element_mapping.get(char, None)

def determine_needed_elements_for_position(pos_label, char_val, full_saju_dict):
    """사주의 특정 위치 글자를 기준으로, 주변 글자와의 관계를 분석하여 해당 위치에 필요한 오행을 결정합니다."""
    center_element = get_saju_element(char_val)
    if not center_element:
        return []

    # '나를 돕는 세력'과 '내 힘을 빼는 세력'의 수를 셉니다.
    in_force_count = 0
    out_force_count = 0
    
    neighbor_positions = saju_position_info[pos_label]['neighbors']
    for neighbor_pos in neighbor_positions:
        neighbor_char = full_saju_dict.get(neighbor_pos, "")
        neighbor_element = get_saju_element(neighbor_char)

        if not neighbor_element or neighbor_element == center_element:
            continue
        
        # 나를 생(生)하거나 극(剋)하는, 'in' 관계 카운트
        if neighbor_element == saju_sheng_in.get(center_element) or neighbor_element == saju_ke_in.get(center_element):
            in_force_count += 1
        # 내가 생(生)하거나 극(剋)하는, 'out' 관계 카운트
        if neighbor_element == SHENG_RELATIONS.get(center_element) or neighbor_element == KE_RELATIONS.get(center_element):
            out_force_count += 1
            
    # 세력 비교를 통해 필요한 오행 관계를 반환합니다.
    if in_force_count > out_force_count: # 세력이 강할 때 (신강)
        # 내 힘을 빼는 관계 (식상, 재성)가 필요함
        return [(KE_RELATIONS[center_element], 1), (SHENG_RELATIONS[center_element], 2)]
    elif out_force_count > in_force_count: # 세력이 약할 때 (신약)
        # 나를 돕는 관계 (인성, 관성)가 필요함
        return [(saju_sheng_in[center_element], 1), (saju_ke_in[center_element], 2)]
    else: # 세력이 중화될 때
        return []

def derive_keyword(saju_8_chars_dict, pne1_element, pne2_element):
    """
    1, 2순위 필요오행을 바탕으로 사주 원국의 키워드(대표 글자)를 도출합니다.
    순서대로 원국에 해당 오행이 있는지 찾고, 없으면 일간을 반환합니다.
    """
    # 검색할 위치의 우선순위를 미리 정의합니다.
    search_positions = ["월간", "시간", "연간", "월지", "일지", "시지", "연지"]

    # --- 1. 1순위 필요오행(pne1_element)으로 키워드 검색 ---
    if pne1_element:
        for pos in search_positions:
            char = saju_8_chars_dict.get(pos)
            if char and get_saju_element(char) == pne1_element:
                return char, pos # 찾으면 즉시 반환

    # --- 2. 2순위 필요오행(pne2_element)으로 키워드 검색 (1순위를 못 찾았을 경우) ---
    if pne2_element:
        for pos in search_positions:
            char = saju_8_chars_dict.get(pos)
            if char and get_saju_element(char) == pne2_element:
                return char, pos # 찾으면 즉시 반환

    # --- 3. 최종 수단: 일간을 키워드로 반환 ---
    day_master_char = saju_8_chars_dict.get("일간")
    if day_master_char:
        return day_master_char, "일간"
    
    return None, None # 일간조차 없는 예외적인 경우

def calculate_needed_element_scores(saju_8_chars_dict):
    """사주팔자 전체의 균형을 보고, 각 오행이 얼마나 필요한지 점수를 계산합니다."""
    scores = defaultdict(float)
    for pos_label, char_val in saju_8_chars_dict.items():
        if not char_val or pos_label not in saju_position_info:
            continue
        
        needed_for_this_pos = determine_needed_elements_for_position(pos_label, char_val, saju_8_chars_dict)
        position_weight = saju_position_info[pos_label]['weight']
        
        for needed_el, priority_val in needed_for_this_pos:
            scores[needed_el] += position_weight
            
    max_total_possible = sum(info['weight'] for info in saju_position_info.values())
    all_elements = ["목", "화", "토", "금", "수"]
    final_scores = {el: 0.0 for el in all_elements}
    
    if max_total_possible > 0:
        for el, score in scores.items():
            final_scores[el] = round(score / max_total_possible, 2)
            
    return final_scores, {}

def calculate_luck_quantity_auto(saju_8_chars_dict, keyword_element, keyword_pos_label):
    """키워드 오행을 기준으로, 원국 내 다른 글자들과의 관계를 통해 '행운량'을 계산합니다."""
    reference_el = keyword_element
    total_score = 0
    
    if reference_el:
        keyword_to_participating_positions = {
            "연간": ["연지", "월간", "월지"], "연지": ["연간", "월간", "월지"],
            "월간": ["연간", "연지", "월지", "일간", "일지"], "월지": ["연간", "연지", "월간", "일간", "일지"],
            "일간": ["월간", "월지", "일지", "시간", "시지"], "일지": ["월간", "월지", "일간", "시간", "시지"],
            "시간": ["일간", "일지", "시지"], "시지": ["일간", "일지", "시간"]
        }
        positions_for_score_calc = keyword_to_participating_positions.get(keyword_pos_label, [])
        
        if positions_for_score_calc:
            for pos_label in positions_for_score_calc:
                char_val = saju_8_chars_dict.get(pos_label, "")
                if not char_val: continue
                
                el = get_saju_element(char_val)
                if not el or el == reference_el: continue
                
                if el == saju_sheng_in.get(reference_el): total_score += 1
                elif el == saju_ke_in.get(reference_el): total_score -= 2
                elif el == SHENG_RELATIONS.get(reference_el): total_score -= 1
                elif el == KE_RELATIONS.get(reference_el): total_score -= 2
    
    luck_quantity = abs(total_score)
    return 1 if luck_quantity == 0 else luck_quantity

def get_base_value_for_score(char_of_luck, position_key, element_of_luck, p_el, p_val, s_el, s_val):
    """운의 글자에 대한 기본 가중치를 계산합니다. (진술축미 고려)"""
    if position_key.endswith("지지") and char_of_luck in jinnsulchukmi_base_values:
        if p_el and p_el in jinnsulchukmi_base_values[char_of_luck]:
            return jinnsulchukmi_base_values[char_of_luck][p_el]
            
    if p_el and element_of_luck == p_el:
        return p_val
    elif s_el and element_of_luck == s_el:
        return s_val
    else:
        return 1.0
    
def get_yearly_gapja(year):
    """특정 연도의 간지를 계산합니다."""
    # 1984년(갑자년)을 기준으로 60갑자 순환을 계산
    index = (year - 1984) % 60
    gan = yearly_heavenly_stems[index % 10]
    ji = yearly_earthly_branches[index % 12]
    return gan, ji

def get_yearly_daewoon_list(birth_year_solar, month_gan_char_saju, month_ji_char_saju, daewoon_soo_val, forward_bool):
    """100년치 대운의 흐름을 리스트로 생성합니다."""
    first_daewoon_start_year = birth_year_solar + daewoon_soo_val - 1
    
    if month_gan_char_saju not in yearly_heavenly_stems:
        raise ValueError(f"월주 천간('{month_gan_char_saju}') 오류")
    if month_ji_char_saju not in yearly_earthly_branches:
        raise ValueError(f"월주 지지('{month_ji_char_saju}') 오류")
        
    gan_idx = yearly_heavenly_stems.index(month_gan_char_saju)
    ji_idx = yearly_earthly_branches.index(month_ji_char_saju)
    
    daewoons = []
    for i in range(10):  # 10개의 대운 (100년)
        offset = (i + 1) if forward_bool else -(i + 1)
        
        g_idx_effective = (gan_idx + offset) % 10
        j_idx_effective = (ji_idx + offset) % 12
        
        gan = yearly_heavenly_stems[g_idx_effective]
        ji = yearly_earthly_branches[j_idx_effective]
        
        start_year = first_daewoon_start_year + (i * 10)
        end_year = start_year + 9
        
        daewoons.append({"start": start_year, "end": end_year, "gan": gan, "ji": ji})
        
    return daewoons

def calculate_yearly_luck_final(birth_year_solar, saju_8_chars, keyword_char, sorted_needed_elements, auto_luck_amount, daewoon_list_val):
    """
    100년치 대운/연운의 흐름에 따른 '행운강도'를 계산하여 리스트로 반환합니다.
    """
    original_primary_el = sorted_needed_elements[0][0] if len(sorted_needed_elements) >= 1 else None
    original_secondary_el = sorted_needed_elements[1][0] if len(sorted_needed_elements) >= 2 else None

    # 필요한 오행이 없거나, 상호작용 데이터가 없으면 빈 결과를 반환하고 함수를 종료합니다.
    if not original_primary_el or original_primary_el not in COMPLEX_INTERACTION_TABLE:
        return [], 0.0, 0.0

    weights_factor = {"대운_천간": 1.0, "대운_지지": 3.0, "연운_천간": 3.0, "연운_지지": 5.0}

    def get_theoretical_extremes():
        """모든 간지 조합을 시뮬레이션하여 이론적인 최대/최소 효과 점수를 계산하는 내부 함수"""
        max_total, min_total = float('-inf'), float('inf')
        for daewoon_ganji in valid_60_gapja_list:
            dg, dj = daewoon_ganji
            for yeonwoon_ganji in valid_60_gapja_list:
                yg, yj = yeonwoon_ganji
                current_total_for_extreme = 0
                
                def score_extreme(ch_extreme, pos_key_extreme):
                    el_extreme_local = saju_reverse_element_mapping.get(ch_extreme) # 수정된 부분
                    if not el_extreme_local: return 0
                    base_extreme = get_base_value_for_score(ch_extreme, pos_key_extreme, el_extreme_local, original_primary_el, 12.0, original_secondary_el, 10.0)
                    interaction_coeffs = COMPLEX_INTERACTION_TABLE.get(original_primary_el, {})
                    inter_extreme = interaction_coeffs.get(el_extreme_local, 0.0)
                    return base_extreme * inter_extreme * weights_factor[pos_key_extreme]

                current_total_for_extreme = sum([score_extreme(dg, '대운_천간'), score_extreme(dj, '대운_지지'), score_extreme(yg, '연운_천간'), score_extreme(yj, '연운_지지')])
                if current_total_for_extreme > max_total: max_total = current_total_for_extreme
                if current_total_for_extreme < min_total: min_total = current_total_for_extreme
        return (max_total, min_total)

    max_effect, min_effect = get_theoretical_extremes()
    results = []

    for age in range(101):
        year = birth_year_solar + age
        y_gan, y_ji = get_yearly_gapja(year)
        daewoon = next((d for d in daewoon_list_val if d['start'] <= year <= d['end']), None)
        if not daewoon: continue
            
        effective_pne1, effective_pne2, was_adjusted = adjust_needed_elements_for_haps(
            saju_8_chars, keyword_char, sorted_needed_elements, daewoon['ji'], y_ji
        )

        def score_yearly_final_inner(ch_yearly, pos_key_yearly):
            el_yearly = saju_reverse_element_mapping.get(ch_yearly) # 수정된 부분
            if not el_yearly: return 0
            base_yearly = get_base_value_for_score(ch_yearly, pos_key_yearly, el_yearly, effective_pne1, 12.0, effective_pne2, 10.0)
            interaction_coeffs = COMPLEX_INTERACTION_TABLE.get(effective_pne1, {})
            inter_yearly = interaction_coeffs.get(el_yearly, 0.0)
            return base_yearly * inter_yearly * weights_factor[pos_key_yearly]

        total_score_for_year = sum([score_yearly_final_inner(daewoon['gan'], '대운_천간'), score_yearly_final_inner(daewoon['ji'], '대운_지지'), score_yearly_final_inner(y_gan, '연운_천간'), score_yearly_final_inner(y_ji, '연운_지지')])
        
        effective_score = 0.0
        if total_score_for_year > 0 and max_effect != 0:
            effective_score = total_score_for_year / max_effect
        elif total_score_for_year < 0 and min_effect != 0:
            effective_score = total_score_for_year / abs(min_effect)

        intermediate_luck_strength = (effective_score * auto_luck_amount) / 10.0
        final_luck_strength = intermediate_luck_strength / 10.0 if was_adjusted else intermediate_luck_strength
        luck_strength = round(final_luck_strength, 3)

        results.append({"나이": age, "연도": year, "행운강도": luck_strength, "대운천간": daewoon['gan'], "대운지지": daewoon['ji'], "연운천간": y_gan, "연운지지": y_ji, "대운시작": daewoon['start'], "대운종료": daewoon['end']})

    return results, max_effect, min_effect

def calculate_overall_luck_score(luck_results_list):
    """행운강도 리스트를 받아, 특정 나이 구간의 평균 점수를 계산합니다."""
    if not luck_results_list: return 0.0
    
    df = pd.DataFrame(luck_results_list)
    df_filtered = df[(df['나이'] >= 21) & (df['나이'] <= 70)]
    
    if df_filtered.empty:
        return df['행운강도'].mean() if not df.empty else 0.0
    
    return df_filtered['행운강도'].mean()

def get_ganji_for_year(year):
    """특정 연도의 간지를 문자열로 반환합니다. (예: "갑자")"""
    # 1984년(갑자년)을 기준으로 60갑자 순환을 계산
    index = (year - 4) % 60
    gan = CHEONGAN[index % 10]
    ji = JIJI[index % 12]
    return f"{gan}{ji}"

def calculate_saju_hjs_total_scores(jiji_siju, jiji_ilju, jiji_wolju, jiji_yeonju, hjs_scores_dict, yeonun_jiji=None):
    """
    사주 원국의 한난조습 점수와, 연운이 적용된 동적 점수를 계산합니다.
    """
    if not hjs_scores_dict:
        return {'한': 0, '난': 0, '조': 0, '습': 0}

    # 1. 원국(타고난 사주)의 한난조습 점수를 먼저 계산합니다. (위치별 가중치 적용)
    base_han_raw = (hjs_scores_dict.get(jiji_siju, {}).get('한', 0) * 0.15) + \
                   (hjs_scores_dict.get(jiji_ilju, {}).get('한', 0) * 0.20) + \
                   (hjs_scores_dict.get(jiji_wolju, {}).get('한', 0) * 0.60) + \
                   (hjs_scores_dict.get(jiji_yeonju, {}).get('한', 0) * 0.05)

    base_jo_raw = (hjs_scores_dict.get(jiji_siju, {}).get('조', 0) * 0.05) + \
                  (hjs_scores_dict.get(jiji_ilju, {}).get('조', 0) * 0.60) + \
                  (hjs_scores_dict.get(jiji_wolju, {}).get('조', 0) * 0.20) + \
                  (hjs_scores_dict.get(jiji_yeonju, {}).get('조', 0) * 0.15)

    # 0~1 사이의 값으로 정규화
    base_han_normalized = base_han_raw / 10.0
    base_jo_normalized = base_jo_raw / 10.0

    # 2. '연운(yeonun_jiji)'이 입력되었는지 확인하여 분기 처리
    if yeonun_jiji:
        # 연운의 영향을 반영한 동적 점수 계산
        yeonun_han_score = hjs_scores_dict.get(yeonun_jiji, {}).get('한', 0) / 10.0
        yeonun_jo_score = hjs_scores_dict.get(yeonun_jiji, {}).get('조', 0) / 10.0
        
        # 원국과 연운의 영향력을 7:3으로 혼합
        dynamic_han = (base_han_normalized * 0.7) + (yeonun_han_score * 0.3)
        dynamic_jo = (base_jo_normalized * 0.7) + (yeonun_jo_score * 0.3)

        return {
            '한': round(dynamic_han, 2), '난': round(1 - dynamic_han, 2),
            '조': round(dynamic_jo, 2), '습': round(1 - dynamic_jo, 2)
        }
    else:
        # 연운 입력이 없으면 원국 점수만 반환
        return {
            '한': round(base_han_normalized, 2), '난': round(1 - base_han_normalized, 2),
            '조': round(base_jo_normalized, 2), '습': round(1 - base_jo_normalized, 2)
        }

def get_ju_hjs_details(gan_siju, jiji_siju, gan_ilju, jiji_ilju, gan_wolju, jiji_wolju, gan_yeonju, jiji_yeonju, hjs_scores_dict):
    """각 주(柱)별 한난조습의 상세 점수를 리스트로 반환합니다."""
    ju_details = []
    ju_list = [
        ('시주', gan_siju, jiji_siju),
        ('일주', gan_ilju, jiji_ilju),
        ('월주', gan_wolju, jiji_wolju),
        ('연주', gan_yeonju, jiji_yeonju)
    ]

    for gubun, gan, ji in ju_list:
        scores = hjs_scores_dict.get(ji, {'한': 0, '난': 0, '조': 0, '습': 0})
        ju_details.append({
            '구분': gubun, '천간': gan, '지지': ji,
            '한': scores.get('한', 0), '난': scores.get('난', 0),
            '조': scores.get('조', 0), '습': scores.get('습', 0)
        })
    return ju_details

# --- 합충 분석 함수 ---

def find_cheongan_hap(saju_8_chars: dict) -> list:
    """사주 원국 내 천간합을 찾습니다."""
    results = []
    # 확인할 천간 쌍 정의 (인접한 위치만)
    adjacent_cheongan_pairs = [
        (saju_8_chars.get("연간"), saju_8_chars.get("월간"), "연간", "월간"),
        (saju_8_chars.get("월간"), saju_8_chars.get("일간"), "월간", "일간"),
        (saju_8_chars.get("일간"), saju_8_chars.get("시간"), "일간", "시간"),
    ]
    for c1, c2, p1_key, p2_key in adjacent_cheongan_pairs:
        if c1 and c2 and (c1, c2) in CHEONGAN_HAP_PAIRS:
            results.append(f"{CHEONGAN_HAP_PAIRS[(c1, c2)]} ({p1_key} {c1} - {p2_key} {c2})")
    
    # 중복 제거 후 반환
    return list(set(results))

def find_jiji_yukhap(saju_8_chars: dict) -> list:
    """사주 원국 내 지지육합(六合)을 찾습니다."""
    results = []
    # 확인할 인접한 지지 쌍 정의
    adjacent_jiji_pairs = [
        (saju_8_chars.get("연지"), saju_8_chars.get("월지"), "연-월"),
        (saju_8_chars.get("월지"), saju_8_chars.get("일지"), "월-일"),
        (saju_8_chars.get("일지"), saju_8_chars.get("시지"), "일-시"),
    ]
    for j1, j2, pos_key in adjacent_jiji_pairs:
        if j1 and j2 and (j1, j2) in JIJI_YUKHAP_PAIRS:
            results.append(f"{JIJI_YUKHAP_PAIRS[(j1, j2)]} ({pos_key})")
    return list(set(results))

def find_jiji_samhap_or_banghap(saju_8_chars: dict, hap_list_data: list, hap_type_name: str) -> list:
    """사주 원국 내 지지 삼합/방합 및 반합을 찾습니다."""
    results = []
    saju_jijis_with_pos = [
        (saju_8_chars.get("연지"), "연"), (saju_8_chars.get("월지"), "월"),
        (saju_8_chars.get("일지"), "일"), (saju_8_chars.get("시지"), "시"),
    ]
    saju_jijis_present = [(char, pos_key) for char, pos_key in saju_jijis_with_pos if char]

    if len(saju_jijis_present) < 2:
        return []

    for required_jijis, hap_name in hap_list_data:
        # 1. 완전한 합 (세 글자)
        found_full_hap = False
        if len(saju_jijis_present) >= 3:
            for group in itertools.combinations(saju_jijis_present, 3):
                group_chars = tuple(sorted(item[0] for item in group))
                if Counter(group_chars) == Counter(required_jijis):
                    display_elements = [f"{char}({pos_key})" for char, pos_key in group]
                    results.append(f"{hap_name} ({'-'.join(display_elements)})")
                    found_full_hap = True
                    break
        
        if found_full_hap:
            continue

        # 2. 반합 (두 글자) - 삼합에만 해당하며, 왕지를 포함해야 함
        if hap_type_name == "삼합" and len(saju_jijis_present) >= 2:
            saengji, wangji, goji = required_jijis
            valid_banhap_pairs = [tuple(sorted((saengji, wangji))), tuple(sorted((wangji, goji)))]
            
            for group in itertools.combinations(saju_jijis_present, 2):
                group_chars = tuple(sorted(item[0] for item in group))
                if group_chars in valid_banhap_pairs:
                    display_elements = [f"{char}({pos_key})" for char, pos_key in group]
                    results.append(f"{hap_name.replace('삼합','반합')} ({'-'.join(display_elements)})")

    return list(set(results))

def find_all_jiji_interactions(saju_8_chars: dict) -> list:
    """
    인접한 지지들 사이의 모든 상호작용(충, 육합, 반합)을 찾아 리스트로 반환합니다.
    """
    # 이 함수 내에서만 사용할 간단한 정의들
    POSITION_NAMES_SIMPLE = {'연지': '연', '월지': '월', '일지': '일', '시지': '시'}
    
    # 확인할 인접한 쌍 정의 (시-일, 일-월, 월-연)
    adjacent_pairs_to_check = [('시지', '일지'), ('일지', '월지'), ('월지', '연지')]
    
    found_interactions = []

    for pos1_key, pos2_key in adjacent_pairs_to_check:
        jiji1 = saju_8_chars.get(pos1_key)
        jiji2 = saju_8_chars.get(pos2_key)

        # 두 지지 중 하나라도 없거나, 같은 글자이면 건너뜁니다.
        if not jiji1 or not jiji2 or jiji1 == jiji2:
            continue

        # --- 1. 충(沖) 검사 ---
        if (jiji1, jiji2) in JIJI_CHUNG_PAIRS:
            pos_str = f"({POSITION_NAMES_SIMPLE.get(pos1_key)}-{POSITION_NAMES_SIMPLE.get(pos2_key)})"
            found_interactions.append(f"{JIJI_CHUNG_PAIRS[(jiji1, jiji2)]}{pos_str}")

        # --- 2. 육합(六合) 검사 ---
        if (jiji1, jiji2) in JIJI_YUKHAP_PAIRS:
            pos_str = f"({POSITION_NAMES_SIMPLE.get(pos1_key)}-{POSITION_NAMES_SIMPLE.get(pos2_key)})"
            found_interactions.append(f"{JIJI_YUKHAP_PAIRS[(jiji1, jiji2)]}{pos_str}")
        
        # --- 3. 반합(半合) 검사 (수정된 로직) ---
        for required_jijis, hap_name in JIJI_SAMHAP_LIST:
            # 1. 두 글자가 같은 삼합 그룹에 속하는지 확인
            if jiji1 in required_jijis and jiji2 in required_jijis:
                # 2. 해당 삼합의 왕지(子, 午, 卯, 酉)를 정의 (리스트의 항상 두 번째 글자)
                wangji = required_jijis[1]

                # 3. 새로운 규칙: jiji1이나 jiji2 중 하나가 반드시 왕지여야만 반합으로 인정
                if jiji1 == wangji or jiji2 == wangji:
                    pos_str = f"({POSITION_NAMES_SIMPLE.get(pos1_key)}-{POSITION_NAMES_SIMPLE.get(pos2_key)})"
                    # 예: "인오술 화국 삼합" -> "인오 화국 반합"
                    found_interactions.append(f"{jiji1}{jiji2} {hap_name.replace('삼합','반합')}{pos_str}")
                    break

    return sorted(list(set(found_interactions)))

# --- 신살 분석 함수들 ---

def get_dohwa_target_char(saju_8_chars: dict) -> str | None:
    """도화살(桃花煞)에 해당하는 지지를 찾습니다."""
    ilji = saju_8_chars.get("일지")
    if not ilji or ilji not in ILJI_TO_SAMHAP_GROUP: return None
    samhap_group = ILJI_TO_SAMHAP_GROUP.get(ilji)
    return SAMHAP_TO_DOHWA.get(samhap_group)

def get_hwagae_target_char(saju_8_chars: dict) -> str | None:
    """화개살(華蓋煞)에 해당하는 지지를 찾습니다."""
    ilji = saju_8_chars.get("일지")
    if not ilji or ilji not in ILJI_TO_SAMHAP_GROUP: return None
    samhap_group = ILJI_TO_SAMHAP_GROUP.get(ilji)
    return SAMHAP_TO_HWAGAE.get(samhap_group)

def get_hongyeom_target_char(saju_8_chars: dict) -> str | None:
    """홍염살(紅艶煞)에 해당하는 지지를 찾습니다."""
    ilgan = saju_8_chars.get("일간")
    if not ilgan or ilgan not in ILGAN_TO_HONGYEOM: return None
    return ILGAN_TO_HONGYEOM.get(ilgan)

def get_yeongma_target_char(saju_8_chars: dict) -> str | None:
    """역마살(驛馬煞)에 해당하는 지지를 찾습니다."""
    ilji = saju_8_chars.get("일지")
    if not ilji or ilji not in ILJI_TO_SAMHAP_GROUP: return None
    samhap_group = ILJI_TO_SAMHAP_GROUP.get(ilji)
    samhap_first_char = SAMHAP_GROUP_TO_FIRST_CHAR.get(samhap_group)
    if not samhap_first_char: return None
    return SAMHAP_FIRST_CHAR_TO_YEONGMA.get(samhap_first_char)

def get_all_sinsal_and_hapchung(saju_8_chars: dict) -> dict:
    """
    사주팔자를 입력받아, 모든 신살과 합충 관계를 계산하여 딕셔너리로 반환합니다.
    """
    analysis_results = {"신살": {}, "합충": {}}
    
    # 8글자가 모두 있는지 기본 검증
    if not isinstance(saju_8_chars, dict) or not all(saju_8_chars.get(key) for key in ["연간","연지","월간","월지","일간","일지","시간","시지"]):
        analysis_results["오류"] = "사주 8글자를 모두 올바르게 입력해야 합니다."
        return analysis_results

    # --- 신살 계산 ---
    sinsal_at_position = defaultdict(list)
    
    # 각 신살 헬퍼 함수를 호출하여 결과를 취합
    dohwa_target = get_dohwa_target_char(saju_8_chars)
    if dohwa_target:
        for pos_key in ["연지", "월지", "일지", "시지"]:
            if saju_8_chars.get(pos_key) == dohwa_target:
                sinsal_at_position[pos_key].append("도화")

    hwagae_target = get_hwagae_target_char(saju_8_chars)
    if hwagae_target:
        for pos_key in ["연지", "월지", "일지", "시지"]:
            if saju_8_chars.get(pos_key) == hwagae_target:
                sinsal_at_position[pos_key].append("화개")
    
    hongyeom_target = get_hongyeom_target_char(saju_8_chars)
    if hongyeom_target and saju_8_chars.get("일지") == hongyeom_target:
        sinsal_at_position["일지"].append("홍염")

    yeongma_target = get_yeongma_target_char(saju_8_chars)
    if yeongma_target:
        for pos_key in ["연지", "월지", "일지", "시지"]:
            if saju_8_chars.get(pos_key) == yeongma_target:
                sinsal_at_position[pos_key].append("역마")
    
    # 결과가 있는 위치만 필터링하여 최종 신살 결과 생성
    analysis_results["신살"] = {k: sorted(list(set(v))) for k, v in sinsal_at_position.items() if v}

    # --- 합충 계산 ---
    analysis_results["합충"] = {
        "천간합": find_cheongan_hap(saju_8_chars),
        "지지관계": find_all_jiji_interactions(saju_8_chars) # 통합된 지지 상호작용 함수 호출
    }
    
    return analysis_results

def get_sipseong(day_master_char, target_jiji_char):
    """일간을 기준으로 특정 지지의 십성을 구합니다."""
    if not day_master_char or not target_jiji_char: return ""
    if day_master_char not in CHEONGAN or target_jiji_char not in JIJI: return ""

    dm_element = get_saju_element(day_master_char)
    dm_yinyang = CHEONGAN_YIN_YANG.get(day_master_char)
    target_element = JIJI_ELEMENTS_PRIMARY.get(target_jiji_char)
    # 지지의 음양은 기능적 음양(절기에 따른 음양)을 사용합니다.
    target_yinyang = JIJI_YIN_YANG_FUNCTIONAL.get(target_jiji_char)

    if not all([dm_element, target_element, dm_yinyang, target_yinyang]): return ""

    if dm_element == target_element:
        return "비견" if dm_yinyang == target_yinyang else "겁재"
    elif SHENG_RELATIONS.get(dm_element) == target_element: # 일간이 생함
        return "식신" if dm_yinyang == target_yinyang else "상관"
    elif saju_sheng_in.get(dm_element) == target_element: # 일간을 생함
        return "편인" if dm_yinyang == target_yinyang else "정인"
    elif KE_RELATIONS.get(dm_element) == target_element: # 일간이 극함
        return "편재" if dm_yinyang == target_yinyang else "정재"
    elif saju_ke_in.get(dm_element) == target_element: # 일간을 극함
        return "편관" if dm_yinyang == target_yinyang else "정관"
    return ""

def get_sipseong_cheongan(base_cheongan_char, target_cheongan_char):
    """기준 천간을 바탕으로 다른 천간의 십성을 구합니다."""
    if not base_cheongan_char or not target_cheongan_char: return ""
    if base_cheongan_char not in CHEONGAN or target_cheongan_char not in CHEONGAN: return ""

    base_element = get_saju_element(base_cheongan_char)
    base_yinyang = CHEONGAN_YIN_YANG.get(base_cheongan_char)
    target_element = get_saju_element(target_cheongan_char)
    target_yinyang = CHEONGAN_YIN_YANG.get(target_cheongan_char)

    if not all([base_element, target_element, base_yinyang, target_yinyang]): return ""

    if base_element == target_element:
        return "비견" if base_yinyang == target_yinyang else "겁재"
    elif SHENG_RELATIONS.get(base_element) == target_element:
        return "식신" if base_yinyang == target_yinyang else "상관"
    elif saju_sheng_in.get(base_element) == target_element:
        return "편인" if base_yinyang == target_yinyang else "정인"
    elif KE_RELATIONS.get(base_element) == target_element:
        return "편재" if base_yinyang == target_yinyang else "정재"
    elif saju_ke_in.get(base_element) == target_element:
        return "편관" if base_yinyang == target_yinyang else "정관"
    return ""

def get_dynamic_monthly_ranking(saju_8_chars_dict, keyword_char_input, pne1_element_input, pne2_element_input):
    """필요오행에 따른 월운의 순위를 반환합니다."""
    # '1순위 필요오행'별 월별 행운 순위 (선생님의 핵심 데이터)
    fixed_monthly_rankings = {
        "목": ["묘(3월)", "인(2월)", "진(4월)", "해(11월)", "미(7월)", "사(5월)", "오(6월)", "술(10월)", "축(1월)", "자(12월)", "유(9월)", "신(8월)"],
        "화": ["오(6월)", "사(5월)", "미(7월)", "술(10월)","인(2월)", "묘(3월)", "신(8월)", "유(9월)", "해(11월)", "자(12월)", "진(4월)", "축(1월)"],
        "토": ["진(4월)", "미(7월)", "술(10월)", "축(1월)", "사(5월)", "오(6월)", "신(8월)", "유(9월)", "해(11월)", "자(12월)", "인(2월)", "묘(3월)"],
        "금": ["신(8월)", "유(9월)", "오(6월)", "사(5월)", "축(1월)", "해(11월)", "자(12월)", "술(10월)", "묘(3월)", "인(2월)", "진(4월)", "미(7월)"],
        "수": ["해(11월)", "자(12월)", "신(8월)", "유(9월)", "인(2월)", "묘(3월)", "축(1월)", "진(4월)", "사(5월)", "오(6월)", "술(10월)", "미(7월)"]
    }
    if pne1_element_input in fixed_monthly_rankings:
        return fixed_monthly_rankings[pne1_element_input]
    else:
        # 1순위 필요오행이 유효하지 않은 경우, 기본 지지 순서대로 반환
        jiji_to_month_map = {"자":"12월", "축":"1월", "인":"2월", "묘":"3월", "진":"4월", "사":"5월", "오":"6월", "미":"7월", "신":"8월", "유":"9월", "술":"10월", "해":"11월"}
        return [f"{jiji_char}({jiji_to_month_map.get(jiji_char, '')})" for jiji_char in JIJI]
    
def adjust_needed_elements_for_haps(saju_8_chars, keyword_char, original_pne_list, daewoon_ji, yeonwoon_ji):
    """
    대운/세운의 지지에 의해 합이 완성될 경우, 필요오행을 동적으로 조정합니다.
    조정이 발생했는지 여부도 함께 반환합니다.
    """
    # 원본 1, 2, 3순위 필요오행 추출
    original_pne1 = original_pne_list[0][0] if len(original_pne_list) >= 1 else None
    original_pne2 = original_pne_list[1][0] if len(original_pne_list) >= 2 else None
    original_pne3 = original_pne_list[2][0] if len(original_pne_list) >= 3 else None
    
    was_adjusted = False # 조정 여부 플래그

    # 이 로직이 적용되는 특정 키워드 글자들이 아니면 바로 원본값 반환
    required_keyword_chars = ["인", "신", "사", "해", "진", "술", "축", "미"]
    if keyword_char not in required_keyword_chars:
        return original_pne1, original_pne2, was_adjusted

    # 합을 완성시키는 특정 '운의 글자'가 오지 않으면 바로 원본값 반환
    trigger_luck_chars = ["자", "오", "묘", "유"]
    if daewoon_ji not in trigger_luck_chars and yeonwoon_ji not in trigger_luck_chars:
        return original_pne1, original_pne2, was_adjusted

    # 합을 구성하는 조건 정의
    HAP_COMBOS = [
        ((("인", "술"), "오"), ["인", "술"]), ((("인", "진"), "묘"), ["인", "진"]),
        ((("사", "축"), "유"), ["사", "축"]), ((("사", "미"), "오"), ["사", "미"]),
        ((("신", "진"), "자"), ["신", "진"]), ((("신", "술"), "유"), ["신", "술"]),
        ((("해", "미"), "묘"), ["해", "미"]), ((("해", "축"), "자"), ["해", "축"]),
    ]

    # 원국의 인접한 지지 쌍
    adjacent_pairs = [
        (saju_8_chars.get("연지"), saju_8_chars.get("월지")),
        (saju_8_chars.get("월지"), saju_8_chars.get("일지")),
        (saju_8_chars.get("일지"), saju_8_chars.get("시지")),
    ]

    for (natal_pair, luck_char), hap_members in HAP_COMBOS:
        # 운에서 합을 완성하는 글자가 왔는지 확인
        if daewoon_ji == luck_char or yeonwoon_ji == luck_char:
            # 원국에 합을 구성하는 나머지 두 글자가 인접해 있는지 확인
            for pair in adjacent_pairs:
                if Counter(pair) == Counter(natal_pair):
                    # 키워드가 합을 구성하는 글자 중 하나라면, 필요오행을 조정
                    if keyword_char in hap_members:
                        new_pne1 = original_pne2       # 2순위를 새로운 1순위로
                        new_pne2 = original_pne3       # 3순위를 새로운 2순위로
                        was_adjusted = True            # 조정 발생
                        return new_pne1, new_pne2, was_adjusted

    # 어떤 조건에도 해당하지 않으면 원본 필요오행을 그대로 반환
    return original_pne1, original_pne2, was_adjusted

def get_city_info(city_name: str) -> tuple:
    """
    도시 이름을 입력받아 시간대 이름과 경도를 반환합니다.
    성공 시: (시간대 이름, 경도) 튜플, 실패 시: (None, None)
    """
    #try:
        # Nominatim은 고유한 user_agent를 요구합니다.
        #geolocator = Nominatim(user_agent="saju_api_v1.0")
        #location = geolocator.geocode(city_name, language='en')

        #if location:
            #tf = TimezoneFinder()
            # 위도와 경도를 사용해 시간대 이름을 찾음
            #tz_name = tf.timezone_at(lng=location.longitude, lat=location.latitude)
            #return (tz_name, location.longitude)
        #else:
            # 도시를 찾지 못한 경우
            #return (None, None)
            
    #except Exception as e:
        # 서버에서는 print 대신 로깅(logging)을 사용하는 것이 더 좋습니다.
        #print(f"도시 정보 조회 중 오류 발생: {e}")
        #return (None, None)

def run_saju_engine(cal_type, date_str, time_str, gender_input, is_leap_input, is_time_unknown, is_overseas, city_name=""):
    """
    모든 사주 분석 계산을 수행하고, 가공되지 않은 순수 결과 데이터 묶음을 반환하는 단일 엔진.
    """
    try:
    # 1. 입력값 검증, 날짜/시간 변환
        if not(len(date_str) == 8 and date_str.isdigit()): raise ValueError("YYYYMMDD 형식")
        year_val_initial_input = int(date_str[0:4]); kst_original_month_input = int(date_str[4:6]); kst_original_day_input = int(date_str[6:8])
        if not(1 <= kst_original_month_input <= 12): raise ValueError("월(01-12)")
        if not(1 <= kst_original_day_input <= 31): raise ValueError("일(01-31)")
        hour_input, minute_input = (12, 30) if is_time_unknown else (int(time_str[0:2]), int(time_str[2:4]))
        if not is_time_unknown:
            if not(0 <= hour_input <= 23): raise ValueError("시(00-23)")
            if not(0 <= minute_input <= 59): raise ValueError("분(00-59)")
        if cal_type == "음":
            solar_date_obj = convert_lunar_to_solar(year_val_initial_input, kst_original_month_input, kst_original_day_input, is_leap_input)
            if solar_date_obj is None: raise ValueError("음력->양력 변환 실패. 유효하지 않은 날짜입니다.")
            calc_target_solar_year, calc_target_solar_month, calc_target_solar_day = solar_date_obj.year, solar_date_obj.month, solar_date_obj.day
        else:
            calc_target_solar_year, calc_target_solar_month, calc_target_solar_day = year_val_initial_input, kst_original_month_input, kst_original_day_input
        if is_overseas:
            if not city_name: raise ValueError("해외 출생 선택 시, 도시 이름은 필수입니다.")
            tz_name, _ = get_city_info(city_name)
            if not tz_name: raise ValueError(f"'{city_name}' 도시 정보를 찾을 수 없습니다.")
            local_tz = pytz.timezone(tz_name); naive_dt = datetime.datetime(calc_target_solar_year, calc_target_solar_month, calc_target_solar_day, hour_input, minute_input)
            local_dt = local_tz.localize(naive_dt, is_dst=None); kst_tz = pytz.timezone('Asia/Seoul')
            datetime_kst = local_dt.astimezone(kst_tz); datetime_lmt = to_LMT(datetime_kst).replace(tzinfo=None)
        else:
            datetime_kst = datetime.datetime(calc_target_solar_year, calc_target_solar_month, calc_target_solar_day, hour_input, minute_input)
            datetime_lmt = to_LMT(datetime_kst).replace(tzinfo=None)

    # 2. 사주팔자, 대운, 필요오행 등 분석 재료 계산
        year_gan_char, year_ji_char, year_gan_idx, astro_year = get_year_pillar(datetime_lmt)
        month_gan_char, month_ji_char, month_gan_idx, current_month_period_idx = get_month_pillar(datetime_lmt, year_gan_idx, astro_year)
        day_gan_char, day_ji_char, day_gan_idx = get_day_pillar(datetime_lmt)
        hour_gan_char, hour_ji_char = get_hour_pillar(datetime_lmt, day_gan_idx)
        saju_8_chars_calculated = {"연간": year_gan_char, "연지": year_ji_char, "월간": month_gan_char, "월지": month_ji_char, "일간": day_gan_char, "일지": day_ji_char, "시간": hour_gan_char, "시지": hour_ji_char}
        daewoon_direction_str = get_daewoon_direction(year_gan_char, gender_input)
        daewoon_su_val = get_daewoon_su(datetime_lmt, daewoon_direction_str, astro_year, current_month_period_idx)
        daewoon_start_year_val = (calc_target_solar_year + daewoon_su_val) - 1
        needed_element_scores, _ = calculate_needed_element_scores(saju_8_chars_calculated)
        sorted_elements = sorted(needed_element_scores.items(), key=lambda item: item[1], reverse=True)
        pne1_element = sorted_elements[0][0] if len(sorted_elements) > 0 else None
        pne2_element = sorted_elements[1][0] if len(sorted_elements) > 1 else None
        keyword_char, keyword_pos_label = derive_keyword(saju_8_chars_dict=saju_8_chars_calculated, pne1_element=pne1_element, pne2_element=pne2_element)
        final_keyword_element = get_saju_element(keyword_char)
        auto_luck_amount = calculate_luck_quantity_auto(saju_8_chars_calculated, final_keyword_element, keyword_pos_label)
        daewoon_list_val = get_yearly_daewoon_list(calc_target_solar_year, month_gan_char, month_ji_char, daewoon_su_val, (daewoon_direction_str=="순행"))
    
    # 3. 100년치 행운강도 계산
        yearly_luck_results, _, _ = calculate_yearly_luck_final(
            birth_year_solar=calc_target_solar_year, saju_8_chars=saju_8_chars_calculated,
            keyword_char=keyword_char, sorted_needed_elements=sorted_elements,
            auto_luck_amount=auto_luck_amount, daewoon_list_val=daewoon_list_val
        )

    # 4. 다른 정적 분석 결과들도 계산
        saju_8_chars_hangeul = saju_8_chars_calculated.copy()
        sinsal_results = get_all_sinsal_and_hapchung(saju_8_chars_hangeul)
        hjs_totals = calculate_saju_hjs_total_scores(jiji_siju=saju_8_chars_calculated.get("시지"), jiji_ilju=saju_8_chars_calculated.get("일지"), jiji_wolju=saju_8_chars_calculated.get("월지"), jiji_yeonju=saju_8_chars_calculated.get("연지"), hjs_scores_dict=HJS_SCORES_GLOBAL)
        day_master_char = saju_8_chars_calculated.get("일간")
        keyword_sipsin = ""
        if day_master_char and keyword_char and keyword_pos_label:
            if '간' in keyword_pos_label: keyword_sipsin = get_sipseong_cheongan(day_master_char, keyword_char)
            else: keyword_sipsin = get_sipseong(day_master_char, keyword_char)
        formatted_keyword_string = f"{keyword_char}({keyword_sipsin})" if keyword_sipsin else keyword_char
    
    # 5. 모든 계산 결과를 하나의 딕셔너리로 묶어 반환
        return {
            "raw_inputs": {"gender_input": gender_input, "cal_type": cal_type, "date_str": date_str, "time_str": time_str, "is_leap_input": is_leap_input, "is_time_unknown": is_time_unknown, "is_overseas": is_overseas, "city_name": city_name, "year_val_initial_input": year_val_initial_input, "kst_original_month_input": kst_original_month_input, "kst_original_day_input": kst_original_day_input, "hour_input": hour_input, "minute_input": minute_input},
            "saju_basics": {"pillars": saju_8_chars_calculated, "daewoon_direction": daewoon_direction_str, "daewoon_su": daewoon_su_val, "start_year_ad": daewoon_start_year_val},
            "core_analysis_results": {"pne1": pne1_element, "pne2": pne2_element, "keyword": formatted_keyword_string, "keyword_pos": keyword_pos_label},
            "yearly_luck_raw_data": yearly_luck_results,
            "sinsal_results": sinsal_results,
            "hjs_totals": hjs_totals,
            "calc_target_solar_year": calc_target_solar_year
        }

    except Exception as e:
        return {"error": "엔진 실행 중 에러", "details": str(e), "traceback": traceback.format_exc()}

def get_saju_analysis_for_api(cal_type, date_str, time_str, gender_input, is_leap_input, is_time_unknown, is_overseas, city_name=""):
    """
    [최종본] 기본 사주 및 3년치 운세 등 모든 초기 분석 정보를 반환합니다.
    """
    try:
        # 1. 마스터 엔진을 호출하여 모든 기본 계산을 한 번에 수행합니다.
        engine_results = run_saju_engine(
            cal_type, date_str, time_str, gender_input, 
            is_leap_input, is_time_unknown, is_overseas, city_name
        )
        
        if "error" in engine_results:
            return engine_results

        # 2. 엔진 결과물에서 필요한 데이터들을 안전하게 꺼냅니다.
        raw_inputs = engine_results.get("raw_inputs", {})
        saju_basics = engine_results.get("saju_basics", {})
        core_analysis = engine_results.get("core_analysis_results", {})
        sinsal_results_raw = engine_results.get("sinsal_results", {})
        hjs_totals = engine_results.get("hjs_totals", {})
        yearly_luck_raw_data = engine_results.get("yearly_luck_raw_data", [])
        saju_8_chars_calculated = saju_basics.get("pillars", {})
        current_year = datetime.datetime.now().year # 현재 연도 (예: 2025)

        # 3. 3년치 한난조습 동적 변화를 계산합니다.
        yearly_hjs_scores = []
        current_year = datetime.datetime.now().year
        years_to_check = [
            {'label': '작년', 'year': current_year - 1},
            {'label': '올해', 'year': current_year},
            {'label': '내년', 'year': current_year + 1}
        ]
        for year_info in years_to_check:
            ganji_str = get_ganji_for_year(year_info['year'])
            yeonun_jiji = ganji_str[1]
            dynamic_scores = calculate_saju_hjs_total_scores(
                jiji_siju=saju_8_chars_calculated.get("시지"), jiji_ilju=saju_8_chars_calculated.get("일지"),
                jiji_wolju=saju_8_chars_calculated.get("월지"), jiji_yeonju=saju_8_chars_calculated.get("연지"),
                hjs_scores_dict=HJS_SCORES_GLOBAL, yeonun_jiji=yeonun_jiji
            )
            yearly_hjs_scores.append({"label": year_info['label'], "year": year_info['year'], "ganji": ganji_str, "scores": dynamic_scores})

        # 4. 평생 운세(Momentum) 데이터 계산 (100년치로 수정)
        lifetime_luck_data = []
        if yearly_luck_raw_data:
            df = pd.DataFrame(yearly_luck_raw_data)
            df['행운강도'] = pd.to_numeric(df['행운강도'], errors='coerce')
            df['momentum'] = df['행운강도'].rolling(window=3, center=True).mean()
        
            start_year = raw_inputs.get("year_val_initial_input", 1900)
            end_year = start_year + 100
        
            df_filtered = df[(df['연도'] >= start_year) & (df['연도'] <= end_year)].copy()

        for _, row in df_filtered.iterrows():
            momentum = round(row['momentum'], 2) if pd.notna(row['momentum']) else None
            if momentum is not None and momentum == -0.0:
                momentum = 0.0
            
            lifetime_luck_data.append({
                "year": int(row['연도']),
                "age": int(row['나이']),
                "daewoon": f"{row.get('대운천간', '')}{row.get('대운지지', '')}",
                "yeonun": f"{row.get('연운천간', '')}{row.get('연운지지', '')}",
                "luck_value": round(row['행운강도'], 2),
                "luck_momentum": momentum
            })

        # 5. 최종 결과물을 보기 좋은 구조로 조립합니다.
        sinsal_summary_parts = []
        if sinsal_results_raw.get("신살"):
            for pos, a_list in sinsal_results_raw.get("신살", {}).items():
                sinsal_summary_parts.append(f"{pos}({', '.join(a_list)})")
        hapchung_data = sinsal_results_raw.get("합충", {})
        cheongan_hap_summary = ", ".join(hapchung_data.get("천간합", []))
        jiji_gwangye_summary = ", ".join(hapchung_data.get("지지관계", []))
        summary_list = []
        if sinsal_summary_parts: summary_list.append(f"[신살] : {', '.join(sinsal_summary_parts)}")
        if cheongan_hap_summary: summary_list.append(f"[천간합] : {cheongan_hap_summary}")
        if jiji_gwangye_summary: summary_list.append(f"[지지관계] : {jiji_gwangye_summary}")
        interaction_summary_string = ", ".join(summary_list)
        
        hour = raw_inputs.get('hour_input', 0); minute = raw_inputs.get('minute_input', 0)
        year_val = raw_inputs.get('year_val_initial_input', '??'); month_val = raw_inputs.get('kst_original_month_input', '??'); day_val = raw_inputs.get('kst_original_day_input', '??')
        user_birth_time_str = f"{hour:02d}시 {minute:02d}분" if not raw_inputs.get("is_time_unknown") else "시간 모름"
        user_birth_date_str = f"{year_val}년 {month_val}월 {day_val}일"
        yeonju_str = f"{saju_8_chars_calculated.get('연간','')}{saju_8_chars_calculated.get('연지','')}년"
        wolju_str = f"{saju_8_chars_calculated.get('월간','')}{saju_8_chars_calculated.get('월지','')}월"
        ilju_str = f"{saju_8_chars_calculated.get('일간','')}{saju_8_chars_calculated.get('일지','')}일"
        siju_str = f"{saju_8_chars_calculated.get('시간','')}{saju_8_chars_calculated.get('시지','')}시"

        final_result = {
            "userInput": {"gender": raw_inputs.get("gender_input"), "calendar": "음력" if raw_inputs.get("cal_type") == "음" else "양력", "birthDateTime": f"{user_birth_date_str} {user_birth_time_str}", "isLeapMonth": raw_inputs.get("is_leap_input")},
            "sajuInfo": { "pillars": {"summary": f"{yeonju_str} {wolju_str} {ilju_str} {siju_str}"}, "daewoon": {"direction": saju_basics.get("daewoon_direction"),"start_age_korean": saju_basics.get("daewoon_su"),"start_year_ad": saju_basics.get("start_year_ad")}},
            "analysis": {
                "core": core_analysis,
                "interaction_summary": interaction_summary_string,
                "balance": {"base_hjs": hjs_totals, "hjs_trend": yearly_hjs_scores},
                "lifetime_luck_trend": lifetime_luck_data
            }
        }
        return final_result

    except Exception as e:
        return {"error": "API 처리 중 에러", "details": str(e), "traceback": traceback.format_exc()}

# Code End.