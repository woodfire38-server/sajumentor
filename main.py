import json
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

# 이제 저희가 만든 단 하나의 통합 함수만 가져옵니다.
from sajumentor import get_saju_analysis_for_api

app = FastAPI()

# --- 이 API가 모든 데이터를 반환하도록 합니다. ---
@app.get("/analysis")
def analysis(birth: str, gender: str, cal_type: str = '양', time: str = '1230', is_leap: bool = False, is_time_unknown: bool = False, is_overseas: bool = False, city: str = ""):
    # 저희가 새로 만든 통합 함수를 호출합니다.
    result = get_saju_analysis_for_api(
        cal_type=cal_type, date_str=birth, time_str=time, gender_input=gender,
        is_leap_input=is_leap, is_time_unknown=is_time_unknown,
        is_overseas=is_overseas, city_name=city
    )
    
    json_string = json.dumps(result, ensure_ascii=False, indent=4)
    status_code = 400 if "error" in result else 200
    
    return Response(content=json_string, status_code=status_code, media_type="application/json; charset=utf-8")


# --- 더 이상 필요 없는 /lifetime-luck API는 삭제되었습니다. ---


# --- 루트 경로는 그대로 둡니다. ---
@app.get("/")
def read_root():
    return {"message": "Saju Analysis API is running."}