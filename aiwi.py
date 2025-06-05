import os
import re
import requests
from dotenv import load_dotenv
from typing import TypedDict, List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "")

# 🔗 LLM 초기화
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# 🟠 1단계: 상품 리스트 생성 (노드 1)
def generate_checklist(state):
    user_input = state["question"]
    prompt = f"""
    다음은 사용자의 상황입니다: '{user_input}'

    사용자의 말에 자연스럽게 반응해주고, 어떤 물건을 추천하는지 간단히 설명해줘.
    그 뒤에 [추천 리스트: 항목1, 항목2, 항목3, ...] 형식으로, **물건 이름만 간단하게 나열**하고 **한두 단어짜리 명사 형태로만** 항목을 써줘.

    그리고 각 항목에 대해:
    - 아래 목록 중 가장 적절한 **카테고리**를 붙여줘.

    카테고리 목록:
    여성의류, 남성의류, 패션잡화, 신발, 화장품/미용, 신선식품, 가공식품, 건강식품,
    출산/유아동, 반려동물용품, 가전, 휴대폰/카메라, PC/주변기기, 가구,
    조명/인테리어, 패브릭/홈데코, 주방용품, 생활용품, 스포츠/레저, 자동차/오토바이,
    키덜트/취미, 건강의료용품, 악기/문구, 공구, 렌탈관, e쿠폰/티켓/생활편의, 여행


    형식 예시:
    - 사용자가 '내일 캠핑갈건데 뭐가 필요할까?'라고 하면 →  
    캠핑 가신다니 설레네요! 텐트, 침낭, 매트는 꼭 챙기시면 좋아요.  
    [추천 리스트: 텐트 | 스포츠/레저, 랜턴 | 생활용품, 매트 | 가구]

    - 사용자가 '물놀이 가게 됐어'라고 하면 →  
    재밌겠네요! 수영복, 물안경, 방수팩은 필수예요.  
    [추천 리스트: 수영복 | 스포츠/레저, 물안경 | 스포츠/레저, 방수팩 | 생활용품]

    지금 사용자 입력: '{user_input}'
    """

    full_response = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    # 리스트 추출
    match = re.search(r"\[추천 리스트: (.*?)\]", full_response, re.DOTALL)
    checklist_raw = match.group(1) if match else ""
    
    checklist_message = full_response.split("[추천 리스트:")[0].strip()

    # 항목별 파싱
    checklist = []
    keywords = []
    for part in checklist_raw.split(","):
        if "|" in part:
            name, category = part.split("|", 1)
            name = name.strip()
            category = category.strip()
            checklist.append(name)
            keywords.append({"keyword": name, "category": category})

    print("\n📝 [ChecklistGenerator]")
    print("📥 입력 질문:", user_input)
    print("📤 LLM 반응:", checklist_message)
    print("📤 추출된 리스트:", checklist)
    print("📤 추출된 카테고리:", keywords)

    return {
        "question": user_input,
        "checklist": ", ".join(checklist),
        "checklist_message": checklist_message,
        "keywords": keywords
    }


# 🟠 2단계: 네이버 쇼핑 API 검색
def search_naver_items(state):
    results = []
    for keyword_obj  in state["keywords"]:
        keyword_text = keyword_obj["keyword"]  # ✅ keyword만 추출

        url = "https://openapi.naver.com/v1/search/shop.json"
        headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
        }
        params = {"query": keyword_text, "display": 3, "start": 1, "sort": "sim"}
        response = requests.get(url, headers=headers, params=params)
        items = response.json().get("items", []) if response.status_code == 200 else []

        print(f"\n🔍 [NaverSearcher] 키워드: '{keyword_text}'")
        print(f"🔎 검색된 상품 수: {len(items)}")
        print(f"🔎 상품 JSON 예시:")
        for item in items:
            print(item)

        results.append({"keyword": keyword_text, "items": items})

    return {"keywords": state["keywords"], "search_results": results}

# 🟠 4단계: 상품 비교 및 추천 물품별 {title, 가격}
def compare_and_recommend(state):
    CATEGORY_PROMPT_MAP = {
        "여성의류": "디자인, 계절 적합성, 소재, 사이즈 다양성, 가격 등을 기준으로 비교해줘.",
        "남성의류": "스타일, 활용도, 소재 품질, 브랜드, 가격 등을 기준으로 비교해줘.",
        "패션잡화": "디자인, 실용성, 브랜드, 재질, 가격 등을 기준으로 비교해줘.",
        "신발": "착화감, 용도, 재질, 내구성, 디자인, 가격 등을 기준으로 비교해줘.",
        "화장품/미용": "성분, 피부타입 적합성, 사용감, 브랜드, 가격 등을 기준으로 비교해줘.",
        "신선식품": "신선도, 영양 성분, 수확 시기, 원산지, 유통기한 등을 기준으로 비교해줘.",
        "가공식품": "맛, 유통기한, 성분, 조리 편의성, 가격 등을 기준으로 비교해줘.",
        "건강식품": "주요 영양소, 건강에 미치는 효과, 복용 편의성, 인증 여부, 가격 등을 비교해줘.",
        "출산/유아동": "안전성, 피부 친화성, 연령 적합성, 기능성, 가격 등을 기준으로 비교해줘.",
        "반려동물용품": "안전성, 반려동물 선호도, 기능, 재질, 가격 등을 기준으로 비교해줘.",
        "가전": "성능, 에너지 효율, 브랜드 신뢰도, 편의 기능, 가격 등을 기준으로 비교해줘.",
        "휴대폰/카메라": "성능, 브랜드, 배터리 수명, 기능, 가격 등을 기준으로 비교해줘.",
        "PC/주변기기": "성능, 호환성, 브랜드, 기능성, 가격 등을 기준으로 비교해줘.",
        "가구": "디자인, 내구성, 크기, 수납 기능, 가격 등을 기준으로 비교해줘.",
        "조명/인테리어": "밝기, 디자인, 에너지 효율, 설치 편의성, 가격 등을 기준으로 비교해줘.",
        "패브릭/홈데코": "재질, 세탁 용이성, 디자인, 계절성, 가격 등을 기준으로 비교해줘.",
        "주방용품": "재질, 내구성, 세척 용이성, 기능성, 가격 등을 기준으로 비교해줘.",
        "생활용품": "실용성, 내구성, 사용 편의성, 디자인, 가격 등을 기준으로 비교해줘.",
        "스포츠/레저": "내구성, 사용 목적 적합성, 기능성, 휴대성, 가격 등을 기준으로 비교해줘.",
        "자동차/오토바이": "성능, 브랜드, 연비, 안전 기능, 유지비용 등을 기준으로 비교해줘.",
        "키덜트/취미": "희소성, 디자인, 만족도, 수집 가치, 가격 등을 기준으로 비교해줘.",
        "건강의료용품": "정확성, 안전성, 사용 편의성, 인증 여부, 가격 등을 기준으로 비교해줘.",
        "악기/문구": "사용감, 내구성, 기능성, 브랜드, 가격 등을 기준으로 비교해줘.",
        "공구": "내구성, 사용 용이성, 기능, 브랜드, 가격 등을 기준으로 비교해줘.",
        "렌탈관": "렌탈 기간, 비용, 유지보수 조건, 최신 모델 여부, 브랜드 신뢰도 등을 기준으로 비교해줘.",
        "e쿠폰/티켓/생활편의": "사용처, 유효기간, 할인율, 사용 조건, 가격 등을 기준으로 비교해줘.",
        "여행": "여행지 매력도, 일정 구성, 가격, 포함 혜택, 후기 평점 등을 기준으로 비교해줘.",
    }   

    recommendations = []
    for idx, group in enumerate(state["search_results"]):
        keyword_info = state["keywords"][idx]
        keyword = keyword_info["keyword"]
        category = keyword_info["category"]
        items = group["items"]

        print(f"\n🤖 [Recommender] 키워드: '{keyword}' | 카테고리: {category}")
        print(f"🛒 상품 수: {len(items)}")

        if len(items) < 3:
            summary = f"'{keyword}'에 대한 상품이 부족하여 비교할 수 없습니다."
        else:
            compare_criteria = CATEGORY_PROMPT_MAP.get(
                category, "가격, 성능, 사용자 리뷰, 가성비 등을 기준으로 비교해줘."
            )
            summary_prompt = f"""
            다음은 네이버 쇼핑에서 '{keyword}' 키워드로 검색한 상품 3개야:

            1. {items[0]['title']} - {items[0]['lprice']}원  
            2. {items[1]['title']} - {items[1]['lprice']}원  
            3. {items[2]['title']} - {items[2]['lprice']}원  

            {compare_criteria}
            각 제품을 기준에 따라 비교하고, 가장 추천할 제품 1개를 골라 이유와 함께 말해줘.

            🔍 출력 예시 (키워드: '대형 스테인리스 얼음컵', 기준: 실용성, 내구성, 사용 편의성, 디자인, 가격 등):

            제품 비교:  
            1번은 플라스틱 소재로 가볍지만 내구성이 떨어지고,  
            2번은 스테인리스 재질로 보온 유지가 뛰어나며 세척도 쉬워요.  
            3번은 유리 제품으로 고급스럽지만 무겁고 깨지기 쉬워요.

            추천:  
            2번 제품은 내구성, 기능성, 위생 면에서 가장 균형이 좋고, 가격도 합리적입니다. 따라서 추천드립니다.
            """
            
            summary = llm.invoke([HumanMessage(content=summary_prompt)]).content.strip()

        recommendations.append({
            "keyword": keyword,
            "category": category,
            "summary": summary
        })

    return {"search_results": state["search_results"], "recommendations": recommendations}

# 💡 상태 스키마 정의
class AppState(TypedDict):
    question: str
    checklist_message: str  # 사용자 응답용 자연어 문장
    checklist: str
    keywords: List[Dict[str, str]]  # 각 dict는 {"keyword": SEO 키워드, "category": 카테고리}
    search_results: List[Dict]
    recommendations: List[Dict]

# 🧠 LangGraph 구성
builder = StateGraph(state_schema=AppState)
builder.add_node("ChecklistGenerator", RunnableLambda(generate_checklist))
builder.add_node("NaverSearcher", RunnableLambda(search_naver_items))
builder.add_node("Recommender", RunnableLambda(compare_and_recommend))

builder.set_entry_point("ChecklistGenerator")
builder.add_edge("ChecklistGenerator", "NaverSearcher")
builder.add_edge("NaverSearcher", "Recommender")
builder.add_edge("Recommender", END)

app = builder.compile()

# 🚀 실행
# output = app.invoke({"question": "나는 내일 캠핑갈건데 뭐가 필요할까?"})
output = app.invoke({"question": "아무 상품이나 추천해줘."})

# ✅ 최종 결과 출력
print("\n🎯 최종 추천 요약 결과")
for rec in output["recommendations"]:
    print(f"\n📌 [{rec['keyword']}] 추천 요약:\n{rec['summary']}")