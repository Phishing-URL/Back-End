from flask import Flask, request, jsonify
from flasgger import Swagger
import joblib
import re
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse

app = Flask(__name__)
swagger = Swagger(app)

# XGBoost 모델 로드 (joblib 사용)
try:
    model = joblib.load('xgb_model2.pkl')
except Exception as e:
    raise RuntimeError(f"모델 로드 중 오류 발생: {e}")

# 특성 추출 함수
def extract_features(url):
    # 'http'나 'https'로 시작하지 않으면 'https://'를 자동으로 붙임
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    features = []
    
    try:
        # URL에 대한 요청을 보내 페이지 내용을 가져옴
        response = requests.get(url)
        page_content = response.content

        # BeautifulSoup을 이용해 HTML 파싱
        soup = BeautifulSoup(page_content, 'lxml')

        # LineOfCode: 페이지 소스 코드에서 라인의 수
        lines = page_content.splitlines()
        line_of_code = len(lines)
        features.append(line_of_code)

        # LargestLineLength: 가장 긴 줄의 길이
        largest_line_length = max(len(line) for line in lines) if lines else 0
        features.append(largest_line_length)

    except requests.exceptions.RequestException as e:
        print(f"URL 요청 오류: {str(e)}")
        features.extend([0, 0])  # 요청 실패 시 기본값

    # URLLength: URL의 전체 길이
    url_length = len(url)
    features.append(url_length)
    
    # NoOfImage: 페이지 내 <img> 태그의 개수
    no_of_image = len(soup.find_all('img')) if 'soup' in locals() else 0
    features.append(no_of_image)
    
    # NoOfExternalRef: 외부 링크 (<a href="...">) 중 외부 도메인의 개수
    parsed_url = urlparse(url)
    external_refs = [a['href'] for a in soup.find_all('a', href=True) if urlparse(a['href']).netloc != parsed_url.netloc] if 'soup' in locals() else []
    no_of_external_ref = len(external_refs)
    features.append(no_of_external_ref)
    
    # SpacialCharRationURL: URL에서 특수 문자의 비율
    special_chars = re.findall(r'[^a-zA-Z0-9]', url)
    special_char_ratio = len(special_chars) / url_length if url_length > 0 else 0
    features.append(special_char_ratio)
    
    # LetterRatioInURL: URL 내 알파벳의 비율
    letters = re.findall(r'[a-zA-Z]', url)
    letter_ratio = len(letters) / url_length if url_length > 0 else 0
    features.append(letter_ratio)
    
    # DomainLength: 도메인의 길이 추출
    domain_length = len(parsed_url.netloc)
    features.append(domain_length)

    # 특성 값 출력 (디버깅용)
    print(f"Extracted features for URL {url}: {features}")
    return features

@app.route('/predict', methods=['POST'])
def predict():
    """
    악성 URL 분류 모델 예측 API
    ---
    parameters:
      - name: url
        in: body
        type: string
        required: true
        description: 예측할 URL
    responses:
      200:
        description: 모델의 예측 결과
        schema:
          id: Prediction
          properties:
            prediction:
              type: array
              items:
                type: number
              description: 모델의 예측 결과
      400:
        description: 요청에 URL이 없을 경우
      500:
        description: 서버 오류
    """
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({'error': 'URL이 제공되지 않았습니다.'}), 400
    
    try:
        features = extract_features(url)
        prediction = model.predict([features])
        result = {'prediction': prediction.tolist()}
        return jsonify(result)
    except Exception as e:
        # 예외 상세 메시지 출력
        return jsonify({'error': f'예측 중 오류 발생: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
