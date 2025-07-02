# apps/iris/views.py
from flask import Flask, flash, redirect, request, render_template, jsonify, abort, current_app, url_for, g
import pickle, os 
import logging, functools
from apps.extensions import csrf # 이 줄을 추가하여 정의된 csrf 객체를 임포트
from apps.dbmodels import IRIS, db, User, APIKey, UsageLog, UsageType
import numpy as np
from flask_login import current_user, login_required
from apps.iris.forms import IrisUserForm
from . import iris
 
from datetime import datetime, timedelta

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

#TARGET_NAMES = ['setosa', 'versicolor', 'virginica']
from apps.config import Config
TARGET_NAMES = Config.IRIS_LABELS   # 라벨 읽기

# AI 사용량 제한 데코레이터
def rate_limit(limit_config_key):
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            limit_str = getattr(Config, limit_config_key)
            # 간단한 속도 제한 구현 (프로덕션에서는 Redis 등을 활용하는 것이 좋습니다)
            # 현재는 UsageLog 테이블을 사용하여 카운트합니다.
            # 이 로직은 실제 배포 시 성능 문제가 될 수 있으므로 주의가 필요합니다.
            # Flask-Limiter와 같은 외부 라이브러리를 사용하는 것을 권장합니다.
            # API Key 사용량 제한
            if 'api_key_id' in kwargs and kwargs['api_key_id']:
                api_key = APIKey.query.get(kwargs['api_key_id'])
                if not api_key:
                    logging.warning(f"Rate Limit: Invalid API Key ID {kwargs['api_key_id']}")
                    return jsonify({"error": "Invalid API Key"}), 401
                # 현재 분의 시작 시간 계산
                now = datetime.now()
                minute_ago = now - timedelta(minutes=1)
                usage_count = UsageLog.query.filter(
                    UsageLog.api_key_id == api_key.id,
                    UsageLog.endpoint == request.path,
                    UsageLog.timestamp >= minute_ago
                ).count()
                limit_value = int(limit_str.split('/')[0])
                if usage_count >= limit_value:
                    logging.warning(f"Rate Limit Exceeded for API Key {api_key.key_string}. Count: {usage_count}, Limit: {limit_value}")
                    return jsonify({"error": "API Key usage limit exceeded. Please try again later."}), 429
                # API Key 사용량 업데이트
                api_key.usage_count += 1
                api_key.last_used = now
                db.session.commit() # 여기서 커밋하여 바로 반영
            # 로그인 사용자 사용량 제한
            elif current_user.is_authenticated:
                # 현재 시간의 시작 시간 (시간당)
                now = datetime.now()
                hour_ago = now - timedelta(hours=1)
                usage_count = UsageLog.query.filter(
                    UsageLog.user_id == current_user.id,
                    UsageLog.endpoint == request.path,
                    UsageLog.timestamp >= hour_ago
                ).count()
                limit_value = int(limit_str.split('/')[0])
                if usage_count >= limit_value:
                    logging.warning(f"Rate Limit Exceeded for User {current_user.email}. Count: {usage_count}, Limit: {limit_value}")
                    flash('시간당 사용량 제한을 초과했습니다. 잠시 후 다시 시도해주세요.', 'warning')
                    return redirect(url_for('iris.predict')) # 또는 에러 페이지로 리디렉션
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@iris.route('/predict', methods=['GET', 'POST'])
@login_required
def iris_predict():
    form = IrisUserForm()
    if form.validate_on_submit():
        sepal_length = form.sepal_length.data
        sepal_width = form.sepal_width.data
        petal_length = form.petal_length.data
        petal_width = form.petal_width.data
    
        features=np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        pred = model.predict(features)[0]
        new_usage_log=UsageLog( 
            user_id=current_user.id,
            #api_key_id=None
            usage_type=UsageType.LOGIN, # LOGIN 타입으로 저장
            endpoint=request.path,
            remote_addr=request.remote_addr,
            response_status_code=200
            )
        db.session.add(new_usage_log)     # 새로운 객체를 데이터베이스 세션에 추가
        db.session.commit() # 변경 사항을 데이터베이스에 실제로 저장
        return render_template('iris/predict.html',
                               result=TARGET_NAMES[pred],
                               sepal_length=sepal_length, sepal_width=sepal_width,
                               petal_length=petal_length, petal_width=petal_width, form=form,
                               TARGET_NAMES=TARGET_NAMES
                              )
    return render_template('iris/predict.html',form=form)

@iris.route('/save_iris_data', methods=['POST'])
@login_required
def save_iris_data():
    if request.method == 'POST':        # HTML 폼에서 전송된 모든 데이터 가져오기
        sepal_length = request.form.get('sepal_length')
        sepal_width = request.form.get('sepal_width')
        petal_length = request.form.get('petal_length')
        petal_width = request.form.get('petal_width')
        predicted_class = request.form.get('predicted_class')
        print(f"predicted_class: '{predicted_class}'") #
        confirmed_class = request.form.get('confirmed_class') # 이게 핵심
        new_iris_entry = IRIS( user_id=current_user.id,
            #api_key_id=
            sepal_length=float(sepal_length), # 저장하기 전에 숫자로 바꿔줘요
            sepal_width=float(sepal_width),
            petal_length=float(petal_length),
            petal_width=float(petal_width),
            predicted_class=predicted_class,
            confirmed_class=confirmed_class,  # 수정되었거나 저장된 값을 저장해요
            confirm =True
            )
        db.session.add(new_iris_entry)     # 새로운 객체를 데이터베이스 세션에 추가
        db.session.commit() # 변경 사항을 데이터베이스에 실제로 저장
        return redirect(url_for('iris.iris_predict')) # 예측 페이지로 다시 이동
    flash('데이터 저장 중 오류가 발생했습니다.', 'danger')
    return redirect(url_for('iris.iris_predict'))

@iris.route('/ai_results')
@login_required
def ai_results():
    # 현재 로그인한 사용자의 추론 결과를 가져오기
    # 생성 순서대로 가져오기
    user_results = IRIS.query.filter_by(user_id=current_user.id).order_by(IRIS.created_at.desc()).all()
    return render_template('iris/user_results.html',title='추론결과', results=user_results)

@iris.route('/ai_logs')
@login_required
def ai_logs():
    # 현재 로그인한 사용자의 AI 추론 로그 가져오기
    # 생성 순서대로 가져오기
    user_logs = UsageLog.query.filter_by(user_id=current_user.id).order_by(UsageLog.timestamp.desc()).all()
    return render_template('iris/user_logs.html',title='AI로그이력', results=user_logs)

@iris.route('/api/predict', methods=['POST'])
#@rate_limit('API_KEY_RATE_LIMIT')  count login 포함됨
@csrf.exempt
def api_predict():
    auth_header = request.headers.get('X-API-Key')
    print(f"auth_header: '{auth_header}'") #
    if not auth_header:
        #logging.warning("API Predict: No X-API-Key header provided.")
        return jsonify({"error": "API Key is required"}), 401
    # 유효키인지 확인, 초기 설정
    is_valid_key = False
    # 모든 유효키 가져오기
    active_api_keys = APIKey.query.filter_by(is_active=True).all()
    # 유효키 집합에 포함되는지 확인
    for api_key_entry in active_api_keys:
#        if check_password_hash(api_key_entry.key_hash, auth_header):  # 암호화시
        if api_key_entry.key_string == auth_header:
            is_valid_key = True
            print(f"is_valid_key: '{is_valid_key}'") #
            # Optional: You might want to store the user associated with this key in flask.g
            # 다른 것도 가능, For example: flask.g.current_user_id = api_key_entry.user_id
            # user_id와 api_key_id를 g 객체에 저장하여 UsageLog에서 참조
            g.user_id_for_log = api_key_entry.user_id
            api_key_id_for_log = api_key_entry.id
            usage_type_for_log = UsageType.API_KEY
            break

    if not is_valid_key:
        #logging.warning(f"API Predict: Invalid or inactive API Key '{auth_header}'")
        return jsonify({"error": "Invalid or inactive API Key"}), 401
 
    data = request.get_json()
    if not data:
        logging.warning("API Predict: No JSON data provided.")
        return jsonify({"error": "Invalid JSON"}), 400

    required_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for field in required_fields:
        if field not in data:
            logging.warning(f"API Predict: Missing field '{field}' in request data.")
            return jsonify({"error": f"Missing field: {field}"}), 400

    try:
        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])
        petal_length = float(data['petal_length'])
        petal_width = float(data['petal_width'])
    except ValueError:
        #logging.warning("API Predict: Invalid data type for Iris features.")
        return jsonify({"error": "Invalid data type for Iris features. Must be numbers."}), 400

    try:
        features=np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        pred = model.predict(features)[0]
# curl -X POST "http://localhost:5000/iris/api/predict" -H "Content-Type: application/json" -H "X-API-Key: your_api_key" -d "{\"sepal_length\":6.0,\"sepal_width\":3.5,\"petal_length\":4.5,\"petal_width\":1.5}"
        print(f"pred: '{pred}'") #
        print(f"pred: '{TARGET_NAMES[pred-1]}'") #TARGET_NAMES[pred]
        print(f": '{sepal_length} {sepal_width} {petal_length} {petal_width}' ") #

        predicted_class_int = int(pred) if isinstance(pred, (int, float, object)) else pred
        #result_id_int = int(new_iris_entry.id) if hasattr(new_iris_entry, 'id') else None

        new_usage_log=UsageLog(
            user_id=g.user_id_for_log, # 테스트 2 사용
            api_key_id=api_key_id_for_log, # 테스트 1 사용
            usage_type=usage_type_for_log,  # UsageType.API_KEY 하드 코딩 가능
            endpoint=request.path,
            remote_addr=request.remote_addr,
            response_status_code=200,
            request_data_summary=str(data)[:200]
        )
        db.session.add(new_usage_log)     # 새로운 객체를 데이터베이스 세션에 추가

        new_iris_entry = IRIS(
            user_id=g.user_id_for_log, # 테스트 2 사용
            api_key_id=api_key_id_for_log, # 테스트 1 사용
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
#            predicted_class=str(pred),
            predicted_class=TARGET_NAMES[pred-1],
            confirmed_class=None,
            confirm=False
        )
        db.session.add(new_iris_entry)     # 새로운 객체를 데이터베이스 세션에 추가
        db.session.commit() # 변경 사항을 데이터베이스에 실제로 저장

        return jsonify({
            "predicted_class": predicted_class_int,
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
#            "result_id": result_id_int # 저장된 결과의 ID 반환
        }), 200
    except RuntimeError as e:
        logging.error(f"AI Model Error in /api/predict (API Key): {e}")
        db.session.rollback() # 오류 발생 시 롤백
        return jsonify({"error": "AI model error. Please try again later."}), 500
    except ValueError as e:
        logging.error(f"Input Data Error in /api/predict (API Key): {e}")
        db.session.rollback()
        return jsonify({"error": f"Input data validation error: {e}"}), 400
    except Exception as e:
        logging.error(f"Unexpected error in /api/predict (API Key): {e}", exc_info=True)
        db.session.rollback()
        return jsonify({"error": "An unexpected error occurred."}), 500

