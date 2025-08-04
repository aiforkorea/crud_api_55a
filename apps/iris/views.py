# apps/iris/views.py
from flask import Flask, flash, redirect, request, render_template, jsonify, abort, current_app, url_for, g
import pickle, os
import logging, functools
from apps.extensions import csrf
# apps.dbmodels 에서 User, APIKey, UsageLog, UsageType 등을 가져오고
from apps.dbmodels import PredictionResult, db, APIKey, UsageLog, UsageType, Service
# apps.iris.dbmodels 에서 IrisResult 모델을 가져옵니다.
from apps.iris.dbmodels import IrisResult
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
      
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        pred = model.predict(features)[0]
        
        # 1. 'iris' 서비스의 ID를 조회합니다.
        # 만약 서비스가 없으면 None으로 처리하거나 오류를 낼 수 있습니다.
        #iris_service = Service.query.filter_by(servicename='iris').first()
        #iris_service_id = iris_service.id if iris_service else None
        iris_service_id = 1   # 서비스 번호는 임의로 설정, 향후 다중 서비스인 경우, 해당 ID 할당 예정

        # 2. UsageLog 객체 생성 시 service_id에 찾은 값을 할당합니다.
        new_usage_log = UsageLog(
            user_id=current_user.id,
            usage_type=UsageType.WEB_UI, # WEB_UI 사용으로 변경
            endpoint=request.path,
            remote_addr=request.remote_addr,
            response_status_code=200,
            service_id=iris_service_id # 이 부분을 수정
        )
        db.session.add(new_usage_log)
        db.session.commit()
        
        return render_template('iris/predict.html',
                               result=TARGET_NAMES[pred],
                               sepal_length=sepal_length, sepal_width=sepal_width,
                               petal_length=petal_length, petal_width=petal_width, form=form,
                               TARGET_NAMES=TARGET_NAMES)
    return render_template('iris/predict.html', form=form)


@iris.route('/save_iris_data', methods=['POST'])
@login_required
def save_iris_data():
    if request.method == 'POST':
        sepal_length = request.form.get('sepal_length')
        sepal_width = request.form.get('sepal_width')
        petal_length = request.form.get('petal_length')
        petal_width = request.form.get('petal_width')
        predicted_class = request.form.get('predicted_class')
        confirmed_class = request.form.get('confirmed_class')

        # 'iris' 서비스의 ID를 조회합니다.
        #iris_service = Service.query.filter_by(servicename='iris').first()
        #iris_service_id = iris_service.id if iris_service else None
        iris_service_id = 1 # 서비스 번호는 임의로 설정, 향후 다중 서비스인 경우, 해당 ID 할당 예정

        # service_id가 None이면 오류를 반환합니다.
        if not iris_service_id:
            flash('IRIS 서비스 ID를 찾을 수 없습니다.', 'danger')
            return redirect(url_for('iris.iris_predict'))
        
        new_iris_entry = IrisResult(
            user_id=current_user.id,
            service_id=iris_service_id,   
            sepal_length=float(sepal_length),
            sepal_width=float(sepal_width),
            petal_length=float(petal_length),
            petal_width=float(petal_width),
            predicted_class=predicted_class,
            confirmed_class=confirmed_class,
            confirm=True
        )
        db.session.add(new_iris_entry)
        db.session.commit()
        
        flash('데이터가 성공적으로 저장되었습니다.', 'success')
        return redirect(url_for('iris.iris_predict'))
    
    flash('데이터 저장 중 오류가 발생했습니다.', 'danger')
    return redirect(url_for('iris.iris_predict'))

@iris.route('/results')
@login_required
def results():
    # `PredictionResult`의 하위 클래스인 `IrisResult`를 쿼리합니다.
    user_results = IrisResult.query.filter_by(user_id=current_user.id).order_by(IrisResult.created_at.desc()).all()
    return render_template('iris/user_results.html', title='추론결과', results=user_results)

@iris.route('/logs')
@login_required
def logs():
    # UsageLog 모델은 변경사항이 없으므로 그대로 사용합니다.
    user_logs = UsageLog.query.filter_by(user_id=current_user.id).order_by(UsageLog.timestamp.desc()).all()
    return render_template('iris/user_logs.html', title='AI로그이력', results=user_logs)

@iris.route('/api/predict', methods=['POST'])
#@rate_limit('API_KEY_RATE_LIMIT')
@csrf.exempt
def api_predict():
    auth_header = request.headers.get('X-API-Key')
    if not auth_header:
        return jsonify({"error": "API Key is required"}), 401
    
    # API 키 검증 및 관련 정보 조회
    api_key_entry = APIKey.query.filter_by(key_string=auth_header, is_active=True).first()
    
    if not api_key_entry:
        return jsonify({"error": "Invalid or inactive API Key"}), 401
    
    # 'iris' 서비스 ID 조회 (API 요청 처리 전에 미리 조회)
    iris_service = Service.query.filter_by(servicename='iris').first()
    if not iris_service:
        return jsonify({"error": "Iris service not found"}), 500
    
    iris_service_id = iris_service.id

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    required_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    try:
        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])
        petal_length = float(data['petal_length'])
        petal_width = float(data['petal_width'])
    except ValueError:
        return jsonify({"error": "Invalid data type for Iris features. Must be numbers."}), 400

    try:
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        pred_index = model.predict(features)[0] - 1  # 모델 예측 결과는 1부터 시작하므로 -1
        predicted_class_name = TARGET_NAMES[pred_index]
        
        # UsageLog 객체 생성
        new_usage_log = UsageLog(
            user_id=api_key_entry.user_id,
            service_id=iris_service_id, # service_id 할당
            api_key_id=api_key_entry.id,
            usage_type=UsageType.API_KEY,
            endpoint=request.path,
            remote_addr=request.remote_addr,
            response_status_code=200,
            request_data_summary=str(data)[:200]
        )
        db.session.add(new_usage_log)
        
        # IrisResult (이전 IRIS) 객체 생성
        new_iris_entry = IrisResult(
            user_id=api_key_entry.user_id,
            service_id=iris_service_id, # service_id 할당
            api_key_id=api_key_entry.id,
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
            predicted_class=predicted_class_name,
            model_version='1.0',
            confirmed_class=None,
            confirm=False,
            type='iris' # IrisResult에 type 컬럼이 있다면 추가
        )
        db.session.add(new_iris_entry)
        db.session.commit()

        return jsonify({
            "predicted_class": predicted_class_name,
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }), 200

    except Exception as e:
        # 광범위한 예외 처리를 하나로 통합
        logging.error(f"Unexpected error in /api/predict (API Key): {e}", exc_info=True)
        db.session.rollback()
        return jsonify({"error": "An unexpected error occurred."}), 500

@iris.route('/confirms')
@login_required
def confirms():
    # confirmed_class가 None인 모든 예측 결과를 조회합니다.
    unconfirmed_results = PredictionResult.query.filter(
        PredictionResult.confirmed_class.is_(None)
    ).all()
    return render_template('iris/confirms.html', unconfirmed_results=unconfirmed_results)

@iris.route('/confirms/update', methods=['POST'])
@login_required
def update_confirm():
    try:
        # 폼 데이터에서 ID와 confirmed_class 값을 가져옵니다.
        prediction_id = request.form['id']
        confirmed_class_value = request.form['confirmed_class']

        # ID로 해당 PredictionResult 객체를 찾습니다.
        result_to_update = PredictionResult.query.get(prediction_id)

        if result_to_update:
            # confirmed_class와 confirm 필드를 업데이트합니다.
            result_to_update.confirmed_class = confirmed_class_value
            result_to_update.confirm = True
            db.session.commit()

    except Exception as e:
        print(f"Error updating prediction: {e}")
        db.session.rollback() # 오류 발생 시 롤백

    # 확인 페이지로 리다이렉트합니다.
    return redirect(url_for('iris.confirms'))
"""
윈도우 CMD
curl -X POST "http://localhost:5000/iris/api/predict" -H "Content-Type: application/json" -H "X-API-Key: your_api_key" -d "{\"sepal_length\":6.0,\"sepal_width\":3.5,\"petal_length\":4.5,\"petal_width\":1.5}"
윈도우 파워쉘
$headers = @{
    "Content-Type" = "application/json"
    "X-API-Key" = "your_api_key"
}

$body = @{
    sepal_length = 6.0
    sepal_width = 3.5
    petal_length = 4.5
    petal_width = 1.5
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/iris/api/predict" -Method Post -Headers $headers -Body $body
"""
