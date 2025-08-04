# apps/__init__.py
import os
from flask import Flask
from werkzeug.security import generate_password_hash

from apps.dbmodels import Service
from .extensions import db, migrate, login_manager, csrf
from apps.admin import MyAdminIndexView, UserAdminView
#from apps.admin import MyAdminIndexView, UserAdminView,APIKeyAdminView,UsageLogAdminView
from .config import Config
from flask_admin import Admin, AdminIndexView, expose
from flask_admin.contrib.sqla import ModelView
#from apps.dbmodels import User, APIKey, UsageLog

# 전역 변수/인스턴스 초기화 (extensions.py에서 정의)
def create_app():     #  factory 함수
    app = Flask(__name__)
    app.config.from_object(Config) # config.py에서 설정 로드
    # 확장 기능 초기화 연계
    db.init_app(app)                      # flask 앱에 db연결
    migrate.init_app(app,db)              # 없으면, flask db 명령어를 사용불가
    login_manager.init_app(app)  # flask 앱에 로그인 관리 연결
    csrf.init_app(app)                    # flask 앱에 CSRF 보호 연결       

    # Flask-Login: 사용자 로더 설정 (auth 블루프린트에서 import하여 사용)
    # create_app() 정의 또는 auth/__init__.py 정의하여 login_manager.user_loader 데코레이터와 함께 사용
    from apps.dbmodels import User  # User 모델 임포트
    @login_manager.user_loader
    def load_user(user_id):   # Flask-Login이 user_id를 기반으로 사용자 객체를 로드
        return User.query.get(int(user_id))

    # Flask-Login: Unauthorized Error 핸들링, login_view와 같은 기능이나, next 값 자동전달
    @login_manager.unauthorized_handler
    def unauthorized():
        """로그인되지 않은 사용자가 @login_required 페이지에 접근 시 redirect"""
        from flask import flash, redirect, url_for, request
        flash('로그인이 필요합니다.', 'warning')
        return redirect(url_for('auth.login', next=request.path))
    # 블루프린트 등록
    from .main import main
    from .auth import auth
    from .iris import iris
    from .mypage import mypage
    app.register_blueprint(main)
    app.register_blueprint(auth, url_prefix='/auth')
    app.register_blueprint(iris, url_prefix='/iris')
    app.register_blueprint(mypage, url_prefix='/mypage')
    # flask-admin 은 블루프린트 등록이 필요없음
    # Flask-Admin 설정 (관리자 페이지)  # flask-admin 인스턴스 생성 및 관리자 페이지의 첫 화면 설정
    admin=Admin(app,name='Flask Admin', template_mode='bootstrap3', index_view = MyAdminIndexView())
    # 모델 뷰 추가 (User, APIKey, UsageLog 모델을 관리자 페이지에서 관리할 수 있도록)
    admin.add_view(UserAdminView(User, db.session, name='사용자 관리'))
    #admin.add_view(APIKeyAdminView(APIKey, db.session, name='API 키 관리'))
    #admin.add_view(UsageLogAdminView(UsageLog, db.session, name='사용량 로그'))
    # db 테이블 생성 및 관리자 초기계정 생성
    with app.app_context():
        #db.drop_all()         # 운영시에는 커멘트 처리 필요
        db.create_all()       # 테이블 생성

        # 'iris' 서비스가 없으므로 강제 추가
        if not Service.query.filter_by(servicename='iris').first():
            iris_service = Service(
                servicename='iris',
                price=0,
                description='꽃잎/꽃받침 길이와 너비로 붓꽃 종류를 예측하는 AI 서비스',
                keywords='iris, prediction, ai, ml',
                #service_endpoint='/iris/api/predict'
                service_endpoint='iris.api.predict'
            )
            db.session.add(iris_service)
            db.session.commit()
            print("INFO: 'iris' 서비스가 데이터베이스에 추가되었습니다.")

        # 최초 관리자 계정 생성
        admin_username = app.config.get('ADMIN_USERNAME')
        admin_email = app.config.get('ADMIN_EMAIL')
        admin_password = app.config.get('ADMIN_PASSWORD')

        if admin_username and admin_password:
            admin_user = User.query.filter_by(username=admin_username).first()
            if not admin_user:
                hashed_password = generate_password_hash(admin_password)
                new_admin = User(username=admin_username, email=admin_email, password_hash=hashed_password, is_admin=True)
                db.session.add(new_admin)
                db.session.commit()
                print(f"관리자 계정 '{admin_username}', '{admin_password}' 이(가) 생성되었습니다.")
            else:
                print(f"관리자 계정 '{admin_username}'이(가) 이미 존재합니다.")
        else:
            print("ADMIN_USERNAME 또는 ADMIN_PASSWORD 환경 변수가 설정되지 않았습니다.")
        return app
