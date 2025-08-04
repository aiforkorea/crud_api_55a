# apps/dbmodels.py
import enum
import uuid
from flask import current_app
from apps.extensions import db  # apps.extensions에서 db를 import, 순환충돌방지
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash 
#hash import
from datetime import datetime
from apps.config import Config
class User(db.Model, UserMixin):  # db.Model, UserMixin 상속하는 User 클래스 생성
    __tablename__= "users"  # 삭제시, 테이블 이름은 Model 이름 소문자 
    id=db.Column(db.Integer,primary_key=True)
    username=db.Column(db.String, index=True)
    email=db.Column(db.String, unique=True, index= True)
    password_hash=db.Column(db.String)
    is_admin=db.Column(db.Boolean,default=False)
    is_active=db.Column(db.Boolean, default=True)  # 활성화 여부
    usage_count = db.Column(db.Integer, default=0)
    # 특정 User에 대한 일일/월간 사용량 제한을 위한 필드 추가 가능 (예: daily_limit, monthly_limit)
    daily_limit = db.Column(db.Integer, default=1000)
    monthly_limit = db.Column(db.Integer, default=5000)
    created_at=db.Column(db.DateTime, default= datetime.now)
    updated_at=db.Column(db.DateTime, default= datetime.now, onupdate=datetime.now)
    subscriptions = db.relationship('Subscription', backref='user', lazy=True, cascade='all, delete-orphan')
    api_keys = db.relationship('APIKey', backref='user', lazy=True, cascade='all, delete-orphan')
    usage_logs = db.relationship('UsageLog', backref='user', lazy=True, cascade='all, delete-orphan')
    prediction_results = db.relationship('PredictionResult', backref='user', lazy=True, cascade="all, delete-orphan")
    @property
    def password(self):
        """
        Prevent direct reading of the password.
        Attempting to read `user.password` will raise an AttributeError.
        """
        raise AttributeError('password is not a readable attribute')
    @password.setter
    def password(self, password):
        """
        Hashes the plain-text password and stores it in password_hash.
        This allows you to do `user.password = "mysecretpassword"`.
        """
        self.password_hash = generate_password_hash(password)
    # Method to check password (you likely already have this or check_password)
    def check_password(self, password):
        """Checks if the provided password matches the stored hash."""
        if self.password_hash is None:
            return False
        return check_password_hash(self.password_hash, password)
    # 비밀번호 체크
    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)
    # Flask-Login required methods
    @property
    def is_authenticated(self):
        return True
    @property
    def is_anonymous(self):
        return False
    def __repr__(self):
        return f'<User {self.username}>'
    # 이메일 중복 체크
    def is_duplicate_email(self):
        return User.query.filter_by(email=self.email).first() is not None
class Service(db.Model):
    __tablename__ = "services"
    id = db.Column(db.Integer, primary_key=True)
    servicename = db.Column(db.String(100), unique=True, nullable=False)
    is_active=db.Column(db.Boolean, default=True)  # 활성화 여부
    is_auto=db.Column(db.Boolean, default=True)  # 자동승인 여부
    price = db.Column(db.Integer, default=0, nullable=False)  # 서비스 단가
    description = db.Column(db.Text, nullable=False)
    keywords = db.Column(db.String(200), nullable=False)
    service_endpoint = db.Column(db.String(255), nullable=True)  # 서비스 엔드포인트 함수, 일단 True
    created_at=db.Column(db.DateTime, default= datetime.now)
    updated_at=db.Column(db.DateTime, default= datetime.now, onupdate=datetime.now)

    # 관계
    subscriptions = db.relationship('Subscription', backref='service', lazy=True, cascade="all, delete-orphan")
    usage_logs = db.relationship('UsageLog', backref='service', lazy=True, cascade="all, delete-orphan")
    prediction_results = db.relationship('PredictionResult', backref='service', lazy=True, cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Service(name='{self.servicename}')>" # servicename으로 변경
class Subscription(db.Model):
    __tablename__ = "subscriptions"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    service_id = db.Column(db.Integer, db.ForeignKey('services.id'), nullable=False) 
    status = db.Column(db.String(20), default='pending', nullable=False) # pending, approved, rejected
    request_date = db.Column(db.DateTime, nullable=False, default=datetime.now)
    approval_date = db.Column(db.DateTime, nullable=True)

    # 한 사용자가 특정 서비스를 여러 번 구독 요청하지 못하도록 제약하는 방식
    __table_args__ = (db.UniqueConstraint('user_id', 'service_id', name='_user_service_uc'),)

    def __repr__(self) -> str:
        return f"<Subscription(user_id={self.user_id}, service_id={self.service_id}, status='{self.status}')>"
class APIKey(db.Model):
    __tablename__ = 'api_keys'
    id = db.Column(db.Integer, primary_key=True)
    key_string = db.Column(db.String(32), unique=True, nullable=False, default=lambda: str(uuid.uuid4()).replace('-', '')[:32])
    description = db.Column(db.String(100), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    last_used = db.Column(db.DateTime)
    usage_count = db.Column(db.Integer, default=0) # 이 API 키를 통한 총 사용 횟수
    daily_limit = db.Column(db.Integer, default=1000)
    monthly_limit = db.Column(db.Integer, default=5000)
    usage_logs = db.relationship('UsageLog', backref='api_key', lazy=True, cascade="all, delete-orphan")
    prediction_results = db.relationship('PredictionResult', backref='api_key', lazy=True, cascade="all, delete-orphan")
    def __init__(self, user_id: int, description: str = None):
        self.user_id = user_id
        self.description = description
    def generate_key(self) -> None:
        self.key_string = str(uuid.uuid4()).replace('-', '')[:32]
    def __repr__(self) -> str:
        return f"<APIKey(key_string='{self.key_string}')>"
class UsageType(enum.Enum):
    LOGIN = 'login'
    API_KEY = 'api_key'
    WEB_UI = 'web_ui'
class UsageLog(db.Model):
    __tablename__ = 'usage_logs'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), index=True)
    service_id = db.Column(db.Integer, db.ForeignKey('services.id'), nullable=False)
    api_key_id = db.Column(db.Integer, db.ForeignKey('api_keys.id'), index=True)
    endpoint = db.Column(db.String(120), nullable=False)
    usage_type = db.Column(db.Enum(UsageType), nullable=False)
    usage_count = db.Column(db.Integer, default=1, nullable=False) # 각 로그 항목은 기본적으로 1회 사용
    login_confirm = db.Column(db.String(10), nullable=True) # 로그인 여부 확인용 (예: 'success', 'fail')
    timestamp = db.Column(db.DateTime, default=datetime.now, index=True)
    last_used = db.Column(db.DateTime, nullable=False, default=datetime.now) # 마지막 사용 시간
    remote_addr = db.Column(db.String(45))
    request_data_summary = db.Column(db.Text)
    response_status_code = db.Column(db.Integer)
    def __repr__(self) -> str:
        return f"<UsageLog(api_service_id={self.service_id}, usage_type='{self.usage_type}', timestamp={self.timestamp})>"
# ----------- 예측 결과 기본 모델 (PredictionResult) -----------
class PredictionResult(db.Model):
    __tablename__ = 'prediction_results'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    service_id = db.Column(db.Integer, db.ForeignKey('services.id'), nullable=False)
    api_key_id = db.Column(db.Integer, db.ForeignKey('api_keys.id'), index=True)
    predicted_class = db.Column(db.String(50))
    model_version = db.Column(db.String(20), default='1.0')
    confirmed_class = db.Column(db.String(50))
    confirm = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.now, index=True)
    confirmed_at = db.Column(db.DateTime, default=datetime.now, index=True)
    # 다형성 설정: 어떤 예측 결과 유형인지 구분
    # polymorphic_on과 polymorphic_identity를 사용한 싱글 테이블 상속(Single Table Inheritance) 구조
    # 이를 통해 IrisResult와 LoanResult 같은 특정 서비스의 예측 결과를 유연하게 확장,SQLAlchemy의 고급 기능
    type = db.Column(db.String(50))
    __mapper_args__ = {
        'polymorphic_on': type,
        'polymorphic_identity': 'prediction_result'
    }
    def __repr__(self) -> str:
        return f"<PredictionResult(user_id={self.user_id}, service_id={self.service_id}, predicted_class='{self.predicted_class}')>"
# 이 예측 결과 기본 모델을 상속하여 각 서비스별 특수화 필드 추가