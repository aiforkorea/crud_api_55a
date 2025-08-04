# apps/auth/views.py
from apps.extensions import db, login_manager
from .forms import SignUpForm, LoginForm
from apps.dbmodels import User
from flask import render_template, flash, url_for, redirect, request
from flask_login import login_user, logout_user
from apps import auth  # views.py import해서 라우팅 등록
from . import auth ## 추가

# index 엔드포인트
@auth.route("/")
def index():
    return render_template("auth/index.html")
# signup 엔드포인트
@auth.route("/signup",methods=["GET", "POST"])     
def signup():
    form = SignUpForm()  # SingUpForm 객체 form 생성
    if form.validate_on_submit():
        user = User(
            username=form.username.data,
            email=form.email.data,
            password=form.password.data 
        )
        if user.is_duplicate_email():  #이메일주소중복체크
            flash("입력한 이메일은 이미 등록됨")
            return redirect(url_for("auth.signup"))
        db.session.add(user)  # 사용자 정보 DB 등록
        db.session.commit()
        login_user(user)   # 사용자 정보 세션 저장
        # GET패러미터에 next키가 존재하면(보호페이지 사전 방문한 경우) 원래 방문 페이지로 redirect
        # next 값이 없는 경우는 사용자의 목록 페이지로 redirect
        next_ = request.args.get(next)
        if next_ is None or not next_.startswith("/"):
            next_ = url_for("main.index")
        return redirect(next_)
    return render_template("auth/signup.html",form= form)
# login 엔드포인트
@auth.route("/login",methods=["GET", "POST"])     
def login():
    form = LoginForm()    # LoginForm 회원로그인 객체 form 생성
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        # DB에 사용자 존재 및 비번 일치하면
        if user is not None and user.verify_password(form.password.data): 
            # 사용자 정보 세션 저장
            login_user(user)
            return redirect(url_for("main.index"))   # main.index로 이동
        # 로그인 실패 메시지
        flash("이메일 주소 및 비번 확인 필요")
    return render_template("auth/login.html",form=form)
# logout 엔드포인트 
@auth.route("/logout")     
def logout():
    # 사용자 로그아웃
    logout_user()
    return redirect(url_for("auth.login"))
