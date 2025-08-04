from flask_wtf import FlaskForm
from wtforms import PasswordField, StringField, SubmitField
from wtforms.validators import DataRequired, Email, length, EqualTo, ValidationError #유효성

class ApiKeyForm(FlaskForm):
    # Assuming you might want to create a new API key, or manage existing ones
    # You might have a field for the key itself, or just a submit button
    # For now, let's just add a submit button if the purpose is just CSRF token
    submit = SubmitField('API_Key')
    
class ChangePasswordForm(FlaskForm):
    current_password = PasswordField('현재 비밀번호', validators=[DataRequired()])
    new_password = PasswordField('새 비밀번호', validators=[DataRequired(), length(min=1)])
    confirm_new_password = PasswordField('새 비밀번호 확인', validators=[DataRequired(), EqualTo('new_password')])
    submit = SubmitField('비밀번호 변경')    