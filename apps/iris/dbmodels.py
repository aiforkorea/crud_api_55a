# apps/iris/dbmodels.py
from apps.dbmodels import db, PredictionResult # PredictionResult를 임포트

class IrisResult(PredictionResult):
    __tablename__ = 'iris_results'
    id = db.Column(db.Integer, db.ForeignKey('prediction_results.id'), primary_key=True)
    sepal_length = db.Column(db.Float, nullable=False)
    sepal_width = db.Column(db.Float, nullable=False)
    petal_length = db.Column(db.Float, nullable=False)
    petal_width = db.Column(db.Float, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'iris'
    }

    def __repr__(self) -> str:
        return (f"<IrisResult(sepal_length={self.sepal_length}, sepal_width={self.sepal_width}, "
                f"petal_length={self.petal_length}, petal_width={self.petal_width}, predicted_class='{self.predicted_class}')>")