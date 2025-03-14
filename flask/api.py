from flask import Flask
from flask_sqlalchemy import SQLAlchemy

DEBUG=True

app=Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']
db=SQLAlchemy(app)

class UserModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)

    def __repr__(self):
        return f"User(name = {self.name})"


@app.route('/')

def index():
    return '<h1>Flask REST Api</h1>'

if __name__=="__main__":
    app.run(debug=DEBUG)
