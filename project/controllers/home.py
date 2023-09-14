from project import app
from flask import render_template, redirect, url_for

# Import models here


@app.route('/', methods = ['GET'])
def index():
    data = {
        "title": "Face Detection",
        "body": "Flask MVC Pattern"
    }
    
    return render_template('index.html', data= data)