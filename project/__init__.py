from flask import Flask

app = Flask("project")
    
from project.routes import *
from project.controllers import *
from project.libs import *