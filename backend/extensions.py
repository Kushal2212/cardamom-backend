
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from apscheduler.schedulers.background import BackgroundScheduler
 
bcrypt = Bcrypt()
jwt    = JWTManager()