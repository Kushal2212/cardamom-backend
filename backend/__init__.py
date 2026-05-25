# backend/__init__.py

from flask import Flask
 


def create_app():
    app = Flask(__name__)

    # Optional Flask configs
    app.config["SECRET_KEY"] = "2663841ec162496da62dbc94f92f700245db690b7ecba93fcc6d242969755a74"

    # Create MongoDB indexes


    return app