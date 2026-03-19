from flask import Flask, render_template
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

app = Flask(__name__, template_folder="../templates", static_folder="../statics")

@app.route("/")
def home():
    return render_template("index.html")

# Vercel handler
def handler(request):
    return app(request)