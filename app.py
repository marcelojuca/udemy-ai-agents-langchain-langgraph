from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
import os
from ice_breaker import ice_break_with


load_dotenv()

# Point Flask to the top-level templates directory
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    name = request.form["name"]
    summary, profile_picture_url = ice_break_with(name=name)
    return jsonify({
        "summary_and_facts": summary.to_dict(),
        "photoUrl": profile_picture_url
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5055"))
    app.run(host="0.0.0.0", port=port, debug=True)