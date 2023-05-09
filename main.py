from flask import Flask, request, jsonify

from rnn import is_suspicious_request

app = Flask(__name__)


@app.before_request
def check_request():
    if is_suspicious_request(request):
        return jsonify({"message": "Acesso negado. Requisição suspeita."}), 403


@app.route("/", methods=["GET", "POST"])
def home():
    return "Hello, World!"


if __name__ == "__main__":
    app.run(debug=True)
