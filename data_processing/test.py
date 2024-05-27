import json

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import ibcf.ibcf as IBCF

app = Flask(__name__, template_folder="template")
CORS(app)


def find_value_by_label(file_path, target_label):
    with open(file_path, 'r') as file:
        data = json.load(file)
        for item in data:
            if item.get("label") == target_label:
                return int(item.get("value"))


@app.route("/")
def hello():
    return render_template('test.html')


@app.route('/process', methods=['POST'])
def process():
    label = request.form.get('data')
    movieId = find_value_by_label('movies_no_tags.json', label)
    IBCF.print_similar_movies(movieId)
    with open('movies_recommend_name_list.json', 'r') as file:
        data = file.read()
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
