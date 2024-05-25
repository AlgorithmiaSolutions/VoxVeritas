from flask import Flask, render_template

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index_get():
    return render_template('index.html', message='Hello World!')


@app.route('/submit', methods=['POST'])
def submit_audio():
    return render_template('index.html', message='Huuuu')


if __name__ == '__main__':
    app.run(debug=True)
