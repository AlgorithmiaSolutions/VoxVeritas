import os

from flask import Flask, render_template, request, redirect, url_for

from ai import ai_process

app = Flask(__name__)
app.secret_key = 'mknafihia;nfiuqwnpf982no2iqn'
app.config['UPLOAD_FOLDER'] = 'uploads/'


@app.route('/', methods=['GET'])
@app.route('/<message>', methods=['GET'])
def index_get(message=None):
    return render_template('index.html', message=message)


@app.route('/submit', methods=['POST'])
def submit_audio():
    if 'ai_audio' not in request.files:
        return redirect(url_for('index_get', message='No file part!'))

    file = request.files['ai_audio']

    if file.filename == '':
        return redirect(url_for('index_get', message='No selected file!'))

    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        result = ai_process(filename)
        return redirect(url_for('index_get', message='The audio is: ' + result))
    return redirect(url_for('index_get', message='File failed to upload!'))


if __name__ == '__main__':
    app.run(debug=True)
