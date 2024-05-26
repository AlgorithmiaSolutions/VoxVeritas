# VoxVeritas

AI Model that detects deep fake audio

## Setup

### Virtual env

Create virtual environment

``python -m venv venv``

activate the virtual env

``source venv/bin/activate`` for linux and mac\
``venv\Scripts\activate`` for windows

Install packages

``pip install -r requirements.txt``

Run flask

```
export FLASK_APP=app.py
export FLASK_ENV=development
export FLASK_DEBUG=true
flask run
`