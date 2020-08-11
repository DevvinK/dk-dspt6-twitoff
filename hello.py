from flask import Flask

# Create a flask object, __name__ references the name of the file
app = Flask(__name__)

# decorator that creates a directory
@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/about')
def about():
    return 'About Me!'
