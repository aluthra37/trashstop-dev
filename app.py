from flask import Flask
from flask import render_template
app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return 'Welcome to TrashStop'

@app.route('/nothing')
def nothing():
    return render_template('nothing.html')

@app.route('/landfill')
def landfill():
    return render_template('landfill.html')

@app.route('/glass')
def glass():
    return render_template('glass.html')

@app.route('/cardboard')
def cardbaord():
    return render_template('cardboard.html')
    
@app.route('/metal')
def metal():
    return render_template('metal.html')

@app.route('/plastic')
def plastic():
    return render_template('plastic.html')

@app.route('/paper')
def paper():
    return render_template('plastic.html')
