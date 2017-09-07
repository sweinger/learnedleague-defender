from flask import (
    Flask,
    Response,
    Request,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_bootstrap import Bootstrap
import numpy as np
from model_builder import *

app = Flask(__name__)
Bootstrap(app)

PORT_NUMBER = 8081 

@app.route('/', methods=['POST', 'GET'])
def get_homepage():
    return render_template('index.html')

@app.route('/score', methods=['GET','POST'])
def score():
	q1 = request.form["q1"]
	q2 = request.form["q2"]
	q3 = request.form["q3"]
	q4 = request.form["q4"]
	q5 = request.form["q5"]
	q6 = request.form["q6"]
	username = request.form["username"]
	points = np.array([3,2,2,1,1,0])
	# ranks = score_questions(username, q1, q2, q3, q4, q5, q6)
	# print ranks
	# assigned_points = points[ranks]
	# print scores
	scores, auc = score_questions(username, q1, q2, q3, q4, q5, q6)
	seq = sorted(scores)
	ranks = [seq.index(v) for v in scores]
	print ranks
	points = np.array([3, 2, 2, 1, 1, 0])
	assigned_points = points[ranks]	
	return render_template('index.html', q1=q1, q2=q2, q3=q3, q4=q4, q5=q5, q6=q6, 
		q1_pts=assigned_points[0], q2_pts=assigned_points[1], q3_pts=assigned_points[2],
		q4_pts=assigned_points[3], q5_pts=assigned_points[4], q6_pts=assigned_points[5],
		ranks=ranks)

if __name__ == '__main__':
	app.debug = True
	app.run(host='0.0.0.0', port=PORT_NUMBER)

