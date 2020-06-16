
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 15:44:37 2020

@author: dhanam
"""
from flask import Flask
from flask import jsonify, request
app = Flask(__name__)

from nlp_twitter import getSentiment

@app.route("/sentiment", methods = ['POST'])
def get_sentiment():

    json = request.json

    if json == None:
        return jsonify({"null":"Enter valid input"})

    for x in json:
        s = json[x]

    z = getSentiment(s)
    y = {"sentiment": str(z)}

    return jsonify(y)



if __name__ == '__main__':
    app.run(debug=True, use_reloader = False)