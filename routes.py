""" routes.py - Flask route definitions

Flask requires routes to be defined to know what data to provide for a given
URL. The routes provided are relative to the base hostname of the website, and
must begin with a slash."""
from flaskapp import app
from flask import render_template
import csv

@app.route('/')
@app.route('/pca')
def get_pca():
    return None
