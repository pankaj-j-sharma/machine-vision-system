from flask import Flask, request, send_from_directory
from flask import jsonify
from flask_cors import CORS, cross_origin
from rq import Queue
from rq_worker import conn

queue = Queue(connection=conn)
# result = q.enqueue(count_words_at_url, 'http://heroku.com')
