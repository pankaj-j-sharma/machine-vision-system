from flask import Flask, request, send_from_directory
from flask import jsonify
from flask_cors import CORS
from user_auth import *
from api_setup import *

app = Flask(__name__)
CORS(app)

localDbData = {}
authToken = None

################################# Test if working ########################################


@app.route('/apitest', methods=["POST"])
def TestAPI():
    return {'success': 'API connected successfully'}


########################## ML API for User Authentication ################################

@app.route('/v1/predict', methods=["POST"])
def PredictFromApi():
    # print('request', request.__dict__)
    req_url = request.url.replace('predict', '')
    data = {'form': request.form.to_dict(), 'files': list(
        request.files.to_dict().values())}
    data['form']['base_url'] = req_url
    return inferenceModelApi(data)


@app.route('/v1/getfile/<path:image_name>', methods=['GET', 'POST'])
def GetAllFiles(image_name):
    try:
        return send_from_directory(os.path.join(os.getcwd(), 'uploads'), image_name, as_attachment=False)
    except FileNotFoundError:
        app.abort(404)


########################## Main Program Begins #####################################
if __name__ == "__main__":
    app.run(debug=True, port='5003')
