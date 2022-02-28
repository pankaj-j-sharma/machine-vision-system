import pickle
from difflib import SequenceMatcher
import os
import numpy as np
import pandas as pd
from flask import Flask, request, send_from_directory
from flask import jsonify
from flask_cors import CORS, cross_origin
import json
from prediction_facade import predictfrommodel
from appitem_setup import *
from image_augmentation import *
from prediction_setup import *
from process_tracking_setup import *
from project_setup import *
from user_auth import *
from custom_train_setup import *
from custom_labelling_setup import *
from data_connector_setup import *
from gpu_check import *

app = Flask(__name__)
CORS(app)

localDbData = {}
authToken = None


@app.route('/test', methods=["GET", "POST"])
def Test():
    return {'data': 'hi this is working'}


########################## ML API for Alerts #############################################


########################## ML API for User Authentication ################################


@app.route('/generateapikey', methods=["GET", "POST"])
def GenerateApiKey():
    data = {'form': request.form.to_dict()}
    return generateApiKey(data)


@app.route('/loaduserapikey', methods=["GET", "POST"])
def LoadUserApiKey():
    data = {'form': request.form.to_dict()}
    return loadUserApiKey(data)


@app.route('/login', methods=["GET", "POST"])
def LogIn():
    return loginUser(request.json)


@app.route('/logout', methods=["GET", "POST"])
def LogOut():
    return logoutUser(request.json)


@app.route('/validatetoken', methods=["GET"])
def ValidateToken():
    return validateToken(request.json)


@app.route('/userprofile', methods=["GET", "POST"])
def UserProfile():
    data = {'form': request.form.to_dict()}
    return userProfile(data)


@app.route('/saveuserprofile', methods=["GET", "POST"])
def SaveUserProfile():
    data = {'form': request.form.to_dict(), 'files': list(
        request.files.to_dict().values())}
    return saveUserProfile(data)


def __checkInLocalDb(user, passwd):
    userData = None
    with open('localDb.json') as r:
        localDbData = json.load(r)
    for d in localDbData['loginData']:
        if d["username"] == user and d["passwd"] == passwd:
            userData = d
            authToken = userData["token"]
    return {'token': authToken, 'user': userData}

########################## ML API for Navigation bar ###############################


@app.route('/getmenuitem', methods=['GET', 'POST'])
def GetMenuItems():
    return getmenuItems(request.json)


@app.route('/gethomeapps', methods=['GET', 'POST'])
def GetHomeApps():
    return gethomeApps()


@app.route('/getappalerts', methods=["GET", "POST"])
def GetAppAlerts():
    data = request.json
    return getappAlerts(data)


@app.route('/getusermsgs', methods=['GET', 'POST'])
def GetUserMsgs():
    return getuserMsgs()

########################## Data Connnectors #####################################


@app.route('/getalldataconnectors', methods=['GET', 'POST'])
def GetAllAvailableDataConnectors():
    data = {'form': request.form.to_dict()}
    return getAllDataConnectors(data)


@app.route('/loadconnectordirs', methods=['GET', 'POST'])
def RetrieveConnectorData():
    data = {'form': request.form.to_dict(), 'files': list(
        request.files.to_dict().values())}
    return getConnectorData(data)


####################### ML API for Custom train screen ##############################


@app.route('/loaddraftcustomtrain', methods=['GET', 'POST'])
def LoadDraftCustomTrainData():
    data = {'form': request.form.to_dict()}
    return loadDraftCustomTrainId(data)


@app.route('/savefilesforcustomtrain', methods=['GET', 'POST'])
def SaveFilesForCustomTrain():
    data = {'form': request.form.to_dict(), 'files': list(
        request.files.to_dict().values())}
    return saveFilesForCustomTrain(data)


@app.route('/runcustomtrain', methods=['GET', 'POST'])
def RunCustomTraining():
    data = {'form': request.form.to_dict()}
    return run_custom_training(data)

####################### ML API for Custom label screen ##############################


@app.route('/loaddraftcustomlabel', methods=['GET', 'POST'])
def LoadDraftCustomLabelData():
    data = {'form': request.form.to_dict()}
    return loadDraftCustomLabelId(data)


@app.route('/loadcustomlabeldata', methods=['GET', 'POST'])
def LoadAllCustomLabelData():
    data = {'form': request.form.to_dict()}
    return loadCustomLabelData(data)


@app.route('/savefilesforcustomlabelling', methods=['GET', 'POST'])
def SaveFilesForCustomLabelling():
    data = {'form': request.form.to_dict(), 'files': list(
        request.files.to_dict().values())}
    return saveFilesForCustomLabelling(data)


@app.route('/savecustomlabels', methods=['GET', 'POST'])
def SaveCustomLabelData():
    data = {'form': request.form.to_dict()}
    return saveCustomLabelForImages(data)


########################## ML API for Project screen ################################


@app.route('/createproject', methods=['GET', 'POST'])
def CreateNewProject():
    return createnewproject(request.json)
    # return project_setup()


@app.route('/getallprojects', methods=['GET'])
def GetAllProjects():
    return getallprojects()


@app.route('/getprojectmetadata', methods=['GET'])
def GetProjectMetaData():
    return getprojectmetadata()


@app.route('/saveprojectstepdata', methods=['GET', 'POST'])
def SaveProjectStepData():
    data = {'form': request.form.to_dict(), 'files': list(
        request.files.to_dict().values())}
    return saveprojectstepdata(data)


@app.route('/loadprojectstepsdata', methods=['GET', 'POST'])
def LoadProjectStepsData():
    data = {'form': request.form.to_dict(), 'files': list(
        request.files.to_dict().values())}
    return loadprojectstepsdata(data)


@app.route('/savetraintagsforproject', methods=['GET', 'POST'])
def SaveTrainTagsForProject():
    return saveprojecttraintags(request.form.to_dict())


@app.route('/runtrainingforproject', methods=['GET', 'POST'])
def RunTrainingForProject():
    return runtrainingforproject(request.form.to_dict())


@app.route('/runinferencingforproject', methods=['GET', 'POST'])
def RunInferencingForProject():
    return runinferencingforproject(request.form.to_dict())


@app.route('/deleteprojectimages', methods=['GET', 'POST'])
def DeleteProjectImages():
    return deleteprojectimages(request.form.to_dict())


@app.route('/deleteprojectiterations', methods=['GET', 'POST'])
def DeleteProjectIterations():
    return deleteprojectiterations(request.form.to_dict())


@app.route('/deleteproject', methods=['GET', 'POST'])
def DeleteProject():
    return deleteproject(request.form.to_dict())


@app.route('/loadalliterations', methods=['GET', 'POST'])
def LoadAllProjectIterations():
    return loadallprojectiters(request.form.to_dict())


########################## ML API for Process Tracking ##############################

@app.route('/loadallprocessvideos', methods=['GET', 'POST'])
def LoadAllProcessVideos():
    return loadallprocessvideos(request.form.to_dict())


########################## ML API for Image Augmentation ############################


@app.route('/augmentimages', methods=['GET', 'POST'])
def RunImageAugmentation():
    data = {'form': request.form.to_dict(), 'files': list(
        request.files.to_dict().values())}
    return runImageAugmentation(data)


@app.route('/loadaugmentationdata', methods=['GET', 'POST'])
def LoadAugmentationData():
    data = {'form': request.form.to_dict(), 'files': list(
        request.files.to_dict().values())}
    clearUnsavedAugmentation(data)
    return loadAugmentationData(data)


@app.route('/loadallresourcesaug', methods=['GET', 'POST'])
def LoadAllResourcesAug():
    return loadallresourcesaug(request.form.to_dict())


@app.route('/loadallresourcesurl', methods=['GET', 'POST'])
def LoadAllResources():
    return loadallresources(request.form.to_dict())


@app.route('/saveaugresultlabel', methods=['GET', 'POST'])
def SaveAugResultsLabel():
    data = {'form': request.form.to_dict()}
    return saveAugResultsLabel(data)


@app.route('/savefilesforaugmentation', methods=['GET', 'POST'])
def SaveFileForAug():
    data = {'form': request.form.to_dict(), 'files': list(
        request.files.to_dict().values())}
    return saveFilesForAugmentation(data)


########################## ML API for prediction #####################################


@app.route('/detectsurfacedefects', methods=['GET', 'POST'])
def DetectSurfaceDefects():
    print('detectsurfacedefects files', request.files.to_dict())
    response = predictfrommodel(
        'Surface Defects', list(request.files.to_dict().values()))
    data = {'form': request.form.to_dict(), 'files': list(
        request.files.to_dict().values()), 'response': response}
    saveDataUseCaseModels(data)
    return response


@app.route('/detectmetalcastdefects', methods=['GET', 'POST'])
def DetectMetalCastDefects():
    print('files', request.files.to_dict())
    response = predictfrommodel(
        'Metal Casting Defects', list(request.files.to_dict().values()))
    data = {'form': request.form.to_dict(), 'files': list(
        request.files.to_dict().values()), 'response': response}
    saveDataUseCaseModels(data)
    return response


@app.route('/detecthardhatpresent', methods=['GET', 'POST'])
def DetectHardHatPresent():
    print('files', request.files.to_dict())
    response = predictfrommodel(
        'Hard Hat Present', list(request.files.to_dict().values()))
    data = {'form': request.form.to_dict(), 'files': list(
        request.files.to_dict().values()), 'response': response}
    saveDataUseCaseModels(data)
    return response


@app.route('/detectsteeldefects', methods=['GET', 'POST'])
def DetectSteelDefectPresent():
    print('files', request.files.to_dict())
    return predictfrommodel('Steel Defects', list(request.files.to_dict().values()))


@app.route('/packagedamagedetection', methods=['GET', 'POST'])
def PackagingInspection():
    print('files', request.files.to_dict())
    return predictfrommodel('Package Damage Detection', list(request.files.to_dict().values()))


@app.route('/inferenceforusermodel', methods=['GET', 'POST'])
def InferenceForUserModel():
    data = {'form': request.form.to_dict(), 'files': list(
        request.files.to_dict().values())}
    return inferenceWithUserModel(data)


@app.route('/savepredictionfiles', methods=['GET', 'POST'])
def SavePredictioDataFiles():
    data = {'form': request.form.to_dict(), 'files': list(
        request.files.to_dict().values())}
    return savePredictionFiles(data)


@app.route('/loaddraftpreddata', methods=['GET', 'POST'])
def LoadDraftPredictionData():
    data = {'form': request.form.to_dict()}
    return loadDraftPredictionId(data)


@app.route('/getfile/<path:image_name>', methods=['GET', 'POST'])
def Get_Files(image_name):
    print('image_name', image_name)
    try:
        return send_from_directory(os.path.join(os.getcwd(), 'tmp'), image_name, as_attachment=False)
    except FileNotFoundError:
        app.abort(404)


@app.route('/getallfile/<path:image_name>', methods=['GET', 'POST'])
def GetAllFiles(image_name):
    try:
        return send_from_directory(os.path.join(os.getcwd(), 'uploads'), image_name, as_attachment=False)
    except FileNotFoundError:
        app.abort(404)


@app.route('/getS3Url', methods=['GET', 'POST'])
def GetS3UrlForFile():
    data = request.json
    return getSavedS3Url(data['fileName'])


@app.route('/getallavlmodels', methods=['GET', 'POST'])
def GetAllAvailableModels():
    data = request.json
    return getallavlmodels(data)


@app.route('/loadpredictionhistory', methods=['GET', 'POST'])
def LoadPredictionHistory():
    data = request.json
    return loadpredictionhistory(data)


@app.route('/removepredictionhistory', methods=['GET', 'POST'])
def RemovePredictionHistory():
    data = request.json
    return deletepredictionhistory(data)

########################## Check GPU ###############################################


@app.route('/gpucheck', methods=['GET', 'POST'])
def CheckAvailableGPU():
    data = request.json
    return check_available_gpus(data)


########################## Main Program Begins #####################################
if __name__ == "__main__":
    app.run(debug=True, port='5002')
