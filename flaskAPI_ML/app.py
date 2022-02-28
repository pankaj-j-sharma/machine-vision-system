import pickle
from difflib import SequenceMatcher
import os
import numpy as np
import pandas as pd
from flask import Flask, request
from flask import jsonify
from flask_cors import CORS, cross_origin
import json
from prediction_facade import predictfrommodel
from flask import Flask, request, send_from_directory


app = Flask(__name__)
CORS(app)

localDbData={}
authToken = None

@app.route('/test', methods = ["GET","POST"])
def Test():
    return {'data':'hi this is working'}


@app.route('/login', methods = ["GET","POST"])
def LogIn():
    data = request.json
    return __checkInLocalDb(data['username'],data['password'])

def __checkInLocalDb(user,passwd):
    userData = None
    with open('localDb.json') as r:
        localDbData=json.load(r)
    for d in localDbData['loginData']:
        if d["username"] == user and d["passwd"] == passwd:
            userData = d
            authToken = userData["token"]
    return {'token':authToken,'user':userData}


@app.route('/validatetoken', methods = ["GET"])
def ValidateToken(token):
    data = request.json
    return authToken==token


@app.route('/logout', methods = ["GET","POST"])
def LogOut():
    data = request.json
    authToken=None
    return {'token':authToken}


@app.route('/repairorderpredict', methods = ["GET","POST"])
def RepairOrderPrediction():
    data = request.json
    txt_challenge = None
    txt_accept = None

    if request.method == "POST":
        part = data['PartNo']
        SrNo = data['SerialNo']
        flatRate = data['FlatRate']
        manHourCost = data['ManHourCost']
        workToBePerformed = data["WorkToBePerformed"]
        components = componentList(workToBePerformed)
        componentsPrice = ComponetPrice(components)
        totalHour = componentTotalHours(components)
        TotalQuotationPrice = 0
        if float(componentsPrice[1]) != None:
            TotalQuotationPrice += float(flatRate) + (float(manHourCost) * float(totalHour)) + float(componentsPrice[1])
        else:
            TotalQuotationPrice += float(flatRate) + (float(manHourCost) * float(totalHour))
        partMean = 0
        avpp = avgPartPrice(part)
        if len(componentsPrice[0].keys()) > 5:
            partMean += round(avpp + ((len(componentsPrice[0].keys()) - 5) * 1200), 2)
        else:
            partMean += round(avpp, 2)

        maxPrice = (partMean + partMean * (10 / 100))
        if maxPrice > TotalQuotationPrice:
            txt_accept = "Accepted"
        else:
            txt_challenge = "Challenged"
        return jsonify( PartNo=part, SrNo=SrNo, FlatRate=flatRate, \
                               WorkToBePerformed=workToBePerformed, Components=components, ComponentPrice=componentsPrice[0], TotalQuotationPrice=TotalQuotationPrice,
                               PartMean=partMean, Result=txt_accept if txt_accept else txt_challenge,
                               TotalHour=totalHour)


@app.route('/historicalrepairsanalysis', methods=['POST'])
def HistoricalRepairAnalysis():
    data = request.json
    result = {}
    if request.method == 'POST':
        input_val = data['SerialNo']
        df = pd.read_excel("FCU_SFRs_Feb_2021.xlsx")
        df =df[df['Serial No.'] == input_val]
        for col in df.columns:
            name = col.replace(" ","").replace(".","")
            result[name] = str(df[col].values[0]) if not df[col].isnull().values.any() else ""
        return result


@app.route('/engineoverhaultimecost', methods=['POST'])
def EngineOverhaulTimeCostPredict():
    data = request.json
    if request.method == "POST":
        engine_part = data['PartNo']
        manHourCost = data['ManHourCost']
        Overhaul_to_Performed = data['OverhaulToBePerformed']
        components = Overhaul_to_Performed.split(",")

        dollar_list = []
        hour_list = []
        minutes_list = []
        seconds_list = []

        for item in components:
            predictedCost = engine_overhaul_cost_predict_model.predict([item])
            predictedHour = engine_overhaul_hour_predict_model.predict([item])
            predictedMin = engine_overhaul_min_predict_model.predict([item])
            predictedSec = engine_overhaul_sec_predict_model.predict([item])

            dollar_list.append(predictedCost[0])
            hour_list.append(predictedHour[0])
            minutes_list.append(predictedMin[0])
            seconds_list.append(predictedSec[0])

    return jsonify(EngineOverhaulCost='{}'.format(sum(dollar_list)),
                   SumHour='{}'.format(sum(hour_list)),
                   SumMinutes='{}'.format(sum(minutes_list)),
                   SumSeconds='{}'.format(sum(seconds_list)))


@app.route('/warrantyclaimsprediction', methods=['POST'])
def WarrantyClaimsPrediction():
    data = request.json
    if request.method == "POST":
        select_engine = int(data['Engine'])
        select_engine_type = data['EngineType']
        issue_count = int(data['IssueCount'])
        call_detail = int(data['CallDetails'])
        part_age = int(data['PartAge'])

        predicted_claim_gb = []
        if select_engine_type == "Turbofan1":
            predicted_claim_gb = model_warranty_claim_predict_GB.predict(
                [[0, select_engine, 0, 0, 0, call_detail, part_age, issue_count, 0]])
        elif select_engine_type == "Turbofan2":
            predicted_claim_gb = model_warranty_claim_predict_GB.predict(
                [[0, select_engine, 0, 0, 0, call_detail, part_age, 0, issue_count]])
        elif select_engine_type == "Turbofan3":
            predicted_claim_gb = model_warranty_claim_predict_GB.predict(
                [[issue_count, select_engine, 0, 0, 0, call_detail, part_age, 0, 0]])
        elif select_engine_type == "Turbojet1":
            predicted_claim_gb = model_warranty_claim_predict_GB.predict(
                [[0, select_engine, issue_count, 0, 0, call_detail, part_age, 0, 0]])
        elif select_engine_type == "Turbojet2":
            predicted_claim_gb = model_warranty_claim_predict_GB.predict(
                [[0, select_engine, 0, 0, issue_count, call_detail, part_age, 0, 0]])
        elif select_engine_type == "Turbojet3":
            predicted_claim_gb = model_warranty_claim_predict_GB.predict(
                [[0, select_engine, 0, issue_count, 0, call_detail, part_age, 0, 0]])

        predicted_claim_gb = round(predicted_claim_gb[0], 2)
    return jsonify( PredictedClaimValue='{}'.format(predicted_claim_gb))


@app.route('/engineoverhaul', methods=['GET','POST'])
def EngineOverhaulTimeCosts():
    df = pd.read_excel("Engine_Overhaul_component_List.xlsx")
    df["Price"] = df["Price"].round(decimals=0).astype(int)
    df["Component"] = df["Component"].apply(lambda s : s.upper().split(" ")[0])
    df= df[["Component","Price","Hours"]].groupby(["Component"]).agg('sum')
    df = df.sort_values(by=['Price'], ascending=False)
    print(df)
    return df.to_json()


@app.route('/vendoraggquotation', methods=['GET','POST'])
def VendorBasedAggQuotation():
    df = pd.read_excel("quotation.xlsx")
    # df= df.groupby(["Vendor"])["Quotation Price"].agg('sum')
    df= df[["Vendor","Quotation Price"]].groupby(["Vendor"]).agg('sum')
    df['Quotation Price'] = df["Quotation Price"].round(decimals=0).astype(int)
    return df.to_json()


@app.route('/vendoravgrepaircost', methods=['GET','POST'])
def VendorAvgRepairCost():
    df = pd.read_excel("FCU_SFRs_Feb_2021.xlsx")
    df['Vendor Name'] = df['Vendor Name'].str.replace('[#,@,&]', '')
    df= df[["Vendor Name","Repair Cost"]].groupby(["Vendor Name"]).agg('mean')
    df['Repair Cost'] = df["Repair Cost"].round(decimals=0).astype(int)
    return df.to_json()



########################## ML API for prediction #####################################

@app.route('/detectsurfacedefects', methods=['GET','POST'])
def DetectSurfaceDefects():
    print('files',request.files.to_dict())
    files = ['test_images//In_100.bmp','test_images//Sc_103.bmp']
    return predictfrommodel('Surface Defects',list(request.files.to_dict().values()))


@app.route('/detectmetalcastdefects', methods=['GET','POST'])
def DetectMetalCastDefects():
    print('files',request.files.to_dict())
    return predictfrommodel('Metal Casting Defects',list(request.files.to_dict().values()))


@app.route('/detecthardhatpresent', methods=['GET','POST'])
def DetectHardHatPresent():
    print('files',request.files.to_dict())
    return predictfrommodel('Hard Hat Present',list(request.files.to_dict().values()))


@app.route('/detectsteeldefects', methods=['GET','POST'])
def DetectSteelDefectPresent():
    print('files',request.files.to_dict())
    return predictfrommodel('Steel Defects',list(request.files.to_dict().values()))


@app.route('/packagedamagedetection', methods=['GET','POST'])
def PackagingInspection():
    print('files',request.files.to_dict())
    return predictfrommodel('Package Damage Detection',list(request.files.to_dict().values()))


@app.route('/getfile/<path:image_name>', methods=['GET','POST'])
def Get_Files(image_name):  
    print('image_name',image_name)
    try:
        return send_from_directory(os.path.join(os.getcwd(),'tmp'), filename=image_name, as_attachment=False)
    except FileNotFoundError:
        app.abort(404)

#--------------------------------------------------------------------------------------------
# Utility functions to be written here

def componentList(workNeedToDo, threshold=0.35):
    compenentList = []

    if "." not in workNeedToDo:
        workNeedToDo = workNeedToDo.replace(",", ".")
    listOfWork = workNeedToDo.split(".")

    df1 = pd.read_excel("component.xlsx")
    listOfComponent = df1["Component"].to_list()
    for w in listOfWork:
        for c in listOfComponent:
            s = similar(w, c)
            if s > threshold:
                # print(c)
                compenentList.append(c)
    for c1 in listOfWork:
        if "knob" in c1.lower() or "knobs" in c1.lower():
            # print("F1482519 KNOB")
            compenentList.append("F1482519 KNOB")
    for c2 in listOfWork:
        if "battery" in c2.lower():
            # print("3B6880 BATTERY 3,6V")
            compenentList.append("3B6880 BATTERY 3,6V")
    compenentList = list(set(compenentList))
    return compenentList

def componentPrice(cmp):
    data = pd.read_excel('component.xlsx')
    price = data['Price'].loc[data['Component'].isin([cmp])]
    price = int(price)
    return price

def ComponetPrice(comp):
    total = 0
    cp = {}
    for i in comp:
        cp[i] = componentPrice(i)
        total += componentPrice(i)
    return cp, total

def componentTotalHours(cmp):
    total = 0
    for i in cmp:
        total += componentHour(i)
    return total

def avgPartPrice(part):
    df = pd.read_excel("quotation.xlsx")
    data = df.loc[df['Part No'] == part]
    return float(data[['Quotation Price']].mean(axis=0))

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def componentHour(cmp):
    data = pd.read_excel('component.xlsx')
    hours = data['Hours'].loc[data['Component'].isin([cmp])]
    hours = int(hours)
    return hours

engine_overhaul_cost_predict_model = None #pickle.load(open('engine_overhaul_cost_model.pkl','rb'))
engine_overhaul_hour_predict_model = None #pickle.load(open('engine_overhaul_hour_model.pkl','rb'))
engine_overhaul_min_predict_model = None #pickle.load(open('engine_overhaul_minute_model.pkl','rb'))
engine_overhaul_sec_predict_model = None #pickle.load(open('engine_overhaul_second_model.pkl','rb'))
claim_predict_model = None #pickle.load(open('regression_model_warranty_claims.pkl','rb'))
model_warranty_claim_predict_GB = None #pickle.load(open('model_warranty_claim_predict_GB.pkl','rb'))

if __name__ == "__main__":
    app.run(debug=True,port='5002')
