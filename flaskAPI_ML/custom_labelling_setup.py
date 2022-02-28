from sqlite_db_connect import SQLiteConnect
import os
from datetime import datetime
import json

now = datetime.now().strftime('%Y-%M-%d %H:%M:%S')


class CustomLabelling(SQLiteConnect):
    def ___init__(self):
        super.__init__()


def loadDraftCustomLabelId(data):
    labller = CustomLabelling()
    labelId = -1
    resourceId = -1
    response = {'success': 'Data retrieved', 'results': []}
    tmp1 = labller.getRecords(
        'select id,ResourceId from CUSTOM_LABELLING_HISTORY where userId={user} and status is null'.format(user=data['form']['userId']))
    if tmp1:
        labelId = str(tmp1[0]['id'])
        resourceId = str(tmp1[0]['ResourceId'])
    results = {'labelId': labelId, 'resourceId': resourceId}
    response['results'] = results
    return response


def loadCustomLabelData(data):

    labeller = CustomLabelling()
    response = {'success': 'Data retrieved', 'results': []}
    select = 'select rd.FilePath,rd.FileName,rd.id from RESOURCE_DETAILS rd where rd.resourceId={id} '
    resp = labeller.getRecords(select.format(
        id=data['form']['resourceId']))
    response['results'] = [
        {'name': r['FileName'], 'path':r['FilePath'], 'id':r['id']} for r in resp]
    return response


def saveFilesForCustomLabelling(data):
    labeller = CustomLabelling()
    response = {'success': 'Data retrieved', 'results': []}
    workdir = os.getcwd()
    resourceId = data['form']['resourceId']
    formData = data['form']
    labelId = formData['labelId']
    if formData['labelId'] == '-1':
        tmp1 = labeller.getRecords(
            'select id,ResourceId from CUSTOM_LABELLING_HISTORY where userId={user} and status is null'.format(user=formData['userId']))
        print('tmp1', tmp1)
        if not tmp1:
            insert = 'insert into ALL_RESOURCES(Name,Description) values ("{name}","{desc}")'.format(
                name=data['form']['name'], desc=data['form']['desc'])
            resourceId = labeller.executeQuery('insert', insert)
            insert = 'insert into CUSTOM_LABELLING_HISTORY(ResourceId,labelledOn,userId) values ("{resource}","{now}","{user}")'.format(
                resource=resourceId, now=now, user=formData['userId'])
            labelId = str(labeller.executeQuery('insert', insert))
        else:
            labelId = str(tmp1[0]['id'])
            resourceId = str(tmp1[0]['ResourceId'])

    if not os.getcwd().endswith('uploads'):
        os.chdir('uploads')

    if not os.path.isdir('CustomLabel'):
        os.mkdir('CustomLabel')

    if not os.path.isdir(os.path.join('CustomLabel', labelId)):
        os.mkdir(os.path.join('CustomLabel', labelId))

    for file in data['files']:
        path = os.path.join(
            "CustomLabel", labelId, file.filename)
        file.save(path)
        response['results'].append(path)

    os.chdir(workdir)

    insertdata = []
    for i, path in enumerate(response['results']):
        insertdata.append("({id},{no},'{name}','{path}','{now}')".format(
            id=resourceId, no=i, name=os.path.split(path)[-1], path=path, now=now))

    insertdata = ','.join(insertdata)
    insert = 'insert into RESOURCE_DETAILS (ResourceId,FileNo,FileName,FilePath,UpdatedOn) values {insertData}'.format(
        insertData=insertdata)
    labeller.executeQuery('insert', insert)
    print('inser2 ->', insert)

    data['form']['resourceId'] = resourceId
    data['form']['labelId'] = labelId
    data['form']['numberOfFiles'] = str(len(data['files']))
    response['form'] = data['form']
    response['results'] = loadCustomLabelData(data)['results']
    print('resp', response)
    return response


def saveCustomLabelForImages(data):
    print('data', data)
    lines = []
    labeller = CustomLabelling()
    response = {'success': 'Data retrieved', 'results': []}
    formData = data['form']
    labelId = formData['labelId']
    resourceDetailId = formData['resourceDetailId']
    fname = formData['fname']
    annotation_path = os.path.join(
        os.getcwd(), 'uploads', 'CustomLabel', str(labelId), fname+'.txt')
    with open(annotation_path, 'w') as f:
        for d in json.loads(formData['canvasItems']):
            line = " ".join([str(l) for l in list(d.values())])+"\n"
            lines.append(line)
        lines = list(set(lines))
        f.writelines(lines)
        labeller.executeQuery('update', 'update RESOURCE_DETAILS set Annotation="{annot}" where id ={detailId}'.format(
            annot=annotation_path, detailId=resourceDetailId))
        labeller.executeQuery('update', 'update ALL_RESOURCES set Description="{desc}",Name="{desc}" where id ={resourceId}'.format(
            desc=formData['desc'], resourceId=formData['resourceId']))
    return response
