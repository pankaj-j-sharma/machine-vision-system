from image_classifier import *
from sqlite_db_connect import SQLiteConnect
from datetime import datetime
import os
import shutil


now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class Project(SQLiteConnect):
    def ___init__(self):
        super.__init__()

    def createnew(self, data):
        result = None
        resourceId = data['resourceType']

        if 'projectId' in data and data['projectId'] != '-1':
            print('my data', data)
            if data['resourceType'] == 0:
                resourceName = data['projectName']+" " + \
                    datetime.now().strftime('%Y-%M-%d %H:%M:%S')
                insert = 'insert into ALL_RESOURCES (Name,Description,ProjectId) values ("{name}","{desc}","{project}")'
                insert = insert.format(name=resourceName, desc="resource created for " +
                                       data['projectName'], project=data['projectId'])
                resourceId = self.executeQuery('update', insert)

            update = 'update ALL_PROJECTS set projectName="{name}",description="{desc}",resource="{resource}",type="{type}",ExportOption="{export}",UpdatedOn="{update}",thumbnail="{thumbnail}" where id={id}'
            update = update.format(id=data['projectId'], name=data['projectName'], desc=data['projectDesc'], resource=resourceId, type=data['projectType'],
                                   export=data['exportOption'], update=datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), thumbnail=data['thumbnail'])
            newProjectId = self.executeQuery('update', update)
            result = {'success': 'Project updated successfully',
                      'newProjectId': data['projectId']}

        else:
            if not self.getRecords("select projectName from ALL_PROJECTS where projectName='{}'".format(data['projectName'])):
                insert = "insert into ALL_PROJECTS(projectName,description,resource,type,ExportOption,UpdatedOn,thumbnail) "
                insert += 'values("{name}","{desc}","{resource}","{type}","{export}","{update}","{thumbnail}")'
                insert = insert.format(name=data['projectName'], desc=data['projectDesc'], resource=data['resourceType'], type=data['projectType'],
                                       export=data['exportOption'], update=datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), thumbnail=data['thumbnail'])
                print('insert', insert, 'data', data)
                newProjectId = self.executeQuery('insert', insert)
                result = {'success': 'Project created successfully',
                          'newProjectId': newProjectId}
            else:
                result = {'error': 'A project with same name already exists'}

        result = result if not self.dbexec_error else {
            'error': self.dbexec_error}
        return result

    def getall(self):
        data = self.getRecords("select * from ALL_PROJECTS ")
        return {'success': 'Data retrieved', 'results': data}

    def getmetadata(self):
        results = {}
        results['exportOptions'] = self.getRecords(
            "select * from EXPORT_OPTIONS ")
        results['projectTypes'] = self.getRecords(
            "select * from PROJECT_TYPES ")
        results['allResources'] = self.getRecords(
            "select * from ALL_RESOURCES ")
        return {'success': 'Data retrieved', 'results': results}

    def savetraining(self, data):
        print('data', data)
        results = data['results'][0]
        insert = 'insert into ALL_MODELS (name, createdOn, status, filePath, labelPath, f1, precision, recall) VALUES '
        insert += '("{name}", "{now}", "ACTIVE", "{file}", "{label}", "{f1}", "{precision}", "{recall}")'.format(
            name=results['model_name'], now=now, file=results['model_file_path'], label=results['encoder_file'], f1=results['f1'], precision=results['precision'], recall=results['recall'])
        model_id = self.executeQuery('insert', insert)

        insert = 'insert into ALL_ITERATIONS (projectId, modelId , UpdatedOn) VALUES '
        insert += '("{project}","{model}","{now}")'.format(
            project=data['projectId'], model=model_id, now=now)
        iter_id = self.executeQuery('insert', insert)


def createnewproject(data):
    project = Project()
    return project.createnew(data)


def getallprojects():
    project = Project()
    return project.getall()


def getprojectmetadata():
    project = Project()
    return project.getmetadata()


def saveprojectstepdata(data):
    project = Project()

    print('debug data', data)
    response = {'success': 'Data retrieved', 'results': []}

    workdir = os.getcwd()
    formData = data['form']
    if not os.getcwd().endswith('uploads'):
        os.chdir('uploads')

    if not os.path.isdir('Projects'):
        os.mkdir('Projects')

    if not os.path.isdir(os.path.join('Projects', formData['projectId'])):
        os.mkdir(os.path.join('Projects', formData['projectId']))

    if not os.path.isdir(os.path.join('Projects', formData['projectId'], formData['projectStep'])):
        os.mkdir(os.path.join(
            'Projects', formData['projectId'], formData['projectStep']))

    for file in data['files']:
        path = os.path.join(
            "Projects", formData['projectId'], formData['projectStep'], file.filename)
        file.save(path)
        response['results'].append(path)

    os.chdir(workdir)

    if formData['resourceId'] == '0':
        tmp = project.getRecords('select Resource from ALL_PROJECTS where id ={project}'.format(
            project=formData['projectId']))
        if tmp:
            tmp = tmp[0]
            formData['resourceId'] = tmp['Resource']

    fileNo = project.getRecords('select count(1) "no" from RESOURCE_DETAILS where ResourceId={resource} and stepno={step}'.format(
        resource=formData['resourceId'], step=formData['projectStep']))
    fileNo = fileNo[0]["no"] if fileNo else 0

    insertdata = []
    for i, path in enumerate(response['results']):
        insertdata.append("({id},{no},'{name}','{path}','{now}',{step})".format(
            id=formData['resourceId'], no=fileNo+i, name=os.path.split(path)[-1], path=path, now=now, step=formData['projectStep']))

    insertdata = ','.join(insertdata)
    insert = 'insert into RESOURCE_DETAILS (ResourceId,FileNo,FileName,FilePath,UpdatedOn,stepno) values {insertData}'.format(
        insertData=insertdata)
    project.executeQuery('insert', insert)
    data['form']['projectStepNo'] = data['form']['projectStep']
    return loadprojectstepsdata(data)


def loadprojectstepsdata(data):

    project = Project()
    response = {'success': 'Data retrieved', 'results': []}

    select = 'select rd.* from RESOURCE_DETAILS rd INNER join ALL_PROJECTS ap on rd.resourceId = ap.Resource where ap.Id={id} and rd.stepno={step}'
    resp = project.getRecords(select.format(
        id=data['form']['projectId'], step=data['form']['projectStepNo']))
    response['results'] = [{'name': r['FileName'], 'id':r['id'],
                            'no':r['FileNo'], 'path':r['FilePath'], 'tag': r['Tag'], 'updated': r['UpdatedOn']} for r in resp]

    # workdir = os.getcwd()
    # if not os.getcwd().endswith('uploads'):
    #     os.chdir('uploads')
    # for root, dirs, files in os.walk(os.path.join('Projects', data['form']['projectId'], data['form']['projectStepNo']), topdown=False):
    #     for name in files:
    #         path = os.path.join(root, name)
    #         response['results'].append(path)
    #         print('path', path)

    # os.chdir(workdir)
    return response


def saveprojecttraintags(data):
    project = Project()
    print('data', data)

    tag = data['tagname']
    files = data['selectedFiles']
    resource = data['resourceId']

    update = "update RESOURCE_DETAILS set tag = '{tag}' where ResourceId={resource} and FileNo in ({selectedFiles}) "
    update = update.format(resource=resource, tag=tag, selectedFiles=files)
    print('update ->', update)
    project.executeQuery('update', update)

    return data


def runtrainingforproject(data):
    result = None
    project = Project()
    print('train for ', data['resourceId'])
    select = "select projectName, (select count(1) from ALL_ITERATIONS where projectId = p.id) [iters] from ALL_PROJECTS p where id ={project}".format(
        project=data['projectId'])
    tmp = project.getRecords(select)
    if tmp:
        tmp = tmp[0]
        data['projectName'] = tmp['projectName']
        data['projectIter'] = tmp['iters']

    response = {**start_training(data), **data}
    project.savetraining(response)
    return response


def runinferencingforproject(data):
    project = Project()
    response = {'success': 'Data retrieved',
                'results': []}
    select = 'select * from ALL_ITERATIONS i inner join ALL_MODELS m on i.modelId = m.id where i.id = {iteration} and i.projectId={project}'.format(
        iteration=data['iterationId'], project=data['projectId'])
    tmp = project.getRecords(select)
    if tmp:
        tmp = tmp[0]
    print('tmp', tmp)
    data['modelFilePath'] = tmp['filePath']
    data['labelFilePath'] = tmp['labelPath']

    response['results'] = start_inferencing(data)
    print('data for inference', data, 'response ', response)
    return response


def deleteprojectimages(data):
    project = Project()
    projectId = data['projectId']
    projectStepNo = data['projectStepNo']
    delete = 'delete from RESOURCE_DETAILS where ResourceId in ( select resource from ALL_PROJECTS where id = {id}) and stepno = {step}'.format(
        id=projectId, step=projectStepNo)
    project.executeQuery('delete', delete)
    return data


def deleteprojectiterations(data):
    project = Project()
    response = {'success': 'Data deleted',
                'results': []}
    projectId = data['projectId']
    select = 'select id,modelId from ALL_ITERATIONS where projectId={project}'.format(
        project=projectId)
    tmp = project.getRecords(select)
    iterIds = ''
    modelIds = ''
    if tmp:
        iterIds = ','.join([str(t['id']) for t in tmp])
        modelIds = ','.join([str(t['modelId']) for t in tmp])

    delete = 'delete from ALL_ITERATIONS where id in ({iters})'.format(
        iters=iterIds)
    project.executeQuery('delete', delete)

    delete = 'delete from ALL_MODELS where id in ({modelIds})'.format(
        modelIds=modelIds)
    project.executeQuery('delete', delete)
    return response


def deleteproject(data):
    project = Project()
    response = {'success': 'Data deleted',
                'results': []}
    projectId = data['projectId']
    delete = 'delete from RESOURCE_DETAILS rd WHERE rd.resourceId in (select Resource from ALL_PROJECTS where id = {project} )'.format(
        project=projectId)
    project.executeQuery('delete', delete)
    delete = 'delete from ALL_RESOURCES ar WHERE ar.Id in (select Resource from ALL_PROJECTS where id = {project} )'.format(
        project=projectId)
    project.executeQuery('delete', delete)
    delete = 'delete from ALL_PROJECTS where id = {project}'.format(
        project=projectId)
    project.executeQuery('delete', delete)
    # remove project directory
    try:
        shutil.rmtree(os.path.join(os.getcwd(), 'uploads',
                      'Projects', str(projectId)))
    except Exception as e:
        print('error', str(e))
    return response


def loadallprojectiters(data):
    project = Project()
    response = {'success': 'Data retrieved', 'results': []}
    select = 'select i.id,m.name,i.UpdatedOn,m.f1,m.precision,m.recall,m.filePath,m.labelPath  from ALL_ITERATIONS i inner join ALL_MODELS m on i.modelId = m.id where i.projectId={project}'.format(
        project=data['projectId'])
    results = project.getRecords(select)
    response['results'] = results
    print('respo->', response)
    return response


def loadallresources(data):
    project = Project()
    print('data', data)
    response = {'success': 'Data retrieved', 'results': []}
    results = project.getRecords(
        'select * from ALL_RESOURCES where ProjectId is null and name not like "mymodel pred%"')
    response['results'] = results
    return response
