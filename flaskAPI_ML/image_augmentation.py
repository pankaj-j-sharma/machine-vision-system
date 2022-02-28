# !pip install imgaug
from distutils import filelist
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import glob
import os
import cv2
from PIL import Image
from imgaug.augmenters.meta import Augmenter
import numpy as np
import uuid
from logger import Logger
from sqlite_db_connect import SQLiteConnect
from datetime import datetime
import json
import shutil

now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class ImageAugmentation(SQLiteConnect):
    def ___init__(self):
        super.__init__()


def translate(img, x, y):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rows, cols, c = image.shape
    M = np.float32([[1, 0, x], [0, 1, y]])
    imgTranslated = cv2.warpAffine(image, M, (cols, rows))
    return imgTranslated


def resize(img, h, w):
    image = cv2.imread(img)
    imgResized = cv2.resize(image, (h, w))
    return imgResized


def rotate(img, x):
    image = cv2.imread(img)
    imgRotate = cv2.rotate(image, x)
    return imgRotate


def shear(img, x, y):
    image = imageio.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    shear = iaa.Affine(shear=(x, y))
    shear_image = shear.augment_image(image)
    return shear_image


def noise(img):
    image = imageio.imread(img)
    gaussian_noise = iaa.AdditiveGaussianNoise(0, 20)
    noise_image = gaussian_noise.augment_image(image)
    return noise_image


def crop(img, x, y):
    image = imageio.imread(img)
    crop = iaa.Crop(percent=(x, y))  # crop image
    crop_image = crop.augment_image(image)
    return crop_image


def flip(img):
    image = imageio.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    flip_hr = iaa.Fliplr(p=1.0)
    flip_hr_image = flip_hr.augment_image(image)
    return flip_hr_image


def bright(img):
    image = imageio.imread(img)
    contrast = iaa.GammaContrast(gamma=2.0)
    contrast_image = contrast.augment_image(image)
    return contrast_image


def scale(img):
    image = imageio.imread(img)
    scale_im = iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})
    scale_image = scale_im.augment_image(image)
    return scale_image


def send_img_or_url(image_np, sendUrl=True):
    img = Image.fromarray(image_np)
    if sendUrl:
        filename = str(uuid.uuid4())+".jpeg"
        cv2.imwrite(os.path.join('tmp', filename), image_np)
        img = filename
    return img

##########################################################################################


def runImageAugmentation(data):
    augmenter = ImageAugmentation()
    augmenter.log_message('run image augmentation ', 'data', data)
    response = {'success': 'Data retrieved', 'results': []}
    formData = data['form']
    augId = formData['augId']
    workdir = os.getcwd()

    try:
        resourceId = formData['resourceId']
        if augId == '-1':
            tmp1 = augmenter.getRecords(
                'select id,ResourceId from AUGMENTATION_HISTORY where userId={user} and status is null'.format(user=formData['userId']))
            print('tmp1', tmp1)
            if not tmp1:
                insert = 'insert into AUGMENTATION_HISTORY(ResourceId,UpdatedOn,userId) values ("{resource}","{now}","{user}")'.format(
                    resource=resourceId, now=now, user=formData['userId'])
                augId = str(augmenter.executeQuery('insert', insert))
            else:
                augId = str(tmp1[0]['id'])
                resourceId = str(tmp1[0]['ResourceId'])

        # introduces a bug if the below lines are added after the os.chdir commands
        data['form']['resourceId'] = resourceId
        data['form']['augId'] = augId
        fileList = loadAugmentationData(data)['results']

        if not os.getcwd().endswith('uploads'):
            os.chdir('uploads')

        if not os.path.isdir('AugmentedImages'):
            os.mkdir('AugmentedImages')

        if augId == -1 or resourceId == -1:
            return response  # exit the program if the aug id is still -1

        if not os.path.isdir(os.path.join('AugmentedImages', augId)):
            os.mkdir(os.path.join('AugmentedImages', augId))

        results = []
        insertdata = []

        sampleParam = 999
        rescaleParam = ""
        flipParam = ""
        translationParam = ""
        rotationParam = ""
        shearingParam = ""

        update_aug_params = 'update AUGMENTATION_HISTORY set samples="{sample}",rescale="{rescale}",flip="{flip}",translation="{translation}",rotation="{rotation}",shearing="{shearing}",UpdatedOn="{now}" where id={augId} '

        for file in fileList:
            tmp = {'id': file['id'], 'name': file['name'], 'path': file['path'], 'resize': '',
                   'translate': '', 'flip': '', 'rotate': '', 'shear': ''}

            if 'rescale' in formData and formData['rescale']:
                rescaleParam = formData['rescale']
                h = int(rescaleParam.split('*')[0])
                w = int(rescaleParam.split('*')[1])
                tmp['resize'] = os.path.join(
                    "AugmentedImages", str(augId), "resize-{name}".format(name=file['name']))
                print('rescale', tmp['resize'])
                cv2.imwrite(tmp['resize'], resize(file['path'], h, w))
            insertdata.append("({AugId},{ResDetailId},'{AugType}','{FilePath}','{UpdatedOn}')".format(
                AugId=augId, ResDetailId=file['id'], AugType="rescale", FilePath=tmp['resize'], UpdatedOn=now))

            if 'translation' in formData and formData['translation']:
                translationParam = formData['translation']
                h = int(translationParam.split('*')[0])
                w = int(translationParam.split('*')[1])
                tmp['translate'] = os.path.join(
                    "AugmentedImages", str(augId), "translate-{name}".format(name=file['name']))
                cv2.imwrite(tmp['translate'], resize(file['path'], h, w))
            insertdata.append("({AugId},{ResDetailId},'{AugType}','{FilePath}','{UpdatedOn}')".format(
                AugId=augId, ResDetailId=file['id'], AugType="translation", FilePath=tmp['translate'], UpdatedOn=now))

            if 'flip' in formData and formData['flip'] == 'Horizontal':
                flipParam = formData['flip']
                tmp['flip'] = os.path.join(
                    "AugmentedImages", str(augId), "flip-{name}".format(name=file['name']))
                cv2.imwrite(tmp['flip'], flip(file['path']))
            insertdata.append("({AugId},{ResDetailId},'{AugType}','{FilePath}','{UpdatedOn}')".format(
                AugId=augId, ResDetailId=file['id'], AugType="flip", FilePath=tmp['flip'], UpdatedOn=now))

            if 'rotation' in formData and formData['rotation'] == '90-Degree':
                rotationParam = formData['rotation']
                tmp['rotate'] = os.path.join(
                    "AugmentedImages", str(augId), "rotate-{name}".format(name=file['name']))
                cv2.imwrite(tmp['rotate'], rotate(
                    file['path'], cv2.ROTATE_90_CLOCKWISE))
            insertdata.append("({AugId},{ResDetailId},'{AugType}','{FilePath}','{UpdatedOn}')".format(
                AugId=augId, ResDetailId=file['id'], AugType="rotation", FilePath=tmp['rotate'], UpdatedOn=now))

            if 'shearing' in formData and formData['shearing'] == 'Yes':
                shearingParam = formData['shearing']
                tmp['shear'] = os.path.join(
                    "AugmentedImages", str(augId), "shear-{name}".format(name=file['name']))
                cv2.imwrite(tmp['shear'], shear(file['path'], 0, 40))
            insertdata.append("({AugId},{ResDetailId},'{AugType}','{FilePath}','{UpdatedOn}')".format(
                AugId=augId, ResDetailId=file['id'], AugType="shearing", FilePath=tmp['shear'], UpdatedOn=now))

            results.append(tmp)

        update_aug_params = update_aug_params.format(augId=augId, sample=sampleParam, rescale=rescaleParam, flip=flipParam,
                                                     translation=translationParam, rotation=rotationParam, shearing=shearingParam, now=now)
        augmenter.executeQuery('update', update_aug_params)

        os.chdir(workdir)
        response['results'] = results

        # save augmentation results
        insertdata = ','.join(insertdata)
        insert = 'insert into AUGMENTATION_RESULTS (AugId , ResDetailId, AugType, FilePath , UpdatedOn ) values {insertdata}'.format(
            insertdata=insertdata)

        augmenter.executeQuery('insert', insert)
    except Exception as e:
        os.chdir(workdir)
        print('error occurred', e)

    augmenter.log_message('run image augmentation ', 'response', response)
    return response


def run_augmentation(options, fileList):
    augmenter = ImageAugmentation()
    augmenter.log_message('run augmentation ', 'option',
                          options, 'fileList', filelist)

    response = {'success': 'Data retrieved', 'results': []}

    workdir = os.getcwd()

    try:
        if not os.getcwd().endswith('uploads'):
            os.chdir('uploads')

        if not os.path.isdir('AugmentedImages'):
            os.mkdir('AugmentedImages')

        results = []
        for file in fileList:
            file.save(file.filename)
            tmp = {'path': file.filename}

            if 'rescale' in options and options['rescale']:
                h = int(options['rescale'].split('*')[0])
                w = int(options['rescale'].split('*')[1])
                tmp['resize'] = os.path.join(
                    "AugmentedImages", "resize-{name}".format(name=file.filename))
                cv2.imwrite(tmp['resize'], resize(file.filename, h, w))

            if 'translation' in options and options['translation']:
                h = int(options['translation'].split('*')[0])
                w = int(options['translation'].split('*')[1])
                tmp['translate'] = os.path.join(
                    "AugmentedImages", "translate-{name}".format(name=file.filename))
                cv2.imwrite(tmp['translate'], resize(file.filename, h, w))

            if 'flip' in options and options['flip'] == 'Horizontal':
                tmp['flip'] = os.path.join(
                    "AugmentedImages", "flip-{name}".format(name=file.filename))
                cv2.imwrite(tmp['flip'], flip(file.filename))

            if 'rotation' in options and options['rotation'] == '90-Degree':
                tmp['rotate'] = os.path.join(
                    "AugmentedImages", "rotate-{name}".format(name=file.filename))
                cv2.imwrite(tmp['rotate'], rotate(
                    file.filename, cv2.ROTATE_90_CLOCKWISE))

            if 'shearing' in options and options['shearing'] == 'Yes':
                tmp['shear'] = os.path.join(
                    "AugmentedImages", "shear-{name}".format(name=file.filename))
                cv2.imwrite(tmp['shear'], shear(file.filename, 0, 40))

            results.append(tmp)
        os.chdir(workdir)
        response['results'] = results
        clearUnsavedAugmentation(response)

    except Exception as e:
        os.chdir(workdir)
        print('error occurred', e)

    print('response', response)
    return response


def loadallresourcesaug(data):
    augmenter = ImageAugmentation()
    response = {'success': 'Data retrieved', 'results': {'all_resources': []}}
    results = augmenter.getRecords(
        'select id,Name from ALL_RESOURCES where ProjectId is null and name not like "mymodel pred%"')
    response['results']['all_resources'] = results
    return response


def saveFilesForAugmentation(data):
    augmenter = ImageAugmentation()
    # clearUnsavedAugmentation(data)
    augmenter.log_message('save files for augmentation', 'data', data)
    response = {'success': 'Data retrieved', 'results': {
        'augId': None, 'files': [], 'resId': None}}

    try:
        formData = data['form']
        resourceId = formData['resourceId']
        augId = formData['augId']
        if formData['augId'] == '-1':
            insert = 'insert into ALL_RESOURCES(Name,Description) values ("{name}","{desc}")'.format(
                name=data['form']['name'], desc=data['form']['desc'])
            resourceId = augmenter.executeQuery('insert', insert)
            insert = 'insert into AUGMENTATION_HISTORY(ResourceId,UpdatedOn,userId) values ("{resource}","{now}","{user}")'.format(
                resource=resourceId, now=now, user=formData['userId'])
            augId = str(augmenter.executeQuery('insert', insert))
        else:
            tmp2 = augmenter.getRecords(
                'select id,ResourceId from AUGMENTATION_HISTORY where id={augId}'.format(augId=augId))
            if tmp2:
                resourceId = str(tmp2[0]['ResourceId'])

        response['results']['augId'] = augId
        response['results']['resId'] = resourceId
        filePaths = []

        pathList = [os.getcwd(), 'uploads', 'AugmentedImages', augId]
        for i, _ in enumerate(pathList):
            if not os.path.isdir(os.path.join(*pathList[:i+1])):
                os.mkdir(os.path.join(*pathList[:i+1]))
            augmentation_data_path = os.path.join(*pathList[:i+1])

        for file in data['files']:
            path = os.path.join(augmentation_data_path, file.filename)
            file.save(path)
            # store only relative path
            relatve_path = os.path.join(os.getcwd(), 'uploads', '')
            path = path.replace(relatve_path, '')
            filePaths.append(path)

        fileNo = augmenter.getRecords('select count(1) "no" from RESOURCE_DETAILS where ResourceId={resource} and stepno is null'.format(
            resource=formData['resourceId']))
        fileNo = fileNo[0]["no"] if fileNo else 0

        insertdata = []
        for i, path in enumerate(filePaths):
            insertdata.append("({id},{no},'{name}','{path}','{now}')".format(
                id=resourceId, no=fileNo+i, name=os.path.split(path)[-1], path=path, now=now))

        insertdata = ','.join(insertdata)
        insert = 'insert into RESOURCE_DETAILS (ResourceId,FileNo,FileName,FilePath,UpdatedOn) values {insertData}'.format(
            insertData=insertdata)
        augmenter.executeQuery('insert', insert)
        augmenter.log_message('insert ', insert)

        response['results']['files'] = augmenter.getRecords(
            'select FileNo as no,FileName as name,FilePath path,Tag as result,UpdatedOn as updated from RESOURCE_DETAILS where ResourceId={resId}'.format(resId=resourceId))
        response['results']['all_resources'] = loadallresourcesaug(data)[
            'results']['all_resources']
    except Exception as e:
        augmenter.log_message('error occurred in save files', str(e))
    finally:
        augmenter.log_message(
            'save files for augmentation', 'response', response)
        return response


def saveAugResultsLabel(data):
    print('saveresult label', data)

    augmenter = ImageAugmentation()
    response = {'success': 'Data retrieved', 'results': []}
    formData = data['form']
    augId = formData['augId']
    augDesc = formData['desc']
    resourceId = formData['resourceId']
    if augId != -1 and resourceId != -1:
        augmenter.executeQuery('update', 'update ALL_RESOURCES set name="{desc}",description ="{desc}" where id = {resource}'.format(
            desc=augDesc, resource=resourceId))
        augmenter.executeQuery(
            'update', 'update AUGMENTATION_HISTORY set status="SAVED" where id = {augId}'.format(augId=augId))
    return response


def loadAugmentationData(data):
    augmenter = ImageAugmentation()

    response = {'success': 'Data retrieved', 'results': []}
    selectAug = 'select rd.FilePath,rd.FileName,rd.id from RESOURCE_DETAILS rd where rd.resourceId={id} '.format(
        id=data['form']['resourceId'])
    resp = augmenter.getRecords(selectAug)
    results = []
    resp = [] if not resp else resp

    response['augId'] = data['form']['augId']

    for r in resp:
        aug1 = []
        r1 = {'name': r['FileName'], 'path': r['FilePath'],
              'id': r['id'], 'augImgs': []}
        select1 = 'select distinct AugType,FilePath from AUGMENTATION_RESULTS where ResDetailId={resDetailId} order by id '.format(
            resDetailId=r1['id'])
        r2 = augmenter.getRecords(select1)
        if r2:
            for augData in r2:
                aug1.append(
                    {'name': augData['AugType'], 'url': augData['FilePath']})

        r1['augImgs'] = aug1
        results.append(r1)

    response['results'] = results
    augmenter.log_message('load augmentation data', data, 'response', response)

    return response


def clearUnsavedAugmentation(data):
    augmenter = ImageAugmentation()
    response = {'success': 'Data removed', 'results': []}

    select = "select id as AugId,ResourceId as Resource from AUGMENTATION_HISTORY where resourceId in (\
                select ar.id FROM ALL_RESOURCES ar inner join AUGMENTATION_HISTORY ah\
                where ar.id = ah.resourceId and ar.name='Custom Augmentation')"
    tmp = augmenter.getRecords(select)
    if tmp:
        augIds, resIds = zip(
            *[(str(t['AugId']), str(t['Resource'])) for t in tmp])

        delete_aug_hist = 'delete from AUGMENTATION_HISTORY where id in ({augIds})'.format(
            augIds=','.join(augIds))
        delete_aug_res = 'delete from AUGMENTATION_RESULTS where AugId in ({augIds})'.format(
            augIds=','.join(augIds))
        delete_all_res = 'delete from ALL_RESOURCES where id in ({resIds})'.format(
            resIds=','.join(resIds))
        delete_res_det = 'delete from RESOURCE_DETAILS where ResourceId in ({resIds})'.format(
            resIds=','.join(resIds))

        for aug in augIds:
            pathToRemove = os.path.join(
                os.getcwd(), 'uploads', 'AugmentedImages', str(aug))
            if os.path.isdir(pathToRemove):
                shutil.rmtree(pathToRemove)
                augmenter.log_message('removed', pathToRemove)

        augmenter.executeQuery('delete', delete_aug_hist)
        augmenter.executeQuery('delete', delete_all_res)
        augmenter.executeQuery('delete', delete_res_det)
        augmenter.executeQuery('delete', delete_aug_res)        

    augmenter.log_message('delete unsaved augmentation data',
                          data, 'response', response)


if __name__ == '__main__':
    # resize(150,150)
    # rotate(cv2.cv2.ROTATE_90_CLOCKWISE)
    # shear(0,40)
    # noise()
    # crop(0, 0.3)
    # flip()
    # bright()
    # scale()
    pass
