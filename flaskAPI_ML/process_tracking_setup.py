from flask.wrappers import Response

from sqlite_db_connect import SQLiteConnect


class ProcessTrack(SQLiteConnect):
    def ___init__(self):
        super.__init__()


def loadallprocessvideos(data):
    processTrack = ProcessTrack()
    print('data', data)
    select = 'select * from ALL_PROCESS_VIDEOS'
    tmp = processTrack.getRecords(select)
    response = {'success': 'Data retrieved',
                'results': []}
    if tmp:
        response['results'] = tmp
    return response


def getSavedS3Url(key):
    processTrack = ProcessTrack()
    url = None
    tmp = processTrack.getRecords(
        'select s3Url from PRESIGNED_URLS where replace(s3Key,char(92),"/") like "%{key}" or s3Key like "%{key}"  limit 1'.format(key=key))

    if tmp:
        tmp = tmp[0]
        url = tmp['s3Url']

    print('getSavedS3Url ', url)
    return {'s3Url': url}
