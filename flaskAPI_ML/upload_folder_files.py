import os
from pprint import pprint
from s3connect import *
from urllib.parse import unquote, parse_qs, urlsplit
from pprint import pprint
from datetime import datetime, timedelta
from sqlite_db_connect import SQLiteConnect


s3Files = []
localFiles = []
fileExt = []
extMimeMaps = {'mp4': 'video/mpeg', 'jfif': 'image/jpeg', 'db': None, 'jpg': 'image/jpeg', 'pkl': None, 'yaml': None,'pt':None,
               'bmp': 'image/bmp', 'csv': 'text/csv', 'png': 'image/png', 'hdf5': None, 'txt': 'text/plain', 'jpeg': 'image/jpeg'}

objectWrapper = S3Wrapper()
dbObject = SQLiteConnect()

filesCounter = 0

# datetime.strptime('20220117T042935Z', "%Y%m%dT%H%M%SZ") AMZ Timestamp
# +timedelta(days=7)

# for root, dirs, files in os.walk(os.path.join(os.getcwd(), 'uploads'), topdown=False):
    # for name in files:
        # fullPath = os.path.join(root, name)
        # relativePath = os.path.join(os.path.relpath(root, os.getcwd()), name)
        # ext = name.split('.')[-1]
        # s3Path = objectWrapper.get_file_url(relativePath)
        # if not s3Path:
            # print('not exists for ', relativePath)
            # # objectWrapper.upload_files(
            # #     filename=relativePath, contentType=extMimeMaps[ext.lower()])
        # else:
            # rPath = unquote(urlsplit(s3Path).path)
            # # print('rPath->',rPath,'fPath->',relativePath)
        # localFiles.append({'rPath': relativePath, 'ext': ext,
                          # 'mime': extMimeMaps[ext.lower()]})
        # filesCounter += 1
        # print(filesCounter, end="\r")

        # break

print('Getting all files from s3 ', end="\n")
s3Files = objectWrapper.get_all_files_v2('uploads')

insertdata = []
for file in s3Files:
    createDate = file['query_str']['X-Amz-Date'][0]
    createDate = datetime.strptime(
        createDate, "%Y%m%dT%H%M%SZ").strftime('%Y-%m-%d %H:%M:%S')

    expDate = datetime.strptime(
        createDate, "%Y-%m-%d %H:%M:%S")+timedelta(days=7)
    expDate = expDate.strftime('%Y-%m-%d %H:%M:%S')

    insertdata.append("('{s3Key}','{s3Url}','{expiry}','{created}')".format(
        s3Key=file['rPath'].replace("/", ""), s3Url=file['url'], created=createDate, expiry=expDate))


# remove the expired URLS
dbObject.executeQuery(
    'Delete', 'delete from PRESIGNED_URLS where CURRENT_DATE >expiry ')

# create presigned urls
insertdata = ','.join(insertdata)
insert = 'insert into PRESIGNED_URLS (s3Key,s3Url,expiry,createDate) values {insertData}'.format(
    insertData=insertdata)
dbObject.executeQuery('insert', insert)

print('s3 files ', len(s3Files))
print('local files', len(localFiles))
# objectWrapper.remove_files('')
