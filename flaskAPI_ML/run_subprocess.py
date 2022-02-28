import subprocess
import re
cmd = 'python script.py'


def run_sub(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    # result = out.split('\n')
    print('out', out)
    response = {'results': re.sub(
        "[^a-zA-Z]+", " ", out.decode('utf-8')), 'error': err, 'results1': out.decode('utf-8')}
    return response


def run_sub_v2(cmd):
    data = subprocess.check_output(cmd, shell=True).strip().split('\n')
    response = {'results': data}
