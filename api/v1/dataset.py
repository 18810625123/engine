from .helpers import *

class File():
    def __init__(self, path):
        self.dirs = []
        names = os.listdir(path)
        print(12341)
        print(names)
        for dir_name in names:

            if os.path.isdir(path + '/' + dir_name):
                self.dirs.append({
                    'dir_name': dir_name
                })

    def read(self):
        2

class DatasetApi():

    def search(request):
        params = get_params(request)
        f = File('static/images')
        return custom_success({'list':f.dirs})

