import os
import pdb
import json
import collections
import codecs

def dict2namedtuple(dic):
    return collections.namedtuple("Namespace", dic.keys())(**dic)

class ConfigParser():
    def __init__(self, config_file):
        print('config filename:' + config_file)
        self.config_file = config_file
        self.model_parameters = dict2namedtuple(
                json.load(codecs.open(os.path.join(self.config_file), 'r', encoding='utf-8')))
        pass

    def __repr__(self):
        return 'ConfigParser(config_file)'
        pass

    def __str__(self):
        return str(self.model_parameters)
        pass
