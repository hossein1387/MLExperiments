import ipdb as pdb
import sys
import yaml
import random

class Configuration():

    def __init__(self, config_type, config_file):
        self.config_file = config_file
        self.config_type = config_type
        self.config_dict =  self.parse_cofig_file()

    def parse_cofig_file(self):
        configs = self.read_config()
        try:
            return configs[self.config_type]
        except yaml.YAMLError as exc:
            print ("Config type {0} was not found in {1}".format(self.config_type, self.config_file))
            sys.exit()

    def get_config_str(self):
        config_str = "data set: {0} \n"\
                     "initial learning rate is set to: {1}\n"\
                     "number of epochs: {2} \n"\
                     "batch size: {3} \n"\
                     "model type: {4} \n".\
                     format(self.config_dict['dataset'], self.config_dict['lr'], self.config_dict['num_epochs'],\
                            self.config_dict['batchsize'], self.config_dict['model_type'])
        return "========================================================\nConfiguration:\n========================================================\n{0}".format(config_str)

    def read_config(self):
        configs = None
        with open(self.config_file, 'r') as stream:
            try:
                configs = yaml.load(stream)
            except yaml.YAMLError as exc:
                print ("Error loading YAML file {0}".format(self.config_file))
                sys.exit()
        return configs

