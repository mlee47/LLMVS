import os
import argparse
import torch
import pprint

def str2bool(v):
    """ Transcode string to boolean.

    :param str v: String to be transcoded.
    :return: The boolean transcoding of the string.
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        print ("Device being used:", self.device)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.set_dataset_dir()
        

    def set_dataset_dir(self):
        """ Function that sets as class attributes the necessary directories for logging important training information.
        """

        self.save_dir_root = f'Summaries/{self.model}/{self.dataset}/{self.tag}'
        os.makedirs(self.save_dir_root, exist_ok = True)

        f = open(os.path.join(self.save_dir_root, 'configuration.txt'), 'w')
        f.write(str(self) + '\n\n\n\n')
        f.flush()
        f.close()

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str