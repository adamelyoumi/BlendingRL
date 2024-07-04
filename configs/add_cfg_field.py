# https://docs.google.com/spreadsheets/d/1qYxyulHMTBfmBNzkTz9xo487xnKjwOy5LTgq75R-mkQ/edit?gid=0#gid=0

import os
import yaml


attrib_name_1 = 'env'
attrib_name_2 = 'maxflow'
default_value = 50

folder_path = './configs'

def add(attrib_name_1, default_value, attrib_name_2 = None):
    for filename in os.listdir(folder_path):
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            if attrib_name_2 is not None:
                data[attrib_name_1][attrib_name_2] = default_value
            else:
                data[attrib_name_1] = default_value
            with open(file_path, 'w') as f:
                yaml.dump(data, f)
            
def remove(attrib_name_1, attrib_name_2 = None):
    for filename in os.listdir(folder_path):
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r') as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
                
            if attrib_name_2 is None:
                del data[attrib_name_1]
            else:
                del data[attrib_name_1][attrib_name_2]
                
            with open(file_path, 'w') as f:
                yaml.dump(data, f)
                
if __name__ == "__main__":
    add()