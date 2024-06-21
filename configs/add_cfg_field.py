# https://docs.google.com/spreadsheets/d/1qYxyulHMTBfmBNzkTz9xo487xnKjwOy5LTgq75R-mkQ/edit?gid=0#gid=0

import os
import yaml

attrib_name = 'custom_softmax'
default_value = False

folder_path = './configs'

for filename in os.listdir(folder_path):
    if filename.endswith('.yaml') or filename.endswith('.yml'):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        data[attrib_name] = default_value
        with open(file_path, 'w') as f:
            yaml.dump(data, f)