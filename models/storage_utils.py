import os
from collections import OrderedDict


def save_statistics(statistics_to_save, file_path, file_action_key='w'):
    '''
    :param statistics_to_save: dict, val type is float
    :param file_path: e.g. file_path = "C:/test_storage_utils/dir2/test.txt"
    :param file_action_key: 'w' or 'a+' for if new file is written or existing file is appended to
    '''
    if type(statistics_to_save) is not OrderedDict:
        raise TypeError('statistics_to_save must be OrderedDict instead got {}'.format(type(statistics_to_save)))

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, file_action_key) as f: # append mode 'a+' + creates if doesn't exist
        header = ""
        line = ""
        for i,key in enumerate(statistics_to_save.keys()):
            val = statistics_to_save[key]
            if i==0:
                line = line + str(val)
                header = header + key
            else:
                line = line + "\t" + str(val)
                header = header + "\t" + key
        if os.stat(file_path).st_size == 0:  # if empty
            f.write(header+"\n")
        f.write(line+"\n")
