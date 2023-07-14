import json
import hashlib
from itertools import zip_longest

def readJson(jsonFile):
    with open(jsonFile, 'r') as file:
        data = json.load(file)
    return data

def groupListByBatch(iterable, batch_size):
    """Returns batches of lists each of size specified in batch size"""
    args = [iter(iterable)] * batch_size
    return zip_longest(*args)


def hasherfunc(name,lst):
    lst_str = str(lst)   # Convert to string
    hasher = hashlib.sha256()
    hasher.update(lst_str.encode('utf-8'))
    return name+"_"+hasher.hexdigest()[:8]  # Return 8 characters of the list
