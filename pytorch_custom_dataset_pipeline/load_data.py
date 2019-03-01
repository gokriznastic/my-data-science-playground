import re
import numpy as np
import torch
from os import listdir
from os.path import join
from PIL import Image

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*" 
            b"(\d+)\s(?:\s*#.*[\r\n])*" 
            b"(\d+)\s(?:\s*#.*[\r\n])*" 
            b"(\d+)\s)", buffer).groups()

    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


def load_images(path):
    my_path = path+'\\train'
    folders = [f for f in listdir(my_path)]

    files = []
    for folder_name in folders:
        folder_path = join(my_path, folder_name)
        files.append([f for f in listdir(folder_path) if f.endswith('.pgm')])

    pathname_list = []
    train_labels = []
    for fo in range(len(folders)):
        for fi in files[fo]:
            pathname_list.append(join(my_path, join(folders[fo], fi)))
            train_labels.append(folders[fo])

    train_images = np.empty((4316, 128, 128))

    for i in range(len(pathname_list)):
#         train_images[i] = read_pgm(pathname_list[i])
        train_images[i] = Image.open(pathname_list[i]).convert('L')      

    for i in range(len(train_labels)):
        train_labels[i] = train_labels[i][1:]
        train_labels[i] = int(train_labels[i])

    train_labels = np.asarray(train_labels)

    train_labels = torch.from_numpy(train_labels).long()

    my_path = path+'\\test'
    folders = [f for f in listdir(my_path)]

    files = []
    for folder_name in folders:
        folder_path = join(my_path, folder_name)
        files.append([f for f in listdir(folder_path) if f.endswith('.pgm')])

    pathname_list = []
    test_labels = []
    for fo in range(len(folders)):
        for fi in files[fo]:
            pathname_list.append(join(my_path, join(folders[fo], fi)))
            test_labels.append(folders[fo])

    test_images = np.empty((605, 128, 128))

    for i in range(len(pathname_list)):
#         test_images[i] = read_pgm(pathname_list[i])
        test_images[i] = Image.open(pathname_list[i]).convert('L')

    for i in range(len(test_labels)):
        test_labels[i] = test_labels[i][1:]
        test_labels[i] = int(test_labels[i])

    test_labels = np.asarray(test_labels)

    test_labels = torch.from_numpy(test_labels).long()

    return np.array(train_images), train_labels, np.array(test_images), test_labels
