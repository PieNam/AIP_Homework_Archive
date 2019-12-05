import numpy
import os
import pickle
import scipy.misc

def load_batch(fp):
    print("loading batch from:", fp)
    with open(fp, 'rb') as f:
        data_dir = pickle.load(f, encoding='bytes')
        imgs = data_dir[b'data']
        labels = data_dir[b'labels']
        imgs = imgs.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        labels = numpy.array(labels)
        return imgs, labels

def load_cifar10(directory):
    img_set_temp = []
    label_set_temp = []
    for batch in range(1, 6):
        fp = os.path.join(directory, 'data_batch_%d' % (batch, ))
        imgs, labels = load_batch(fp)
        img_set_temp.append(imgs)
        label_set_temp.append(labels)
    img_set = numpy.concatenate(img_set_temp)
    label_set = numpy.concatenate(label_set_temp)
    del imgs, labels
    test_image_set, test_label_set = load_batch(os.path.join(directory, "test_batch"))
    return img_set, label_set, test_image_set, test_label_set

