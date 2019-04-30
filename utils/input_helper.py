import os
from glob import glob

import cv2
import numpy as np
from keras.datasets import cifar10, cifar100, fashion_mnist

CATDOG_PATH = "/mnt/workspace/sjh/anomaly-gpnd/dataset_utils/cat_dog/dogs-vs-cats/train"
CATDOG_LABEL = {"cat": 0, "dog": 1}


class keras_inbuilt_dataset(object):
    def __init__(self, dataset, normal_class_label, batch_size=64, test=False):
        if dataset == 'cifar10':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            y_train = y_train[:, 0]
            y_test = y_test[:, 0]
        elif dataset == 'cifar100':
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()
            y_train = y_train[:, 0]
            y_test = y_test[:, 0]
        elif dataset == 'fashion_mnist':
            """
            Fashion-MNIST database of fashion articles
            Dataset of 60,000 28x28 grayscale images of 10 fashion categories, along with a test set of 10,000 images.
            This dataset can be used as a drop-in replacement for MNIST. The class labels are:
                Label	Description
                0	T-shirt/top
                1	Trouser
                2	Pullover
                3	Dress
                4	Coat
                5	Sandal
                6	Shirt
                7	Sneaker
                8	Bag
                9	Ankle boot
            """
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
            x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'constant')
            x_train = np.expand_dims(x_train, axis=3)
            x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2)), 'constant')
            x_test = np.expand_dims(x_test, axis=3)
        elif dataset == 'catdog':
            (x_train, y_train), (x_test, y_test) = load_cats_vs_dogs(CATDOG_PATH, normal_class_label)
        else:
            raise NotImplementedError

        if test:
            if dataset == "catdog":
                y_test = np.asarray(y_test)

            images = x_test
            labels = y_test == normal_class_label
        else:
            if dataset == "catdog":
                images = x_train
                y_train = np.array(y_train)
            else:
                images = x_train[y_train == normal_class_label]
            labels = y_train[y_train == normal_class_label]

        self.images = (images - 127.5) / 127.5
        self.labels = labels
        self.batch_size = batch_size
        self.num_samples = len(self.images)
        self.shuffle_samples()
        self.next_batch_pointer = 0

        self.current_epoch = 0

    def shuffle_samples(self):
        image_indices = np.random.permutation(np.arange(self.num_samples))
        self.images = self.images[image_indices]
        self.labels = self.labels[image_indices]

    def get_next_batch(self):
        num_samples_left = self.num_samples - self.next_batch_pointer
        if num_samples_left >= self.batch_size:
            batch_images = self.images[self.next_batch_pointer:self.next_batch_pointer + self.batch_size]
            batch_labels = self.labels[self.next_batch_pointer:self.next_batch_pointer + self.batch_size]
            self.next_batch_pointer += self.batch_size
        else:
            partial_batch_images_1 = self.images[self.next_batch_pointer:self.num_samples]
            partial_batch_labels_1 = self.labels[self.next_batch_pointer:self.num_samples]
            self.shuffle_samples()
            partial_batch_images_2 = self.images[0:self.batch_size - num_samples_left]
            partial_batch_labels_2 = self.labels[0:self.batch_size - num_samples_left]
            batch_images = np.vstack((partial_batch_images_1, partial_batch_images_2))
            batch_labels = np.concatenate((partial_batch_labels_1, partial_batch_labels_2), axis=0)
            self.next_batch_pointer = self.batch_size - num_samples_left

            self.current_epoch += 1
        return batch_images, batch_labels

    def get_data_hwc(self):
        return self.images[0].shape[0], self.images[0].shape[1], self.images[0].shape[2]

    def get_current_epoch(self):
        return self.current_epoch

    def __len__(self):
        return self.num_samples


def resize_and_crop_image(input_file, output_side_length, greyscale=False):
    # This code was borrowed from https://github.com/izikgo/AnomalyDetectionTransformations/blob/master/utils.py
    img = cv2.imread(input_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB if not greyscale else cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    new_height = output_side_length
    new_width = output_side_length
    if height > width:
        new_height = int(output_side_length * height / width)
    else:
        new_width = int(output_side_length * width / height)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    height_offset = (new_height - output_side_length) // 2
    width_offset = (new_width - output_side_length) // 2
    cropped_img = resized_img[height_offset:height_offset + output_side_length,
                  width_offset:width_offset + output_side_length]
    assert cropped_img.shape[:2] == (output_side_length, output_side_length)

    return cropped_img


def load_cats_vs_dogs(paths, normal_class_label):
    # This code was borrowed from https://github.com/izikgo/AnomalyDetectionTransformations/blob/master/utils.py
    glob_path = os.path.join(paths, '*.*.jpg')
    imgs_paths = glob(glob_path)

    images = [resize_and_crop_image(p, 64) for p in imgs_paths]
    x = np.stack(images)
    y = [CATDOG_LABEL[os.path.split(p)[-1][:3]] for p in imgs_paths]
    y = np.array(y)

    abnormal_class_label = np.abs(normal_class_label - 1)
    normal = x[y == normal_class_label]
    abnormal = x[y == abnormal_class_label]
    x_train = normal[:10000]
    y_train = [normal_class_label] * 10000
    x_test = np.concatenate((normal[10000:], abnormal[:2500]), axis=0)
    y_test = [normal_class_label] * 2500 + [abnormal_class_label] * 2500

    return (x_train, y_train), (x_test, y_test)
