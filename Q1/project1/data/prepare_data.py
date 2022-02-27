import cv2
import os.path as osp
import os
import numpy as np

TRAIN_IMAGE_DATA = 'train-images.idx3-ubyte'
TRAIN_LABEL_DATA = 'train-labels.idx1-ubyte'
TEST_IMAGE_DATA = 't10k-images.idx3-ubyte'
TEST_LABEL_DATA = 't10k-labels.idx1-ubyte'

PATH_TO_SAVE = './mnist'

def convert_images(data_split, label_need):
    if data_split == 'train':
        images = open(TRAIN_IMAGE_DATA, 'rb')
        labels = open(TRAIN_LABEL_DATA, 'rb')
        save_path = osp.join(PATH_TO_SAVE, 'train')
    elif data_split == 'test':
        images = open(TEST_IMAGE_DATA, 'rb')
        labels = open(TEST_LABEL_DATA, 'rb')
        save_path = osp.join(PATH_TO_SAVE, 'test')
    else:
        raise ValueError('Dataset Split Type Error')

    number_dict = {}
    
    image_magic_number = int.from_bytes(images.read(4), byteorder='big', signed=False)
    image_num = int.from_bytes(images.read(4), byteorder='big', signed=False)
    image_row = int.from_bytes(images.read(4), byteorder='big', signed=False)
    image_col = int.from_bytes(images.read(4), byteorder='big', signed=False)

    label_magic_number = int.from_bytes(labels.read(4), byteorder='big', signed=False)
    label_num = int.from_bytes(labels.read(4), byteorder='big', signed=False)
    
    for i in label_need:
        if not osp.exists(osp.join(save_path, str(i))):
            os.makedirs(osp.join(save_path, str(i)))

    for i in range(image_num):
        
        # read images and labels from bytes
        img = [int.from_bytes(images.read(1), byteorder='big', signed=False) for j in range(image_row * image_col)]
        img = np.array(img, dtype=np.uint8).reshape((image_row, image_col))
        label = int.from_bytes(labels.read(1), byteorder='big', signed=False)
        
        if label not in label_need:
            continue

        if label not in number_dict.keys():
            number_dict[label] = 0
        number_dict[label] += 1
        image_path = osp.join(save_path, str(label),  str(number_dict[label]) + ".jpg")

        # save the img
        cv2.imwrite(image_path, img)

        # info
        if (i + 1) % 1000 == 0:
            print("Running, " + data_split + " images: " + str(i + 1) + "/" + str(image_num))

        
    images.close()
    labels.close()

    print(data_split + " dataset finished.")

if __name__ == '__main__':
    label_need = [0, 7]
    convert_images('train', label_need)
    convert_images('test', label_need)