try:
    import cPickle as pickle
except:
    import pickle
import os
from PIL import Image
import numpy as np

GEOMETRY = (256, 84)
PATCH_GEOMETRY = (84, 84)
PATCH_COUNT = 5

def load_gender_dataset():
    # Return dataset - numpy array, label
    # 0 : Femail / 1 : Male

	num_classes = 2
	gender_path = "/home/dj/HighFashionProject/image_analysis/gender_data/scaled_84x256/"

	female_path = gender_path + "img_female/";  female_label = 0
	male_path = gender_path + "img_male/";      male_label = 1

	path_label_list = [(female_path, female_label), (male_path, male_label)]
	dataset_list = map(lambda (p,l): dir2arr(p, l, num_classes), path_label_list)
	total_count = sum(map(lambda (img,label): label.shape[0], dataset_list))

	img_data = np.empty((total_count, PATCH_GEOMETRY[0], PATCH_GEOMETRY[1], 3))
	label_data = label_data = np.zeros((total_count, num_classes), dtype="uint8")


	index = 0
	for i in range(num_classes):
		imgs, labels = dataset_list[i]
		data_count = labels.shape[0]

		img_data[index:index+data_count, :, :, :] = imgs	
		label_data[index:index+data_count, :] = labels
			
		index += data_count

	dataset = {
		'data' : img_data,
		'label' : label_data,
		'geometry': PATCH_GEOMETRY,
		'num_classes': num_classes
	}

	print "dataset : ", img_data.shape, " label : ", label_data.shape
	return dataset

def load_category_dataset():
    # Return dataset - numpy array, label
    # Labels =>
    #	0: Street / 1: Casual / 2: Sexy / 3: Unique / 4: Work Wear / 5: Classic

    num_classes = 6
    category_path = "/home/dj/HighFashionProject/image_analysis/category_data/scaled_84x256/"

    street_path = category_path + "img_street/";      street_label = 0
    casual_path = category_path + "img_casual/";      casual_label = 1
    sexy_path = category_path + "img_sexy/";          sexy_label = 2
    unique_path = category_path + "img_unique/";      unique_label = 3
    workwear_path = category_path + "img_workwear/";  workwear_label = 4
    classic_path = category_path + "img_classic/";    classic_label = 5

    path_label_list = [(street_path, street_label), (casual_path, casual_label), (sexy_path, sexy_label),
                       (unique_path, unique_label), (workwear_path, workwear_label), (classic_path, classic_label)]
    dataset_list = map(lambda (p,l): dir2arr(p, l, num_classes), path_label_list)
    total_count = sum(map(lambda (img,label): label.shape[0], dataset_list))

    img_data = np.empty((total_count, PATCH_GEOMETRY[0], PATCH_GEOMETRY[1], 3))
    label_data = label_data = np.zeros((total_count, num_classes), dtype="uint8")

    index = 0
    for i in range(num_classes):
		imgs, labels = dataset_list[i]
		data_count = labels.shape[0]

		img_data[index:index+data_count, :, :, :] = imgs	
		label_data[index:index+data_count, :] = labels
			
		index += data_count

    dataset = {
        'data' : img_data,
        'label' : label_data,
        'geometry': PATCH_GEOMETRY,
        'num_classes': num_classes
    }

    print "dataset : ", img_data.shape, " label : ", label_data.shape
    return dataset

def dir2arr(dir_path, label_value, num_classes):
	print("Load Data Path...   " + os.path.basename(os.path.normpath(dir_path)) )
	dir_list = os.listdir(dir_path)
	dir_count = len(dir_list)*PATCH_COUNT

	img_data = np.empty((dir_count, PATCH_GEOMETRY[0], PATCH_GEOMETRY[1], 3))
	label_data = np.zeros((dir_count, num_classes), dtype="uint8")

	for i in range(len(dir_list)):
		patch_images = generate_patches(img2numpy_arr(dir_path+dir_list[i]))
		for j in range(PATCH_COUNT):
			img_data[(i*PATCH_COUNT)+j,:,:,:] = patch_images[j]

	targets = np.full((dir_count), label_value, dtype="uint8")
	label_data[np.arange(targets.shape[0]), targets] = 1
	return (img_data, label_data)

def img2numpy_arr(img_path):
    return np.array(Image.open(img_path))

def generate_patches(ndarr):
    step = (GEOMETRY[0] - PATCH_GEOMETRY[0]) / (PATCH_COUNT-1)
    return [ndarr[i*step:i*step+PATCH_GEOMETRY[0], :, : ] for i in range(PATCH_COUNT)]

def save2p(data, fname):
    try:
        with open(fname, "wb") as fh:
            pickle.dump(data, fh)
    except IOError as ioerr:
        print ("File Error: %s" % str(ioerr))
    except pickle.PickleError as pklerr:
        print ("Pickle Error: %s" % str(pklerr))

def load(fname):
    savedItems = []
    try:
        with open(fname, "rb") as fh:
            savedItems = pickle.load(fh)
    except IOError as ioerr:
        save2p(savedItems, fname)
    except pickle.PickleError as pklerr:
        print ("Pickle Error: %s" % str(pklerr))
    finally:
        return savedItems
