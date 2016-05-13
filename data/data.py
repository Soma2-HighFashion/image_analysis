try:
   import cPickle as pickle
except:
    import pickle
import os
from PIL import Image
import numpy as np

def load_gender_dataset():
	# Return dataset - numpy array, label 
	# 0 : Femail / 1 : Male

	gender_path = "/home/dj/HighFashionProject/image_analysis/gender_data/scaled_84x256/"
	geometry = (256, 84)

	patch_geometry = (84, 84)
	patch_count = 5

	# Make Female & Male DataSet
	female_path = gender_path + "img_female/"
	female_list = os.listdir(female_path)
	female_count = len(female_list) * patch_count
	
	male_path = gender_path + "img_male/"
	male_list = os.listdir(male_path)
	male_count = len(male_list) * patch_count

	img_data = np.empty((female_count+male_count, patch_geometry[0], patch_geometry[1], 3))
	label_data = np.zeros((female_count+male_count, 2), dtype="uint8")

	print "Load Female DataSet....."
	for i in range(len(female_list)):
		patch_images = generate_patches(img2numpy_arr(female_path+female_list[i]))
		for j in range(patch_count):
			img_data[(i*patch_count)+j,:,:,:] = patch_images[j]	

	label_data[:female_count, 0] = 1
	print "Complete!"

	print "Load Male DataSet ..."
	for i in range(len(male_list)):
		patch_images = generate_patches(img2numpy_arr(male_path+male_list[i]))
		for j in range(patch_count):
			img_data[female_count+(i*patch_count)+j,:,:,:] = patch_images[j]	
	label_data[female_count+1:, 1] = 1
	print "Complete!"				

	dataset = {
		'data' : img_data,
		'label' : label_data,
		'geometry': patch_geometry,
		'num_classes': 2
	}

	print "dataset : ", img_data.shape, " label : ", label_data.shape
	return dataset 

def load_category_dataset():
	# Return dataset - numpy array, label 
	# Labels => 
	#	0: Street / 1: Casual / 2: Sexy / 3: Unique / 4: Work wear / 5: Classic

	gender_path = "/home/dj/HighFashionProject/image_analysis/category_data/scaled_84x256/"
	geometry = (256, 84)

	patch_geometry = (84, 84)
	patch_count = 5

	# Make Female DataSet
	female_path = gender_path + "img_female/"
	female_list = os.listdir(female_path)
	female_count = len(female_list) * patch_count
	
	male_path = gender_path + "img_male/"
	male_list = os.listdir(male_path)
	male_count = len(male_list) * patch_count

	img_data = np.empty((female_count+male_count, patch_geometry[0], patch_geometry[1], 3))
	label_data = np.zeros((female_count+male_count, 2), dtype="uint8")

	print "Load Female DataSet....."
	for i in range(len(female_list)):
		patch_images = generate_patches(img2numpy_arr(female_path+female_list[i]))
		for j in range(patch_count):
			img_data[(i*patch_count)+j,:,:,:] = patch_images[j]	

	label_data[:female_count, 0] = 1
	print "Complete!"

	# Make Male DataSet

	print "Load Male DataSet ..."
	for i in range(len(male_list)):
		patch_images = generate_patches(img2numpy_arr(male_path+male_list[i]))
		for j in range(patch_count):
			img_data[female_count+(i*patch_count)+j,:,:,:] = patch_images[j]	
	label_data[female_count+1:, 1] = 1
	print "Complete!"				

	dataset = {
		'data' : img_data,
		'label' : label_data,
		'geometry': patch_geometry,
		'num_classes': 2
	}

	print "dataset : ", img_data.shape, " label : ", label_data.shape

#	print "Save to Pickle data for cache"
#	save2p(dataset, cache_path)
	return dataset 


def img2numpy_arr(img_path):
	return np.array(Image.open(img_path))

def generate_patches(ndarr):
	geometry = (256, 84)
	patch_geometry = (84, 84)
	patch_count = 5

	step = (geometry[0] - patch_geometry[0]) / (patch_count-1)
	return [ndarr[i*step:i*step+patch_geometry[0], :, : ] for i in range(patch_count)]

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


