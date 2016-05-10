try:
   import cPickle as pickle
except:
    import pickle
import os
from PIL import Image
import numpy as np

def img2numpy_arr(img_path):
	return np.array(Image.open(img_path))

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

def load_gender_dataset():
	# Return dataset - numpy array, label 
	# 0 : Femail / 1 : Male

# Pickle Memory Error
#cache_path = "gender_data/gender_dataset.p"
#	if os.path.isfile(cache_path):
#		print "Load data cache!"
#		return load(cache_path)
#	else:

	# Make Female DataSet
	female_path = "gender_data/img_female/"
	female_list = os.listdir(female_path)

	print "Load Female DataSet....."
	female_data = np.empty((len(female_list), 128, 128, 3))
	for i in range(len(female_list)):
		female_data[i,:,:,:] = img2numpy_arr(female_path+female_list[i])

	female_label = np.zeros((len(female_list), 2), dtype="uint8")
	female_label[:, 0] = 1
	print "Complete!"

	# Make Male DataSet
	male_path = "gender_data/img_male/"
	male_list = os.listdir(male_path)

	print "Load Male DataSet ..."
	male_data = np.empty((len(male_list), 128, 128, 3))
	for i in range(len(male_list)):
		male_data[i,:,:,:] = img2numpy_arr(male_path+male_list[i])
	
	male_label = np.zeros((len(male_list), 2), dtype="uint8")
	male_label[:, 1] = 1
	print "Complete!"				

	dataset = {
		'data' : np.concatenate((male_data, female_data),axis=0),
		'label' : np.concatenate((male_label, female_label), axis=0),
		'geometry': (128, 128),
		'num_classes': 2
	}

#	print "Save to Pickle data for cache"
#	save2p(dataset, cache_path)
	return dataset 
