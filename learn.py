import sys
import os
from sift import Sift
from sklearn.cluster import KMeans
import numpy as np
from math import sqrt
import cPickle
from scipy.cluster import vq


from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

''' these must be set to the correct path dependent on the system '''
TRAINING_PATH = "meta/Train/"
TESTING_PATH = "meta/Test/"
LABELS_FILE = "meta/labels.txt"
SIFT_PATH = "meta/Sift/"
PRE_ALLOCATION_BUFFER = 1000

def read_images(path = TRAINING_PATH):
	''' returns a list of filenames of the images at the `path` in fixed sorted order '''
	return sorted(os.listdir(os.path.abspath(path)))

def read_labels():
	''' returns a dictionary mapping training image to its class and a set of all classes '''
	classes = {}
	categories = set()
	with open(LABELS_FILE) as f:
		for l in f:
			img, cat = l.strip().split()
			categories.add(cat)
			classes[img] = cat
	return classes, categories


def get_sift_dict(training_imgs, base_folder = TRAINING_PATH):
	''' given a list of image names (`training_imgs`) contained in the folder `base_folder`, return a dictionary mapping each image filename to its sift matrix (a matrix of keypoints) '''
	sift = Sift()
	sift_dict = {}
	mat_indices = {}
	for i, img in enumerate(training_imgs):
		sift_dict[img] = sift.extract_sift(os.path.join(base_folder, img))
	return sift_dict


def dict2matrix(d):
	''' convert a dictionary of sift feature matrices to a single feature matrix '''
	nkeys = len(d)
	array = np.zeros((nkeys * PRE_ALLOCATION_BUFFER, 128))
	pivot = 0
	for key in d.keys():
		value = d[key]
		nelements = value.shape[0]
		while pivot + nelements > array.shape[0]:
			padding = zeros_like(array)
			array = vstack((array, padding))
		array[pivot : pivot + nelements] = value
		pivot += nelements
	array = np.resize(array, (pivot, 128))
	return array


def get_histogram(sift_matrix, codebook):
	''' get the clusters to which each row (i.e feature) in sift_matrix belongs '''
	code, dist = vq.vq(sift_matrix, codebook)

	''' predictions is a list of integers corresponding to the indices of the clusters (it is NOT a list of the cluster centroids themselves). Therefore, we can generate their histogram easily by np.histogram. np.histogram will go through each element in the list and decide which bin it belongs to and increment the value of that bin. the bins are labeled from 0 to codebook.shape[0] + 1. Therefore, np.histogram will return a frequency distribution of the words in the image '''
	hist, bin_edges = np.histogram(code, bins = range(codebook.shape[0] + 1))
	return hist


def get_classifier(cname):
	''' returns a classifier object as specified by the argument '''
	if cname == "svc": return SVC()
	elif cname == "mnb": return MultinomialNB()
	elif cname == "rfc": return RandomForestClassifier()
	elif cname == "dt": return DecisionTreeClassifier()
	else: raise Exception("invalid classifier name!")


'''
===== ACTUAL EXECUTION STARTS HERE =====
'''

if not os.path.exists("meta/all_features_matrix") or not os.path.exists("meta/sift_dict"):
	training_imgs = read_images()
	sift_dict = get_sift_dict(training_imgs)
	all_features_matrix = dict2matrix(sift_dict)
	with open("meta/all_features_matrix", "w") as f, open("meta/sift_dict", "w") as g:
		cPickle.dump(all_features_matrix, f)
		cPickle.dump(sift_dict, g)



if not os.path.exists("meta/codebook"):
	''' number of features, number of clusters for KMeans '''

	''' load all_features_matrix '''
	with open("meta/all_features_matrix") as f:
		all_features_matrix = cPickle.load(f)
	num_features = all_features_matrix.shape[0]
	num_clusters = int(sqrt(num_features))

	''' create a KMeans object and train it with the all features matrix. this will generate a KMeans classifier. the cluster_centers_ attribute of the classifier (a.k.a "codebook") is a list of all the cluster centroids and will be used for generating the histograms '''
	codebook, distortion = vq.kmeans(all_features_matrix, num_clusters, thresh = 1)
	with open("meta/codebook", "w") as f:
		cPickle.dump(codebook, f)

'''
At the end of all this, we have done the following:
1. Taken each training image and extracted the SIFT feature points from it. Each feature point is a 128 size vector. Therefore, each image is represented by a n x 128 matrix where n = number of feature points in the image. The mapping of image to matrix is stored in `sift_dict`.
2. Combine all of the feature points from `sift_dict` into a single matrix of size N x 128, where N = n1 + n2 + ... + nr where ni = no of feature points in the ith image. This is stored in `all_features_matrix`.
3. Cluster the feature point vectors in `all_features_matrix` using KMeans Clustering. This will lead to a set of cluster centroids stored in the variable `codebook`.
4. All of the above stuff is written to files for quick retrieval.
'''

'''
At this point, we have 1. the cluster centroids (`codebook`), 2. the classifier (`k`) and 3. image -> sift matrix mapping (`sift_dict`).
1. Now, we represent each training image by a set of cluster centroids instead of feature points. Basically, we are quantizing the feature points. Each feature point for an image is replaced by the centroid of the cluster to which it belongs.
2. At the end of this, given a set of cluster centroids: c1, c2, ..., ck, each image will be represented by a subset of these cluster centroids.
3. Then, when a new image is to be tested, we perform the same steps as above and arrive at a representation by the above cluster centroids.
We then match the image to the training images and classify it.
'''

'''
Now, we generate the training data: each training image is represented by a TF vector of size codebook.shape[0] (= no of centroids).
TF = term frequency
'''


if not os.path.exists("meta/training_data"):
	with open("meta/codebook") as f, open("meta/sift_dict") as g:
		codebook = cPickle.load(f)
		sift_dict = cPickle.load(g)

	training_data = []
	training_imgs = read_images()
	for img in training_imgs:
		sift_matrix = sift_dict[img]
		training_data.append(get_histogram(sift_matrix, codebook))
	training_labels_dict, categories = read_labels()
	training_labels = [training_labels_dict[x] for x in read_images()]

	with open("meta/training_data", "w") as f, open("meta/training_labels", "w") as g:
		cPickle.dump(training_data, f)
		cPickle.dump(training_labels, g)


'''
we now have training_data, training_labels in the format to train a classifier.
choose any classifier and train it if not already trained, and then store it in a file with the filename = name of the classifier
to change the classifier to X, set CLASSIFIER_FILE_NAME = 'X' , and change the line: classifier = SVC() to classifier = X()
'''

CLASSIFIERS_NAMES = ["svc", "mnb", "dt"]
classifiers = []

for CLASSIFIER_FILE_NAME in CLASSIFIERS_NAMES:
	if not os.path.exists("meta/" + CLASSIFIER_FILE_NAME):
		with open("meta/training_data") as f, open("meta/training_labels") as g:
			training_data = cPickle.load(f)
			training_labels = cPickle.load(g)

		classifier = get_classifier(CLASSIFIER_FILE_NAME)
		classifier.fit(training_data, training_labels)

		''' store the classifier for easy reuse '''
		with open("meta/" + CLASSIFIER_FILE_NAME, "w") as f:
			cPickle.dump(classifier, f)

	with open("meta/" + CLASSIFIER_FILE_NAME) as f:
		classifiers.append(cPickle.load(f))


'''
This will read all of the images present at the TESTING_PATH folder and predict their class.
Now, given a new test image, convert it to the same form as the training images (as a vector of cluster centroids) using kmeans.
Therefore, given the codebook, k means classifier and the sift feature points matrix (which can be calculated), we can now call `get_histogram` and obtain the representation of the test image in the same format as that of the training images
'''

testing_images = read_images(TESTING_PATH)
if not testing_images: sys.exit(0)
test_sift_dict = get_sift_dict(testing_images, TESTING_PATH)
testing_data = []
with open("meta/codebook") as f:
	codebook = cPickle.load(f)
for img in testing_images:
	sys.stderr.write("processing:%s\n" % img)
	sift_matrix = test_sift_dict[img]
	testing_data.append(get_histogram(sift_matrix, codebook))


''' generates one file per classifier outputting the predicted class of each testing image '''
for classifier in classifiers:
	with open("meta/%s.result" % classifier.__class__, "w") as f:
		for i, c in zip(testing_images, classifier.predict(testing_data)):
			f.write("%s %s\n" % (i, c))
