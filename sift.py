""" 
ADAPTED FROM:
Python module for use with David Lowe's SIFT code available at:
http://www.cs.ubc.ca/~lowe/keypoints/
adapted from the matlab code examples.
Jan Erik Solem, 2009-01-30
http://www.janeriksolem.net/2009/02/sift-python-implementation.html
"""

import os
import subprocess
from PIL import Image
from numpy import *
import numpy
import cPickle
import pylab
from os.path import exists, basename

''' set accordingly '''
MAXSIZE = 1024

''' sift feature points will be stored in files to prevent them from being recalculated '''
SIFT_PATH = "meta/Sift/"

class Sift:

	def __init__(self):
		try: os.mkdir(SIFT_PATH)
		except OSError: pass

	def extract_sift(self, imagename):
		''' extract SIFT feature points and store them in file '''
		tempname = os.path.join(SIFT_PATH, "%s.sift" % basename(imagename))
		if not exists(tempname):
			import sys
			sys.exit(1)
			self.process_image(imagename, tempname)
		else: print "FOUND!"
		return self.read_features_from_file(tempname)[1]


	def process_image(self, imagename, resultname='temp.sift'):
		''' process an image and save the results in a .key ascii file '''
		if imagename[-3:] != 'pgm':
			size = (MAXSIZE, MAXSIZE)
			im = Image.open(imagename).convert('L')
			im.thumbnail(size, Image.ANTIALIAS)
			im.save('tmp.pgm')
			imagename = 'tmp.pgm'
		
		# assume linux
		cmd = "./sift < %s > %s" % (imagename, resultname)
		
		# run extraction command, assume success :)
		subprocess.call(cmd, shell = True)


	def read_features_from_file(self, filename='temp.sift'):
		""" read feature properties and return in matrix form"""
		
		if exists(filename) != False | os.path.getsize(filename) == 0:
			raise IOError("wrong file path or file empty: "+ filename)
		with open(filename) as f:
			header = f.readline().split()
			
			num = int(header[0])  # the number of features
			featlength = int(header[1])  # the length of the descriptor
			if featlength != 128:  # should be 128 in this case
				raise RuntimeError('Keypoint descriptor length invalid (should be 128).')
					 
			locs = zeros((num, 4))
			descriptors = zeros((num, featlength));		

			#parse the .key file
			e = f.read().split()  # split the rest into individual elements
			pos = 0
			for point in range(num):
				#row, col, scale, orientation of each feature
				for i in range(4):
					locs[point, i] = float(e[pos + i])
				pos += 4
				
				#the descriptor values of each feature
				for i in range(featlength):
					descriptors[point, i] = int(e[pos + i])
				pos += 128
				
				#normalize each input vector to unit length
				descriptors[point] = descriptors[point] / linalg.norm(descriptors[point])
				
		return locs, descriptors
