'''

K-Nearest Neighbors Implementation

Allison Aprile
Assignment 2: Word Vectors (2.c.iii)
March 15, 2021

'''

import numpy as np


'''
Function: cosine_similarity

  Calculates cosine similarity between a vector and a matrix (but also 
  can calculate cosine similarity between two vectors)

Arguments:
  vec -- word embedding vector
  mat -- word embedding matrix

Return:
  Cosine similarity vector between the vector and matrix, having shape (# matrix columns, 1)

'''
def cosine_similarity(vec, mat):
	
	# Calculate numerator (vector of the dot products between each matrix row and the vector)
	num = np.dot(mat, vec)

	# Check shape of the matrix
	if mat.ndim > 1:
		# Matrix - for vec, take the square root of the elementwise sum of squares
		#        - for mat, take the square root of the rowwise sum of squares
		#		 - Multipy the results
		denom = (np.sqrt(np.sum(vec**2))) * (np.sqrt(np.sum(mat**2, axis=1)))

	else:
		# Vector - for both vec and mat, take the square root of the elementwise sum of squares
		#        - Multipy the results
		denom = (np.sqrt(np.sum(vec**2))) * (np.sqrt(np.sum(mat**2)))
	
	return num / denom



'''
Function: knn

  Implements the K-Nearest Neighbors algorithm, using cosine similarity as a distance metric.

Arguments:
  word_vector -- word embedding vector
  word_matrix -- word embedding matrix
  k -- integer, number of nearest neighbors to return

Return:
  K indices of the matrix's rows that are closest to the vector

'''
def knn(word_vector, word_matrix, k):
	# Calculate cosine similarity vector
	cos_sim = cosine_similarity(word_vector, word_matrix)

	# Return the k nearest neighbors for the word 
	# - Because larger cosine similarity means closer distance, 
	#   sort the cosine similarity vector DESCENDINGLY and slice the first k elements
	return (-cos_sim).argsort()[:k]

