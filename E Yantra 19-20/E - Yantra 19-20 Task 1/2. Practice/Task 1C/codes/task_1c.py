
'''
*****************************************************************************************
*
*        		===============================================
*           		Rapid Rescuer (RR) Theme (eYRC 2019-20)
*        		===============================================
*
*  This script is to implement Task 1C of Rapid Rescuer (RR) Theme (eYRC 2019-20).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*  
*  e-Yantra - An MHRD project under National Mission on Education using ICT (NMEICT)
*
*****************************************************************************************
'''


# Team ID:			[ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:			task_1c.py
# Functions:		computeSum
# 					[ Comma separated list of functions in this file ]
# Global variables:	None
# 					[ List of global variables defined in this file ]


# Import necessary modules
import cv2
import numpy as np
import os
import sys


#############	You can import other modules here	#############

from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
from joblib import load

#################################################################


# Function Name:	computeSum
# Inputs: 			img_file_path [ file path of image ]
# 					shortestPath [ list of coordinates of shortest path from initial_point to final_point ]
# Outputs:			digits_list [ list of digits present in the maze image ]
# 					digits_on_path [ list of digits present on the shortest path in the maze image ]
# 					sum_of_digits_on_path [ sum of digits present on the shortest path in the maze image ]
# Purpose: 			the function takes file path of original image and shortest path in the maze image
# 					to return the list of digits present in the image, list of digits present on the shortest
# 					path in the image and sum of digits present on the shortest	path in the image
# Logic:			[ write the logic in short of how this function solves the purpose ]
# Example call: 	digits_list, digits_on_path, sum_of_digits_on_path = computeSum(img_file_path, shortestPath)


def computeSum(img_file_path, shortestPath):

	"""
	Purpose:
	---
	the function takes file path of original image and shortest path as argument and returns list of digits, digits on path and sum of digits on path

	Input Arguments:
	---
	`img_file_path` :		[ str ]
		file path of image
	`shortestPath` :		[ list ]
		list of coordinates of shortest path from initial_point to final_point

	Returns:
	---
	`digits_list` :	[ list ]
		list of all digits on image
	`digits_on_path` :	[ list ]
		list of digits adjacent to the path from initial_point to final_point
	`sum_of_digits_on_path` :	[ int ]
		sum of digits on path

	Example call:
	---
	original_binary_img = readImage(img_file_path)

	"""

	digits_list = []
	digits_on_path = []
	sum_of_digits_on_path = 0

	#############  Add your Code here   ###############
	img = cv2.imread(img_file_path,0)
	binary_img = (img>128)*1.0
	dr = load_model("digit_recognizer.h5")
	x,y = img.shape
	ncw,nch = int(y/40),int(x/40)
	OHE = load("ohe.joblib")
	pos = []
	kernel = np.ones((3,3),np.uint8)
	for i in range(0,int(x/40)):
		for j in range(0,int(y/40)):
			pimg = binary_img[40*i+4:40*(i+1)-5,40*j+4:40*(j+1)-5]
			pimg = (pimg<0.5)*1.0
			if pimg.sum()>0:
				#print(pimg.sum())
				#pimg = cv2.dilate(pimg,kernel,iterations=1)
				pimg = cv2.resize(pimg,(28,28))
				#cv2.imshow("image",pimg)
				#cv2.waitKey(0)
				pimg.resize(1,28,28,1)
				ans = dr.predict(pimg)
				pos = pos + [(i,j)]
				#print(i,j)
				ans = (ans==ans.max())*1
				ans = OHE.inverse_transform(ans)
				ans = ans[0,0]
				ans = int(ans)
				digits_list = digits_list + [ans]
	adj = np.zeros((nch*ncw,nch*ncw))
	for i in range(0,ncw):
		for j in range(0,nch):
			a = binary_img[j*40:j*40+40,i*40:i*40+40]
			r = (a[20,39]==1)*1
			d = (a[39,20]==1)*1
			if(r==1 and i!=ncw-1):
				adj[i*nch+j,(i+1)*nch+j],adj[(i+1)*nch+j,i*nch+j] = 1,1
			if(d==1 and j!=nch-1):
				adj[i*nch+j,i*nch+j+1],adj[i*nch+j+1,i*nch+j] = 1,1
	adj = (adj==1)*1
	k=-1
	for l in pos:
		x1,y1 = l
		k=k+1
		for l1 in shortestPath:
			if(l1 == (x1-1,y1)):
				if (adj[x1+y1*nch,(x1-1)+y1*nch]==1):
					digits_on_path = digits_on_path + [digits_list[k]]
					sum_of_digits_on_path = sum_of_digits_on_path + digits_list[k]
					break
			elif(l1 == (x1+1,y1)):
				if (adj[x1+y1*nch,(x1+1)+y1*nch]==1):
					digits_on_path = digits_on_path + [digits_list[k]]
					sum_of_digits_on_path = sum_of_digits_on_path+digits_list[k]
					break
			elif(l1 == (x1,y1-1)):
				if (adj[x1+y1*nch,x1+(y1-1)*nch]==1):
					digits_on_path = digits_on_path + [digits_list[k]]
					sum_of_digits_on_path = sum_of_digits_on_path + digits_list[k]
					break
			elif(l1 == (x1,y1+1)):
				if (adj[x1+y1*nch,(x1)+(y1+1)*nch]==1):
					digits_on_path = digits_on_path + [digits_list[k]]
					sum_of_digits_on_path = sum_of_digits_on_path + digits_list[k]
					break
	###################################################

	return digits_list, digits_on_path, sum_of_digits_on_path


#############	You can add other helper functions here		#############



#########################################################################


# NOTE:	YOU ARE NOT ALLOWED TO MAKE ANY CHANGE TO THIS FUNCTION
# 
# Function Name:	main
# Inputs:			None
# Outputs: 			None
# Purpose: 			the function first takes 'maze00.jpg' as input and solves the maze by calling computeSum
# 					function, it then asks the user whether to repeat the same on all maze images
# 					present in 'task_1c_images' folder or not

if __name__ != '__main__':
	
	curr_dir_path = os.getcwd()

	# Importing task_1a and image_enhancer script
	try:

		task_1a_dir_path = curr_dir_path + '/../../Task 1A/codes'
		sys.path.append(task_1a_dir_path)

		import task_1a
		import image_enhancer

	except Exception as e:

		print('\ntask_1a.py or image_enhancer.pyc file is missing from Task 1A folder !\n')
		exit()

if __name__ == '__main__':
	
	curr_dir_path = os.getcwd()
	img_dir_path = curr_dir_path + '/../task_1c_images/'				# path to directory of 'task_1c_images'
	
	file_num = 0
	img_file_path = img_dir_path + 'maze0' + str(file_num) + '.jpg'		# path to 'maze00.jpg' image file

	# Importing task_1a and image_enhancer script
	try:

		task_1a_dir_path = curr_dir_path + '/../../Task 1A/codes'
		sys.path.append(task_1a_dir_path)

		import task_1a
		import image_enhancer

	except Exception as e:

		print('\n[ERROR] task_1a.py or image_enhancer.pyc file is missing from Task 1A folder !\n')
		exit()

	# modify the task_1a.CELL_SIZE to 40 since maze images
	# in task_1c_images folder have cell size of 40 pixels
	task_1a.CELL_SIZE = 40

	print('\n============================================')

	print('\nFor maze0' + str(file_num) + '.jpg')

	try:
		
		original_binary_img = task_1a.readImage(img_file_path)
		height, width = original_binary_img.shape

	except AttributeError as attr_error:
		
		print('\n[ERROR] readImage function is not returning binary form of original image in expected format !\n')
		exit()

	
	no_cells_height = int(height/task_1a.CELL_SIZE)					# number of cells in height of maze image
	no_cells_width = int(width/task_1a.CELL_SIZE)					# number of cells in width of maze image
	initial_point = (0, 0)											# start point coordinates of maze
	final_point = ((no_cells_height-1),(no_cells_width-1))			# end point coordinates of maze

	try:

		shortestPath = task_1a.solveMaze(original_binary_img, initial_point, final_point, no_cells_height, no_cells_width)

		if len(shortestPath) > 2:

			img = image_enhancer.highlightPath(original_binary_img, initial_point, final_point, shortestPath)
			
		else:

			print('\n[ERROR] shortestPath returned by solveMaze function is not complete !\n')
			exit()
	
	except TypeError as type_err:
		
		print('\n[ERROR] solveMaze function is not returning shortest path in maze image in expected format !\n')
		exit()

	print('\nShortest Path = %s \n\nLength of Path = %d' % (shortestPath, len(shortestPath)))

	digits_list, digits_on_path, sum_of_digits_on_path = computeSum(img_file_path, shortestPath)

	print('\nDigits in the image = ', digits_list)
	print('\nDigits on shortest path in the image = ', digits_on_path)
	print('\nSum of digits on shortest path in the image = ', sum_of_digits_on_path)

	print('\n============================================')

	cv2.imshow('canvas0' + str(file_num), img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	choice = input('\nWant to run your script on all maze images ? ==>> "y" or "n": ')

	if choice == 'y':

		file_count = len(os.listdir(img_dir_path))

		for file_num in range(file_count):

			img_file_path = img_dir_path + 'maze0' + str(file_num) + '.jpg'		# path to 'maze00.jpg' image file

			print('\n============================================')

			print('\nFor maze0' + str(file_num) + '.jpg')

			try:
				
				original_binary_img = task_1a.readImage(img_file_path)
				height, width = original_binary_img.shape

			except AttributeError as attr_error:
				
				print('\n[ERROR] readImage function is not returning binary form of original image in expected format !\n')
				exit()

			
			no_cells_height = int(height/task_1a.CELL_SIZE)					# number of cells in height of maze image
			no_cells_width = int(width/task_1a.CELL_SIZE)					# number of cells in width of maze image
			initial_point = (0, 0)											# start point coordinates of maze
			final_point = ((no_cells_height-1),(no_cells_width-1))			# end point coordinates of maze

			try:

				shortestPath = task_1a.solveMaze(original_binary_img, initial_point, final_point, no_cells_height, no_cells_width)

				if len(shortestPath) > 2:

					img = image_enhancer.highlightPath(original_binary_img, initial_point, final_point, shortestPath)
					
				else:

					print('\n[ERROR] shortestPath returned by solveMaze function is not complete !\n')
					exit()
			
			except TypeError as type_err:
				
				print('\n[ERROR] solveMaze function is not returning shortest path in maze image in expected format !\n')
				exit()

			print('\nShortest Path = %s \n\nLength of Path = %d' % (shortestPath, len(shortestPath)))

			digits_list, digits_on_path, sum_of_digits_on_path = computeSum(img_file_path, shortestPath)

			print('\nDigits in the image = ', digits_list)
			print('\nDigits on shortest path in the image = ', digits_on_path)
			print('\nSum of digits on shortest path in the image = ', sum_of_digits_on_path)

			print('\n============================================')

			cv2.imshow('canvas0' + str(file_num), img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

	else:

		print('')


