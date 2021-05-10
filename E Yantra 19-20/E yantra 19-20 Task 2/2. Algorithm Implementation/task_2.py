
'''
*****************************************************************************************
*
*        		===============================================
*           		Rapid Rescuer (RR) Theme (eYRC 2019-20)
*        		===============================================
*
*  This script is to implement Task 2 of Rapid Rescuer (RR) Theme (eYRC 2019-20).
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


# Team ID:			369
# Author List:		Anshul Agrawal,Rajat Pal
# Filename:			task_2.py
# Functions:		findCombination
# 					[ Comma separated list of functions in this file ]
# Global variables:	[ List of global variables defined in this file ]


# Import necessary modules
# Do not import any other modules
import cv2
import numpy as np
import os


def findCombination(Digits_list, Sum_integer):

    """
    Purpose:
    ---
    the function takes list of Digits and integer of Sum and returns the list of combination of digits

    Input Arguments:
    ---
    `Digits_list` :		[ list ]
        list of Digits on the first line of text file
    `Sum_integer` :     [ int ]
        integer of Sum on the second line of text file

    Returns:
    ---
    `combination_of_digits` :	[ list ]
        list of digits whose total is equal to Sum_integer

    Example call:
    ---
    combination_of_digits = findCombination(Digits_list, Sum_integer)

    """

    combination_of_digits = []
    #############	Add your Code here	###############
    k=1
    c=-1
    while 1:
        #print ("hat")
        c=c+1
        if k==1:
                combination_of_digits,k = find(Digits_list,Sum_integer,c,[])
                #print(combination_of_digits,k)
        else:
                break
    ###################################################

    return combination_of_digits


#############	You can add other helper functions here		#############
def find(Digits_list,Sum_integer,c,cod):
    l = 0
    k=1
    for k in Digits_list:
        l=l+1
    for i in range(0,l):
        cod1 = cod + [Digits_list[i]]
        if i!=l and i!=0:
                dl1 = Digits_list[0:i]+Digits_list[i+1:l]
        if i==0:
                dl1 = Digits_list[1:l]
        if i==l:
                dl1 = Digits_list[0:l-1]
        if c>0:
            c1,k=find(dl1,Sum_integer,c-1,cod1)
        if(k==0):
                return c1,0
        else:
            sm=0
            for j in cod1:
                sm = sm+j
            #print("gad",i,Digits_list,c,cod1,Sum_integer,sm)
            if(sm==Sum_integer):
                return cod1,0
    return [],1
#########################################################################


# NOTE:	YOU ARE NOT ALLOWED TO MAKE ANY CHANGE TO THIS FUNCTION
# 
# Function Name:	main
# Inputs:			None
# Outputs: 			None
# Purpose: 			the function first takes 'Test_input0.txt' as input, finds the combination
# 					of digits and selects one of them using findCombination function, it then asks
#                   the user whether to repeat the same on other Test input txt files
#                   present in 'Test_inputs' folder or not

if __name__ == '__main__':

    curr_dir_path = os.getcwd()
    txt_input_dir_path = curr_dir_path + '/Test_inputs/'				            # path to directory of 'Test_inputs'

    file_num = 0
    txt_file_path = txt_input_dir_path + 'Test_input' + str(file_num) + '.txt'      # path to 'Test_input0.txt' text file

    txt_input_obj = open(txt_file_path, 'r')

    print('\n============================================')

    print('\nFor Test_input' + str(file_num) + '.txt')

    Digits_list = [int(digit) for digit in txt_input_obj.readline().split()]
    Sum_integer = [int(integer) for integer in txt_input_obj.readline().split()][0]

    print('\nGiven List of Digits = %s \n\nGiven Integer of Sum = %d' % (Digits_list, Sum_integer))

    try:
        
        combination_of_digits = findCombination(Digits_list, Sum_integer)

    except Exception as e:
        
        print('\n[ERROR] findCombination function ran into an error, check your code !\n')
        raise(e)
        exit()
    
    print('\nCombination of digits chosen = %s \n\nNumber of digits chosen = %d' % (combination_of_digits, len(combination_of_digits)))
    
    print('\n============================================')
    
    choice = input('\nWant to run your script on other text input files ? ==>> "y" or "n": ')

    if choice == 'y':

        file_count = len(os.listdir(txt_input_dir_path))

        for file_num in range(1, file_count):

            txt_file_path = txt_input_dir_path + 'Test_input' + str(file_num) + '.txt'      # path to 'Test_input0.txt' text file

            txt_input_obj = open(txt_file_path, 'r')

            print('\n============================================')

            print('\nFor Test_input' + str(file_num) + '.txt')

            Digits_list = [int(digit) for digit in txt_input_obj.readline().split()]
            Sum_integer = [int(integer) for integer in txt_input_obj.readline().split()][0]

            print('\nGiven List of Digits = %s \n\nGiven Integer of Sum = %d' % (Digits_list, Sum_integer))

            try:
                
                combination_of_digits = findCombination(Digits_list, Sum_integer)

            except Exception as e:
                
                print('\n[ERROR] findCombination function ran into an error, check your code !\n')
                raise(e)
                exit()
            
            print('\nCombination of digits chosen = %s \n\nNumber of digits chosen = %d' % (combination_of_digits, len(combination_of_digits)))
            
            print('\n============================================')
    
    else:

        print('')


