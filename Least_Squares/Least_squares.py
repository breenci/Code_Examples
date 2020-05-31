# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:07:28 2020

@author: ciara
"""

import numpy as np
from scipy import linalg
from sympy import Matrix as mat

#task 1
A = np.loadtxt('A_matrix.txt') #load the matrix file
b_tran = np.arange(1,14) #create the b vector

A_tran = A.T #transpose A
b = b_tran.T

#Least squares problem:
#A A^T x = A^T b   (1)
lhs = np.dot(A_tran, A) #left hand side of (1)
rhs = np.dot(A_tran, b) #right hand side of (1)

#Solve using scipy module. For checking result later
soln = linalg.solve(lhs, rhs)
easy_soln = linalg.lstsq(A,b)

#Cholesky decompostion into upper (U) and Lower (L) triangular factors
L = linalg.cholesky(lhs, True)
U = linalg.cholesky(lhs, False)

#function for forward substitution
def forward_sub(L,b):
    #initialise solution vector
    x = np.zeros(len(b))
    
    #solve for each x by forward subsitution
    for i in range(len(b)):
        x[i] = (b[i] - np.dot(L[i], x))/L[i][i]
        
    return x

#function for backward subsitution
#similar to forward_sub but loop from bottom to top instead
def back_sub(U,b):
    x = np.zeros(len(b))
    
    for i in np.arange(len(b)-1,-1,-1):
        x[i] = (b[i] - np.dot(U[i], x))/U[i][i]
        
    return x

#Solve L z = A^T b via forward subsitution
z = forward_sub(L, np.dot(A_tran,b))
#Solve U x = z via back subsitution 
cho_sub_soln = back_sub(U, z)

#solve using scipy modules
u_low  = linalg.cho_factor(lhs)
cho_soln = linalg.cho_solve(u_low, rhs)

print('Task 1\n')
print('Method: Cholesky Subsitiution (manual substitution)')
print('Solution: ', cho_sub_soln)
print('\nMethod: Cholesky Subsitiution (scipy modules)')
print('Solution: ', cho_soln)
print('\nMethod: Least Squares (scipy module)')
print('Solution: ', easy_soln)


#task 2
B1 = np.array([[3,6,3,3], [2,5,0,3], [3,9,3,6], [1,2,-1,1]]) #create B1 array
B1_tran = B1.T #find B1^T

#RREF for both matrices via row operations (calulated by hand)
B1_rref = np.array([[1,0,0,-1], [0,1,0,1], [0,0,1,0], [0,0,0,0]])
B1_tran_rref = np.array([[1,0,0,0], [0,1,0,1], [0,0,1,-1/3], [0,0,0,0]])

print('\nTask 2\n')
print('B1 RREF: \n', B1_rref)
print('B1.T RREF: \n', B1_tran_rref)

#basis
B1_null = np.array([1,-1,0,1]).T
B1_row = B1_rref[0:3].T
B1_column = B1[:, 0:3]
B1_left_null = np.array([0,-1,1/3,1]).T

print('\nBasis from row operations:')
print('Null: ', B1_null)
print('Row:\n', B1_row)
print('Column:\n', B1_column)
print('Left Null: ', B1_left_null)

B1T_null = B1_left_null
B1T_row = B1_tran_rref[0:3].T
B1T_column = B1_tran[:, 0:3]
B1T_left_null = B1_null

#check A x_null = 0, y_leftnull A = 0
print('\nB1 x_null =', np.dot(B1, B1_null))
print('y_leftnull B1 =', np.dot(B1_left_null.T, B1))

#Basis can also be found by singular value decomp.
U, s, Vh = linalg.svd(B1)
SVD_null = Vh[3].T #last row is the null space
SVD_row = Vh[0:3].T #first three rows are row space
SVD_col = U[:,0:3] #First three columns of U are column space
SVD_lns = U[:,3] #final column is LNS space

print('\nBasis from SVD:')
print('Null: ', SVD_null)
print('Row:\n',SVD_row)
print('Column:\n',SVD_col)
print('Left Null: ',SVD_lns)

print('\nB1 x_null =', np.dot(B1, SVD_null))
print('y_leftnull B1 =', np.dot(SVD_lns, B1))

#task 3
#set up matrices
A1 = B1
A2 = np.delete(A1, 3,1)
A3 = np.delete(A1, 0,0)

#and b vectors
b12 = np.array([1,2,3,4]).T
b3 = np.array([1,2,3]).T

#find rank of each matrix
rank1 = np.linalg.matrix_rank(A1)
rank2 = np.linalg.matrix_rank(A2)
rank3 = np.linalg.matrix_rank(A3)

print('\nTask 3:')

#print(linalg.eigvals(A1_lhs))

#first matrix
#min norm method
# x = A^+ b where A^+ is the pseudoinverse
A1_soln = np.dot(np.linalg.pinv(A1), b12)
print('\nA1: ')
print('Rank:', rank1)
print('Method: Min Norm')
print('Solution: ',A1_soln)


#second matrix
#solve by cholesky decomp. and subsitution as above
A2_lhs = np.dot(A2.T, A2)
A2_rhs = np.dot(A2.T, b12)

#print(linalg.eigvals(A2_lhs))
L2 = linalg.cholesky(A2_lhs, True)
U2 = linalg.cholesky(A2_lhs, False)

z2 = forward_sub(L2, np.dot(A2.T, b12))
A2_soln = back_sub(U2, z2)

#print(np.dot(np.linalg.pinv(A2), b12))

print('\nA2: ')
print('Rank:', rank2)
print('Method: Cholesky')
print('Solution: ', A2_soln)
#print(linalg.lstsq(A2,b12)[0])

#third Matrix
#solve by min norm method
A3_soln = np.dot(np.linalg.pinv(A3), b3)
print('\nA3: ')
print('Rank:', rank3)
print('Method: Min Norm')
print('Solution: ',A3_soln)
#print(linalg.lstsq(A3,b3)[0])


