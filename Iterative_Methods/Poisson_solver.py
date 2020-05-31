# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:28:23 2020

@author: ciaran
"""



import numpy as np 
from scipy import linalg 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
np.set_printoptions(precision=3, suppress=True, linewidth=140) 

plt.close('all')

#set N the dimension of the 2D grid 
N = 20
print('Dimension: ', N)

#Liebmann's Method

#Set the number of cycles
cycles = 1000
h = 1

#initialise 2D grid u and matrix representing source f
u = np.zeros((N,N))
f = np.zeros((N,N))

#set source term
f[N//2][N//2] = 1000

#Dirichlet BC
u[:,0] = 0
u[0, :] = 0
u[:,-1] = 0
u[-1,:] = 0

#Liebmann Method
#iterate over grid until solution converges
for c in range(cycles):
    for i in range(1, N-1):
        for n in range(1, N-1):
            u[i][n] = (u[i][n-1]+u[i][n+1]+u[n][i+1]+u[n][i-1]+(h**2)*f[i][n])/4

#plotting        
fig = plt.figure() 
ax = fig.gca(projection='3d') 
x = np.arange(0,N,1) 
y = np.arange(0,N,1) 
x, y = np.meshgrid(x, y) 
 
surf = ax.plot_surface(x, y, u) # generate surface plot using x, y mesh points
plt.title('Liebmann') 

#Iterative methods
#A.u_i = b (1)
#Dirichlet BC gives solution for boundary points: u_ij = 0
#Only need to solve equation (1) for interior points
#Initialise matrices for iterative methods
u = np.zeros((N,N))
N_i = N-2 #dimension of interior grid
A = np.zeros(((N_i)**2,(N_i)**2))
b = np.zeros((N_i)**2)

#Setting up discrete laplacian matrix for interior points
#set up equation for each grid point 
for i in range(N_i):
    for j in range(N_i):
        k = i*(N_i) + j #grid point index
        A[k][k] = 4 #diagonal
        
        #off diagonals
        if j > 0:
            A[k][k-1] = -1
            
        if j < (N_i - 1):
            A[k][k+1] = -1

        if i > 0:
            A[k][k-N_i] = -1
            
        if i < (N-3):
            A[k][k+N_i] = -1

#set source term in the same place as in Liebmann method for comparison
b[(N_i)//2*(N_i) + (N_i)//2] = 1000


#Functions for iterative methods
#Matrix Splits: A = U + D + L
#U = upper triangular, D = diagonal, L = lower triangular
#Each method iterates x_n+1 = M_inv*n*x_n + M_inv*b = T*x_n + c (2)

#jacobi method
def jacobi(A, x, b):
    diag_vals = np.diag(A) #extract diagonal values
    
    D = np.diagflat(diag_vals)
    M_inv = np.diagflat(1/diag_vals)
    n = -1*(A - D)
    
    #set up c and T terms
    c = np.dot(M_inv, b)
    T = np.dot(M_inv, n)
    
    max_ev = np.max(abs(linalg.eigvals(T))) #find max eigenvalue of T matrix

    sol_err = 1 #initialise solution error
    sol_err_list = [] #list for storing solution error
    
    #iterate until convergence
    while abs(sol_err) > 10**(-6):
        tx = np.dot(T,x)
        x_new = tx + c #Implement equation (2)
        
        #solution error = difference in sum of square elements
        sol_err = np.sum(x_new**2) - np.sum(x**2)
        sol_err_list.append(abs(sol_err))
        
        x = x_new #update x
            
    return x, sol_err_list, max_ev

    
#Gauss_seidel
def Guass_Seidel(A,x,b):
    M = np.tril(A) #extract L + D 
    M_inv = np.linalg.inv(M)
    U = A - M
    n = -1*U
    
    c = np.dot(M_inv, b)
    T = np.dot(M_inv, n)
    
    max_ev = np.max(abs(linalg.eigvals(T)))

    sol_err = 1
    sol_err_list = []
    while abs(sol_err) > 10**(-6):
        tx = np.dot(T,x)
        x_new = tx + c
        
        sol_err = np.sum(x_new**2) - np.sum(x**2)
        sol_err_list.append(abs(sol_err))
        
        x = x_new
            
    return x, sol_err_list, max_ev

#SOR
def SOR(A,x,b,omega):
    diag_vals = np.diag(A)
    D = np.diagflat(diag_vals)
    U = np.triu(A) - D
    L = np.tril(A) - D
    M = D + omega*L
    M_inv = np.linalg.inv(M)
    n = (1-omega)*D - omega*U
    
    c = np.dot(M_inv, b)
    T = np.dot(M_inv, n)
    
    max_ev = np.max(abs(linalg.eigvals(T)))
    
    sol_err = 1
    sol_err_list = []
    while abs(sol_err) > 10**(-6):
        tx = np.dot(T,x)
        x_new = tx + c
        
        sol_err = np.sum(x_new**2) - np.sum(x**2)
        sol_err_list.append(abs(sol_err))
        
        x = x_new
    
    return x, sol_err_list, max_ev
    
x_guess = np.zeros((N_i)**2) #intial guess for x

#Implement iterative methods
JC = jacobi(A, x_guess, b)
GS = Guass_Seidel(A, x_guess, b)
SOR_run = SOR(A, x_guess, b, 1.5)
SOR_low = SOR(A, x_guess, b, 1.25)
SOR_high = SOR(A, x_guess, b, 1.75)


print('\nJacobi')
print('Iterations:' , len(JC[1]))
print('|lambda_max|:', JC[2])

print('\nGauss_Seidel')
print('Iterations:' , len(GS[1]))
print('|lambda_max|:', GS[2])


#plotting convergence
fig1 = plt.figure(figsize = (12,5))
ax1 = fig1.add_subplot(121)
ax1.set_yscale('log')
ax1.plot(np.arange(0,len(SOR_run[1])),SOR_run[1], label = 'SOR ($\omega = 1.5$)', linestyle = '-')
ax1.plot(np.arange(0,len(SOR_high[1])),SOR_high[1], label = 'SOR ($\omega = 1.75$)', linestyle = '-')
ax1.plot(np.arange(0,len(SOR_low[1])),SOR_low[1], label = 'SOR ($\omega = 1.25$)', linestyle = '-')
ax1.plot(np.arange(0,len(GS[1])),GS[1], label = 'GS', linestyle = '--')
ax1.plot(np.arange(0,len(JC[1])),JC[1], label = 'Jacobi', linestyle = ':')
plt.xlabel('No. of Iterations')
plt.ylabel('Solution Error')
plt.legend()

#convergence of SOR at different omega
omega_list = np.arange(1., 2., .01)
SOR_conv_iter = []
SOR_ev = []
for i in omega_list:
    SOR_conv = SOR(A, x_guess, b, i)
    SOR_iter = len(SOR_conv[1])
    SOR_ev.append(SOR_conv[2])
    SOR_conv_iter.append(SOR_iter)

#optimal SOR results
opt_SOR_index = np.argmin(SOR_conv_iter) 
print('\nBest SOR')
print('Iterations:', SOR_conv_iter[opt_SOR_index])
print('|lambda_max|:', SOR_ev[opt_SOR_index])
print('omega:', omega_list[opt_SOR_index])
    
ax2 = fig1.add_subplot(122)
ax2.plot(omega_list, SOR_conv_iter)
plt.xlabel('$\omega$')
plt.ylabel('No. of Iterations')
plt.tight_layout()
plt.savefig('omega.png')

fig3 = plt.figure()
ax = fig3.add_subplot(111)
ax.plot(omega_list, SOR_ev)
plt.xlabel('$\omega$')
plt.ylabel('Max Eigenvalue')
plt.tight_layout()
plt.savefig('eigenvales.png')

#plotting solutions
#Gauss Seidel
GS_sol = GS[0] 
GS_grid = np.array(GS_sol).T.reshape((N-2, N-2))

#put solved interior grid back into u
for i in range(N-2):
    u[i+1][1:N-1] = GS_grid[i]

fig4 = plt.figure() 
ax = fig4.gca(projection='3d') 
x = np.arange(0,N,1) 
y = np.arange(0,N,1) 
x, y = np.meshgrid(x, y) 
 
surf = ax.plot_surface(x, y, u) # generate surface plot using x, y mesh points 
plt.title('Gauss-Seidel')
plt.show()

#Jacobi
JC_sol = JC[0] 
JC_grid = np.array(JC_sol).T.reshape((N-2, N-2))
for i in range(N-2):
    u[i+1][1:N-1] = JC_grid[i]

fig5 = plt.figure() 
ax = fig5.gca(projection='3d') 
x = np.arange(0,N,1) 
y = np.arange(0,N,1) 
x, y = np.meshgrid(x, y) 
 
surf = ax.plot_surface(x, y, u)
plt.title('Jacobi')
plt.show() 

#SOR @ omega = 1.75
SOR_sol = SOR_high[0] 
SOR_grid = np.array(SOR_sol).T.reshape((N-2, N-2))
for i in range(N-2):
    u[i+1][1:N-1] = SOR_grid[i]


fig6 = plt.figure() 
ax = fig6.gca(projection='3d') 
x = np.arange(0,N,1) 
y = np.arange(0,N,1) 
x, y = np.meshgrid(x, y) 
 
surf = ax.plot_surface(x, y, u)
plt.title('SOR')
plt.savefig('Solution.png')
plt.show()
            
        
        
