# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 09:57:09 2020

@author: ciaran
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as con


#define constants
T_sun = 5778.
R_sun = 6.957e8
AU = 1.49598e11

R_earth = 6.3781e6
M_earth = 5.9722e24


A_moon = 0.12 #Moon albedo
R_moon = 1.727e6
M_moon = 7.342e22

#A function to calculate the escape velocity
def v_esc(M,r):
    return np.sqrt((2*con.gravitational_constant*M)/r)

#This function calculates the equilibrium temperature of a body
def T_eq(T_star,R_star, d, A):
    return T_star*((R_star/(d*AU))**(1/2))*(((1-A)/4)**(1/4))

#Function to calculate the jeans flux / particle number density
#takes Temp in kelvin, molecular mass in amu, and escape velocity in m/s
def jeans(T,m,v):
    #V_0 defined according to equation in lecture notes
    v_0 = np.sqrt((2*con.Boltzmann*T)/(m*con.atomic_mass))
    #define lambda
    lam = v**2/v_0**2
    #calculate flux/number density
    flux = (v_0*(1+lam)*np.exp(-1*lam))/(2*np.sqrt(np.pi))
    
    return flux


#calculate escape velocity for earth
v_earth = v_esc(M_earth, R_earth)

#calulate escape velocity and eq. temp. for moon
v_moon = v_esc(M_moon, R_moon)
T_moon = T_eq(T_sun,R_sun,1,A_moon)

print('Earth:')
print('Escape Velocity:', v_earth, 'm/s')

print('\nMoon:')
print('Escape Velocity:', v_moon, 'm/s')

#Range of molecular masses 
atom_mass = np.arange(1., 50,0.1)

#plotting
fig1 = plt.figure(figsize = (10,10))
ax1 = fig1.add_subplot(111)
ax1.plot(atom_mass,jeans(1000.,atom_mass, v_earth), 'g-', label = 'Earth, T = 1000K')
ax1.plot(atom_mass,jeans(T_moon, atom_mass, v_moon), 'b--', label = 'Moon, T = ' + str(round(T_moon, 1))+'K')
ax1.set_xscale('log')
ax1.set_yscale('log')
plt.xlabel('Atomic Mass (amu)')
plt.ylabel('$\Phi_J$ /$n_{exo}$ (m/s)')
plt.legend()
plt.savefig('jeans_flux.png')
plt.show()