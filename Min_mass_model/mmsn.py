#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:42:41 2019

@author: ciaran
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from matplotlib.ticker import ScalarFormatter

#inputting data into arrays
planet = np.array(['Mercury', 'Venus', 'Earth', 'Mars', 'Asteroids', 
                   'Jupiter', 'Saturn', 'Uranus', 'Neptune'])
mass = np.array([3.3, 48.7, 59.8, 6.4, 0.1, 19040., 5695., 870., 1032.])
f = np.array([350, 270, 235, 235, 200, 5., 8., 15., 20.])
dist = np.array([0.387, 0.723, 1, 1.524, 2.7, 5.203, 9.523, 19.208, 30.087])

#converting from AU to cgs
dist_cgs = dist*1.49e13

#using scaling factor to convert to mass of solar composition
m_sol_comp = mass*f
print('Total Mass: ', np.sum(m_sol_comp))

#defining the min and max radius for orbital zones
diff = np.diff(dist) #Calculates the difference in orbital distance between planets

#empty arrays to store values
r_min = np.zeros_like(dist) 
r_max = np.zeros_like(dist)

#Zone boundaries are halfway between each planet i.e. diff/2
#Also have to define R min for Mercury and R max for Neptune by extending equal distances in and out.
r_min[0] = dist[0] - diff[0]/2
r_min[1:] = dist[1:] - diff/2 #Mercury

r_max[0:-1] = dist[0:-1] + diff/2
r_max[-1] = dist[-1] + diff[-1]/2 #Neptune

print('rmin:', r_min)
print('rmax:', r_max)

#zone area = area of annulus. Distances converted to cgs
area = np.pi*((1.49e13*r_max)**2 - (1.49e13*r_min)**2)
print('Area:',area )

#zone surface density in g/cm^3
sur_den = (m_sol_comp*1e26/area)
print('Surface Density: ',sur_den)

#defining power law function for plotting fit
def power_law(r, sigma_0, alpha):
    sigma_r = (sigma_0)*(r)**(alpha)
    return sigma_r

#function takes list of distances and surface densities, fits a straight line to log-log plot, and returns
#parameters of power law and errors in those parameters
def fitter(dist, sigma):    
    log_x = np.log10(dist)
    log_y = np.log10(sigma)
    
    #fit returns slope, intercept and covariance matrix of the straight line fit
    fit = np.polyfit(log_x, log_y, 1, cov= True)
    
    slope = fit[0][0]
    intercept = fit[0][1]
    slope_stderr = np.sqrt(fit[1][0,0])
    intercept_stderr = np.sqrt(fit[1][1,1])
    
    #converting to ufloats for error propagation
    #see uncertainties package
    m = ufloat(slope, slope_stderr)
    c = ufloat(intercept, intercept_stderr)
    
    #converting back to powerlaw parameters
    A = 10**c
    
    #return power law parameters and errors
    return m.nominal_value, A.nominal_value, m.std_dev, A.std_dev

#fitting to distance and surface density    
alpha, A, alpha_err, A_err = fitter(dist, sur_den)

#removing Mercury, Mars and Asteroids values for reduced fit
red_dist = np.delete(dist, [0,3,4])
red_sur_den = np.delete(sur_den, [0,3,4])

red_alpha, red_A, red_alpha_err, red_A_err = fitter(red_dist, red_sur_den)

#power law integrand for numerical intergration
def integrand(r, a, b):
    sigma_r = 2*np.pi*r*a*(r/1.49e13)**(b)
    return sigma_r

#integrating surface density profile over disc for full and reduced fits to find mass of solar nebula
M_int = quad(integrand, r_min[0]*1.49e13, r_max[-1]*1.49e13, args = (A, alpha))
red_M_int = quad(integrand, r_min[0]*1.49e13, r_max[-1]*1.49e13, args = (A, alpha))
print('Total Int Min Mass: ', M_int)
print('Reduced Int Min Mass: ', M_int)

#converting zone boundaries to 'errors' for plotting
x_err_low = dist-r_min
x_err_up = r_max-dist

#plotting
#Mass of solar comp plot
figure1 = plt.figure(figsize = (10,6))
ax1 = figure1.add_subplot(111)
sol_comp_scat = ax1.scatter(dist, m_sol_comp, marker = 'o', color = 'blue')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.xaxis.set_major_formatter(ScalarFormatter())
ax1.ticklabel_format(axis = 'x', useOffset=False, style='plain')
for i, txt in enumerate(planet):
    ax1.annotate(txt, (dist[i], m_sol_comp[i]+1), xytext = (0,5), textcoords = 'offset points')
plt.xlim(.3,60)
plt.xlabel('r (AU)')
plt.ylabel('Mass (x$10^{26}$ g)')
plt.savefig('M_solar_comp.png')

#Surface density plots
figure2 = plt.figure(figsize = (10,6))
ax1 = figure2.add_subplot(111)
sol_comp_scat = ax1.errorbar(dist, sur_den, xerr = np.vstack((x_err_low, x_err_up)), 
                             fmt = 'o', label = 'Planets',color = 'black')
fit = ax1.plot(dist, power_law(dist, A, alpha), label = 'Full Fit', color = 'red')
fit2 = ax1.plot(dist, power_law(dist, red_A, red_alpha), label = 'Reduced Fit', color = 'blue', linestyle = '--')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.xaxis.set_major_formatter(ScalarFormatter())
ax1.ticklabel_format(axis = 'x', useOffset=False, style='plain')
plt.xlabel('r (AU)')
plt.ylabel('$\sigma$ (g $cm^2$)')
plt.legend()
plt.savefig('surface_dense.png', dpi = 200)


plt.show()
