import pandas as pd
import numpy as np
from scipy import spatial
import re
from biopandas.pdb import PandasPdb
from biopandas.mol2 import PandasMol2
import numba as nb
from Bio.PDB import *


@nb.jit(nopython=True)
def mean_curvature(Jacobian, Hessian):
    """Suppose rho is a smooth function on a open subset, for a fixed point (x0, y0, z0), 
    rho = rho(x0, y0, z0) is a surface inside R^3 that (x0, y0, z0) lies in. We compute the mean curvature of this
    surface at (x0, y0, z0). For a sphere of radius r, the result is -1/r. 
    Parameters
    ----------
    Jacobian: a array of shape (3, ). 
        [rho_x, rho_y, rho_z]
    Hessian: a symmetric matrix of shape (3, 3)
        [[rho_xx, rho_xy, rho_xz],
         [rho_yx, rho_yy, rho_yz],
         [rho_zx, rho_zy, rho_zz]]
    Returns
    -------
    the mean curvature at (x0, y0, z0)
    """
    #if np.all(Jacobian == 0):
    #    print('zeros Jacobian is encountered', Jacobian)
    #print(Jacobian)
    g = np.sum(Jacobian**2)
    if g**(3/2) < np.finfo(np.float64).tiny:
        return 0
    mean_curvature = (1/(2*g**(3/2)))*(2*Jacobian[0]*Jacobian[1]*Hessian[0,1]+2*Jacobian[0]*Jacobian[2]*Hessian[0,2]+2*Jacobian[1]*Jacobian[2]*Hessian[1,2] - (Jacobian[1]**2+Jacobian[2]**2)*Hessian[0,0]-(Jacobian[0]**2+Jacobian[2]**2)*Hessian[1,1] - (Jacobian[0]**2+Jacobian[1]**2)*Hessian[2,2])
    return mean_curvature

@nb.njit
def exp_jacobian(R, R_j, eta, kappa):
    """Returns the Jacobian of exp kernel (equation (4)).
    theta(R, R_j, eta, kappa) := exp(-(|R-R_j|/eta)^kappa)
    Parameters
    ----------
    R: an array of shape (3, )
        R = (x, y, z)
    R_j: an array of shape (3, )
        the position of j-th atom R_j = (x_j, y_j, z_j)
    eta: real number > 0
    kappa: real number > 0
    
    Returns
    -------
    Jacobian: an array of shape (3, )
    """
    x, y, z = R
    x_j, y_j, z_j = R_j
    Jacobian =     np.array([-(kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*x - 2*x_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 1))/(2*eta*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)), 
              -(kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*y - 2*y_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 1))/(2*eta*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)), 
              -(kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 1))/(2*eta*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2))])
    return Jacobian

@nb.njit
def exp_hessian(R, R_j, eta, kappa):
    """Returns the Hessian of exp kernel.
    """
    x, y, z = R
    x_j, y_j, z_j = R_j
    Hessian =    np.array([[(kappa**2*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*x - 2*x_j)**2*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(2*kappa - 2))/(4*eta**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)) - (kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 1))/(eta*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)) + (kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*x - 2*x_j)**2*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 1))/(4*eta*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(3/2)) - (kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*x - 2*x_j)**2*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 2)*(kappa - 1))/(4*eta**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)),                                                                                                                                                                 (kappa**2*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*x - 2*x_j)*(2*y - 2*y_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(2*kappa - 2))/(4*eta**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)) + (kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*x - 2*x_j)*(2*y - 2*y_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 1))/(4*eta*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(3/2)) - (kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*x - 2*x_j)*(2*y - 2*y_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 2)*(kappa - 1))/(4*eta**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)),                                                                                                                                                                 (kappa**2*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*x - 2*x_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(2*kappa - 2))/(4*eta**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)) + (kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*x - 2*x_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 1))/(4*eta*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(3/2)) - (kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*x - 2*x_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 2)*(kappa - 1))/(4*eta**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2))],
              [                                                                                                                                                                 (kappa**2*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*x - 2*x_j)*(2*y - 2*y_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(2*kappa - 2))/(4*eta**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)) + (kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*x - 2*x_j)*(2*y - 2*y_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 1))/(4*eta*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(3/2)) - (kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*x - 2*x_j)*(2*y - 2*y_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 2)*(kappa - 1))/(4*eta**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)), (kappa**2*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*y - 2*y_j)**2*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(2*kappa - 2))/(4*eta**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)) - (kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 1))/(eta*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)) + (kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*y - 2*y_j)**2*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 1))/(4*eta*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(3/2)) - (kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*y - 2*y_j)**2*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 2)*(kappa - 1))/(4*eta**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)),                                                                                                                                                                 (kappa**2*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*y - 2*y_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(2*kappa - 2))/(4*eta**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)) + (kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*y - 2*y_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 1))/(4*eta*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(3/2)) - (kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*y - 2*y_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 2)*(kappa - 1))/(4*eta**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2))],
              [                                                                                                                                                                 (kappa**2*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*x - 2*x_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(2*kappa - 2))/(4*eta**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)) + (kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*x - 2*x_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 1))/(4*eta*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(3/2)) - (kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*x - 2*x_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 2)*(kappa - 1))/(4*eta**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)),                                                                                                                                                                 (kappa**2*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*y - 2*y_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(2*kappa - 2))/(4*eta**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)) + (kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*y - 2*y_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 1))/(4*eta*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(3/2)) - (kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*y - 2*y_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 2)*(kappa - 1))/(4*eta**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)), (kappa**2*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*z - 2*z_j)**2*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(2*kappa - 2))/(4*eta**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)) - (kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 1))/(eta*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)) + (kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*z - 2*z_j)**2*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 1))/(4*eta*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(3/2)) - (kappa*np.exp(-(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**kappa)*(2*z - 2*z_j)**2*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(kappa - 2)*(kappa - 1))/(4*eta**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2))]])
    return Hessian


@nb.njit
def lorentz_jacobian(R, R_j, eta, nu):
    x, y, z = R
    x_j, y_j, z_j = R_j
    Jacobian =np.array([ -(nu*(2*x - 2*x_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 1))/(2*eta*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)), -(nu*(2*y - 2*y_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 1))/(2*eta*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)), -(nu*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 1))/(2*eta*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2))])
    return Jacobian

@nb.njit
def lorentz_hessian(R, R_j, eta, nu):
    x, y, z = R
    x_j, y_j, z_j = R_j
    Hessian =np.array([[ (nu**2*(2*x - 2*x_j)**2*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(2*nu - 2))/(2*eta**2*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**3*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)) - (nu*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 1))/(eta*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)) + (nu*(2*x - 2*x_j)**2*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 1))/(4*eta*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(3/2)) - (nu*(2*x - 2*x_j)**2*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 2)*(nu - 1))/(4*eta**2*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)),                                                                                                                                                          (nu**2*(2*x - 2*x_j)*(2*y - 2*y_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(2*nu - 2))/(2*eta**2*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**3*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)) + (nu*(2*x - 2*x_j)*(2*y - 2*y_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 1))/(4*eta*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(3/2)) - (nu*(2*x - 2*x_j)*(2*y - 2*y_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 2)*(nu - 1))/(4*eta**2*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)),                                                                                                                                                          (nu**2*(2*x - 2*x_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(2*nu - 2))/(2*eta**2*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**3*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)) + (nu*(2*x - 2*x_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 1))/(4*eta*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(3/2)) - (nu*(2*x - 2*x_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 2)*(nu - 1))/(4*eta**2*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2))],
              [                                                                                                                                                          (nu**2*(2*x - 2*x_j)*(2*y - 2*y_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(2*nu - 2))/(2*eta**2*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**3*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)) + (nu*(2*x - 2*x_j)*(2*y - 2*y_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 1))/(4*eta*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(3/2)) - (nu*(2*x - 2*x_j)*(2*y - 2*y_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 2)*(nu - 1))/(4*eta**2*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)), (nu**2*(2*y - 2*y_j)**2*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(2*nu - 2))/(2*eta**2*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**3*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)) - (nu*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 1))/(eta*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)) + (nu*(2*y - 2*y_j)**2*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 1))/(4*eta*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(3/2)) - (nu*(2*y - 2*y_j)**2*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 2)*(nu - 1))/(4*eta**2*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)),                                                                                                                                                          (nu**2*(2*y - 2*y_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(2*nu - 2))/(2*eta**2*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**3*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)) + (nu*(2*y - 2*y_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 1))/(4*eta*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(3/2)) - (nu*(2*y - 2*y_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 2)*(nu - 1))/(4*eta**2*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2))],
              [                                                                                                                                                          (nu**2*(2*x - 2*x_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(2*nu - 2))/(2*eta**2*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**3*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)) + (nu*(2*x - 2*x_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 1))/(4*eta*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(3/2)) - (nu*(2*x - 2*x_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 2)*(nu - 1))/(4*eta**2*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)),                                                                                                                                                          (nu**2*(2*y - 2*y_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(2*nu - 2))/(2*eta**2*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**3*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)) + (nu*(2*y - 2*y_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 1))/(4*eta*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(3/2)) - (nu*(2*y - 2*y_j)*(2*z - 2*z_j)*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 2)*(nu - 1))/(4*eta**2*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)), (nu**2*(2*z - 2*z_j)**2*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(2*nu - 2))/(2*eta**2*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**3*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)) - (nu*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 1))/(eta*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)) + (nu*(2*z - 2*z_j)**2*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 1))/(4*eta*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(3/2)) - (nu*(2*z - 2*z_j)**2*(((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**(nu - 2)*(nu - 1))/(4*eta**2*((((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2)**(1/2)/eta)**nu + 1)**2*((x - x_j)**2 + (y - y_j)**2 + (z - z_j)**2))]])
    return Hessian

@nb.jit(nopython=True)
def stats_extracter(array):
    """
    Parameters
    ----------
    array: a 1d np array.
    Returns
    -------
    stats: a 1d array of length 10.
        [sum, 
        the sum of absolute values, 
        minimum, 
        the minimum of absolute values, 
        maximum, 
        the maximum of absolute values, 
        mean, 
        the mean of absolute values, 
        standard deviation,
        the standard deviation of absolute values]
    """
    return np.array([np.sum(array), 
           np.sum(np.abs(array)),
           np.amin(array),
           np.amin(np.abs(array)),
           np.amax(array),
           np.amax(np.abs(array)),
           np.mean(array),
           np.mean(np.abs(array)),
           np.std(array),
           np.std(np.abs(array))])


# In[3]:


def read_pdb(path):
    """
    Parameters
    ----------
    path: a string of file path.
    Returns
    -------
    atom_df: pandas dataframe of atom record type data
    L: dictionary of coordinates.
    """
    # use biopython to load atom data
    structure = PDBParser(QUIET=True).get_structure('temp', path)
    # use biopandas to load atom data from a pdb file.
    ppdb = PandasPdb().read_pdb(path)
    atom_df = ppdb.df['ATOM']

    L = {'C':[], 'N':[], 'O':[], 'S':[]}
    i = 0
    for atom in structure.get_atoms():
        atom_type = atom.get_fullname()
        if atom_type[:2] == ' C':
            L['C'].append(i)
        elif atom_type[:2] == ' N':
            L['N'].append(i)
        elif atom_type[:2] == ' O':
            L['O'].append(i)
        elif atom_type[:2] == ' S':
            L['S'].append(i)   
        i += 1
        if i >= atom_df.shape[0]:
            break
    return atom_df, L
        

def read_mol2(path):
    """
    Parameters
    ----------
    path: a string of file path.
    Returns
    -------
    atom_df: pandas dataframe 
    L: dictionary of coordinates.
    """
    # use biopandas to load atom data from a mol2 file.
    pmol = PandasMol2().read_mol2(path)
    atom_df = pmol.df
    L = {'H':[], 'C':[], 'N':[], 'O':[], 'S':[], 'P':[], 'F':[], 'Cl':[], 'Br':[], 'I':[]}
    for i in np.arange(atom_df.shape[0]):
        atom_type = atom_df.iloc[i]['atom_type']
        if atom_type[0] == 'H':
            L['H'].append(i)
        elif atom_type[0] == 'C':
            if len(atom_type) == 1:
                L['C'].append(i)
            elif atom_type[1] == '.':
                L['C'].append(i)
            elif atom_type[1] == 'l':
                L['Cl'].append(i)
        elif atom_type[0] == 'N':
            if len(atom_type) == 1:
                L['N'].append(i)
            elif atom_type[1] == '.':
                L['N'].append(i)
        elif atom_type[0] == 'O':
            L['O'].append(i)  
        elif atom_type[0] == 'S':
            if len(atom_type) == 1:
                L['S'].append(i)  
            elif atom_type[1] == '.':
                L['S'].append(i) 
        elif atom_type[0] == 'P':
            L['P'].append(i)  
        elif atom_type[0] == 'F':
            if len(atom_type) == 1:
                L['F'].append(i)  
            elif atom_type[1] == '.':
                L['F'].append(i)  
        elif atom_type[:2] == 'Br':
            L['Br'].append(i)  
        elif atom_type[0] == 'I':
            L['I'].append(i)  
    return atom_df, L


# In[22]:


class protein_ligand_feature_generator():
    
    def __init__(self, kernel = 'exp', curvature = 'mean', beta=1, tau=1):
        """
        Note: only consider distance <= 20 Angstrom cut off distance. Do not consider
        distance > r_i + r_j + sigma in the paper. 
        kernel: str
            'exp' represents generalized exponential functions and 
            'lorentz' represents generalized Lorentz functions.
        curvature: str
            'mean' representes the mean curvature. 'gaussian' represents Gaussian.
        beta: = kappa if kernel == 'exp'. not used in Lorentz kernel.
              = nu if kernel == 'lorentz'. not used in exp kernel.
        atom_radii: in protein-ligand problem, vdw_radii are employed.
        """
        self.kernel = kernel
        self.curvature = curvature
        self.beta = beta
        self.tau = tau
        self.vdw_radii_dict = {
            'H':1.2,
            'C':1.7,
            'N':1.55,
            'O':1.52,
            'F':1.47,
            'P':1.8,
            'S':1.8,
            'Cl':1.75,
            'Br':1.85,
            'I':1.98
}
        
    def generator(self, protein_path, ligand_path):
        """EIC for protein-ligand binding affinity prediction.
        Parameters
        ----------
        protein_path: a str.    
        ligand_path: a str.
        Returns
        -------
        features: a 1d array of length 400.
            400 features for a certain protein-ligand complex that are going to be 
            incorporated into a dictionary.
        """
        
        patom_df, patom_dict = read_pdb(protein_path) # dictionary of coords
        latom_df, latom_dict = read_mol2(ligand_path) # dictionary of coords

        patom_list = ['C', 'N', 'O', 'S']
        latom_list = ['H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
        features = np.zeros(400)
        for i, patom_type in enumerate(patom_list):
            for j, latom_type in enumerate(latom_list):
                protein_coords = patom_df.iloc[patom_dict[patom_type]][['x_coord','y_coord','z_coord']]
                protein_coords = protein_coords.to_numpy()
                ligand_coords = latom_df.iloc[latom_dict[latom_type]][['x','y','z']]
                ligand_coords = ligand_coords.to_numpy()
                features[(i*10+j)*10:(i*10+j)*10+10] = stats_extracter(self.EIC(protein_coords, ligand_coords, patom_type, latom_type))
        return features 
    
    def EIC(self, protein_coords, ligand_coords, patom_type, latom_type):
        """
        Parameters
        ----------
        protein_coords: a 2d array
            the coordinates of a specific atom type in a protein, eg. C
        ligand_coords: a 2d array
            the coordinates of a specific atom type in a ligand, eg. N
        protein_atom_type: C, N, O, S.
        ligand_atom_type: H, C, N, O, S, P, F, Cl, Br, I
        Returns
        -------
        EICs: a 1d np array 
            It contains the curvature evaluated at any atom in the binding site.
            if either protein_coords or ligand_coords is empty array, return 
            np.zeros(1)
        """
        if protein_coords.size == 0 or ligand_coords.size == 0:
            return np.zeros(1)
        
        dists = spatial.distance.cdist(protein_coords, ligand_coords)
        # find the binding site.
        protein_coords = protein_coords[np.any(dists <= 20, axis = 1)]
        ligand_coords = ligand_coords[np.any(dists <= 20, axis = 0)]
        # if the binding site is empty
        if protein_coords.size == 0:
            return np.zeros(1)

        eta = self.tau*(self.vdw_radii_dict[patom_type] + self.vdw_radii_dict[latom_type])
        dists = spatial.distance.cdist(protein_coords, ligand_coords)
        
        EICs = np.zeros(protein_coords.shape[0]+ ligand_coords.shape[0])
        # For curvature of protein atoms
        Js = np.zeros((protein_coords.shape[0], ligand_coords.shape[0],3))
        Hs = np.zeros((protein_coords.shape[0], ligand_coords.shape[0],3,3))
        # for curvature of ligand atoms
        Js_ = np.zeros((ligand_coords.shape[0], protein_coords.shape[0],3))
        Hs_ = np.zeros((ligand_coords.shape[0], protein_coords.shape[0],3,3))  
        
        if self.kernel == 'exp':
            for i in np.arange(protein_coords.shape[0]):
                for j in np.arange(ligand_coords.shape[0]):
                    # cut off distance is 20 Angstrom
                    if dists[i, j] <= 20:
                        Js[i,j] = exp_jacobian(protein_coords[i], ligand_coords[j], eta, self.beta)
                        Hs[i,j] = exp_hessian(protein_coords[i], ligand_coords[j], eta, self.beta)
                        Js_[j,i] = exp_jacobian(ligand_coords[j], protein_coords[i], eta, self.beta)
                        Hs_[j,i] = exp_hessian(ligand_coords[j], protein_coords[i], eta, self.beta)
                    else:
                        Js[i,j] = np.zeros(3)
                        Hs[i,j] = np.zeros((3,3)) 
                        Js_[j,i] = np.zeros(3)
                        Hs_[j,i] = np.zeros((3,3))    
                        
        if self.kernel == 'lorentz':
            for i in np.arange(protein_coords.shape[0]):
                for j in np.arange(ligand_coords.shape[0]):
                    # cut off distance is 20 Angstrom
                    if dists[i, j] <= 20:
                        Js[i,j] = lorentz_jacobian(protein_coords[i], ligand_coords[j], eta, self.beta)
                        Hs[i,j] = lorentz_hessian(protein_coords[i], ligand_coords[j], eta, self.beta)
                        Js_[j,i] = lorentz_jacobian(ligand_coords[j], protein_coords[i], eta, self.beta)
                        Hs_[j,i] = lorentz_hessian(ligand_coords[j], protein_coords[i], eta, self.beta)
                    else:
                        Js[i,j] = np.zeros(3)
                        Hs[i,j] = np.zeros((3,3)) 
                        Js_[j,i] = np.zeros(3)
                        Hs_[j,i] = np.zeros((3,3))   
        
        
        # if self.curvature == 'mean'. Here we haven't implemented gaussian curvature yet.
        for i in np.arange(protein_coords.shape[0]):
            EICs[i] = mean_curvature(np.sum(Js[i,:], axis=0), np.sum(Hs[i,:], axis=0))
        for j in np.arange(ligand_coords.shape[0]):
            EICs[j+protein_coords.shape[0]] = mean_curvature(np.sum(Js_[j,:], axis=0), np.sum(Hs_[j,:], axis=0))
        
        return EICs          

