#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
from scipy import linalg, spatial
import re
from biopandas.pdb import PandasPdb
from biopandas.mol2 import PandasMol2
import numba as nb
from Bio.PDB import *



@nb.njit
def exp_kernel(d, eta, kappa): 
    """
    d: distance of two points.
    """
    return np.exp(-(d/eta)**kappa)

@nb.njit
def lorentz_kernel(d, eta, nu):
    """
    d: distance of two points.
    """
    return 1/(1+(d/eta)**nu)

@nb.njit
def stats_extracter(array):
    """
    Parameters
    ----------
    array: a 1d np array.
    Returns
    -------
    stats: a 1d array of length 9.
        [sum, 
        minimum, 
        maximum, 
        mean, 
        median,
        standard deviation,
        variance,
        number of eigenvalues
        sum of second power of eigenvalues]
    """
    if array == None:
        return np.zeros(9)
    return np.array([np.sum(array),
                    np.amin(array),
                    np.amax(array),
                    np.mean(array),
                    np.median(array),
                    np.std(array),
                    np.var(array),
                    array.size,
                    np.sum(array**2)])

@nb.njit
def eig_solver(mat):
    return np.linalg.eigvalsh(mat)


def read_pdb(path):
    """Returns a list L of np array. 
        L[0], L[1], L[2], and L[3] corresponds to C, N, O, and S respectively.
        For example, each row of L[0] represents a coordinate of C atom in the protein.
        Note that we ignore the HETATM record type.
        Parameters
        ----------
        path: a string of file path.
    """
    # use biopandas to load atom data from a pdb file.
    structure = PDBParser(QUIET=True).get_structure('temp', path)
    ppdb = PandasPdb().read_pdb(path)
    atom_df = ppdb.df['ATOM']

    L = {'C':[], 'N':[], 'O':[], 'S':[]}
    i = 0
    for atom in structure.get_atoms():
        atom_type = atom.get_fullname()[1]
        if atom_type == 'C':
            L['C'].append(i)
        elif atom_type == 'N':
            L['N'].append(i)
        elif atom_type == 'O':
            L['O'].append(i)
        elif atom_type == 'S':
            L['S'].append(i)   
        i += 1
        if i >= atom_df.shape[0]:
            break
    return atom_df, L

def read_mol2(path):
    """ 
        ----------
        path: a string of file path.
    """
    # use biopandas to load atom data from a mol2 file.
    pmol = PandasMol2().read_mol2(path)
    atom_df = pmol.df
    L = {'H':[], 'C':[], 'N':[], 'O':[], 'S':[], 'P':[], 'F':[], 'Cl':[], 'Br':[], 'I':[]}
    for i in np.arange(atom_df.shape[0]):
        atom_type = atom_df.iloc[i]['atom_type']
        if atom_type[0] == 'H':
            if len(atom_type) == 1:    
                L['H'].append(i)
            elif atom_type[1] == '.':
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
            if len(atom_type) == 1:
                L['O'].append(i)  
            elif atom_type[1] == '.':
                L['O'].append(i) 
        elif atom_type[0] == 'S':
            if len(atom_type) == 1:
                L['S'].append(i)  
            elif atom_type[1] == '.':
                L['S'].append(i) 
        elif atom_type[0] == 'P':
            if len(atom_type) == 1:
                L['P'].append(i) 
            elif atom_type[1] == '.':
                L['P'].append(i) 
        elif atom_type[0] == 'F':
            if len(atom_type) == 1:
                L['F'].append(i)  
            elif atom_type[1] == '.':
                L['F'].append(i)  
        elif atom_type[:2] == 'Br':
            L['Br'].append(i)  
        elif atom_type[0] == 'I':
            if len(atom_type) == 1:
                L['I'].append(i) 
            elif atom_type[1] == '.':
                L['I'].append(i) 
    return atom_df, L

class feature_generator():
    
    def __init__(self, kernel = 'exp', matrix = 'lap', beta=1, tau=1):
        """
        Note: only consider distance < cut off distance. Do not consider
        distance > r_i + r_j in the paper. 
        kernel: str
            'exp' represents generalized exponential functions and 
            'lorentz' represents generalized Lorentz functions.
        matrix: str
            'lap' representes the laplacian. 
            'gaussian' represents adjacency.
            'inv' represents pseudo inverse of laplacian
        beta: = kappa if kernel == 'exp'. not used in Lorentz kernel.
              = nu if kernel == 'lorentz'. not used in exp kernel.
        atom_radii: in protein-ligand problem, vdw_radii are employed.
        """
        self.kernel = kernel
        self.matrix = matrix
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
        """
        Parameters
        ----------
        
        Returns
        -------
        features: a 1d array of length 360.
            400 features for a certain protein-ligand complex that are going to be 
            incorporated into a dictionary.
            List data type is used because we need to build a np array from a dictionary later on. 
        """
        
        patom_df, patom_dict = read_pdb(protein_path) # dictionary of coords
        latom_df, latom_dict = read_mol2(ligand_path) # dictionary of coords

        patom_list = ['C', 'N', 'O', 'S']
        latom_list = ['H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
        features = np.zeros(360)
        for i, patom_type in enumerate(patom_list):
            for j, latom_type in enumerate(latom_list):
                protein_coords = patom_df.iloc[patom_dict[patom_type]][['x_coord','y_coord','z_coord']]
                protein_coords = protein_coords.to_numpy()
                ligand_coords = latom_df.iloc[latom_dict[latom_type]][['x','y','z']]
                ligand_coords = ligand_coords.to_numpy()
                features[(i*10+j)*9:(i*10+j)*9+9] = stats_extracter(self.eigenvalues(protein_coords, ligand_coords, patom_type, latom_type))
        return features 
    
    def eigenvalues(self, protein_coords, ligand_coords, patom_type, latom_type):
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
        eigenvalues: a 1d np array of nontrivial eigenvalues.
        """
        if protein_coords.size == 0 or ligand_coords.size == 0:
            return None
        eta = self.tau*(self.vdw_radii_dict[patom_type] + self.vdw_radii_dict[latom_type])        
        dists = spatial.distance.cdist(protein_coords, ligand_coords)
        # cut off distance 
        cut_off = 12
        protein_coords = protein_coords[np.any(dists <= cut_off, axis = 1)]
        if protein_coords.size == 0:
            return None 
        dists = spatial.distance.cdist(protein_coords, ligand_coords)
        
        mat = np.zeros((protein_coords.shape[0]+ligand_coords.shape[0], protein_coords.shape[0]+ligand_coords.shape[0]))
        
        if self.kernel == 'exp':
            if self.matrix == 'lap':
                for i in range(protein_coords.shape[0]):
                    for j in range(ligand_coords.shape[0]):
                        mat[i, protein_coords.shape[0]+j] = -exp_kernel(dists[i,j], eta, self.beta)
                        mat[protein_coords.shape[0]+j, i] = mat[i, protein_coords.shape[0]+j]
                for i in range(mat.shape[0]):
                    mat[i,i] = -np.sum(mat[i,:])
                
                eigvals = eig_solver(mat)
                # Take care of numerical error. Very small eigenvalues are seen as zero.
                eigvals = eigvals[eigvals > 1e-5]
                if eigvals.size == 0:
                    return None
                return eigvals
            
            if self.matrix == 'adj':
                for i in range(protein_coords.shape[0]):
                    for j in range(ligand_coords.shape[0]):
                        mat[i, protein_coords.shape[0]+j] = -exp_kernel(dists[i,j], eta, self.beta)
                        mat[protein_coords.shape[0]+j, i] = mat[i, protein_coords.shape[0]+j]
                eigvals = eig_solver(mat)
                eigvals = eigvals[eigvals > 1e-5]
                if eigvals.size == 0:
                    return None
                return eigvals
            
        if self.kernel == 'lorentz':
            if self.matrix == 'lap':
                for i in range(protein_coords.shape[0]):
                    for j in range(ligand_coords.shape[0]):
                        mat[i, protein_coords.shape[0]+j] = -lorentz_kernel(dists[i,j], eta, self.beta)
                        mat[protein_coords.shape[0]+j, i] = mat[i, protein_coords.shape[0]+j]
                for i in range(mat.shape[0]):
                    mat[i,i] = -np.sum(mat[i,:])
                
                eigvals = eig_solver(mat)
                eigvals = eigvals[eigvals > 1e-5]
                if eigvals.size == 0:
                    return None
                return eigvals
            
            if self.matrix == 'adj':
                for i in range(protein_coords.shape[0]):
                    for j in range(ligand_coords.shape[0]):
                        mat[i, protein_coords.shape[0]+j] = -lorentz_kernel(dists[i,j], eta, self.beta)
                        mat[protein_coords.shape[0]+j, i] = mat[i, protein_coords.shape[0]+j]
                
                eigvals = eig_solver(mat)
                eigvals = eigvals[eigvals > 1e-5]
                if eigvals.size == 0:
                    return None
                return eigvals

