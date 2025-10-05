# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 19:36:56 2021

@author: Administrator
"""

import numpy as np

def puregen(sample_number=1000, dims=(2, 2)):
    """generate random subdim-dimensional states in dim-dimension space
    sample_number: int; sample numbers.
    dims: int or tuple/list of integers; if it is a tuple, the first (dims[0]) and the last (dims[-1]) 
          values are taken; dims[-1] is the dimensionality of the state; dims[0] is the dimensionality
          with none-zero coefficients. dims[-1] should be >= dims[0].
    """
    # resolve parameters
    if isinstance(dims, int):
        subdim, dim = dims, dims
    elif dims[-1] >= dims[0]:
        subdim, dim = dims[0], dims[-1]
    else:
        print('''Wrong parameters! Parameter "dims" should be a int or a tuple 
              of (subdim, dim) with dim >= subdim.''')
        return      
    # initial the random number generator
    rng = np.random.default_rng()        
                      
    # generate coefficients       
    if subdim == 1:
        coes = np.ones(sample_number).reshape(-1, 1)
    elif subdim >=2:
        # generate modules of the coefficients
        a = rng.standard_normal((sample_number,subdim)) 
        # normalization
        norm = np.linalg.norm(a, axis=1, keepdims=True)
        a = np.abs(a / norm)
        # generate the relative phases
        p = rng.uniform(low=-np.pi, high=np.pi, size=(sample_number, subdim-1))
        
        # insert global phase 0
        phases = ins_globalphase(a, p)
               
        # calculate coefficients
        exph = np.exp(1j * phases)
        coes = np.multiply(a, exph)
    
    # fill up spared-dimensions with zeros
    sparedim = int(dim - subdim)
    if sparedim >= 1:
        temp = fill_sparedims(sparedim, coes)
        coes = rng.permuted(temp, axis=1)
            
    return coes

def ins_globalphase(a, p):
    # locate maximum coefficients
    loc_maxcoe = np.argmax(a, axis=1)
    # insert global phase 0
    phases = []
    for index, arr in zip(loc_maxcoe, p):
        newarr = np.insert(arr, index, 0)
        phases.append(newarr)
    phases = np.array(phases)
    return phases

def fill_sparedims(sparedim, coes):
    # fill up spared-dimensions with zeros
    row_number = coes.shape[0]
    sparecoes = np.zeros((row_number, sparedim))
    y = np.hstack((coes, sparecoes))
    return y

    