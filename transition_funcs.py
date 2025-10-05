# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 13:00:05 2021

@author: Administrator
"""
import numpy as np

def reform_mat_to_flat(matrix):
    dims = matrix.shape
    if len(dims) == 3 and dims[1] == dims[2]:
        flat = np.empty(dims)
        mre = matrix.real
        mim = matrix.imag
        for i in range(dims[1]):
            for j in range(dims[2]):
                if j >= i:
                    flat[:, i, j] = mre[:, i, j]
                elif j < i:
                    flat[:, i, j] = mim[:, i, j]
        flat = flat.reshape(dims[0], dims[1] * dims[2])
        return flat
    else:
        print("Wrong matrix shape!")

def reform_flat_to_mat(flat):
    dims = flat.shape
    a = [4, 9, 16, 25, 36, 49, 64, 81, 100, 144, 256]
    if len(dims) == 2 and dims[1] in a:
        dim = int(np.sqrt(dims[1]))
        flat = flat.reshape(dims[0], dim, dim)
        mre = np.empty((dims[0], dim, dim))
        mim = np.empty((dims[0], dim, dim))
        for i in range(dim):
            for j in range(dim):
                if j == i:
                    mre[:, i, j] = flat[:, i, j]
                    mim[:, i, j] = 0
                elif j > i:
                    mre[:, i, j] = flat[:, i, j]
                    mim[:, i, j] = -flat[:, j, i]
                elif j < i:
                    mre[:, i, j] = flat[:, j, i]
                    mim[:, i, j] = flat[:, i, j]
        ma = mre + 1j * mim
        return ma
    else:
        print("Wrong data shape!")
        

def sv_to_matrix(svs):
    dims = svs.shape
    if len(dims) != 2:
        print("Wrong data shape! Failed to convert!")
        return
    dms = []
    for sv in svs:
        dm = np.outer(sv, np.conj(sv))
        dms.append(dm)
    
    dms = np.array(dms)
    return dms

def cons_pure_sep(svA, svB):
    """
    Construct separable pure states using svA and svB.
    
    Parameters
    ----------
    svA : TYPE numpy.ndarray
        DESCRIPTION coeficients of state A. 
    svB : TYPE numpy.ndarray
        DESCRIPTION coeficients of state B. The number in axis=0 dimension 
        should be the same for A and B.

    Returns
    -------
    results : TYPE ndarray
             DESCRIPTION Coefficients of state AB: svAB.

    """
    shapeA, shapeB = svA.shape, svB.shape
    
    if not (len(shapeA) == 2 and len(shapeB) == 2):
        print("Improper shapes of A and/or B!")
        return
    if shapeA[0] != shapeB[0]:
        print("Sample number of A is not the same as that of B!")
        return
    
    svAB = []
    for A, B in zip(svA, svB):    
        ABouter = np.outer(A, B)    
        AB = ABouter.reshape(-1)
        svAB.append(AB)
        
    svAB = np.array(svAB)
    svAB[np.abs(svAB) == 0.] = 0.
    
    return svAB

def reduced_matrix(svAB, dimA, dimB):
    """
    Calculate reduced matrix.
    
    """
    svAB = svAB.reshape(-1, dimA, dimB)
    
    rdA = []
    rdB = []
    labels = []
    for sv in svAB:
        dmAB = np.tensordot(sv, np.conj(sv), axes=0)
        dmAB_ptoA = np.trace(dmAB, axis1=0, axis2=2) # ptoA: shorthand for partial traced over A
        dmAB_ptoB = np.trace(dmAB, axis1=1, axis2=3) # ptoB: shorthand for partial traced over B

        # create labels
        if np.linalg.matrix_rank(dmAB_ptoA) == 1:
            labels.append(0)
        elif np.linalg.matrix_rank(dmAB_ptoA) > 1:
            labels.append(1)
        else:
            print("unexpected rank of dmAB_ptoA!")
            break
        
        rdA.append(dmAB_ptoB)
        rdB.append(dmAB_ptoA)
        
    labels = np.array(labels)
    rdA = np.array(rdA)
    rdB = np.array(rdB)
    results = (rdA, rdB, labels)
    return results
        
def sort_ndarr(a, i):
    ind = np.argsort(a[:, i])
    return a[ind]
        
        
        
        
    