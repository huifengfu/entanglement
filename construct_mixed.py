# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 15:01:21 2021

@author: Administrator
"""

import numpy as np
from datetime import datetime
import json
import entanglement.puregen as pg
from entanglement.transition_funcs import *

class MixedGenerator:
    """ Generate mixed states of AB, separable and non PPT entangled.
    Attributes:
        dimA, dimB, dimAB, sample_num, 
    
    """
    def __init__(self, dimA=2, dimB=2):
        """        
        dimA : TYPE int, optional
        DESCRIPTION. The default is 2. Should be equal or greater than 2.
        dimB : TYPE int, optional
        DESCRIPTION. The default is 2. Should be equal or greater than 2.
        """
        # parameter check
        if dimA < 2 or dimB < 2:
            print("Failed to create the instance. \
                  dimA and dimB should be equal or grearter than 2.")
            return
        self.dimA = int(dimA)
        self.dimB = int(dimB)  

    def gen_samples(self, sam_subnum=1000):
        """
        Generate samples from _direct_genAB and _gen_sepAB together.

        """
        # parameter check
        if sam_subnum < 1:
            print("Failed to generate. Sample number per subspace should be \
                  equal or greater than 1.")
            return
        # list of mix_nums
        mix_nums = list(range(2, 8, 2))
        # generate samples using _direct_genAB
        data1 = []
        for mix_num in mix_nums:
            self._direct_genAB(sam_subnum, mix_num)
            # store the data
            data1.append(self._data)
        data1 = np.vstack(data1)
        
        # data structure
        datastruc = [data1.shape[0]]
        
        # generate samples using _gen_sepAB
        data2 = []
        for mix_num in mix_nums:
            self._gen_sepAB(sam_subnum, mix_num)
            data2.append(self._data)        
        data2 = np.vstack(data2)
        
        datastruc.append(data2.shape[0])
        # combine
        self.fulldata = np.vstack((data1, data2))
        
        # generate bound entangled states
        if self.dimA == 3 and self.dimB == 3:
            self._const_bound()
            data3 = self._data
            datastruc.append(data3.shape[0])
            # combine
            self.fulldata = np.vstack((data1, data2, data3))                
                
        # other sample information
        self.sample_structure = datastruc
        print(f"sample structure:\n{self.sample_structure}")
        self.sample_subnum = sam_subnum
        self.sample_num = self.fulldata.shape[0]
        # create feature names
        self._create_feature_names()
        self.datatag = 'mix'
        self.results = {'data': self.fulldata.tolist(), 
                   'feature_names': self.feature_names.tolist(), 
                   'dimA': self.dimA, 
                   'dimB': self.dimB, 
                   'sample_num': self.sample_num,
                   'sample_subnum': self.sample_subnum,
                   'sample_structure': self.sample_structure,
                   'datatag': self.datatag,
                   }
        print(f"{self.sample_num} samples ({self.dimA}*{self.dimB}) are generated!")
        
    def to_json(self):
        now = datetime.now()
        nowstr = now.strftime("%b%d%H%M%S")
        filename = (f"mix_{self.sample_num}"+
            f"_{self.dimA}{self.dimB}_{nowstr}.json")
        with open(f"data\{filename}", 'w') as f:
            json.dump(self.results, f)
         
        
    def _direct_genAB(self, sam_subnum=1000, mix_num=4):
        """
        Generate mixed states of AB directly. Wheather these states are separable or
        entangled are unknown.
    
        Parameters
        ----------
        mix_num : TYPE int, optional
    
        """

        # parameter check
        if mix_num <= 1:
            print("Failed to generate. mix_num takes values larger than 1")
            return
        
        dimAB = self.dimA * self.dimB 
        # generate pure state samples
        sv = pg.puregen(sample_number=sam_subnum*mix_num, 
                            dims=dimAB)
        
        # construct mixed states
        self._const_mix(sv, sam_subnum, mix_num)
        # discard ppt states to ensure the states are entangled
        checker = (self._pptlabels == np.ones(sam_subnum))
        self._dmAB = self._dmAB[checker]
        num = self._dmAB.shape[0]
        labels = np.ones(num)        
        # store data
        flatdmAB = reform_mat_to_flat(self._dmAB)
        self._data = np.hstack((flatdmAB, labels.reshape(-1, 1)))
        print(f"{num} Nppt states generated.")
 

    def _gen_sepAB(self, sam_subnum=1000, mix_num=4):
        """
        Generate separable mixed states. 

        """
        # parameter check
        if mix_num <= 1:
            print("Failed to generate. mix_num takes values larger than 1")
            return

        dimA = self.dimA
        dimB = self.dimB
        # generate pure state samples
        svA_ori = pg.puregen(sample_number=sam_subnum*mix_num, dims=dimA)                         
        svB_ori = pg.puregen(sample_number=sam_subnum*mix_num, dims=dimB)
        svAB_ori = cons_pure_sep(svA_ori, svB_ori)
        # construct mixed states
        self._const_mix(svAB_ori, sam_subnum, mix_num)
        labels = np.zeros(sam_subnum)
        checker = (self._pptlabels == labels)
        if checker.all():    
            self.wrong_labels = None
            print(f"label checked! {sam_subnum} separable states generated.")
        else:
            self.wrong_labels = np.nonzero(checker == 0)
            print("label unmatch! check self.wrong_labels!")
        # store data    
        flatdmAB = reform_mat_to_flat(self._dmAB)
        self._data = np.hstack((flatdmAB, labels.reshape(-1, 1)))
            
    def _const_bound(self):
        if self.dimA != 3 or self.dimB != 3:
            return
        # the bound entangled states are taken from Sixia Yu and C. H. Oh, 
        # PhysRevA.95.032111(2017) Eqs. (3-7)
        # basis   00     01 02 10     11          12       20      21            22
        # psi_3    0      0  0  0      0       -1/sqrt(2)   0     1/sqrt(2)      0
        # psi_0 1/sqrt(3) 0  0  0   1/sqrt(3)     0         0       0         1/sqrt(3)
        # psi_1    0      x  0  y    0         -z/sqrt(2)   0    -z/sqrt(2)      0
        # psi_2    0      0  x  0  -z/sqrt(2)     0         y       0         z/sqrt(2)
        
        sqrt2 = np.sqrt(2)
        psi_3 = np.array([0, 0, 0, 0, 0, -1/sqrt2, 0, 1/sqrt2, 0])
        psi_0 = np.array([1/np.sqrt(3), 0, 0, 0, 1/np.sqrt(3), 0, 0, 0, 
                          1/np.sqrt(3)]) 
        dm_3 = np.outer(psi_3, psi_3)
        dm_0 = np.outer(psi_0, psi_0)
        
        counter1 = 0
        counter2 = 0
        dms = []
        bins = 80
        step = 1 / bins
        for i in range(0, bins):
            x = (i + 0.5) * step
            upper = np.sqrt(1 - 0.75 * x**2) - x / 2
            y = 0.5 * step
            while (y < upper):
                counter1 += 1
                z = np.sqrt(1 - x ** 2 - y ** 2)
                t = np.sqrt(x / y)
                if t < 0.5:
                    if z <= (4 * x + y) / 4:
                        #print(f"1:{counter1}_{x}_{y}")
                        y += step
                        continue
                elif t >= 0.5 and t <= 2:
                    if z <= np.sqrt(x * y):
                        #print(f"2:{counter1}_{x}_{y}")
                        y += step
                        continue
                elif t > 2:
                    if z <= (x + 4 * y) / 4:
                        #print(f"3:{counter1}_{x}_{y}")
                        y += step
                        continue
                counter2 += 1
                Del = z ** 2 - x * y
                R = 4 - 2 * x**2 + x * y - 2 * y **2
                lam0 = 3 * x * y / R
                lam3 = 2 * Del / R
                lam1 = 1 / R
                psi_1 = np.array([0, x, 0, y, 0, -z / sqrt2, 0, -z / sqrt2, 0])
                psi_2 = np.array([0, 0, x, 0, -z / sqrt2, 0, y, 0, z / sqrt2])
                dm_1 = np.outer(psi_1, psi_1)
                dm_2 = np.outer(psi_2, psi_2)
                dm_mix = lam1 * dm_1 + lam1 * dm_2 + lam3 * dm_3 + lam0 * dm_0
                
                # check
                trace = np.trace(dm_mix)
                if (abs(trace - 1)) > 1e-15:
                    print(f"something went wrong! trace: {trace}; index: {i}")
                    
                dm_reshaped = dm_mix.reshape(3, 3, 3, 3)
                pt = dm_reshaped.transpose((0, 3, 2, 1))
                dm_pt = pt.reshape(9, 9)
                if not (np.abs(dm_mix-dm_pt) < 1e-15).all():
                    print(f"something went wrong! {dm_mix}")
                    print(dm_pt)
                    print(f"{dm_mix==dm_pt}")
                    break
                
                dms.append(dm_mix)
                y += step
        print(counter1, counter2)
        # store data
        self._dmAB = np.stack(dms)
        num = self._dmAB.shape[0]
        labels = np.ones(num)
        flatdmAB = reform_mat_to_flat(self._dmAB)
        self._data = np.hstack((flatdmAB, labels.reshape(-1, 1)))
        print(f"{num} bound entangled states generated.")
                
        
    def _const_mix(self, svAB, sam_subnum, mix_num):        
        dimA = self.dimA
        dimB = self.dimB
        dimAB = dimA * dimB
        # assign pure states into groups
        svs = np.split(svAB, sam_subnum)
        # generate probilities for the pure states in the mixture
        prob = self._gen_prob(sam_subnum, mix_num)
        
        dms = []
        pptlabels = []
        for vecs, pbs in zip(svs, prob):
            dm_mix = np.zeros((dimAB, dimAB), dtype='complex128')
            for vec, pb in zip(vecs, pbs):                
                dm_pure = np.outer(vec, np.conj(vec))
                dm_mix += pb * dm_pure
        
            # reshape dm_mix, the order such as (dimA, dimB, dimB, dimA)
            # generally mis-assigns the matrix elements, and should be avoided
            dm_reshaped = dm_mix.reshape(dimA, dimB, dimA, dimB)
            # check the positivity of the partial transposed matrix
            pt = dm_reshaped.transpose((0, 3, 2, 1))
            dm_pt = pt.reshape(dimAB, dimAB)
            eigv_pt = np.linalg.eigvalsh(dm_pt)
            isppt = (eigv_pt > -1e-15).all()
            if isppt:
               pptlabels.append(0) 
            else:
               pptlabels.append(1)
                    
            dms.append(dm_mix) 
        
        self._dmAB = np.stack(dms)
        self._pptlabels = np.array(pptlabels)    

    
    def _gen_prob(self, sam_subnum, mix_num):
        # initial the random number generator
        rng = np.random.default_rng() 
        # generate probilities for the pure states in the mixture
        prob = rng.standard_normal((sam_subnum, mix_num)) 
        # normalization
        norm = np.linalg.norm(prob, ord=1, axis=1, keepdims=True)
        prob = np.abs(prob / norm)
        return prob
    
    def _create_feature_names(self):
        # create feature names
        dimA = self.dimA
        dimB = self.dimB
        a = np.array(range(dimA)).astype(str)
        b = np.array(range(dimB)).astype(str)
        
        svbase = []
        for ia in a:
            for ib in b:
                svbase.append(ia + ib)
                
        dmbase = []
        for i, row in enumerate(svbase):
            for j, column in enumerate(svbase):
                if j >= i:
                    dmbase.append('R' + row + '_' + column)
                elif j < i:
                    dmbase.append('I' + row + '_' + column)

        self.feature_names = np.array(dmbase)

        
        
        
        
        
        