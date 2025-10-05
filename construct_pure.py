# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 10:32:07 2021

@author: Administrator
"""

import numpy as np
from datetime import datetime
import json
import entanglement.puregen as pg
from entanglement.transition_funcs import *

class PureGenerator:
    """ Generate and construct pure states of AB, separable and entangled.
    Attributes:
        dimA, dimB, dimAB, sample_num, svAB, flatA, flatB, svre, svim, labels
        rdA, rdB, feature_names, data, datatag, sample_subnum
    
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
        
    def gen_samples(self, snum_rel_subspace=1000):
        """
        Generate samples from direct_genAB and gen_sepAB together.

        Parameters
        ----------
        snum_rel_subspace : TYPE, optional
            DESCRIPTION. The default is 1000.

        Returns
        -------
        None.

        """
        # generate samples using direct_genAB
        self.direct_genAB(snum_rel_subspace)
        # store the data
        data1 = self.data
        sample_structure1 = self.sample_structure
        
        # generate samples using gen_sepAB
        self.gen_sepAB(snum_rel_subspace)
        data2 = self.data
        sample_structure2 = self.sample_structure
        
        # combine
        self.data = np.vstack((data1, data2))
        self.sample_structure = sample_structure1 + sample_structure2
        print(f"sample structure:\n{self.sample_structure}")
        
        # other sample information
        self.sample_subnum = snum_rel_subspace
        self.sample_num = self.data.shape[0]
        self.datatag = 'ds' 
        # create feature names
        self._create_feature_names()

        self.results = {'data': self.data.tolist(), 
                   'feature_names': self.feature_names.tolist(), 
                   'dimA': self.dimA, 
                   'dimB': self.dimB, 
                   'sample_num': self.sample_num,
                   'sample_subnum': self.sample_subnum,
                   'sample_structure': self.sample_structure,
                   'datatag': self.datatag,
                   }
        print(f"{self.sample_num} samples ({self.dimA}*{self.dimB}) are generated!")
        
        
                    
        
    def direct_genAB(self, snum_rel_subspace=1000):
        """
        Generate pure states of AB directly. Wheather these states are separable or
        entangled are unknown.
    
        Parameters
        ----------
        snum_rel_subspace : TYPE int, optional
            DESCRIPTION. The default is 1000.
    
        """
        # initalization
        snum_rel_subspace = int(snum_rel_subspace)
        # parameter check
        if snum_rel_subspace < 1:
            print("Failed to generate. Sample number per subspace should be \
                  equal or greater than 1.")
            return
        
        dimAB = self.dimA * self.dimB        
        # generate samples
        svs = []
        for i in range(1, dimAB):
            sv = pg.puregen(sample_number=snum_rel_subspace, 
                            dims=(dimAB-i+1, dimAB))
            svs.append(sv)
        
        self.svAB = np.vstack(tuple(svs))
        
        # prepare data and label packege
        self.sample_structure = list(range(dimAB, 1, -1))
        self.sample_subnum = snum_rel_subspace
        self.sample_num = self.svAB.shape[0]
        self.datatag = 'd'   # stands for directly generated
        
        self._reduced_matrix()
        self.flatA = reform_mat_to_flat(self.rdA)
        self.flatB = reform_mat_to_flat(self.rdB)
        self.svre = self.svAB.real
        self.svim = self.svAB.imag
        labels = self.labels.reshape(-1, 1)
        self.data = np.hstack((self.flatA, self.flatB, 
                               self.svre, self.svim, labels))
        
        # create feature names
        self._create_feature_names()

        self.results = {'data': self.data.tolist(), 
                   'feature_names': self.feature_names.tolist(), 
                   'dimA': self.dimA, 
                   'dimB': self.dimB, 
                   'sample_num': self.sample_num,
                   'sample_subnum': self.sample_subnum,
                   'sample_structure': self.sample_structure,
                   'datatag': self.datatag,
                   }
        print(f"{self.sample_num} samples ({self.dimA}*{self.dimB})" + 
              "are directly generated!")

        
    def gen_sepAB(self, snum_rel_subspace=1000):
        """
        Generate separable pure states. 

        """
        # initalization
        snum_rel_subspace = int(snum_rel_subspace)
        # parameter check
        if snum_rel_subspace < 1:
            print("Failed to generate. Sample number per subspace should be \
                  equal or greater than 1.")
            return
        
        dimA = self.dimA
        dimB = self.dimB
        # construct sample indicator
        sns = self._sample_indicator()
        # generate sample states
        svAB = []
        svA = []
        svB = []
        sample_structure = []
        for z, y in zip(sns[0], sns[1]):
            if y == 1:
                svA_ori = pg.puregen(sample_number=snum_rel_subspace,
                                 dims=(z[0, 0], dimA))
                svB_ori = pg.puregen(sample_number=snum_rel_subspace,
                                 dims=(z[0, 1], dimB))
                svAB_ori = cons_pure_sep(svA_ori, svB_ori)
                svAB.append(svAB_ori)
                svA.append(svA_ori)
                svB.append(svB_ori)
            elif y > 1:
                snum = snum_rel_subspace // y
                snum_last = snum_rel_subspace - snum * (y - 1)
                for i, x in enumerate(z):
                    # set sample numbers
                    if i == (y - 1):
                        number = snum_last
                    else:
                        number = snum
                    # skip empty sample number    
                    if number == 0:
                        continue
                    
                    svA_ori = pg.puregen(sample_number=number,
                                 dims=(x[0], dimA))
                    svB_ori = pg.puregen(sample_number=number,
                                 dims=(x[1], dimB))
                    svAB_ori = cons_pure_sep(svA_ori, svB_ori)
                    svAB.append(svAB_ori)
                    svA.append(svA_ori)
                    svB.append(svB_ori)
            sample_structure.append(int(z[0, 2]))
                    
        self.svAB = np.vstack(tuple(svAB))
        svA = np.vstack(tuple(svA))
        svB = np.vstack(tuple(svB))
        
        # prepare data and label packege
        self.sample_structure = sample_structure
        self.sample_subnum = snum_rel_subspace
        self.sample_num = self.svAB.shape[0]
        self.datatag = 's'  # stands for generated from A and B, so the states
                            # are seperable
                            
        self.labels = np.zeros(self.sample_num)
        self.rdA = sv_to_matrix(svA)
        self.rdB = sv_to_matrix(svB)
        self.flatA = reform_mat_to_flat(self.rdA)
        self.flatB = reform_mat_to_flat(self.rdB)
        self.svre = self.svAB.real
        self.svim = self.svAB.imag
        labels = self.labels.reshape(-1, 1)
        self.data = np.hstack((self.flatA, self.flatB, 
                               self.svre, self.svim, labels))
        
        # create feature names
        self._create_feature_names()

        self.results = {'data': self.data.tolist(), 
                   'feature_names': self.feature_names.tolist(), 
                   'dimA': self.dimA, 
                   'dimB': self.dimB, 
                   'sample_num': self.sample_num,
                   'sample_subnum': self.sample_subnum,
                   'sample_structure': self.sample_structure,
                   'datatag': self.datatag,
                   }
        print(f"{self.sample_num} separable" + 
              f" samples ({self.dimA}*{self.dimB}) are generated!")

    def to_json(self):
        now = datetime.now()
        nowstr = now.strftime("%b%d%H%M%S")
        filename = (f"pure_{self.datatag}{self.sample_num}"+
            f"_{self.dimA}{self.dimB}_{nowstr}.json")
        with open(f"data\{filename}", 'w') as f:
            json.dump(self.results, f)

    def _reduced_matrix(self):
        """
        Calculate reduced matrix.
        
        """
        dimA = self.dimA
        dimB = self.dimB
        svAB = self.svAB.reshape(-1, dimA, dimB)
        
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
            
        self.labels = np.array(labels)
        self.rdA = np.array(rdA)
        self.rdB = np.array(rdB)
    
    def _sample_indicator(self):
        """" Calculate possible nonzero dimensions of a separable state.
             subdimA(subdimB) stands for nonzero dimensions of A(B);
             A*B = subdimA*subdimB; for each A*B, variable 'y' counts how many
             times it appears: for example, if we have subdimA = subdimB = 2 
             gives A*B = 4 and subdimA = 1, subdimB = 4 (and the reverse case)
             also gives A*B = 4, then the corresponding 'y' is 3.
             
             results: (z, y) 
                 format: ([subdimA, subdimB, A*B], reputation number of A*B)
        """
               
        dimA = list(range(self.dimA, 0, -1))
        dimB = list(range(self.dimB, 0, -1))
        
        seq = []
        for i in dimA:
            for j in dimB:
                seq.append((i, j, i * j))
                
        seq = np.array(seq)
        ind = np.argsort(seq[:, 2])
        ordered_seq = seq[ind[::-1]]
        x = ordered_seq[:, 2]
        y = np.bincount(x)[::-1]
        y = y[y>0]
        
        index_split = []
        j = 0
        for i in y[:-1]:
            j += i
            index_split.append(j)
            
        z = np.split(ordered_seq, index_split)
            
        results = (z, y)
        return results
    
    def _create_feature_names(self):
        # create feature names
        dimA = self.dimA
        dimB = self.dimB
        a = np.array(range(dimA)).astype(str)
        b = np.array(range(dimB)).astype(str)
        svbase = []
        Abase = []
        Bbase = []
        for ia in a:
            for ib in b:
                svbase.append(ia + ib)
        for i in a:
            for j in a:
                Abase.append(i + j)
        for i in b:
            for j in b:
                Bbase.append(i + j)
        svbase = np.array(svbase)
        Abase = np.array(Abase)
        Bbase = np.array(Bbase)

        svbase_re = np.char.add('R', svbase)
        svbase_im = np.char.add('I', svbase)
        AAbase = np.char.add('A', Abase)
        BBbase = np.char.add('B', Bbase)
        self.feature_names = np.concatenate((AAbase, BBbase, 
                                             svbase_re, svbase_im))
        








    
    
    
    
    
            
            