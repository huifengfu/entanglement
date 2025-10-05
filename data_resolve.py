# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 17:45:32 2021

@author: Administrator
"""
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from entanglement.transition_funcs import *

class DataResolve:
    """ resolve data 
    
    Attributes:
        for pure and mixed:
        dimA, dimB, sample_num, sample_subnum, feature_names, datatag,
        data, target, flatAB, ap, a, ph,
        for pure states:
        svreim, ABap, riap, ABriap,  groups, svre, svim, svAB,
        for mixed states:
        ppt, red_cri, dmAB
    ---------------------------------------------------------------
    Useable Methods:
        check_pure, check_mix, rescale_phase, pure_label_check
        
    """
    def __init__(self, inputdata):
        """ resolve data """
        # extract basic data
        self.fulldata = np.array(inputdata['data'])
        self.dimA = inputdata['dimA']
        self.dimB = inputdata['dimB']
        self.sample_num = inputdata['sample_num']
        self.sample_subnum = inputdata['sample_subnum']
        self.feature_names = inputdata['feature_names']
        self.datatag = inputdata['datatag']
        self.sample_structure = inputdata['sample_structure']
        
        if self.datatag == 'ds' or self.datatag == 's' or self.datatag == 'd':
            self._generate_groups()  
            # resolve data
            self._resolve_data()
            # calculate relative phases
            self._rel_phase()
            # construct density matrix
            self._cons_dm()
            # calculate entanglement entropy
            self._ent_entropy()
            # check data
            self.check_pure()
        elif self.datatag == 'mix':
            # resolve data
            self._resolve_mix()
            # calculate entanglement entropy 
            self._ent_entropy()
            # check labels
            self.check_mix()
        
        # set the rescale flag
        self.__rescaled = False
        
        text = """    
        Attributes: 
        for pure and mixed:
        dimA, dimB, sample_num, sample_subnum, feature_names, datatag,
        data, target, flatAB, ap, a, ph,
        for pure states:
        svreim, ABap, riap, ABriap,  groups, svre, svim, svAB,
        for mixed states:
        ppt, red_cri, dmAB
        """
        print(text)
        
    ### methods working for pure states
    ######################################
    def _resolve_data(self):
        # resolve data
        dimAB = self.dimA * self.dimB
        rdAlen = self.dimA ** 2 
        rdBlen = self.dimB ** 2
        rdABlen = rdAlen + rdBlen
        tillre = rdABlen + dimAB
        tillim = tillre + dimAB
       
        self.data = self.fulldata[:, :tillim]
        self.target = self.fulldata[:, tillim:].reshape(-1)
        self.flatA = self.fulldata[:, :rdAlen]
        self.flatB = self.fulldata[:, rdAlen:rdABlen]
        self.flatAB = self.fulldata[:, :rdABlen]
        self.svreim = self.fulldata[:, rdABlen:tillim]

        # derived attrs.
        self.svre = self.fulldata[:, rdABlen:tillre]
        self.svim = self.fulldata[:, tillre:tillim]
        self.svAB = self.svre + 1j * self.svim
        self.a = np.abs(self.svAB)
        self.ph = np.angle(self.svAB)
        self._ap_pure()
        
    def _ap_pure(self):
        self.ap = np.hstack((self.a, self.ph))
        self.ABap = np.hstack((self.flatAB, self.ap))
        self.riap = np.hstack((self.svreim, self.ap))
        self.ABriap = np.hstack((self.data, self.ap))
    
    def _rel_phase(self):
        A = self.dimA
        B = self.dimB
        ph = self.ph
        relph = []
        for i in range(A):
            for j in range(B):
                index = i * B + j
                for k1 in range(j+1, B):
                    relph.append(ph[:, index] - ph[:, i*B+k1])
                for k2 in range(i+1, A):
                    relph.append(ph[:, index] - ph[:, k2*B+j])
        relph = np.column_stack(tuple(relph))
        relph[relph > np.pi] -= 2 * np.pi
        relph[relph <= -np.pi] += 2 * np.pi
        self.relph = relph        
        # construct data sets with relph
        self._arel_pure()
        
    def _arel_pure(self):
        # construct data sets with relph
        self.a_relph = np.hstack((self.a, self.relph))
        self.AB_relph = np.hstack((self.flatAB, self.relph))
        self.ri_relph = np.hstack((self.svreim, self.relph))
        self.ABri_relph = np.hstack((self.data, self.relph))
        self.ap_relph = np.hstack((self.ap, self.relph))
        self.ABap_relph = np.hstack((self.ABap, self.relph))
        self.riap_relph = np.hstack((self.riap, self.relph))
        self.ABriap_relph = np.hstack((self.ABriap, self.relph))  
        
    def _generate_groups(self):
        dimAB = self.dimA * self.dimB
        lis = list(range(dimAB, 0, -1))
        groups = []
        for i in self.sample_structure:
            a = [lis.index(i)] * self.sample_subnum
            groups += a
        self.groups = groups

    def _plot_phase(self):
        phase = self.ph[self.ph != 0]
        plt.hist(phase, bins=10)
        plt.savefig('phase.pdf')
        
    def _plot_abs(self):
        coeab = self.a
        dim_num = self.a.shape[1]
        dims = []
        for i in range(dim_num):
            for j in range(i+1, dim_num):
                dims.append((i, j))
        pair_num = len(dims)
        rng = np.random.default_rng()
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        for i in range(3):
            n = pair_num - i
            index = rng.choice(n)
            pair = dims.pop(index)
            axes[i].scatter(coeab[:,pair[0]], coeab[:,pair[1]], s=1)
        plt.savefig('abs.pdf')
            
    def _norm_check(self):
        norm = np.linalg.norm(self.a, axis=1)
        if (np.abs(norm - 1) < 1e-15).all():
            print("normalized to 1")
        else:
            print("not normalized to 1")            
            
    def _label_check(self):
        svAB = self.svAB.reshape((self.sample_num, self.dimA, self.dimB))
        
        derived_labels = []
        for sv in svAB:
            nonzeros = np.nonzero(sv)
            A = nonzeros[0]
            B = nonzeros[1]
            i = A[0]
            j = B[0]

            if np.unique(A).shape[0] == 1 or np.unique(B).shape[0] == 1:
                derived_labels.append(0)
            else:
                for row in sv[i+1:]:
                    if (row == 0).all():
                        continue
                    factor = row[j] / sv[i, j]
                    newrow = factor * sv[i]
                    flag = (np.abs(row - newrow) < 1e-15).all()
                    if not flag:
                        break
                if flag:
                    derived_labels.append(0)
                else:
                    derived_labels.append(1)
        
        checker = (self.target == derived_labels)
        if checker.all():
            print("label checked")
            self.wrong_labels = None
        else:
            print("label unmatch! check self.wrong_labels!")
            self.wrong_labels = np.nonzero(checker == 0)

    def _cons_dm(self):
        dmAB = []
        for sv in self.svAB:
            dm = np.outer(sv, np.conj(sv))
            dmAB.append(dm)
        dmAB = np.array(dmAB)
        self.dms = reform_mat_to_flat(dmAB)
                        
    def check_pure(self):
        self._norm_check()
        self._label_check()
        self._plot_phase()
        self._plot_abs()
        
    def pure_label_check(self):
        A = reform_flat_to_mat(self.flatA)
        B = reform_flat_to_mat(self.flatB)
        state = reduced_matrix(self.svAB, self.dimA, self.dimB)
        if (np.abs(A - state[0]) < 1e-15).all():
            print("reduced A checked!")
        else:
            print("reduced A wrong!")
        if (np.abs(B - state[1]) < 1e-15).all():
            print("reduced B checked!")
        else:
            print("reduced B wrong!")
        if (self.target == state[2]).all():
            print("label checked!")
        else:
            print("label wrong!")
        
    ### methods working for mixed states
    #############################################    
    def _resolve_mix(self):
        # resolve data
        dimAB = self.dimA * self.dimB
        ABlen = dimAB ** 2 
        self.data = self.fulldata[:, :ABlen]
        self.target = self.fulldata[:, ABlen:].reshape(-1)

        # rebuild dmAB
        self.dmAB = reform_flat_to_mat(self.data)
        # extract flatAB and red_cri
        self._reduce_dmAB()
        # extract and build abs and phase data
        self.a = np.abs(self.dmAB)
        self.ph = np.angle(self.dmAB)
        self._ap_mix()

    
    def _reduce_dmAB(self):
        # calculate reduced dm of A and B
        dimA = self.dimA
        dimB = self.dimB
        dimAB = dimA * dimB
        IA = np.identity(dimA)
        IB = np.identity(dimB)
        rdA = []
        rdB = []
        labels = []
        i = -1
        for dm_mix in self.dmAB:
            # check positivity and trace
            i += 1
            eigv = np.linalg.eigvalsh(dm_mix)
            trace = np.trace(dm_mix)
            if (abs(trace - 1)) > 1e-15:
                print(f"improper trace: {trace}; index: {i}")
                break
            if (eigv < -1e-15).any():
                print(f"positivity failure! eigv:{eigv}; index: {i}")
                break
            
            dmAB = dm_mix.reshape(dimA, dimB, dimA, dimB)
            dmAB_ptoA = np.trace(dmAB, axis1=0, axis2=2) # ptoA: shorthand for 
                                                # partial traced over A
            dmAB_ptoB = np.trace(dmAB, axis1=1, axis2=3) # ptoB: shorthand for 
                                                # partial traced over B
            
            # check
            A = np.tensordot(dmAB_ptoB, IB, axes=0)
            B = np.tensordot(IA, dmAB_ptoA, axes=0)
            A = A.transpose((0, 2, 1, 3))
            B = B.transpose((0, 2, 1, 3))
            RcrA = (A - dmAB).reshape(dimAB, dimAB)
            RcrB = (B - dmAB).reshape(dimAB, dimAB)
#            print(RcrA)
#           print(RcrB)
            eigRA = np.linalg.eigvalsh(RcrA)
            eigRB = np.linalg.eigvalsh(RcrB)
            if (eigRA > -1e-15).all() and (eigRB > -1e-15).all():
                labels.append(0)
            else:
                labels.append(1)
            
            rdA.append(dmAB_ptoB)
            rdB.append(dmAB_ptoA)            
        
        self.red_cri = np.array(labels)
        rdA = np.array(rdA)
        rdB = np.array(rdB)
        self.flatA = reform_mat_to_flat(rdA)
        self.flatB = reform_mat_to_flat(rdB)
        self.flatAB = np.hstack((self.flatA, self.flatB))        
        
    def _ap_mix(self):
        a_diag = reform_mat_to_flat(self.a)
        ph_diag = reform_mat_to_flat(0 + 1j * self.ph)
        self.ap = a_diag + ph_diag
                            
    def check_mix(self):
        # check labels of mixed states
        if self.dimA >= 3 and self.dimB >= 3:
            labels = np.hstack((np.ones(self.sample_structure[0]),
                                  np.zeros(self.sample_structure[1]),
                                  np.ones(self.sample_structure[2]),
                                  ))
            if (labels == self.target).all():
                print("sample structure checked!")
            self.ppt = np.hstack((np.ones(self.sample_structure[0]),
                                  np.zeros(self.sample_structure[1]),
                                  np.zeros(self.sample_structure[2]),
                                  ))
            if (self.ppt - self.red_cri >= 0).all():
                print("label checked!")
        else:
            labels = np.hstack((np.ones(self.sample_structure[0]),
                                  np.zeros(self.sample_structure[1]),
                                  ))
            if (labels == self.target).all():
                print("sample structure checked!")
            self.ppt = self.target
            if (self.target == self.red_cri).all():
                print("label checked!")
                
    ### methods working for both pure and mixed states
    #################################################
    def rescale_phase(self):
        if self.__rescaled:
            print("phases are already rescaled!")
            return
        
        self.ph = self.ph / (2 * np.pi) + 0.5
        
        if self.datatag == 'ds' or self.datatag == 's' or self.datatag == 'd':
            self.relph = self.relph / (2 * np.pi) + 0.5            
            # update relevant data
            self._ap_pure()
            self._arel_pure()
        elif self.datatag == 'mix':
            # update relevant data
            self._ap_mix()
            
        # update the rescale flag
        self.__rescaled = True
        
    def _ent_entropy(self):
        rdA = reform_flat_to_mat(self.flatA)
        rdB = reform_flat_to_mat(self.flatB)
        eigA = []
        eigB = []
        for matA, matB in zip(rdA, rdB):
            eigA.append(np.linalg.eigvalsh(matA))
            eigB.append(np.linalg.eigvalsh(matB))

        s = []
        s2 = []
        for lams in eigA:
            entr = 0
            entr2 = 0
            for lam in lams:
                if lam <= 1e-15:
                    continue
                entr += -lam * np.log(lam)
                entr2 += lam ** 2
            s.append(entr)
            entr2 = - np.log(entr2)
            s2.append(entr2)
        self.s2 = np.array(s2)
        self.s2[self.s2 <= 1e-15] = 0
        self.s = np.array(s)
        self.s[self.s <= 1e-15] = 0
        y = self.s.copy()
        y[y > 0] = 1
        if (y == self.target).all():
            print("entanglement entropy checked.")
        else:
            print("entanglement entropy contradict with targets.")
                
                
        