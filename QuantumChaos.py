import numpy as np
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt
import inspect
from quimb.calc import entropy
from quimb.calc import mutinf_subsys as mi
from quimb import partial_trace as ptr


class quantum_info_scrambler:
    def __init__(self, QS : object, ndots = 1, temp = False, lead = False, T = 10**6, μ = 0):
        self.QS = QS
        try:
            self.H = QS.Htot
        except(AttributeError):
            self.H = QS
        try:
            self.N = QS.l
        except(AttributeError):
            self.N = ndots
        try:
            self.s = QS.s
        except(AttributeError):
            self.s = False
        
        self.temp = temp
        self.lead = lead
        self.T = T
        self.μ = μ
        self.Hildim = 2**(self.N) if self.s is False else 2**(2*self.N)
        self.kets = self.N if self.s is False else 2*self.N
        self.dims = [2] * self.kets * 2
        
        if self.temp is True:
            self.eigvals, self.eigvecs = np.linalg.eig(self.H)
            
        if self.temp is True and self.lead is False:
            self.thermnorm =  np.trace(expm(-(1/self.T) * self.H))

    # (Superposition) statevector from binary:
    def statevec(self, prob : list[float], state :list[str], binary = True) -> np.array:
        self.sts = []
        if binary is True:
            try:
                for i in range(len(state)):
                    self.sts.append(int(state[i],2))
            except(TypeError):
                self.sts.append(int(state,2))
        else:
            try:
                for i in range(len(state)):
                    self.sts.append(int(state[i]))
            except(TypeError):
                self.sts.append(int(state))
        self.vec = np.zeros((self.Hildim,1))

        for i in range(len(sts)):
            self.vecz = np.zeros((self.Hildim,1))
            self.vecz[self.sts[i]] = prob[i]
            self.vec += self.vecz
        self.norm_vec = self.vec / np.linalg.norm(self.vec)
        
        return(self.norm_vec)
    
    # Initialize input state of the quantum channel
    def in_state(self, uniform = False, T = 10**6, μ = 0) -> np.array:
        #stat = []
        self.stat = 0
        if self.temp is True:

                
                if self.lead is False:
                    for i in range(self.Hildim):
                        self.Boltz = np.e**(-(1/(2* T)) * self.eigvals[i])
                        self.stat += self.Boltz * self.eigvecs[:, i]
                    self.stat = np.array([self.stat]).T
                        #stat.append( Boltz * vecs[:,i])
                    # stat = sum(stat) to create the superposition eigenvector

                elif self.lead is True:
                    self.sols = []
                    self.sol_t = []
                    self.solve = self.QS.solv_eqn(μ, T)[1]
                    for i in range(self.Hildim):
                        self.sols.append(self.solve[i,-1])
                        self.stat += np.real(self.sols[i]) * self.QS.states[i].vector


        elif self.temp is not True:
            for j in range(self.Hildim):
                self.stat += self.statevec([1/np.sqrt(self.Hildim)], [bin(j)[2::]])
                #stat.append( self.statevec([1/np.sqrt(self.Hildim)], [bin(j)[2::]]) )
 
        return(self.stat / np.linalg.norm(self.stat)) 
    
    # Define output state of the quantum channel
    def out_state(self, time : float, input_st : np.array) -> np.array:
        out_stat = 0
        if self.temp is True and self.lead is False:
            for i in range(self.Hildim):
                    out_stat += np.e**(-1j * self.eigvals[i] * time ) * self.eigvecs[:, i]
            out_stat = np.array([out_stat]).T
        else:
            out_stat += self.QS.U(time) @ input_st

        return out_stat 
    
    # Function to access arguments of another function
    def get_default_args(self, func):
        signature = inspect.signature(func)
        return {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }

    # Define the state-operator mapping of the quantum channel
    def psi_c(self, input_st = np.array([1]), output_st = np.array([1]), time = 10) -> np.array:
        psi = 0
        if self.lead is True:
            self.temp == True
            self.sols = []
            self.sol_t = []
            self.solve = self.QS.solv_eqn(self.μ, self.T)[1]
            for i in range(self.Hildim):
                self.sols.append(self.solve[i,-1])
                unit_evo = np.e**(-1j * self.QS.states[i].energy * time)
                psi += np.kron(np.sqrt(np.real(self.sols[i])) * self.QS.states[i].vector, unit_evo * self.QS.states[i].vector)
        else:
            if self.temp is True:
                for i in range(self.Hildim):
                    Boltz = np.e**(-(1/(2*self.T)) * self.eigvals[i])
                    unit_evo = np.e**(-1j * self.eigvals[i] * time)
                    psi += np.kron(Boltz * self.eigvecs[:,i], unit_evo * self.eigvecs[:,i])
            else:
                psi = np.kron(output_st, input_st)
        psi = psi / np.linalg.norm(psi)
        
        # check when lead is on to reproduce correctly
        return np.array([psi]).T
                                         
    # Define quantum channel density operator
    def rho(self, state : np.array) -> np.array:
                                         
        return(state @ state.conj().T)
        
    # Define partition indices    
    def idx(self, partition : str) -> list:
        if type(partition[0]) == str:
            
            partition = partition.lower()
            indices_of_all_states = list(range(2 * self.kets))
            if self.s is True:
                indices_of_ups = list(range(int(self.kets/2))) + list(range(int(self.kets), int((3/2) * self.kets)))
                indices_of_downs = list(range(int(self.kets/2), self.kets)) + list(range(int((3/2) * self.kets), 2 * self.kets))
                # not_rhoa_indices = [i for i in range(1, 2 * self.kets)]
                # rhoa_indices = [i for i in indices_of_all_states if i not in not_rhoa_indices]
                rhoa_indices = [indices_of_ups[0]] + [indices_of_downs[0]]

                # not_rhob_indices = [0]+[i for i in range(self.kets, 2 * self.kets)]
                # rhob_indices = [i for i in indices_of_all_states if i not in not_rhob_indices]
                rhob_indices = indices_of_ups[1 : int(self.kets/2)] + indices_of_downs[1 : int(self.kets/2)]

                # not_rhoc_indices = [i for i in range(self.kets)] + [2 * self.kets - 1]
                # rhoc_indices = [i for i in indices_of_all_states if i not in not_rhoc_indices]
                rhoc_indices = indices_of_ups[int(self.kets/2) : -1] + indices_of_downs[int(self.kets/2) : -1]

                # not_rhod_indices = [i for i in range(2 * self.kets - 1)]
                # rhod_indices = [i for i in indices_of_all_states if i not in not_rhod_indices]
                rhod_indices = [indices_of_ups[-1]] + [indices_of_downs[-1]]
            else:
                rhoa_indices = [0]
                rhob_indices = list(range(1,self.kets))
                rhoc_indices = list(range(self.kets, 2 * self.kets - 1))
                rhod_indices = [2 * self.kets - 1]

            if partition == 'a':
                return rhoa_indices

            elif partition == 'b':
                return rhob_indices

            elif partition == 'c':
                return rhoc_indices

            elif partition == 'd':
                return rhod_indices

            elif partition == 'ab': 
                # not_rhoab_indices = [i for i in range(self.kets, 2 * self.kets)]
                # rhoab_indices = [i for i in indices_of_all_states if i not in not_rhoab_indices]
                rhoab_indices = rhoa_indices + rhob_indices

                return rhoab_indices

            elif partition == 'ac': 
                # not_rhoac_indices = [i for i in range(1, self.kets)] + [2 * self.kets - 1]
                # rhoac_indices = [i for i in indices_of_all_states if i not in not_rhoac_indices]
                rhoac_indices = rhoa_indices + rhoc_indices

                return rhoac_indices  

            elif partition == 'bc': 
                # not_rhobc_indices = [0] + [2 * self.kets - 1]
                # rhobc_indices = [i for i in indices_of_all_states if i not in not_rhobc_indices]
                rhobc_indices = rhob_indices + rhoc_indices

                return rhobc_indices

            elif partition == 'ad': 
                # not_rhoad_indices = [i for i in range(1, 2 * self.kets - 1)]
                # rhoad_indices = [i for i in indices_of_all_states if i not in not_rhoad_indices]
                rhoad_indices = rhoa_indices + rhod_indices

                return rhoad_indices 

            elif partition == 'bd': 
                # not_rhobd_indices = [0] + [i for i in range(self.kets, 2 * self.kets - 1)]
                # rhobd_indices = [i for i in indices_of_all_states if i not in not_rhobd_indices]
                rhobd_indices = rhob_indices + rhod_indices

                return rhobd_indices 

            elif partition == 'cd': 
                # not_rhocd_indices = [i for i in range(0, self.kets)]
                # rhocd_indices = [i for i in indices_of_all_states if i not in not_rhocd_indices]
                rhocd_indices = rhoc_indices + rhod_indices

                return rhocd_indices

            elif partition == 'acd': 
                # not_rhoacd_indices = [i for i in range(1, self.kets)]
                # rhoacd_indices = [i for i in indices_of_all_states if i not in not_rhoacd_indices]
                rhoacd_indices = rhoa_indices + rhoc_indices + rhod_indices

                return rhoacd_indices

            elif partition == 'bcd': 
                # not_rhobcd_indices = [0]
                # rhobcd_indices = [i for i in indices_of_all_states if i not in not_rhobcd_indices]
                rhobcd_indices = rhob_indices + rhoc_indices + rhod_indices

                return rhobcd_indices
        
        
        
        
    #Define reduced density matrix
    def rho_red(self, density_matrix : np.array, partition : str) -> np.array:
                                         
        return(ptr(density_matrix, self.dims, self.idx(partition)))

                                    
    # Define entanglement entropy
    def S(self, density_matrix : np.array) -> float:

        return entropy(density_matrix)

    # Define bipartite mutual information
    def I2(self, density_matrix : np.array, partitions : list[str]) -> float:

        if type(partitions[0]) == str:
            
            return mi(density_matrix, self.dims, sysa = self.idx(partitions[0]), sysb = self.idx(partitions[1]))
        else:
            return mi(density_matrix, self.dims, sysa = partitions[0], sysb = partitions[1])
    

    # Define tripartite mutual information
    def I3(self, density_matrix : np.array, partitions :list[str]) -> float:
        if type(partitions[0]) != str:
            I3_mut = self.I2(density_matrix, [partitions[0], partitions[1]]) + self.I2(density_matrix, [partitions[0], partitions[2]]) - self.I2(density_matrix, [partitions[0], partitions[1] + partitions[2]]) 
        
        else:
            I3_mut = self.I2(density_matrix, [self.idx(partitions[0]), self.idx(partitions[1])]) + self.I2(density_matrix, [self.idx(partitions[0]), self.idx(partitions[2])]) - self.I2(density_matrix, [self.idx(partitions[0]), self.idx(partitions[1]) + self.idx(partitions[2])])
        return(I3_mut)

    
    # Define 4-point OTO correlator
    def OTOC(self, oper_1 : np.array, oper_2 : np.array, rho : np.array, space : np.array, time : np.array, pars : list, start_dot = 1):
    
        mid = start_dot
        Oto = []
        onz = pars[0]
        hopz = pars[1]
        T = Tv
        μ = μv
        for t in time:
            time_slice = []
            for s in space:
                com = oper_1(self.kets, int(s)) @ oper_2(self.kets, mid) - oper_2(self.kets, mid) @ oper_1(self.kets, int(s))
                time_slice.append(np.abs(np.real(tr(rho @ com @ com.conj().T))))
            Oto.append(time_slice)
            
        return Oto


