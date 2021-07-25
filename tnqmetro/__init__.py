"""TNQMetro: Tensor-network based package for efficient quantum metrology computations."""


# Table of Contents
#
# 1 Functions for finite size systems......................................29
#   1.1 High level functions...............................................37
#   1.2 Low level functions...............................................257
#       1.2.1 Problems with exact derivative.............................1207
#       1.2.2 Problems with discrete approximation of the derivative.....2411
# 2 Functions for infinite size systems..................................3808
#   2.1 High level functions.............................................3816
#   2.2 Low level functions..............................................4075
# 3 Auxiliary functions..................................................5048


import itertools
import math
import warnings

import numpy as np

from ncon import ncon


########################################
#                                      #
#                                      #
# 1 Functions for finite size systems. #
#                                      #
#                                      #
########################################


#############################
#                           #
# 1.1 High level functions. #
#                           #
#############################


def fin(N, so_before_list, h, so_after_list, BC='O', L_ini=None, psi0_ini=None, imprecision=10**-2, D_L_max=100, D_L_max_forced=False, L_herm=True, D_psi0_max=100, D_psi0_max_forced=False):
    """
    Optimization of the QFI over operator L (in MPO representation) and wave function psi0 (in MPS representation) and check of convergence in their bond dimensions. Function for finite size systems.
    
    User has to provide information about the dynamics by specifying the quantum channel. It is assumed that the quantum channel is translationally invariant and is built from layers of quantum operations.
    User has to provide one defining operation for each layer as a local superoperator. These local superoperators have to be input in order of their action on the system.
    Parameter encoding is a stand out quantum operation. It is assumed that the parameter encoding acts only once and is unitary so the user has to provide only its generator h.
    Generator h has to be diagonal in computational basis, or in other words, it is assumed that local superoperators are expressed in the eigenbasis of h.
    
    Parameters:
      N: integer
        Number of sites in the chain of tensors (usually number of particles).
      so_before_list: list of ndarrays of a shape (d**(2*k),d**(2*k)) where k describes on how many sites a particular local superoperator acts
        List of local superoperators (in order) which act before unitary parameter encoding.
      h: ndarray of a shape (d,d)
        Generator of unitary parameter encoding. Dimension d is the dimension of local Hilbert space (dimension of physical index).
        Generator h has to be diagonal in the computational basis, or in other words, it is assumed that local superoperators are expressed in the eigenbasis of h.
      so_after_list: list of ndarrays of a shape (d**(2*k),d**(2*k)) where k describes on how many sites particular local superoperator acts
        List of local superoperators (in order) which act after unitary parameter encoding.
      BC: 'O' or 'P', optional
        Boundary conditions, 'O' for OBC, 'P' for PBC.
      L_ini: list of length N of ndarrays of a shape (Dl_L,Dr_L,d,d) for OBC (Dl_L, Dr_L can vary between sites) or ndarray of a shape (D_L,D_L,d,d,N) for PBC, optional
        Initial MPO for L.
      psi0_ini: list of length N of ndarrays of a shape (Dl_psi0,Dr_psi0,d) for OBC (Dl_psi0, Dr_psi0 can vary between sites) or ndarray of a shape (D_psi0,D_psi0,d,N) for PBC, optional
        Initial MPS for psi0.
      imprecision: float, optional
        Expected relative imprecision of the end results.
      D_L_max: integer, optional
        Maximal value of D_L (D_L is bond dimension for MPO representing L).
      D_L_max_forced: bool, optional
        True if D_L_max has to be reached, otherwise False.
      L_herm: bool, optional
        True if Hermitian gauge has to be imposed on MPO representing L, otherwise False.
      D_psi0_max: integer, optional
        Maximal value of D_psi0 (D_psi0 is bond dimension for MPS representing psi0).
      D_psi0_max_forced: bool, optional
        True if D_psi0_max has to be reached, otherwise False.
    
    Returns:
      result: float
        Optimal value of figure of merit.
      result_m: ndarray
        Matrix describing the figure of merit as a function of bond dimensions of respectively L [rows] and psi0 [columns].
      L: list of length N of ndarrays of a shape (Dl_L,Dr_L,d,d) for OBC (Dl_L, Dr_L can vary between sites) or ndarray of a shape (D_L,D_L,d,d,N) for PBC
        Optimal L in MPO representation.
      psi0: list of length N of ndarrays of a shape (Dl_psi0,Dr_psi0,d) for OBC (Dl_psi0, Dr_psi0 can vary between sites) or ndarray of a shape (D_psi0,D_psi0,d,N) for PBC
        Optimal psi0 in MPS representation.
    """
    if np.linalg.norm(h - np.diag(np.diag(h))) > 10**-10:
        warnings.warn('Generator h have to be diagonal in computational basis, or in other words it is assumed that local superoperators are expressed in the eigenbasis of h.')
    d = np.shape(h)[0]
    ch = fin_create_channel(N, d, BC, so_before_list + so_after_list)
    ch2 = fin_create_channel_derivative(N, d, BC, so_before_list, h, so_after_list)
    result, result_m, L, psi0 = fin_gen(N, d, BC, ch, ch2, None, L_ini, psi0_ini, imprecision, D_L_max, D_L_max_forced, L_herm, D_psi0_max, D_psi0_max_forced)
    return result, result_m, L, psi0


def fin_gen(N, d, BC, ch, ch2, epsilon=None, L_ini=None, psi0_ini=None, imprecision=10**-2, D_L_max=100, D_L_max_forced=False, L_herm=True, D_psi0_max=100, D_psi0_max_forced=False):
    """
    Optimization of the figure of merit (usually interpreted as the QFI) over operator L (in MPO representation) and wave function psi0 (in MPS representation) and check of convergence when increasing their bond dimensions. Function for finite size systems.
    
    User has to provide information about the dynamics by specifying a quantum channel ch and its derivative ch2 (or two channels separated by small parameter epsilon) as superoperators in MPO representation.
    There are no constraints on the structure of the channel but the complexity of calculations highly depends on the channel's bond dimension.
    
    Parameters:
      N: integer
        Number of sites in the chain of tensors (usually number of particles).
      d: integer
        Dimension of local Hilbert space (dimension of physical index).
      BC: 'O' or 'P'
        Boundary conditions, 'O' for OBC, 'P' for PBC.
      ch: list of length N of ndarrays of a shape (Dl_ch,Dr_ch,d**2,d**2) for OBC (Dl_ch, Dr_ch can vary between sites) or ndarray of a shape (D_ch,D_ch,d**2,d**2,N) for PBC
        Quantum channel as a superoperator in MPO representation.
      ch2: list of length N of ndarrays of a shape (Dl_ch2,Dr_ch2,d**2,d**2) for OBC (Dl_ch2, Dr_ch2 can vary between sites) or ndarray of a shape (D_ch2,D_ch2,d**2,d**2,N) for PBC
        Interpretiaon depends on whether epsilon is specifed (2) or not (1, default approach):
        1) derivative of the quantum channel as a superoperator in the MPO representation,
        2) the quantum channel as superoperator in the MPO representation for the value of estimated parameter shifted by epsilon in relation to ch.
      epsilon: float, optional
        If specified then interpeted as value of a separation between estimated parameters encoded in ch and ch2.
      L_ini: list of length N of ndarrays of a shape (Dl_L,Dr_L,d,d) for OBC (Dl_L, Dr_L can vary between sites) or ndarray of a shape (D_L,D_L,d,d,N) for PBC, optional
        Initial MPO for L.
      psi0_ini: list of length N of ndarrays of a shape (Dl_psi0,Dr_psi0,d) for OBC (Dl_psi0, Dr_psi0 can vary between sites) or ndarray of a shape (D_psi0,D_psi0,d,N) for PBC, optional
        Initial MPS for psi0.
      imprecision: float, optional
        Expected relative imprecision of the end results.
      D_L_max: integer, optional
        Maximal value of D_L (D_L is bond dimension for MPO representing L).
      D_L_max_forced: bool, optional
        True if D_L_max has to be reached, otherwise False.
      L_herm: bool, optional
        True if the Hermitian gauge has to be imposed on MPO representing L, otherwise False.
      D_psi0_max: integer, optional
        Maximal value of D_psi0 (D_psi0 is bond dimension for MPS representing psi0).
      D_psi0_max_forced: bool, optional
        True if D_psi0_max has to be reached, otherwise False.
    
    Returns:
      result: float
        Optimal value of the figure of merit.
      result_m: ndarray
        Matrix describing the figure of merit as a function of bond dimensions of respectively L [rows] and psi0 [columns].
      L: list of length N of ndarrays of a shape (Dl_L,Dr_L,d,d) for OBC (Dl_L, Dr_L can vary between sites) or ndarray of a shape (D_L,D_L,d,d,N) for PBC
        Optimal L in MPO representation.
      psi0: list of length N of ndarrays of a shape (Dl_psi0,Dr_psi0,d) for OBC (Dl_psi0, Dr_psi0 can vary between sites) or ndarray of a shape (D_psi0,D_psi0,d,N) for PBC
        Optimal psi0 in MPS representation.
    """
    if epsilon is None:
        result, result_m, L, psi0 = fin_FoM_FoMD_optbd(N, d, BC, ch, ch2, L_ini, psi0_ini, imprecision, D_L_max, D_L_max_forced, L_herm, D_psi0_max, D_psi0_max_forced)
    else:
        result, result_m, L, psi0 = fin2_FoM_FoMD_optbd(N, d, BC, ch, ch2, epsilon, L_ini, psi0_ini, imprecision, D_L_max, D_L_max_forced, L_herm, D_psi0_max, D_psi0_max_forced)
    return result, result_m, L, psi0


def fin_state(N, so_before_list, h, so_after_list, rho0, BC='O', L_ini=None, imprecision=10**-2, D_L_max=100, D_L_max_forced=False, L_herm=True):
    """
    Optimization of the QFI over operator L (in MPO representation) and check of convergence when increasing its bond dimension. Function for finite size systems and fixed state of the system.
    
    User has to provide information about the dynamics by specifying a quantum channel. It is assumed that the quantum channel is translationally invariant and is built from layers of quantum operations.
    User has to provide one defining operation for each layer  as a local superoperator. Those local superoperator have to be input in order of their action on the system.
    Parameter encoding is a stand out quantum operation. It is assumed that parameter encoding acts only once and is unitary so the user has to provide only its generator h.
    Generator h has to be diagonal in the computational basis, or in other words it is assumed that local superoperators are expressed in the eigenbasis of h.
    
    Parameters:
      N: integer
        Number of sites in the chain of tensors (usually number of particles).
      so_before_list: list of ndarrays of a shape (d**(2*k),d**(2*k)) where k describes on how many sites particular local superoperator acts
        List of local superoperators (in order) which act before unitary parameter encoding.
      h: ndarray of a shape (d,d)
        Generator of unitary parameter encoding. Dimension d is the dimension of local Hilbert space (dimension of physical index).
        Generator h have to be diagonal in computational basis, or in other words it is assumed that local superoperators are expressed in the eigenbasis of h.
      so_after_list: list of ndarrays of a shape (d**(2*k),d**(2*k)) where k describes on how many sites particular local superoperator acts
        List of local superoperators (in order) which act after unitary parameter encoding.
      rho0: list of length N of ndarrays of a shape (Dl_rho0,Dr_rho0,d,d) for OBC (Dl_rho0, Dr_rho0 can vary between sites) or ndarray of a shape (D_rho0,D_rho0,d,d,N) for PBC
        Density matrix describing initial state of the system in MPO representation.
      BC: 'O' or 'P', optional
        Boundary conditions, 'O' for OBC, 'P' for PBC.
      L_ini: list of length N of ndarrays of shape (Dl_L,Dr_L,d,d) for OBC, (Dl_L, Dr_L can vary between sites) or ndarray of shape (D_L,D_L,d,d,N) for PBC, optional
        Initial MPO for L.
      imprecision: float, optional
        Expected relative imprecision of the end results.
      D_L_max: integer, optional
        Maximal value of D_L (D_L is bond dimension for MPO representing L).
      D_L_max_forced: bool, optional
        True if D_L_max has to be reached, otherwise False.
      L_herm: bool, optional
        True if Hermitian gauge has to be imposed on MPO representing L, otherwise False.
    
    Returns:
      result: float
        Optimal value of figure of merit.
      result_v: ndarray
        Vector describing figure of merit in function of bond dimensions of L.
      L: list of length N of ndarrays of a shape (Dl_L,Dr_L,d,d) for OBC (Dl_L, Dr_L can vary between sites) or ndarray of a shape (D_L,D_L,d,d,N) for PBC
        Optimal L in the MPO representation.
    """
    if np.linalg.norm(h - np.diag(np.diag(h))) > 10**-10:
        warnings.warn('Generator h have to be diagonal in computational basis, or in other words it is assumed that local superoperators are expressed in the eigenbasis of h.')
    d = np.shape(h)[0]
    ch = fin_create_channel(N, d, BC, so_before_list + so_after_list)
    ch2 = fin_create_channel_derivative(N, d, BC, so_before_list, h, so_after_list)
    rho = channel_acting_on_operator(ch, rho0)
    rho2 = channel_acting_on_operator(ch2, rho0)
    result, result_v, L = fin_state_gen(N, d, BC, rho, rho2, None, L_ini, imprecision, D_L_max, D_L_max_forced, L_herm)
    return result, result_v, L


def fin_state_gen(N, d, BC, rho, rho2, epsilon=None, L_ini=None, imprecision=10**-2, D_L_max=100, D_L_max_forced=False, L_herm=True):
    """
    Optimization of the the figure of merit (usually interpreted as the QFI) over operator L (in MPO representation) and check of convergence when increasing its bond dimension. Function for finite size systems and fixed state of the system.
    User has to provide information about the dynamics by specifying a quantum channel ch and its derivative ch2 (or two channels separated by small parameter epsilon) as superoperators in the MPO representation.
    There are no constraints on the structure of the channel but the complexity of calculations highly depends on channel's bond dimension.
    
    Parameters:
      N: integer
        Number of sites in the chain of tensors (usually number of particles).
      d: integer
        Dimension of local Hilbert space (dimension of physical index).
      BC: 'O' or 'P'
        Boundary conditions, 'O' for OBC, 'P' for PBC.
      rho: list of length N of ndarrays of a shape (Dl_rho,Dr_rho,d,d) for OBC (Dl_rho, Dr_rho can vary between sites) or ndarray of a shape (D_rho,D_rho,d,d,N) for PBC
        Density matrix at the output of the quantum channel in the MPO representation.
      rho2: list of length N of ndarrays of a shape (Dl_rho2,Dr_rho2,d,d) for OBC (Dl_rho2, Dr_rho2 can vary between sites) or ndarray of a shape (D_rho2,D_rho2,d,d,N) for PBC
        Interpretaion depends on whether epsilon is specifed (2) or not (1, default approach):
        1) derivative of density matrix at the output of quantum channel in MPO representation,
        2) density matrix at the output of quantum channel in MPO representation for the value of estimated parameter shifted by epsilon in relation to rho.
      epsilon: float, optional
        If specified then it is interpeted as the value of separation between estimated parameters encoded in rho and rho2.
      L_ini: list of length N of ndarrays of a shape (Dl_L,Dr_L,d,d) for OBC (Dl_L, Dr_L can vary between sites) or ndarray of a shape (D_L,D_L,d,d,N) for PBC, optional
        Initial MPO for L.
      imprecision: float, optional
        Expected relative imprecision of the end results.
      D_L_max: integer, optional
        Maximal value of D_L (D_L is bond dimension for MPO representing L).
      D_L_max_forced: bool, optional
        True if D_L_max has to be reached, otherwise False.
      L_herm: bool, optional
        True if Hermitian gauge has to be imposed on MPO representing L, otherwise False.
    
    Returns:
      result: float
        Optimal value of figure of merit.
      result_v: ndarray
        Vector describing figure of merit as a function of bond dimensions of L.
      L: list of length N of ndarrays of a shape (Dl_L,Dr_L,d,d) for OBC (Dl_L, Dr_L can vary between sites) or ndarray of a shape (D_L,D_L,d,d,N) for PBC
        Optimal L in MPO representation.
    """
    if epsilon is None:
        result, result_v, L = fin_FoM_optbd(N, d, BC, rho, rho2, L_ini, imprecision, D_L_max, D_L_max_forced, L_herm)
    else:
        result, result_v, L = fin2_FoM_optbd(N, d, BC, rho, rho2, epsilon, L_ini, imprecision, D_L_max, D_L_max_forced, L_herm)
    return result, result_v, L


############################
#                          #
# 1.2 Low level functions. #
#                          #
############################


def fin_create_channel(N, d, BC, so_list, tol=10**-10):
    """
    Creates MPO for a superoperator describing translationally invariant quantum channel from list of local superoperators. Function for finite size systems.
    
    For OBC, tensor-network length N has to be at least 2k-1, where k is the correlation length (number of sites on which acts the biggest local superoperator).
    Local superoperators acting on more then 4 neighbouring sites are not currently supported.
    
    Parameters:
      N: integer
        Number of sites in the chain of tensors (usually number of particles).
        For OBC tensor-network length N has to be at least 2k-1 where k is the correlation length (number of sites on which acts the biggest local superoperator).
      d: integer
        Dimension of local Hilbert space (dimension of physical index).
      BC: 'O' or 'P'
        Boundary conditions, 'O' for OBC, 'P' for PBC.
      so_list: list of ndarrays of a shape (d**(2*k),d**(2*k)) where k describes on how many sites a particular local superoperator acts
        List of local superoperators in order of their action on the system.
        Local superoperators acting on more then 4 neighbour sites are not currently supported.
      tol: float, optional
        Factor which after multiplication by the highest singular value gives a cutoff on singular values that are treated as nonzero.
    
    Returns:
      ch: list of length N of ndarrays of shape (Dl_ch,Dr_ch,d**2,d**2) for OBC (Dl_ch, Dr_ch can vary between sites) or ndarray of shape (D_ch,D_ch,d**2,d**2,N) for PBC
        Quantum channel as a superoperator in the MPO representation.
    """
    if so_list == []:
        if BC == 'O':
            ch = np.eye(d**2,dtype=complex)
            ch = ch[np.newaxis,np.newaxis,:,:]
            ch = [ch]*N
        elif BC == 'P':
            ch = np.eye(d**2,dtype=complex)
            ch = ch[np.newaxis,np.newaxis,:,:,np.newaxis]
            ch = np.tile(ch,(1,1,1,1,N))
        return ch
    if BC == 'O':
        ch = [0]*N
        kmax = max([int(math.log(np.shape(so_list[i])[0],d**2)) for i in range(len(so_list))])
        if N < 2*kmax-1:
            warnings.warn('For OBC tensor-network length N have to be at least 2k-1 where k is correlation length (number of sites on which acts the biggest local superoperator).')
        for x in range(N):
            if x >= kmax and N-x >= kmax:
                ch[x] = ch[x-1]
                continue
            for i in range(len(so_list)):
                so = so_list[i]
                k = int(math.log(np.shape(so)[0],d**2))
                if np.linalg.norm(so-np.diag(np.diag(so))) < 10**-10:
                    so = np.diag(so)
                    if k == 1:
                        bdchil = 1
                        bdchir = 1
                        chi = np.zeros((bdchil,bdchir,d**2,d**2),dtype=complex)
                        for nx in range(d**2):
                            chi[:,:,nx,nx] = so[nx]
                    elif k == 2:
                        so = np.reshape(so,(d**2,d**2),order='F')
                        u,s,vh = np.linalg.svd(so)
                        s = s[s > s[0]*tol]
                        bdchi = np.shape(s)[0]
                        u = u[:,:bdchi]
                        vh = vh[:bdchi,:]
                        us = u @ np.diag(np.sqrt(s))
                        sv = np.diag(np.sqrt(s)) @ vh
                        if x == 0:
                            bdchil = 1
                            bdchir = bdchi
                            chi = np.zeros((bdchil,bdchir,d**2,d**2),dtype=complex)
                            for nx in range(d**2):
                                tensors = [us[nx,:]]
                                legs = [[-1]]
                                chi[:,:,nx,nx] = np.reshape(ncon(tensors,legs),(bdchil,bdchir),order='F')
                        elif x > 0 and x < N-1:
                            bdchil = bdchi
                            bdchir = bdchi
                            chi = np.zeros((bdchil,bdchir,d**2,d**2),dtype=complex)
                            for nx in range(d**2):
                                tensors = [sv[:,nx],us[nx,:]]
                                legs = [[-1],[-2]]
                                chi[:,:,nx,nx] = np.reshape(ncon(tensors,legs),(bdchil,bdchir),order='F')
                        elif x == N-1:
                            bdchil = bdchi
                            bdchir = 1
                            chi = np.zeros((bdchil,bdchir,d**2,d**2),dtype=complex)
                            for nx in range(d**2):
                                tensors = [sv[:,nx]]
                                legs = [[-1]]
                                chi[:,:,nx,nx] = np.reshape(ncon(tensors,legs),(bdchil,bdchir),order='F')
                    elif k == 3:
                        so = np.reshape(so,(d**2,d**4),order='F')
                        u1,s1,vh1 = np.linalg.svd(so,full_matrices=False)
                        s1 = s1[s1 > s1[0]*tol]
                        bdchi1 = np.shape(s1)[0]
                        u1 = u1[:,:bdchi1]
                        vh1 = vh1[:bdchi1,:]
                        us1 = u1 @ np.diag(np.sqrt(s1))
                        sv1 = np.diag(np.sqrt(s1)) @ vh1
                        sv1 = np.reshape(sv1,(bdchi1*d**2,d**2),order='F')
                        u2,s2,vh2 = np.linalg.svd(sv1,full_matrices=False)
                        s2 = s2[s2 > s2[0]*tol]
                        bdchi2 = np.shape(s2)[0]
                        u2 = u2[:,:bdchi2]
                        vh2 = vh2[:bdchi2,:]
                        us2 = u2 @ np.diag(np.sqrt(s2))
                        us2 = np.reshape(us2,(bdchi1,d**2,bdchi2),order='F')
                        sv2 = np.diag(np.sqrt(s2)) @ vh2
                        if x == 0:
                            bdchil = 1
                            bdchir = bdchi1
                            chi = np.zeros((bdchil,bdchir,d**2,d**2),dtype=complex)
                            for nx in range(d**2):
                                tensors = [us1[nx,:]]
                                legs = [[-1]]
                                chi[:,:,nx,nx] = np.reshape(ncon(tensors,legs),(bdchil,bdchir),order='F')
                        elif x == 1:
                            bdchil = bdchi1
                            bdchir = bdchi2*bdchi1
                            chi = np.zeros((bdchil,bdchir,d**2,d**2),dtype=complex)
                            for nx in range(d**2):
                                tensors = [us2[:,nx,:],us1[nx,:]]
                                legs = [[-1,-2],[-3]]
                                chi[:,:,nx,nx] = np.reshape(ncon(tensors,legs),(bdchil,bdchir),order='F')
                        elif x > 1 and x < N-2:
                            bdchil = bdchi2*bdchi1
                            bdchir = bdchi2*bdchi1
                            chi = np.zeros((bdchil,bdchir,d**2,d**2),dtype=complex)
                            for nx in range(d**2):
                                tensors = [sv2[:,nx],us2[:,nx,:],us1[nx,:]]
                                legs = [[-1],[-2,-3],[-4]]
                                chi[:,:,nx,nx] = np.reshape(ncon(tensors,legs),(bdchil,bdchir),order='F')
                        elif x == N-2:
                            bdchil = bdchi2*bdchi1
                            bdchir = bdchi2
                            chi = np.zeros((bdchil,bdchir,d**2,d**2),dtype=complex)
                            for nx in range(d**2):
                                tensors = [sv2[:,nx],us2[:,nx,:]]
                                legs = [[-1],[-2,-3]]
                                chi[:,:,nx,nx] = np.reshape(ncon(tensors,legs),(bdchil,bdchir),order='F')
                        elif x == N-1:
                            bdchil = bdchi2
                            bdchir = 1
                            chi = np.zeros((bdchil,bdchir,d**2,d**2),dtype=complex)
                            for nx in range(d**2):
                                tensors = [sv2[:,nx]]
                                legs = [[-1]]
                                chi[:,:,nx,nx] = np.reshape(ncon(tensors,legs),(bdchil,bdchir),order='F')
                    elif k == 4:
                        so = np.reshape(so,(d**2,d**6),order='F')
                        u1,s1,vh1 = np.linalg.svd(so,full_matrices=False)
                        s1 = s1[s1 > s1[0]*tol]
                        bdchi1 = np.shape(s1)[0]
                        u1 = u1[:,:bdchi1]
                        vh1 = vh1[:bdchi1,:]
                        us1 = u1 @ np.diag(np.sqrt(s1))
                        sv1 = np.diag(np.sqrt(s1)) @ vh1
                        sv1 = np.reshape(sv1,(bdchi1*d**2,d**4),order='F')
                        u2,s2,vh2 = np.linalg.svd(sv1,full_matrices=False)
                        s2 = s2[s2 > s2[0]*tol]
                        bdchi2 = np.shape(s2)[0]
                        u2 = u2[:,:bdchi2]
                        vh2 = vh2[:bdchi2,:]
                        us2 = u2 @ np.diag(np.sqrt(s2))
                        us2 = np.reshape(us2,(bdchi1,d**2,bdchi2),order='F')
                        sv2 = np.diag(np.sqrt(s2)) @ vh2
                        sv2 = np.reshape(sv2,(bdchi2*d**2,d**2),order='F')
                        u3,s3,vh3 = np.linalg.svd(sv2,full_matrices=False)
                        s3 = s3[s3 > s3[0]*tol]
                        bdchi3 = np.shape(s3)[0]
                        u3 = u3[:,:bdchi3]
                        vh3 = vh3[:bdchi3,:]
                        us3 = u3 @ np.diag(np.sqrt(s3))
                        us3 = np.reshape(us3,(bdchi2,d**2,bdchi3),order='F')
                        sv3 = np.diag(np.sqrt(s3)) @ vh3
                        if x == 0:
                            bdchil = 1
                            bdchir = bdchi1
                            chi = np.zeros((bdchil,bdchir,d**2,d**2),dtype=complex)
                            for nx in range(d**2):
                                tensors = [us1[nx,:]]
                                legs = [[-1]]
                                chi[:,:,nx,nx] = np.reshape(ncon(tensors,legs),(bdchil,bdchir),order='F')
                        elif x == 1:
                            bdchil = bdchi1
                            bdchir = bdchi2*bdchi1
                            chi = np.zeros((bdchil,bdchir,d**2,d**2),dtype=complex)
                            for nx in range(d**2):
                                tensors = [us2[:,nx,:],us1[nx,:]]
                                legs = [[-1,-2],[-3]]
                                chi[:,:,nx,nx] = np.reshape(ncon(tensors,legs),(bdchil,bdchir),order='F')
                        elif x == 2:
                            bdchil = bdchi2*bdchi1
                            bdchir = bdchi3*bdchi2*bdchi1
                            chi = np.zeros((bdchil,bdchir,d**2,d**2),dtype=complex)
                            for nx in range(d**2):
                                tensors = [us3[:,nx,:],us2[:,nx,:],us1[nx,:]]
                                legs = [[-1,-3],[-2,-4],[-5]]
                                chi[:,:,nx,nx] = np.reshape(ncon(tensors,legs),(bdchil,bdchir),order='F')
                        elif x > 2 and x < N-3:
                            bdchil = bdchi3*bdchi2*bdchi1
                            bdchir = bdchi3*bdchi2*bdchi1
                            chi = np.zeros((bdchil,bdchir,d**2,d**2),dtype=complex)
                            for nx in range(d**2):
                                tensors = [sv3[:,nx],us3[:,nx,:],us2[:,nx,:],us1[nx,:]]
                                legs = [[-1],[-2,-4],[-3,-5],[-6]]
                                chi[:,:,nx,nx] = np.reshape(ncon(tensors,legs),(bdchil,bdchir),order='F')
                        elif x == N-3:
                            bdchil = bdchi3*bdchi2*bdchi1
                            bdchir = bdchi3*bdchi2
                            chi = np.zeros((bdchil,bdchir,d**2,d**2),dtype=complex)
                            for nx in range(d**2):
                                tensors = [sv3[:,nx],us3[:,nx,:],us2[:,nx,:]]
                                legs = [[-1],[-2,-4],[-3,-5]]
                                chi[:,:,nx,nx] = np.reshape(ncon(tensors,legs),(bdchil,bdchir),order='F')
                        elif x == N-2:
                            bdchil = bdchi3*bdchi2
                            bdchir = bdchi3
                            chi = np.zeros((bdchil,bdchir,d**2,d**2),dtype=complex)
                            for nx in range(d**2):
                                tensors = [sv3[:,nx],us3[:,nx,:]]
                                legs = [[-1],[-2,-3]]
                                chi[:,:,nx,nx] = np.reshape(ncon(tensors,legs),(bdchil,bdchir),order='F')
                        elif x == N-1:
                            bdchil = bdchi3
                            bdchir = 1
                            chi = np.zeros((bdchil,bdchir,d**2,d**2),dtype=complex)
                            for nx in range(d**2):
                                tensors = [sv3[:,nx]]
                                legs = [[-1]]
                                chi[:,:,nx,nx] = np.reshape(ncon(tensors,legs),(bdchil,bdchir),order='F')
                    else:
                        warnings.warn('Local superoperators acting on more then 4 neighbour sites are not currently supported.')
                else:
                    if k == 1:
                        bdchil = 1
                        bdchir = 1
                        chi = so[np.newaxis,np.newaxis,:,:]
                    elif k == 2:
                        u,s,vh = np.linalg.svd(so)
                        s = s[s > s[0]*tol]
                        bdchi = np.shape(s)[0]
                        u = u[:,:bdchi]
                        vh = vh[:bdchi,:]
                        us = u @ np.diag(np.sqrt(s))
                        sv = np.diag(np.sqrt(s)) @ vh
                        us = np.reshape(us,(d**2,d**2,bdchi),order='F')
                        sv = np.reshape(sv,(bdchi,d**2,d**2),order='F')
                        tensors = [sv,us]
                        legs = [[-1,-3,1],[1,-4,-2]]
                        chi = ncon(tensors,legs)
                        if x == 0:
                            tensors = [us]
                            legs = [[-2,-3,-1]]
                            chi = ncon(tensors,legs)
                            bdchil = 1
                            bdchir = bdchi
                        elif x > 0 and x < N-1:
                            tensors = [sv,us]
                            legs = [[-1,-3,1],[1,-4,-2]]
                            chi = ncon(tensors,legs)
                            bdchil = bdchi
                            bdchir = bdchi
                        elif x == N-1:
                            tensors = [sv]
                            legs = [[-1,-2,-3]]
                            chi = ncon(tensors,legs)
                            bdchil = bdchi
                            bdchir = 1
                        chi = np.reshape(chi,(bdchil,bdchir,d**2,d**2),order='F')
                    elif k == 3:
                        so = np.reshape(so,(d**4,d**8),order='F')
                        u1,s1,vh1 = np.linalg.svd(so,full_matrices=False)
                        s1 = s1[s1 > s1[0]*tol]
                        bdchi1 = np.shape(s1)[0]
                        u1 = u1[:,:bdchi1]
                        vh1 = vh1[:bdchi1,:]
                        us1 = u1 @ np.diag(np.sqrt(s1))
                        sv1 = np.diag(np.sqrt(s1)) @ vh1
                        us1 = np.reshape(us1,(d**2,d**2,bdchi1),order='F')
                        sv1 = np.reshape(sv1,(bdchi1*d**4,d**4),order='F')
                        u2,s2,vh2 = np.linalg.svd(sv1,full_matrices=False)
                        s2 = s2[s2 > s2[0]*tol]
                        bdchi2 = np.shape(s2)[0]
                        u2 = u2[:,:bdchi2]
                        vh2 = vh2[:bdchi2,:]
                        us2 = u2 @ np.diag(np.sqrt(s2))
                        us2 = np.reshape(us2,(bdchi1,d**2,d**2,bdchi2),order='F')
                        sv2 = np.diag(np.sqrt(s2)) @ vh2
                        sv2 = np.reshape(sv2,(bdchi2,d**2,d**2),order='F')
                        if x == 0:
                            tensors = [us1]
                            legs = [[-2,-3,-1]]
                            chi = ncon(tensors,legs)
                            bdchil = 1
                            bdchir = bdchi1
                        elif x == 1:
                            tensors = [us2,us1]
                            legs = [[-1,-5,1,-2],[1,-6,-3]]
                            chi = ncon(tensors,legs)
                            bdchil = bdchi1
                            bdchir = bdchi2*bdchi1
                        elif x > 1 and x < N-2:
                            tensors = [sv2,us2,us1]
                            legs = [[-1,-5,1],[-2,1,2,-3],[2,-6,-4]]
                            chi = ncon(tensors,legs)
                            bdchil = bdchi2*bdchi1
                            bdchir = bdchi2*bdchi1
                        elif x == N-2:
                            tensors = [sv2,us2]
                            legs = [[-1,-4,1],[-2,1,-5,-3]]
                            chi = ncon(tensors,legs)
                            bdchil = bdchi2*bdchi1
                            bdchir = bdchi2
                        elif x == N-1:
                            tensors = [sv2]
                            legs = [[-1,-2,-3]]
                            chi = ncon(tensors,legs)
                            bdchil = bdchi2
                            bdchir = 1
                        chi = np.reshape(chi,(bdchil,bdchir,d**2,d**2),order='F')
                    elif k == 4:
                        so = np.reshape(so,(d**4,d**12),order='F')
                        u1,s1,vh1 = np.linalg.svd(so,full_matrices=False)
                        s1 = s1[s1 > s1[0]*tol]
                        bdchi1 = np.shape(s1)[0]
                        u1 = u1[:,:bdchi1]
                        vh1 = vh1[:bdchi1,:]
                        us1 = u1 @ np.diag(np.sqrt(s1))
                        sv1 = np.diag(np.sqrt(s1)) @ vh1
                        us1 = np.reshape(us1,(d**2,d**2,bdchi1),order='F')
                        sv1 = np.reshape(sv1,(bdchi1*d**4,d**8),order='F')
                        u2,s2,vh2 = np.linalg.svd(sv1,full_matrices=False)
                        s2 = s2[s2 > s2[0]*tol]
                        bdchi2 = np.shape(s2)[0]
                        u2 = u2[:,:bdchi2]
                        vh2 = vh2[:bdchi2,:]
                        us2 = u2 @ np.diag(np.sqrt(s2))
                        us2 = np.reshape(us2,(bdchi1,d**2,d**2,bdchi2),order='F')
                        sv2 = np.diag(np.sqrt(s2)) @ vh2
                        sv2 = np.reshape(sv2,(bdchi2*d**4,d**4),order='F')
                        u3,s3,vh3 = np.linalg.svd(sv2,full_matrices=False)
                        s3 = s3[s3 > s3[0]*tol]
                        bdchi3 = np.shape(s3)[0]
                        u3 = u3[:,:bdchi3]
                        vh3 = vh3[:bdchi3,:]
                        us3 = u3 @ np.diag(np.sqrt(s3))
                        us3 = np.reshape(us3,(bdchi2,d**2,d**2,bdchi3),order='F')
                        sv3 = np.diag(np.sqrt(s3)) @ vh3
                        sv3 = np.reshape(sv3,(bdchi3,d**2,d**2),order='F')
                        if x == 0:
                            tensors = [us1]
                            legs = [[-2,-3,-1]]
                            chi = ncon(tensors,legs)
                            bdchil = 1
                            bdchir = bdchi1
                        elif x == 1:
                            tensors = [us2,us1]
                            legs = [[-1,-4,1,-2],[1,-5,-3]]
                            chi = ncon(tensors,legs)
                            bdchil = bdchi1
                            bdchir = bdchi2*bdchi1
                        elif x == 2:
                            tensors = [us3,us2,us1]
                            legs = [[-1,-6,1,-3],[-2,1,2,-4],[2,-7,-5]]
                            chi = ncon(tensors,legs)
                            bdchil = bdchi2*bdchi1
                            bdchir = bdchi3*bdchi2*bdchi1
                        elif x > 2 and x < N-3:
                            tensors = [sv3,us3,us2,us1]
                            legs = [[-1,-7,1],[-2,1,2,-4],[-3,2,3,-5],[3,-8,-6]]
                            chi = ncon(tensors,legs)
                            bdchil = bdchi3*bdchi2*bdchi1
                            bdchir = bdchi3*bdchi2*bdchi1
                        elif x == N-3:
                            tensors = [sv3,us3,us2]
                            legs = [[-1,-6,1],[-2,1,2,-4],[-3,2,-7,-5]]
                            chi = ncon(tensors,legs)
                            bdchil = bdchi3*bdchi2*bdchi1
                            bdchir = bdchi3*bdchi2
                        elif x == N-2:
                            tensors = [sv3,us3]
                            legs = [[-1,-4,1],[-2,1,-5,-3]]
                            chi = ncon(tensors,legs)
                            bdchil = bdchi3*bdchi2
                            bdchir = bdchi3
                        elif x == N-1:
                            tensors = [sv3]
                            legs = [[-1,-2,-3]]
                            chi = ncon(tensors,legs)
                            bdchil = bdchi3
                            bdchir = 1
                        chi = np.reshape(chi,(bdchi,bdchi,d**2,d**2),order='F')
                    else:
                        warnings.warn('Local superoperators acting on more then 4 neighbour sites are not currently supported.')
                if i == 0:
                    bdchl = bdchil
                    bdchr = bdchir
                    ch[x] = chi
                else:
                    bdchl = bdchil*bdchl
                    bdchr = bdchir*bdchr
                    tensors = [chi,ch[x]]
                    legs = [[-1,-3,-5,1],[-2,-4,1,-6]]
                    ch[x] = ncon(tensors,legs)
                    ch[x] = np.reshape(ch[x],(bdchl,bdchr,d**2,d**2),order='F')
    elif BC == 'P':
        for i in range(len(so_list)):
            so = so_list[i]
            k = int(math.log(np.shape(so)[0],d**2))
            if np.linalg.norm(so-np.diag(np.diag(so))) < 10**-10:
                so = np.diag(so)
                if k == 1:
                    bdchi = 1
                    chi = np.zeros((bdchi,bdchi,d**2,d**2),dtype=complex)
                    for nx in range(d**2):
                        chi[:,:,nx,nx] = so[nx]
                elif k == 2:
                    so = np.reshape(so,(d**2,d**2),order='F')
                    u,s,vh = np.linalg.svd(so)
                    s = s[s > s[0]*tol]
                    bdchi = np.shape(s)[0]
                    u = u[:,:bdchi]
                    vh = vh[:bdchi,:]
                    us = u @ np.diag(np.sqrt(s))
                    sv = np.diag(np.sqrt(s)) @ vh
                    chi = np.zeros((bdchi,bdchi,d**2,d**2),dtype=complex)
                    for nx in range(d**2):
                        chi[:,:,nx,nx] = np.outer(sv[:,nx],us[nx,:])
                elif k == 3:
                    so = np.reshape(so,(d**2,d**4),order='F')
                    u1,s1,vh1 = np.linalg.svd(so,full_matrices=False)
                    s1 = s1[s1 > s1[0]*tol]
                    bdchi1 = np.shape(s1)[0]
                    u1 = u1[:,:bdchi1]
                    vh1 = vh1[:bdchi1,:]
                    us1 = u1 @ np.diag(np.sqrt(s1))
                    sv1 = np.diag(np.sqrt(s1)) @ vh1
                    sv1 = np.reshape(sv1,(bdchi1*d**2,d**2),order='F')
                    u2,s2,vh2 = np.linalg.svd(sv1,full_matrices=False)
                    s2 = s2[s2 > s2[0]*tol]
                    bdchi2 = np.shape(s2)[0]
                    u2 = u2[:,:bdchi2]
                    vh2 = vh2[:bdchi2,:]
                    us2 = u2 @ np.diag(np.sqrt(s2))
                    us2 = np.reshape(us2,(bdchi1,d**2,bdchi2),order='F')
                    sv2 = np.diag(np.sqrt(s2)) @ vh2
                    bdchi = bdchi2*bdchi1
                    chi = np.zeros((bdchi,bdchi,d**2,d**2),dtype=complex)
                    for nx in range(d**2):
                        tensors = [sv2[:,nx],us2[:,nx,:],us1[nx,:]]
                        legs = [[-1],[-2,-3],[-4]]
                        chi[:,:,nx,nx] = np.reshape(ncon(tensors,legs),(bdchi,bdchi),order='F')
                elif k == 4:
                    so = np.reshape(so,(d**2,d**6),order='F')
                    u1,s1,vh1 = np.linalg.svd(so,full_matrices=False)
                    s1 = s1[s1 > s1[0]*tol]
                    bdchi1 = np.shape(s1)[0]
                    u1 = u1[:,:bdchi1]
                    vh1 = vh1[:bdchi1,:]
                    us1 = u1 @ np.diag(np.sqrt(s1))
                    sv1 = np.diag(np.sqrt(s1)) @ vh1
                    sv1 = np.reshape(sv1,(bdchi1*d**2,d**4),order='F')
                    u2,s2,vh2 = np.linalg.svd(sv1,full_matrices=False)
                    s2 = s2[s2 > s2[0]*tol]
                    bdchi2 = np.shape(s2)[0]
                    u2 = u2[:,:bdchi2]
                    vh2 = vh2[:bdchi2,:]
                    us2 = u2 @ np.diag(np.sqrt(s2))
                    us2 = np.reshape(us2,(bdchi1,d**2,bdchi2),order='F')
                    sv2 = np.diag(np.sqrt(s2)) @ vh2
                    sv2 = np.reshape(sv2,(bdchi2*d**2,d**2),order='F')
                    u3,s3,vh3 = np.linalg.svd(sv2,full_matrices=False)
                    s3 = s3[s3 > s3[0]*tol]
                    bdchi3 = np.shape(s3)[0]
                    u3 = u3[:,:bdchi3]
                    vh3 = vh3[:bdchi3,:]
                    us3 = u3 @ np.diag(np.sqrt(s3))
                    us3 = np.reshape(us3,(bdchi2,d**2,bdchi3),order='F')
                    sv3 = np.diag(np.sqrt(s3)) @ vh3
                    bdchi = bdchi3*bdchi2*bdchi1
                    chi = np.zeros((bdchi,bdchi,d**2,d**2),dtype=complex)
                    for nx in range(d**2):
                        tensors = [sv3[:,nx],us3[:,nx,:],us2[:,nx,:],us1[nx,:]]
                        legs = [[-1],[-2,-4],[-3,-5],[-6]]
                        chi[:,:,nx,nx] = np.reshape(ncon(tensors,legs),(bdchi,bdchi),order='F')
                else:
                    warnings.warn('Local superoperators acting on more then 4 neighbour sites are not currently supported.')
            else:
                if k == 1:
                    bdchi = 1
                    chi = so[np.newaxis,np.newaxis,:,:]
                elif k == 2:
                    u,s,vh = np.linalg.svd(so)
                    s = s[s > s[0]*tol]
                    bdchi = np.shape(s)[0]
                    u = u[:,:bdchi]
                    vh = vh[:bdchi,:]
                    us = u @ np.diag(np.sqrt(s))
                    sv = np.diag(np.sqrt(s)) @ vh
                    us = np.reshape(us,(d**2,d**2,bdchi),order='F')
                    sv = np.reshape(sv,(bdchi,d**2,d**2),order='F')
                    tensors = [sv,us]
                    legs = [[-1,-3,1],[1,-4,-2]]
                    chi = ncon(tensors,legs)
                elif k == 3:
                    so = np.reshape(so,(d**4,d**8),order='F')
                    u1,s1,vh1 = np.linalg.svd(so,full_matrices=False)
                    s1 = s1[s1 > s1[0]*tol]
                    bdchi1 = np.shape(s1)[0]
                    u1 = u1[:,:bdchi1]
                    vh1 = vh1[:bdchi1,:]
                    us1 = u1 @ np.diag(np.sqrt(s1))
                    sv1 = np.diag(np.sqrt(s1)) @ vh1
                    us1 = np.reshape(us1,(d**2,d**2,bdchi1),order='F')
                    sv1 = np.reshape(sv1,(bdchi1*d**4,d**4),order='F')
                    u2,s2,vh2 = np.linalg.svd(sv1,full_matrices=False)
                    s2 = s2[s2 > s2[0]*tol]
                    bdchi2 = np.shape(s2)[0]
                    u2 = u2[:,:bdchi2]
                    vh2 = vh2[:bdchi2,:]
                    us2 = u2 @ np.diag(np.sqrt(s2))
                    us2 = np.reshape(us2,(bdchi1,d**2,d**2,bdchi2),order='F')
                    sv2 = np.diag(np.sqrt(s2)) @ vh2
                    sv2 = np.reshape(sv2,(bdchi2,d**2,d**2),order='F')
                    tensors = [sv2,us2,us1]
                    legs = [[-1,-5,1],[-2,1,2,-3],[2,-6,-4]]
                    chi = ncon(tensors,legs)
                    bdchi = bdchi2*bdchi1
                    chi = np.reshape(chi,(bdchi,bdchi,d**2,d**2),order='F')
                elif k == 4:
                    so = np.reshape(so,(d**4,d**12),order='F')
                    u1,s1,vh1 = np.linalg.svd(so,full_matrices=False)
                    s1 = s1[s1 > s1[0]*tol]
                    bdchi1 = np.shape(s1)[0]
                    u1 = u1[:,:bdchi1]
                    vh1 = vh1[:bdchi1,:]
                    us1 = u1 @ np.diag(np.sqrt(s1))
                    sv1 = np.diag(np.sqrt(s1)) @ vh1
                    us1 = np.reshape(us1,(d**2,d**2,bdchi1),order='F')
                    sv1 = np.reshape(sv1,(bdchi1*d**4,d**8),order='F')
                    u2,s2,vh2 = np.linalg.svd(sv1,full_matrices=False)
                    s2 = s2[s2 > s2[0]*tol]
                    bdchi2 = np.shape(s2)[0]
                    u2 = u2[:,:bdchi2]
                    vh2 = vh2[:bdchi2,:]
                    us2 = u2 @ np.diag(np.sqrt(s2))
                    us2 = np.reshape(us2,(bdchi1,d**2,d**2,bdchi2),order='F')
                    sv2 = np.diag(np.sqrt(s2)) @ vh2
                    sv2 = np.reshape(sv2,(bdchi2*d**4,d**4),order='F')
                    u3,s3,vh3 = np.linalg.svd(sv2,full_matrices=False)
                    s3 = s3[s3 > s3[0]*tol]
                    bdchi3 = np.shape(s3)[0]
                    u3 = u3[:,:bdchi3]
                    vh3 = vh3[:bdchi3,:]
                    us3 = u3 @ np.diag(np.sqrt(s3))
                    us3 = np.reshape(us3,(bdchi2,d**2,d**2,bdchi3),order='F')
                    sv3 = np.diag(np.sqrt(s3)) @ vh3
                    sv3 = np.reshape(sv3,(bdchi3,d**2,d**2),order='F')
                    tensors = [sv3,us3,us2,us1]
                    legs = [[-1,-7,1],[-2,1,2,-4],[-3,2,3,-5],[3,-8,-6]]
                    chi = ncon(tensors,legs)
                    bdchi = bdchi3*bdchi2*bdchi1
                    chi = np.reshape(chi,(bdchi,bdchi,d**2,d**2),order='F')
                else:
                    warnings.warn('Local superoperators acting on more then 4 neighbour sites are not currently supported.')
            if i == 0:
                bdch = bdchi
                ch = chi
            else:
                bdch = bdchi*bdch
                tensors = [chi,ch]
                legs = [[-1,-3,-5,1],[-2,-4,1,-6]]
                ch = ncon(tensors,legs)
                ch = np.reshape(ch,(bdch,bdch,d**2,d**2),order='F')
        ch = ch[:,:,:,:,np.newaxis]
        ch = np.tile(ch,(1,1,1,1,N))
    return ch


def fin_create_channel_derivative(N, d, BC, so_before_list, h, so_after_list):
    """
    Creates a MPO for the derivative (over estimated parameter) of the superoperator describing the quantum channel. Function for finite size systems.
    
    Function for translationally invariant channels with unitary parameter encoding generated by h.
    Generator h has to be diagonal in the computational basis, or in other words it is assumed that local superoperators are expressed in the eigenbasis of h.
    
    Parameters:
      N: integer
        Number of sites in the chain of tensors (usually number of particles).
      d: integer
        Dimension of local Hilbert space (dimension of physical index).
      BC: 'O' or 'P'
        Boundary conditions, 'O' for OBC, 'P' for PBC.
      so_before_list: list of ndarrays of a shape (d**(2*k),d**(2*k)) where k describes on how many sites particular local superoperator acts
        List of local superoperators (in order) which act before unitary parameter encoding.
      h: ndarray of a shape (d,d)
        Generator of unitary parameter encoding.
        Generator h have to be diagonal in computational basis, or in other words it is assumed that local superoperators are expressed in the eigenbasis of h.
      so_after_list: list of ndarrays of a shape (d**(2*k),d**(2*k)) where k describes on how many sites particular local superoperator acts
        List of local superoperators (in order) which act after unitary parameter encoding.
    
    Returns:
      chd: list of length N of ndarrays of a shape (Dl_chd,Dr_chd,d**2,d**2) for OBC (Dl_chd, Dr_chd can vary between sites) or ndarray of a shape (D_chd,D_chd,d**2,d**2,N) for PBC
        Derivative of superoperator describing quantum channel in MPO representation.
    """
    if np.linalg.norm(h-np.diag(np.diag(h))) > 10**-10:
        warnings.warn('Generator h have to be diagonal in computational basis, or in other words it is assumed that local superoperators are expressed in the eigenbasis of h.')
    if len(so_before_list) == 0:
        if BC == 'O':
            ch1 = np.eye(d**2,dtype=complex)
            ch1 = ch1[np.newaxis,np.newaxis,:,:]
            ch1 = [ch1]*N
        elif BC == 'P':
            ch1 = np.eye(d**2,dtype=complex)
            ch1 = ch1[np.newaxis,np.newaxis,:,:,np.newaxis]
            ch1 = np.tile(ch1,(1,1,1,1,N))
        ch1d = fin_commutator(N,d,BC,ch1,h,1j)
        ch2 = fin_create_channel(N,d,BC,so_after_list)
        if BC == 'O':
            chd = [0]*N
            for x in range(N):
                bdch1dl = np.shape(ch1d[x])[0]
                bdch1dr = np.shape(ch1d[x])[1]
                bdch2l = np.shape(ch2[x])[0]
                bdch2r = np.shape(ch2[x])[1]
                tensors = [ch2[x],ch1d[x]]
                legs = [[-1,-3,-5,1],[-2,-4,1,-6]]
                chd[x] = np.reshape(ncon(tensors,legs),(bdch1dl*bdch2l,bdch1dr*bdch2r,d**2,d**2),order='F')
        elif BC == 'P':
            bdch1d = np.shape(ch1d)[0]
            bdch2 = np.shape(ch2)[0]
            chd = np.zeros((bdch1d*bdch2,bdch1d*bdch2,d**2,d**2,N),dtype=complex)
            for x in range(N):
                tensors = [ch2[:,:,:,:,x],ch1d[:,:,:,:,x]]
                legs = [[-1,-3,-5,1],[-2,-4,1,-6]]
                chd[:,:,:,:,x] = np.reshape(ncon(tensors,legs),(bdch1d*bdch2,bdch1d*bdch2,d**2,d**2),order='F')
    elif len(so_after_list) == 0:
        ch1 = fin_create_channel(N,d,BC,so_before_list)
        chd = fin_commutator(N,d,BC,ch1,h,1j)
    else:
        ch1 = fin_create_channel(N,d,BC,so_before_list)
        ch1d = fin_commutator(N,d,BC,ch1,h,1j)
        ch2 = fin_create_channel(N,d,BC,so_after_list)
        if BC == 'O':
            chd = [0]*N
            for x in range(N):
                bdch1dl = np.shape(ch1d[x])[0]
                bdch1dr = np.shape(ch1d[x])[1]
                bdch2l = np.shape(ch2[x])[0]
                bdch2r = np.shape(ch2[x])[1]
                tensors = [ch2[x],ch1d[x]]
                legs = [[-1,-3,-5,1],[-2,-4,1,-6]]
                chd[x] = np.reshape(ncon(tensors,legs),(bdch1dl*bdch2l,bdch1dr*bdch2r,d**2,d**2),order='F')
        elif BC == 'P':
            bdch1d = np.shape(ch1d)[0]
            bdch2 = np.shape(ch2)[0]
            chd = np.zeros((bdch1d*bdch2,bdch1d*bdch2,d**2,d**2,N),dtype=complex)
            for x in range(N):
                tensors = [ch2[:,:,:,:,x],ch1d[:,:,:,:,x]]
                legs = [[-1,-3,-5,1],[-2,-4,1,-6]]
                chd[:,:,:,:,x] = np.reshape(ncon(tensors,legs),(bdch1d*bdch2,bdch1d*bdch2,d**2,d**2),order='F')
    return chd


def fin_commutator(N, d, BC, a, h, c):
    """
    Calculate MPO for commutator b = [a, c*sum{h}] of MPO a with sum of local generators h and with arbitrary multiplicative scalar factor c.
    
    Generator h have to be diagonal in computational basis, or in other words it is assumed that a is expressed in the eigenbasis of h.
    
    Parameters:
      N: integer
        Number of sites in the chain of tensors (usually number of particles).
      d: integer
        Dimension of local Hilbert space (dimension of physical index).
      BC: 'O' or 'P'
        Boundary conditions, 'O' for OBC, 'P' for PBC.
      a: list of length N of ndarrays of a shape (Dl_a,Dr_a,d,d) for OBC (Dl_a, Dr_a can vary between sites) or ndarray of a shape (D_a,D_a,d,d,N) for PBC
        MPO.
      h: ndarray of a shape (d,d)
        Generator of unitary parameter encoding.
        Generator h have to be diagonal in computational basis, or in other words it is assumed that a is expressed in the eigenbasis of h.
      c: complex
        Scalar factor which multiplies sum of local generators.
    
    Returns:
      b: list of length N of ndarrays of a shape (Dl_b,Dr_b,d,d) for OBC (Dl_b, Dr_b can vary between sites) or ndarray of a shape (D_b,D_b,d,d,N) for PBC
        Commutator [a, c*sum{h}] in MPO representation.
    """
    if np.linalg.norm(h-np.diag(np.diag(h))) > 10**-10:
        warnings.warn('Generator h have to be diagonal in computational basis, or in other words it is assumed that a is expressed in the eigenbasis of h.')
    if BC == 'O':
        bh = [0]*N
        b = [0]*N
        for x in range(N):
            da = np.shape(a[x])[2]
            bda1 = np.shape(a[x])[0]
            bda2 = np.shape(a[x])[1]
            if x == 0:
                bdbh1 = 1
                bdbh2 = 2
                bh[x] = np.zeros((bdbh1,bdbh2,d,d),dtype=complex)
                for nx in range(d):
                    for nxp in range(d):
                        bh[x][:,:,nx,nxp] = np.array([[c*(h[nxp,nxp]-h[nx,nx]),1]])
            elif x > 0 and x < N-1:
                bdbh1 = 2
                bdbh2 = 2
                bh[x] = np.zeros((bdbh1,bdbh2,d,d),dtype=complex)
                for nx in range(d):
                    for nxp in range(d):
                        bh[x][:,:,nx,nxp] = np.array([[1,0],[c*(h[nxp,nxp]-h[nx,nx]),1]])
            elif x == N-1:
                bdbh1 = 2
                bdbh2 = 1
                bh[x] = np.zeros((bdbh1,bdbh2,d,d),dtype=complex)
                for nx in range(d):
                    for nxp in range(d):
                        bh[x][:,:,nx,nxp] = np.array([[1],[c*(h[nxp,nxp]-h[nx,nx])]])
            if da == d:
            # a is operator
                b[x] = np.zeros((bdbh1*bda1,bdbh2*bda2,d,d),dtype=complex)
                for nx in range(d):
                    for nxp in range(d):
                        b[x][:,:,nx,nxp] = np.kron(bh[x][:,:,nx,nxp],a[x][:,:,nx,nxp])
            elif da == d**2:
            # a is superoperator (vectorized channel)    
                bh[x] = np.reshape(bh[x],(bdbh1,bdbh2,d**2),order='F')
                b[x] = np.zeros((bdbh1*bda1,bdbh2*bda2,d**2,d**2),dtype=complex)
                for nx in range(d**2):
                    for nxp in range(d**2):
                        b[x][:,:,nx,nxp] = np.kron(bh[x][:,:,nx],a[x][:,:,nx,nxp])
    elif BC == 'P':
        da = np.shape(a)[2]
        bda = np.shape(a)[0]
        if N == 1:
            bdbh = 1
        else:
            bdbh = 2
        bh = np.zeros((bdbh,bdbh,d,d,N),dtype=complex)
        for nx in range(d):
            for nxp in range(d):
                if N == 1:
                    bh[:,:,nx,nxp,0] = c*(h[nxp,nxp]-h[nx,nx])
                else:
                    bh[:,:,nx,nxp,0] = np.array([[c*(h[nxp,nxp]-h[nx,nx]),1],[0,0]])
                    for x in range(1,N-1):
                        bh[:,:,nx,nxp,x] = np.array([[1,0],[c*(h[nxp,nxp]-h[nx,nx]),1]])
                    bh[:,:,nx,nxp,N-1] = np.array([[1,0],[c*(h[nxp,nxp]-h[nx,nx]),0]])
        if da == d:
        # a is operator
            b = np.zeros((bdbh*bda,bdbh*bda,d,d,N),dtype=complex)
            for nx in range(d):
                for nxp in range(d):
                    for x in range(N):
                        b[:,:,nx,nxp,x] = np.kron(bh[:,:,nx,nxp,x],a[:,:,nx,nxp,x])
        elif da == d**2:
        # a is superoperator (vectorized channel)    
            bh = np.reshape(bh,(bdbh,bdbh,d**2,N),order='F')
            b = np.zeros((bdbh*bda,bdbh*bda,d**2,d**2,N),dtype=complex)
            for nx in range(d**2):
                for nxp in range(d**2):
                    for x in range(N):
                        b[:,:,nx,nxp,x] = np.kron(bh[:,:,nx,x],a[:,:,nx,nxp,x])
    return b


def fin_enlarge_bdl(cold,factor):
    """
    Enlarge bond dimension of SLD MPO. Function for finite size systems.
    
    Parameters:
      cold: SLD MPO, expected list of length n of ndarrays of a shape (bd,bd,d,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,d,n) for PBC
      factor: factor which determine on average relation between old and newly added values of SLD MPO
    
    Returns:
      c: SLD MPO with bd += 1
    """
    rng = np.random.default_rng()
    if type(cold) is list:
        n = len(cold)
        if n == 1:
            warnings.warn('Tensor networks with OBC and length one have to have bond dimension equal to one.')
        else:
            c = [0]*n
            x = 0
            d = np.shape(cold[x])[2]
            bdl1 = 1
            bdl2 = np.shape(cold[x])[1]+1
            c[x] = np.zeros((bdl1,bdl2,d,d),dtype=complex)
            for nx in range(d):
                for nxp in range(d):
                    meanrecold = np.sum(np.abs(np.real(cold[x][:,:,nx,nxp])))/(bdl2-1)
                    meanimcold = np.sum(np.abs(np.imag(cold[x][:,:,nx,nxp])))/(bdl2-1)
                    c[x][:,:,nx,nxp] = (meanrecold*rng.random((bdl1,bdl2))+1j*meanimcold*rng.random((bdl1,bdl2)))*factor
            c[x] = (c[x] + np.conj(np.moveaxis(c[x],2,3)))/2
            c[x][0:bdl1-1,0:bdl2-1,:,:] = cold[x]
            for x in range(1,n-1):
                d = np.shape(cold[x])[2]
                bdl1 = np.shape(cold[x])[0]+1
                bdl2 = np.shape(cold[x])[1]+1
                c[x] = np.zeros((bdl1,bdl2,d,d),dtype=complex)
                for nx in range(d):
                    for nxp in range(d):
                        meanrecold = np.sum(np.abs(np.real(cold[x][:,:,nx,nxp])))/((bdl1-1)*(bdl2-1))
                        meanimcold = np.sum(np.abs(np.imag(cold[x][:,:,nx,nxp])))/((bdl1-1)*(bdl2-1))
                        c[x][:,:,nx,nxp] = (meanrecold*rng.random((bdl1,bdl2))+1j*meanimcold*rng.random((bdl1,bdl2)))*factor
                c[x] = (c[x] + np.conj(np.moveaxis(c[x],2,3)))/2
                c[x][0:bdl1-1,0:bdl2-1,:,:] = cold[x]
            x = n-1
            d = np.shape(cold[x])[2]
            bdl1 = np.shape(cold[x])[0]+1
            bdl2 = 1
            c[x] = np.zeros((bdl1,bdl2,d,d),dtype=complex)
            for nx in range(d):
                for nxp in range(d):
                    meanrecold = np.sum(np.abs(np.real(cold[x][:,:,nx,nxp])))/(bdl1-1)
                    meanimcold = np.sum(np.abs(np.imag(cold[x][:,:,nx,nxp])))/(bdl1-1)
                    c[x][:,:,nx,nxp] = (meanrecold*rng.random((bdl1,bdl2))+1j*meanimcold*rng.random((bdl1,bdl2)))*factor
            c[x] = (c[x] + np.conj(np.moveaxis(c[x],2,3)))/2
            c[x][0:bdl1-1,0:bdl2-1,:,:] = cold[x]
    elif type(cold) is np.ndarray:
        n = np.shape(cold)[4]
        d = np.shape(cold)[2]
        bdl = np.shape(cold)[0]+1
        c = np.zeros((bdl,bdl,d,d,n),dtype=complex)
        for nx in range(d):
            for nxp in range(d):
                for x in range(n):
                    meanrecold = np.sum(np.abs(np.real(cold[:,:,nx,nxp,x])))/(bdl-1)**2
                    meanimcold = np.sum(np.abs(np.imag(cold[:,:,nx,nxp,x])))/(bdl-1)**2
                    c[:,:,nx,nxp,x] = (meanrecold*rng.random((bdl,bdl))+1j*meanimcold*rng.random((bdl,bdl)))*factor
        c = (c + np.conj(np.moveaxis(c,2,3)))/2
        c[0:bdl-1,0:bdl-1,:,:,:] = cold
    return c


def fin_enlarge_bdpsi(a0old,factor):
    """
    Enlarge bond dimension of wave function MPS. Function for finite size systems.
    
    Parameters:
      a0old: wave function MPS, expected list of length n of ndarrays of a shape (bd,bd,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,n) for PBC
      ratio: factor which determine on average relation between last and next to last values of diagonals of wave function MPS
    
    Returns:
      a0: wave function MPS with bd += 1
    """
    rng = np.random.default_rng()
    if type(a0old) is list:
        n = len(a0old)
        if n == 1:
            warnings.warn('Tensor networks with OBC and length one have to have bond dimension equal to one.')
        else:
            a0 = [0]*n
            x = 0
            d = np.shape(a0old[x])[2]
            bdpsi1 = 1
            bdpsi2 = np.shape(a0old[x])[1]+1
            a0[x] = np.zeros((bdpsi1,bdpsi2,d),dtype=complex)
            for nx in range(d):
                meanrea0old = np.sum(np.abs(np.real(a0old[x][:,:,nx])))/(bdpsi2-1)
                meanima0old = np.sum(np.abs(np.imag(a0old[x][:,:,nx])))/(bdpsi2-1)
                a0[x][:,:,nx] = (meanrea0old*rng.random((bdpsi1,bdpsi2))+1j*meanima0old*rng.random((bdpsi1,bdpsi2)))*factor
            a0[x][0:bdpsi1-1,0:bdpsi2-1,:] = a0old[x]
            for x in range(1,n-1):
                d = np.shape(a0old[x])[2]
                bdpsi1 = np.shape(a0old[x])[0]+1
                bdpsi2 = np.shape(a0old[x])[1]+1
                a0[x] = np.zeros((bdpsi1,bdpsi2,d),dtype=complex)
                for nx in range(d):
                    meanrea0old = np.sum(np.abs(np.real(a0old[x][:,:,nx])))/((bdpsi1-1)*(bdpsi2-1))
                    meanima0old = np.sum(np.abs(np.imag(a0old[x][:,:,nx])))/((bdpsi1-1)*(bdpsi2-1))
                    a0[x][:,:,nx] = (meanrea0old*rng.random((bdpsi1,bdpsi2))+1j*meanima0old*rng.random((bdpsi1,bdpsi2)))*factor
                a0[x][0:bdpsi1-1,0:bdpsi2-1,:] = a0old[x]
            x = n-1
            d = np.shape(a0old[x])[2]
            bdpsi1 = np.shape(a0old[x])[0]+1
            bdpsi2 = 1
            a0[x] = np.zeros((bdpsi1,bdpsi2,d),dtype=complex)
            for nx in range(d):
                meanrea0old = np.sum(np.abs(np.real(a0old[x][:,:,nx])))/(bdpsi1-1)
                meanima0old = np.sum(np.abs(np.imag(a0old[x][:,:,nx])))/(bdpsi1-1)
                a0[x][:,:,nx] = (meanrea0old*rng.random((bdpsi1,bdpsi2))+1j*meanima0old*rng.random((bdpsi1,bdpsi2)))*factor
            a0[x][0:bdpsi1-1,0:bdpsi2-1,:] = a0old[x]
            
            tensors = [np.conj(a0[n-1]),a0[n-1]]
            legs = [[-1,-3,1],[-2,-4,1]]
            r1 = ncon(tensors,legs)
            a0[n-1] = a0[n-1]/np.sqrt(np.linalg.norm(np.reshape(r1,-1,order='F')))
            tensors = [np.conj(a0[n-1]),a0[n-1]]
            legs = [[-1,-3,1],[-2,-4,1]]
            r2 = ncon(tensors,legs)
            for x in range(n-2,0,-1):
                tensors = [np.conj(a0[x]),a0[x],r2]
                legs = [[-1,2,1],[-2,3,1],[2,3,-3,-4]]
                r1 = ncon(tensors,legs)
                a0[x] = a0[x]/np.sqrt(np.linalg.norm(np.reshape(r1,-1,order='F')))
                tensors = [np.conj(a0[x]),a0[x],r2]
                legs = [[-1,2,1],[-2,3,1],[2,3,-3,-4]]
                r2 = ncon(tensors,legs)
            tensors = [np.conj(a0[0]),a0[0],r2]
            legs = [[4,2,1],[5,3,1],[2,3,4,5]]
            r1 = ncon(tensors,legs)
            a0[0] = a0[0]/np.sqrt(np.abs(r1))
    elif type(a0old) is np.ndarray:
        n = np.shape(a0old)[3]
        d = np.shape(a0old)[2]
        bdpsi = np.shape(a0old)[0]+1
        a0 = np.zeros((bdpsi,bdpsi,d,n),dtype=complex)
        for nx in range(d):
            for x in range(n):
                meanrea0old = np.sum(np.abs(np.real(a0old[:,:,nx,x])))/(bdpsi-1)**2
                meanima0old = np.sum(np.abs(np.imag(a0old[:,:,nx,x])))/(bdpsi-1)**2
                a0[:,:,nx,x] = (meanrea0old*rng.random((bdpsi,bdpsi))+1j*meanima0old*rng.random((bdpsi,bdpsi)))*factor
        a0[0:bdpsi-1,0:bdpsi-1,:,:] = a0old
        
        if n == 1:
            tensors = [np.conj(a0[:,:,:,0]),a0[:,:,:,0]]
            legs = [[2,2,1],[3,3,1]]
            r1 = ncon(tensors,legs)
            a0[:,:,:,0] = a0[:,:,:,0]/np.sqrt(np.abs(r1))
        else:
            tensors = [np.conj(a0[:,:,:,n-1]),a0[:,:,:,n-1]]
            legs = [[-1,-3,1],[-2,-4,1]]
            r1 = ncon(tensors,legs)
            a0[:,:,:,n-1] = a0[:,:,:,n-1]/np.sqrt(np.linalg.norm(np.reshape(r1,-1,order='F')))
            tensors = [np.conj(a0[:,:,:,n-1]),a0[:,:,:,n-1]]
            legs = [[-1,-3,1],[-2,-4,1]]
            r2 = ncon(tensors,legs)
            for x in range(n-2,0,-1):
                tensors = [np.conj(a0[:,:,:,x]),a0[:,:,:,x],r2]
                legs = [[-1,2,1],[-2,3,1],[2,3,-3,-4]]
                r1 = ncon(tensors,legs)
                a0[:,:,:,x] = a0[:,:,:,x]/np.sqrt(np.linalg.norm(np.reshape(r1,-1,order='F')))
                tensors = [np.conj(a0[:,:,:,x]),a0[:,:,:,x],r2]
                legs = [[-1,2,1],[-2,3,1],[2,3,-3,-4]]
                r2 = ncon(tensors,legs)
            tensors = [np.conj(a0[:,:,:,0]),a0[:,:,:,0],r2]
            legs = [[4,2,1],[5,3,1],[2,3,4,5]]
            r1 = ncon(tensors,legs)
            a0[:,:,:,0] = a0[:,:,:,0]/np.sqrt(np.abs(r1))
    return a0


#########################################
# 1.2.1 Problems with exact derivative. #
#########################################


def fin_FoM_FoMD_optbd(n,d,bc,ch,chp,cini=None,a0ini=None,imprecision=10**-2,bdlmax=100,alwaysbdlmax=False,lherm=True,bdpsimax=100,alwaysbdpsimax=False):
    """
    Iterative optimization of FoM/FoMD over SLD MPO and initial wave function MPS and also check of convergence in bond dimensions. Function for finite size systems.
    
    Parameters:
      n: number of sites in TN
      d: dimension of local Hilbert space (dimension of physical index)
      bc: boundary conditions, 'O' for OBC, 'P' for PBC
      ch: MPO for quantum channel, expected list of length n of ndarrays of a shape (bd,bd,d**2,d**2) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d**2,d**2,n) for PBC
      chp: MPO for generalized derivative of quantum channel, expected list of length n of ndarrays of a shape (bd,bd,d**2,d**2) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d**2,d**2,n) for PBC
      cini: initial MPO for SLD, expected list of length n of ndarrays of a shape (bd,bd,d,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,d,n) for PBC
      a0ini: initial MPS for initial wave function, expected list of length n of ndarrays of a shape (bd,bd,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,n) for PBC
      imprecision: expected imprecision of the end results, default value is 10**-2
      bdlmax: maximal value of bd for SLD MPO, default value is 100
      alwaysbdlmax: boolean value, True if maximal value of bd for SLD MPO have to be reached, otherwise False (default value)
      lherm: boolean value, True (default value) when Hermitian gauge is imposed on SLD MPO, otherwise False
      bdpsimax: maximal value of bd for initial wave function MPS, default value is 100
      alwaysbdpsimax: boolean value, True if maximal value of bd for initial wave function MPS have to be reached, otherwise False (default value)
    
    Returns:
      result: optimal value of FoM/FoMD
      resultm: matrix describing FoM/FoMD in function of bd of respectively SLD MPO [rows] and initial wave function MPS [columns]
      c: optimal MPO for SLD
      a0: optimal MPS for initial wave function
    """
    while True:
        if a0ini is None:
            bdpsi = 1
            a0 = np.zeros(d,dtype=complex)
            for i in range(d):
                a0[i] = np.sqrt(math.comb(d-1,i))*2**(-(d-1)/2) # prod
                # a0[i] = np.sqrt(2/(d+1))*np.sin((1+i)*np.pi/(d+1)) # sine
            if bc == 'O':
                a0 = a0[np.newaxis,np.newaxis,:]
                a0 = [a0]*n
            elif bc == 'P':
                a0 = a0[np.newaxis,np.newaxis,:,np.newaxis]
                a0 = np.tile(a0,(1,1,1,n))
        else:
            a0 = a0ini
            if bc == 'O':
                bdpsi = max([np.shape(a0[i])[0] for i in range(n)])
                a0 = [a0[i].astype(complex) for i in range(n)]
            elif bc == 'P':
                bdpsi = np.shape(a0)[0]
                a0 = a0.astype(complex)
        if cini is None:
            bdl = 1
            rng = np.random.default_rng()
            if bc == 'O':
                c = [0]*n
                c[0] = (rng.random((1,bdl,d,d)) + 1j*rng.random((1,bdl,d,d)))/bdl
                c[0] = (c[0] + np.conj(np.moveaxis(c[0],2,3)))/2
                for x in range(1,n-1):
                    c[x] = (rng.random((bdl,bdl,d,d)) + 1j*rng.random((bdl,bdl,d,d)))/bdl
                    c[x] = (c[x] + np.conj(np.moveaxis(c[x],2,3)))/2
                c[n-1] = (rng.random((bdl,1,d,d)) + 1j*rng.random((bdl,1,d,d)))/bdl
                c[n-1] = (c[n-1] + np.conj(np.moveaxis(c[n-1],2,3)))/2
            elif  bc == 'P':
                c = (rng.random((bdl,bdl,d,d,n)) + 1j*rng.random((bdl,bdl,d,d,n)))/bdl
                c = (c + np.conj(np.moveaxis(c,2,3)))/2
        else:
            c = cini
            if bc == 'O':
                bdl = max([np.shape(c[i])[0] for i in range(n)])
                c = [c[i].astype(complex) for i in range(n)]
            elif bc == 'P':
                bdl = np.shape(c)[0]
                c = c.astype(complex)
        resultm = np.zeros((bdlmax,bdpsimax),dtype=float)
        resultm[bdl-1,bdpsi-1],c,a0 = fin_FoM_FoMD_optm(n,d,bc,c,a0,ch,chp,imprecision,lherm)
        if bc == 'O' and n == 1:
            resultm = resultm[0:bdl,0:bdpsi]
            result = resultm[bdl-1,bdpsi-1]
            return result,resultm,c,a0
        factorv = np.array([0.5,0.25,0.1,1,0.01])
        problem = False
        while True:
            while True:
                if bdpsi == bdpsimax:
                    break
                else:
                    a0old = a0
                    bdpsi += 1
                    i = 0
                    while True:
                        a0 = fin_enlarge_bdpsi(a0,factorv[i])
                        resultm[bdl-1,bdpsi-1],cnew,a0new = fin_FoM_FoMD_optm(n,d,bc,c,a0,ch,chp,imprecision,lherm)
                        if resultm[bdl-1,bdpsi-1] >= resultm[bdl-1,bdpsi-2]:
                            break
                        i += 1
                        if i == np.size(factorv):
                            problem = True
                            break
                    if problem:
                        break
                    if not(alwaysbdpsimax) and resultm[bdl-1,bdpsi-1] < (1+imprecision)*resultm[bdl-1,bdpsi-2]:
                        bdpsi += -1
                        a0 = a0old
                        a0copy = a0new
                        ccopy = cnew
                        break
                    else:
                        a0 = a0new
                        c = cnew
            if problem:
                break
            if bdl == bdlmax:
                if bdpsi == bdpsimax:
                    resultm = resultm[0:bdl,0:bdpsi]
                    result = resultm[bdl-1,bdpsi-1]
                else:
                    a0 = a0copy
                    c = ccopy
                    resultm = resultm[0:bdl,0:bdpsi+1]
                    result = resultm[bdl-1,bdpsi]
                break
            else:
                bdl += 1
                i = 0
                while True:
                    c = fin_enlarge_bdl(c,factorv[i])
                    resultm[bdl-1,bdpsi-1],cnew,a0new = fin_FoM_FoMD_optm(n,d,bc,c,a0,ch,chp,imprecision,lherm)
                    if resultm[bdl-1,bdpsi-1] >= resultm[bdl-2,bdpsi-1]:
                        a0 = a0new
                        c = cnew
                        break
                    i += 1
                    if i == np.size(factorv):
                        problem = True
                        break
                if problem:
                    break
                if not(alwaysbdlmax) and resultm[bdl-1,bdpsi-1] < (1+imprecision)*resultm[bdl-2,bdpsi-1]:
                    if bdpsi == bdpsimax:
                        resultm = resultm[0:bdl,0:bdpsi]
                        result = resultm[bdl-1,bdpsi-1]
                    else:
                        if resultm[bdl-1,bdpsi-1] < resultm[bdl-2,bdpsi]:
                            a0 = a0copy
                            c = ccopy
                            resultm = resultm[0:bdl,0:bdpsi+1]
                            bdl += -1
                            bdpsi += 1
                            result = resultm[bdl-1,bdpsi-1]
                        else:
                            resultm = resultm[0:bdl,0:bdpsi+1]
                            result = resultm[bdl-1,bdpsi-1]
                    break
        if not(problem):
            break
    return result,resultm,c,a0


def fin_FoM_optbd(n,d,bc,a,b,cini=None,imprecision=10**-2,bdlmax=100,alwaysbdlmax=False,lherm=True):
    """
    Optimization of FoM over SLD MPO and also check of convergence in bond dimension. Function for finite size systems.
    
    Parameters:
      n: number of sites in TN
      d: dimension of local Hilbert space (dimension of physical index)
      bc: boundary conditions, 'O' for OBC, 'P' for PBC
      a: MPO for density matrix, expected list of length n of ndarrays of a shape (bd,bd,d,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,d,n) for PBC
      b: MPO for generalized derivative of density matrix, expected list of length n of ndarrays of a shape (bd,bd,d,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,d,n) for PBC
      cini: initial MPO for SLD, expected list of length n of ndarrays of a shape (bd,bd,d,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,d,n) for PBC
      imprecision: expected imprecision of the end results, default value is 10**-2
      bdlmax: maximal value of bd for SLD MPO, default value is 100
      alwaysbdlmax: boolean value, True if maximal value of bd for SLD MPO have to be reached, otherwise False (default value)
      lherm: boolean value, True (default value) when Hermitian gauge is imposed on SLD MPO, otherwise False
    
    Returns:
      result: optimal value of FoM
      resultv: vector describing FoM in function of bd of SLD MPO
      c: optimal MPO for SLD
    """
    while True:
        if cini is None:
            bdl = 1
            rng = np.random.default_rng()
            if bc == 'O':
                c = [0]*n
                c[0] = (rng.random((1,bdl,d,d)) + 1j*rng.random((1,bdl,d,d)))/bdl
                c[0] = (c[0] + np.conj(np.moveaxis(c[0],2,3)))/2
                for x in range(1,n-1):
                    c[x] = (rng.random((bdl,bdl,d,d)) + 1j*rng.random((bdl,bdl,d,d)))/bdl
                    c[x] = (c[x] + np.conj(np.moveaxis(c[x],2,3)))/2
                c[n-1] = (rng.random((bdl,1,d,d)) + 1j*rng.random((bdl,1,d,d)))/bdl
                c[n-1] = (c[n-1] + np.conj(np.moveaxis(c[n-1],2,3)))/2
            elif  bc == 'P':
                c = (rng.random((bdl,bdl,d,d,n)) + 1j*rng.random((bdl,bdl,d,d,n)))/bdl
                c = (c + np.conj(np.moveaxis(c,2,3)))/2
        else:
            c = cini
            if bc == 'O':
                bdl = max([np.shape(c[i])[0] for i in range(n)])
                c = [c[i].astype(complex) for i in range(n)]
            elif bc == 'P':
                bdl = np.shape(c)[0]
                c = c.astype(complex)
        resultv = np.zeros(bdlmax,dtype=float)
        if bc == 'O':
            resultv[bdl-1],c = fin_FoM_OBC_optm(a,b,c,imprecision,lherm)
            if n == 1:
                resultv = resultv[0:bdl]
                result = resultv[bdl-1]
                return result,resultv,c
        elif  bc == 'P':
            resultv[bdl-1],c = fin_FoM_PBC_optm(a,b,c,imprecision,lherm)
        factorv = np.array([0.5,0.25,0.1,1,0.01])
        problem = False
        while True:
            if bdl == bdlmax:
                resultv = resultv[0:bdl]
                result = resultv[bdl-1]
                break
            else:
                bdl += 1
                i = 0
                while True:
                    c = fin_enlarge_bdl(c,factorv[i])
                    if bc == 'O':
                        resultv[bdl-1],cnew = fin_FoM_OBC_optm(a,b,c,imprecision,lherm)
                    elif  bc == 'P':
                        resultv[bdl-1],cnew = fin_FoM_PBC_optm(a,b,c,imprecision,lherm)
                    if resultv[bdl-1] >= resultv[bdl-2]:
                        c = cnew
                        break
                    i += 1
                    if i == np.size(factorv):
                        problem = True
                        break
                if problem:
                    break
                if not(alwaysbdlmax) and resultv[bdl-1] < (1+imprecision)*resultv[bdl-2]:
                    resultv = resultv[0:bdl]
                    result = resultv[bdl-1]
                    break
        if not(problem):
            break
    return result,resultv,c


def fin_FoMD_optbd(n,d,bc,c2d,cpd,a0ini=None,imprecision=10**-2,bdpsimax=100,alwaysbdpsimax=False):
    """
    Optimization of FoMD over initial wave function MPS and also check of convergence in bond dimension. Function for finite size systems.
    
    Parameters:
      n: number of sites in TN
      d: dimension of local Hilbert space (dimension of physical index)
      bc: boundary conditions, 'O' for OBC, 'P' for PBC
      c2d: MPO for square of dual of SLD, expected list of length n of ndarrays of a shape (bd,bd,d,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,d,n) for PBC
      cpd: MPO for dual of generalized derivative of SLD, expected list of length n of ndarrays of a shape (bd,bd,d,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,d,n) for PBC
      a0ini: initial MPS for initial wave function, expected list of length n of ndarrays of a shape (bd,bd,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,n) for PBC
      imprecision: expected imprecision of the end results, default value is 10**-2
      bdpsimax: maximal value of bd for initial wave function MPS, default value is 100
      alwaysbdpsimax: boolean value, True if maximal value of bd for initial wave function MPS have to be reached, otherwise False (default value)
    
    Returns:
      result: optimal value of FoMD
      resultv: vector describing FoMD in function of bd of initial wave function MPS
      a0: optimal MPS for initial wave function
    """
    while True:
        if a0ini is None:
            bdpsi = 1
            a0 = np.zeros(d,dtype=complex)
            for i in range(d):
                a0[i] = np.sqrt(math.comb(d-1,i))*2**(-(d-1)/2) # prod
                # a0[i] = np.sqrt(2/(d+1))*np.sin((1+i)*np.pi/(d+1)) # sine
            if bc == 'O':
                a0 = a0[np.newaxis,np.newaxis,:]
                a0 = [a0]*n
            elif bc == 'P':
                a0 = a0[np.newaxis,np.newaxis,:,np.newaxis]
                a0 = np.tile(a0,(1,1,1,n))
        else:
            a0 = a0ini
            if bc == 'O':
                bdpsi = max([np.shape(a0[i])[0] for i in range(n)])
                a0 = [a0[i].astype(complex) for i in range(n)]
            elif bc == 'P':
                bdpsi = np.shape(a0)[0]
                a0 = a0.astype(complex)
        resultv = np.zeros(bdpsimax,dtype=float)
        if bc == 'O':
            resultv[bdpsi-1],a0 = fin_FoMD_OBC_optm(c2d,cpd,a0,imprecision)
            if n == 1:
                resultv = resultv[0:bdpsi]
                result = resultv[bdpsi-1]
                return result,resultv,a0
        elif  bc == 'P':
            resultv[bdpsi-1],a0 = fin_FoMD_PBC_optm(c2d,cpd,a0,imprecision)
        factorv = np.array([0.5,0.25,0.1,1,0.01])
        problem = False
        while True:
            if bdpsi == bdpsimax:
                resultv = resultv[0:bdpsi]
                result = resultv[bdpsi-1]
                break
            else:
                bdpsi += 1
                i = 0
                while True:
                    a0 = fin_enlarge_bdpsi(a0,factorv[i])
                    if bc == 'O':
                        resultv[bdpsi-1],a0new = fin_FoMD_OBC_optm(c2d,cpd,a0,imprecision)
                    elif  bc == 'P':
                        resultv[bdpsi-1],a0new = fin_FoMD_PBC_optm(c2d,cpd,a0,imprecision)
                    if resultv[bdpsi-1] >= resultv[bdpsi-2]:
                        a0 = a0new
                        break
                    i += 1
                    if i == np.size(factorv):
                        problem = True
                        break
                if problem:
                    break
                if not(alwaysbdpsimax) and resultv[bdpsi-1] < (1+imprecision)*resultv[bdpsi-2]:
                    resultv = resultv[0:bdpsi]
                    result = resultv[bdpsi-1]
                    break
        if not(problem):
            break
    return result,resultv,a0


def fin_FoM_FoMD_optm(n,d,bc,c,a0,ch,chp,imprecision=10**-2,lherm=True):
    """
    Iterative optimization of FoM/FoMD over SLD MPO and initial wave function MPS. Function for finite size systems.
    
    Parameters:
      n: number of sites in TN
      d: dimension of local Hilbert space (dimension of physical index)
      bc: boundary conditions, 'O' for OBC, 'P' for PBC
      c: MPO for SLD, expected list of length n of ndarrays of a shape (bd,bd,d,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,d,n) for PBC
      a0: MPS for initial wave function, expected list of length n of ndarrays of a shape (bd,bd,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,n) for PBC
      ch: MPO for quantum channel, expected list of length n of ndarrays of a shape (bd,bd,d**2,d**2) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d**2,d**2,n) for PBC
      chp: MPO for generalized derivative of quantum channel, expected list of length n of ndarrays of a shape (bd,bd,d**2,d**2) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d**2,d**2,n) for PBC
      imprecision: expected imprecision of the end results, default value is 10**-2
      lherm: boolean value, True (default value) when Hermitian gauge is imposed on SLD MPO, otherwise False
    
    Returns:
      fval: optimal value of FoM/FoMD
      c: optimal MPO for SLD
      a0: optimal MPS for initial wave function
    """
    relunc_f = 0.1*imprecision
    if bc == 'O':
        chd = [0]*n
        chpd = [0]*n
        for x in range(n):
            chd[x] = np.conj(np.moveaxis(ch[x],2,3))
            chpd[x] = np.conj(np.moveaxis(chp[x],2,3))
    elif bc == 'P':
        chd = np.conj(np.moveaxis(ch,2,3))
        chpd = np.conj(np.moveaxis(chp,2,3))
    f = np.array([])
    iter_f = 0
    while True:
        a0_dm = wave_function_to_density_matrix(a0)
        a = channel_acting_on_operator(ch,a0_dm)
        b = channel_acting_on_operator(chp,a0_dm)
        if bc == 'O':
            fom,c = fin_FoM_OBC_optm(a,b,c,imprecision,lherm)
        elif bc == 'P':
            fom,c = fin_FoM_PBC_optm(a,b,c,imprecision,lherm)
        f = np.append(f,fom)
        if iter_f >= 2 and np.std(f[-4:])/np.mean(f[-4:]) <= relunc_f:
            break
        if bc == 'O':
            c2 = [0]*n
            for x in range(n):
                bdl1 = np.shape(c[x])[0]
                bdl2 = np.shape(c[x])[1]
                c2[x] = np.zeros((bdl1**2,bdl2**2,d,d),dtype=complex)
                for nx in range(d):
                    for nxp in range(d):
                        for nxpp in range(d):
                            c2[x][:,:,nx,nxp] = c2[x][:,:,nx,nxp]+np.kron(c[x][:,:,nx,nxpp],c[x][:,:,nxpp,nxp])
        elif bc == 'P':
            bdl = np.shape(c)[0]
            c2 = np.zeros((bdl**2,bdl**2,d,d,n),dtype=complex)
            for nx in range(d):
                for nxp in range(d):
                    for nxpp in range(d):
                        for x in range(n):
                            c2[:,:,nx,nxp,x] = c2[:,:,nx,nxp,x]+np.kron(c[:,:,nx,nxpp,x],c[:,:,nxpp,nxp,x])
        c2d = channel_acting_on_operator(chd,c2)
        cpd = channel_acting_on_operator(chpd,c)
        if bc == 'O':
            fomd,a0 = fin_FoMD_OBC_optm(c2d,cpd,a0,imprecision)
        elif bc == 'P':
            fomd,a0 = fin_FoMD_PBC_optm(c2d,cpd,a0,imprecision)
        f = np.append(f,fomd)
        iter_f += 1
    fval = f[-1]
    return fval,c,a0


def fin_FoM_OBC_optm(a,b,c,imprecision=10**-2,lherm=True):
    """
    Optimization of FoM over MPO for SLD. Function for finite size systems with OBC.
    
    Parameters:
      a: MPO for density matrix, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
      b: MPO for generalized derivative of density matrix, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
      c: MPO for SLD, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
      imprecision: expected imprecision of the end results, default value is 10**-2
      lherm: boolean value, True (default value) when Hermitian gauge is imposed on SLD MPO, otherwise False
    
    Returns:
      fomval: optimal value of FoM
      c: optimal MPO for SLD
    """
    n = len(c)
    tol_fom = 0.1*imprecision/n**2
    if n == 1:
        if np.shape(a[0])[0] == 1 and np.shape(b[0])[0] == 1 and np.shape(c[0])[0] == 1:
            d = np.shape(c[0])[2]
            tensors = [b[0][0,0,:,:]]
            legs = [[-2,-1]]
            l1 = ncon(tensors,legs)
            l1 = np.reshape(l1,-1,order='F')
            tensors = [a[0][0,0,:,:],np.eye(d)]
            legs = [[-2,-3],[-4,-1]]
            l2 = ncon(tensors,legs)
            l2 = np.reshape(l2,(d*d,d*d),order='F')
            dl2 = l2+l2.T
            dl1 = 2*l1
            dl2pinv = np.linalg.pinv(dl2,tol_fom)
            dl2pinv = (dl2pinv+dl2pinv.T)/2
            cv = dl2pinv @ dl1
            c[0][0,0,:,:] = np.reshape(cv,(d,d),order='F')
            if lherm:
                c[0] = (c[0]+np.conj(np.moveaxis(c[0],2,3)))/2
                cv = np.reshape(c[0],-1,order='F')
            fomval = np.real(2*cv @ l1 - cv @ l2 @ cv)
        else:
            warnings.warn('Tensor networks with OBC and length one have to have bond dimension equal to one.')
    else:
        relunc_fom = 0.1*imprecision
        l1f = [0]*n
        l2f = [0]*n
        fom = np.array([])
        iter_fom = 0
        while True:
            tensors = [c[n-1],b[n-1]]
            legs = [[-1,-3,1,2],[-2,-4,2,1]]
            l1f[n-2] = ncon(tensors,legs)
            l1f[n-2] = l1f[n-2][:,:,0,0]
            tensors = [c[n-1],a[n-1],c[n-1]]
            legs = [[-1,-4,1,2],[-2,-5,2,3],[-3,-6,3,1]]
            l2f[n-2] = ncon(tensors,legs)
            l2f[n-2] = l2f[n-2][:,:,:,0,0,0]
            for x in range(n-2,0,-1):
                tensors = [c[x],b[x],l1f[x]]
                legs = [[-1,3,1,2],[-2,4,2,1],[3,4]]
                l1f[x-1] = ncon(tensors,legs)
                tensors = [c[x],a[x],c[x],l2f[x]]
                legs = [[-1,4,1,2],[-2,5,2,3],[-3,6,3,1],[4,5,6]]
                l2f[x-1] = ncon(tensors,legs)
            bdl1,bdl2,d,d = np.shape(c[0])
            tensors = [b[0],l1f[0]]
            legs = [[-5,1,-4,-3],[-2,1]]
            l1 = ncon(tensors,legs)
            l1 = np.reshape(l1,-1,order='F')
            tensors = [a[0],np.eye(d),l2f[0]]
            legs = [[-9,1,-4,-7],[-8,-3],[-2,1,-6]]
            l2 = ncon(tensors,legs)
            l2 = np.reshape(l2,(bdl1*bdl2*d*d,bdl1*bdl2*d*d),order='F')
            dl2 = l2+l2.T
            dl1 = 2*l1
            dl2pinv = np.linalg.pinv(dl2,tol_fom)
            dl2pinv = (dl2pinv+dl2pinv.T)/2
            cv = dl2pinv @ dl1
            c[0] = np.reshape(cv,(bdl1,bdl2,d,d),order='F')
            if lherm:
                c[0] = (c[0]+np.conj(np.moveaxis(c[0],2,3)))/2
                cv = np.reshape(c[0],-1,order='F')
            fom = np.append(fom,np.real(2*cv @ l1 - cv @ l2 @ cv))
            tensors = [c[0],b[0]]
            legs = [[-3,-1,1,2],[-4,-2,2,1]]
            l1c = ncon(tensors,legs)
            l1c = l1c[:,:,0,0]
            tensors = [c[0],a[0],c[0]]
            legs = [[-4,-1,1,2],[-5,-2,2,3],[-6,-3,3,1]]
            l2c = ncon(tensors,legs)
            l2c = l2c[:,:,:,0,0,0]
            for x in range(1,n-1):
                bdl1,bdl2,d,d = np.shape(c[x])
                tensors = [l1c,b[x],l1f[x]]
                legs = [[-1,1],[1,2,-4,-3],[-2,2]]
                l1 = ncon(tensors,legs)
                l1 = np.reshape(l1,-1,order='F')
                tensors = [l2c,a[x],np.eye(d),l2f[x]]
                legs = [[-1,1,-5],[1,2,-4,-7],[-8,-3],[-2,2,-6]]
                l2 = ncon(tensors,legs)
                l2 = np.reshape(l2,(bdl1*bdl2*d*d,bdl1*bdl2*d*d),order='F')
                dl2 = l2+l2.T
                dl1 = 2*l1
                dl2pinv = np.linalg.pinv(dl2,tol_fom)
                dl2pinv = (dl2pinv+dl2pinv.T)/2
                cv = dl2pinv @ dl1
                c[x] = np.reshape(cv,(bdl1,bdl2,d,d),order='F')
                if lherm:
                    c[x] = (c[x]+np.conj(np.moveaxis(c[x],2,3)))/2
                    cv = np.reshape(c[x],-1,order='F')
                fom = np.append(fom,np.real(2*cv @ l1 - cv @ l2 @ cv))
                tensors = [l1c,c[x],b[x]]
                legs = [[3,4],[3,-1,1,2],[4,-2,2,1]]
                l1c = ncon(tensors,legs)
                tensors = [l2c,c[x],a[x],c[x]]
                legs = [[4,5,6],[4,-1,1,2],[5,-2,2,3],[6,-3,3,1]]
                l2c = ncon(tensors,legs)
            bdl1,bdl2,d,d = np.shape(c[n-1])
            tensors = [l1c,b[n-1]]
            legs = [[-1,1],[1,-5,-4,-3]]
            l1 = ncon(tensors,legs)
            l1 = np.reshape(l1,-1,order='F')
            tensors = [l2c,a[n-1],np.eye(d)]
            legs = [[-1,1,-5],[1,-9,-4,-7],[-8,-3]]
            l2 = ncon(tensors,legs)
            l2 = np.reshape(l2,(bdl1*bdl2*d*d,bdl1*bdl2*d*d),order='F')
            dl2 = l2+l2.T
            dl1 = 2*l1
            dl2pinv = np.linalg.pinv(dl2,tol_fom)
            dl2pinv = (dl2pinv+dl2pinv.T)/2
            cv = dl2pinv @ dl1
            c[n-1] = np.reshape(cv,(bdl1,bdl2,d,d),order='F')
            if lherm:
                c[n-1] = (c[n-1]+np.conj(np.moveaxis(c[n-1],2,3)))/2
                cv = np.reshape(c[n-1],-1,order='F')
            fom = np.append(fom,np.real(2*cv @ l1 - cv @ l2 @ cv))
            iter_fom += 1
            if iter_fom >= 2 and all(fom[-2*n:] > 0) and np.std(fom[-2*n:])/np.mean(fom[-2*n:]) <= relunc_fom:
                break
        fomval = fom[-1]
    return fomval,c


def fin_FoM_PBC_optm(a,b,c,imprecision=10**-2,lherm=True):
    """
    Optimization of FoM over MPO for SLD. Function for finite size systems with PBC.
    
    Parameters:
      a: MPO for density matrix, expected ndarray of a shape (bd,bd,d,d,n)
      b: MPO for generalized derivative of density matrix, expected ndarray of a shape (bd,bd,d,d,n)
      c: MPO for SLD, expected ndarray of a shape (bd,bd,d,d,n)
      imprecision: expected imprecision of the end results, default value is 10**-2
      lherm: boolean value, True (default value) when Hermitian gauge is imposed on SLD MPO, otherwise False
    
    Returns:
      fomval: optimal value of FoM
      c: optimal MPO for SLD
    """
    n = np.shape(a)[4]
    d = np.shape(a)[2]
    bdr = np.shape(a)[0]
    bdrp = np.shape(b)[0]
    bdl = np.shape(c)[0]
    tol_fom = 0.1*imprecision/n**2
    if n == 1:
        tensors = [b[:,:,:,:,0],np.eye(bdl)]
        legs = [[1,1,-4,-3],[-2,-1]]
        l1 = ncon(tensors,legs)
        l1 = np.reshape(l1,-1,order='F')
        tensors = [a[:,:,:,:,0],np.eye(d),np.eye(bdl),np.eye(bdl)]
        legs = [[1,1,-4,-7],[-8,-3],[-2,-1],[-6,-5]]
        l2 = ncon(tensors,legs)
        l2 = np.reshape(l2,(bdl*bdl*d*d,bdl*bdl*d*d),order='F')
        dl2 = l2+l2.T
        dl1 = 2*l1
        dl2pinv = np.linalg.pinv(dl2,tol_fom)
        dl2pinv = (dl2pinv+dl2pinv.T)/2
        cv = dl2pinv @ dl1
        c[:,:,:,:,0] = np.reshape(cv,(bdl,bdl,d,d),order='F')
        if lherm:
            c[:,:,:,:,0] = (c[:,:,:,:,0]+np.conj(np.moveaxis(c[:,:,:,:,0],2,3)))/2
            cv = np.reshape(c[:,:,:,:,0],-1,order='F')
        fomval = np.real(2*cv @ l1 - cv @ l2 @ cv)
    else:
        relunc_fom = 0.1*imprecision
        l1f = np.zeros((bdl,bdrp,bdl,bdrp,n-1),dtype=complex)
        l2f = np.zeros((bdl,bdr,bdl,bdl,bdr,bdl,n-1),dtype=complex)
        fom = np.array([])
        iter_fom = 0
        while True:
            tensors = [c[:,:,:,:,n-1],b[:,:,:,:,n-1]]
            legs = [[-1,-3,1,2],[-2,-4,2,1]]
            l1f[:,:,:,:,n-2] = ncon(tensors,legs)
            tensors = [c[:,:,:,:,n-1],a[:,:,:,:,n-1],c[:,:,:,:,n-1]]
            legs = [[-1,-4,1,2],[-2,-5,2,3],[-3,-6,3,1]]
            l2f[:,:,:,:,:,:,n-2] = ncon(tensors,legs)
            for x in range(n-2,0,-1):
                tensors = [c[:,:,:,:,x],b[:,:,:,:,x],l1f[:,:,:,:,x]]
                legs = [[-1,3,1,2],[-2,4,2,1],[3,4,-3,-4]]
                l1f[:,:,:,:,x-1] = ncon(tensors,legs)
                tensors = [c[:,:,:,:,x],a[:,:,:,:,x],c[:,:,:,:,x],l2f[:,:,:,:,:,:,x]]
                legs = [[-1,4,1,2],[-2,5,2,3],[-3,6,3,1],[4,5,6,-4,-5,-6]]
                l2f[:,:,:,:,:,:,x-1] = ncon(tensors,legs)
            tensors = [b[:,:,:,:,0],l1f[:,:,:,:,0]]
            legs = [[2,1,-4,-3],[-2,1,-1,2]]
            l1 = ncon(tensors,legs)
            l1 = np.reshape(l1,-1,order='F')
            tensors = [a[:,:,:,:,0],np.eye(d),l2f[:,:,:,:,:,:,0]]
            legs = [[2,1,-4,-7],[-8,-3],[-2,1,-6,-1,2,-5]]
            l2 = ncon(tensors,legs)
            l2 = np.reshape(l2,(bdl*bdl*d*d,bdl*bdl*d*d),order='F')
            dl2 = l2+l2.T
            dl1 = 2*l1
            dl2pinv = np.linalg.pinv(dl2,tol_fom)
            dl2pinv = (dl2pinv+dl2pinv.T)/2
            cv = dl2pinv @ dl1
            c[:,:,:,:,0] = np.reshape(cv,(bdl,bdl,d,d),order='F')
            if lherm:
                c[:,:,:,:,0] = (c[:,:,:,:,0]+np.conj(np.moveaxis(c[:,:,:,:,0],2,3)))/2
                cv = np.reshape(c[:,:,:,:,0],-1,order='F')
            fom = np.append(fom,np.real(2*cv @ l1 - cv @ l2 @ cv))
            tensors = [c[:,:,:,:,0],b[:,:,:,:,0]]
            legs = [[-1,-3,1,2],[-2,-4,2,1]]
            l1c = ncon(tensors,legs)
            tensors = [c[:,:,:,:,0],a[:,:,:,:,0],c[:,:,:,:,0]]
            legs = [[-1,-4,1,2],[-2,-5,2,3],[-3,-6,3,1]]
            l2c = ncon(tensors,legs)
            for x in range(1,n-1):
                tensors = [l1c,b[:,:,:,:,x],l1f[:,:,:,:,x]]
                legs = [[3,4,-1,1],[1,2,-4,-3],[-2,2,3,4]]
                l1 = ncon(tensors,legs)
                l1 = np.reshape(l1,-1,order='F')
                tensors = [l2c,a[:,:,:,:,x],np.eye(d),l2f[:,:,:,:,:,:,x]]
                legs = [[3,4,5,-1,1,-5],[1,2,-4,-7],[-8,-3],[-2,2,-6,3,4,5]]
                l2 = ncon(tensors,legs)
                l2 = np.reshape(l2,(bdl*bdl*d*d,bdl*bdl*d*d),order='F')
                dl2 = l2+l2.T
                dl1 = 2*l1
                dl2pinv = np.linalg.pinv(dl2,tol_fom)
                dl2pinv = (dl2pinv+dl2pinv.T)/2
                cv = dl2pinv @ dl1
                c[:,:,:,:,x] = np.reshape(cv,(bdl,bdl,d,d),order='F')
                if lherm:
                    c[:,:,:,:,x] = (c[:,:,:,:,x]+np.conj(np.moveaxis(c[:,:,:,:,x],2,3)))/2
                    cv = np.reshape(c[:,:,:,:,x],-1,order='F')
                fom = np.append(fom,np.real(2*cv @ l1 - cv @ l2 @ cv))
                tensors = [l1c,c[:,:,:,:,x],b[:,:,:,:,x]]
                legs = [[-1,-2,3,4],[3,-3,1,2],[4,-4,2,1]]
                l1c = ncon(tensors,legs)
                tensors = [l2c,c[:,:,:,:,x],a[:,:,:,:,x],c[:,:,:,:,x]]
                legs = [[-1,-2,-3,4,5,6],[4,-4,1,2],[5,-5,2,3],[6,-6,3,1]]
                l2c = ncon(tensors,legs)
            tensors = [l1c,b[:,:,:,:,n-1]]
            legs = [[-2,2,-1,1],[1,2,-4,-3]]
            l1 = ncon(tensors,legs)
            l1 = np.reshape(l1,-1,order='F')
            tensors = [l2c,a[:,:,:,:,n-1],np.eye(d)]
            legs = [[-2,2,-6,-1,1,-5],[1,2,-4,-7],[-8,-3]]
            l2 = ncon(tensors,legs)
            l2 = np.reshape(l2,(bdl*bdl*d*d,bdl*bdl*d*d),order='F')
            dl2 = l2+l2.T
            dl1 = 2*l1
            dl2pinv = np.linalg.pinv(dl2,tol_fom)
            dl2pinv = (dl2pinv+dl2pinv.T)/2
            cv = dl2pinv @ dl1
            c[:,:,:,:,n-1] = np.reshape(cv,(bdl,bdl,d,d),order='F')
            if lherm:
                c[:,:,:,:,n-1] = (c[:,:,:,:,n-1]+np.conj(np.moveaxis(c[:,:,:,:,n-1],2,3)))/2
                cv = np.reshape(c[:,:,:,:,n-1],-1,order='F')
            fom = np.append(fom,np.real(2*cv @ l1 - cv @ l2 @ cv))
            iter_fom += 1
            if iter_fom >= 2 and all(fom[-2*n:] > 0) and np.std(fom[-2*n:])/np.mean(fom[-2*n:]) <= relunc_fom:
                break
        fomval = fom[-1]
    return fomval,c


def fin_FoMD_OBC_optm(c2d,cpd,a0,imprecision=10**-2):
    """
    Optimization of FoMD over MPS for initial wave function. Function for finite size systems with OBC.
    
    Parameters:
      c2d: MPO for square of dual of SLD, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
      cpd: MPO for dual of generalized derivative of SLD, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
      a0: MPS for initial wave function, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
      imprecision: expected imprecision of the end results, default value is 10**-2
    
    Returns:
      fomdval: optimal value of FoMD
      a0: optimal MPS for initial wave function
    """
    n = len(a0)
    if n == 1:
        if np.shape(c2d[0])[0] == 1 and np.shape(cpd[0])[0] == 1 and np.shape(a0[0])[0] == 1:
            d = np.shape(a0[0])[2]
            tensors = [c2d[0][0,0,:,:]]
            legs = [[-1,-2]]
            l2d = ncon(tensors,legs)
            l2d = np.reshape(l2d,(d,d),order='F')
            tensors = [cpd[0][0,0,:,:]]
            legs = [[-1,-2]]
            lpd = ncon(tensors,legs)
            lpd = np.reshape(lpd,(d,d),order='F')
            eiginput = 2*lpd-l2d
            eiginput = (eiginput+np.conj(eiginput).T)/2
            fomdval,a0v = np.linalg.eig(eiginput)
            position = np.argmax(np.real(fomdval))
            a0v = np.reshape(a0v[:,position],-1,order='F')
            a0v = a0v/np.sqrt(np.abs(np.conj(a0v) @ a0v))
            a0[0][0,0,:] = np.reshape(a0v,(d),order='F')
            fomdval = np.real(fomdval[position])
        else:
            warnings.warn('Tensor networks with OBC and length one have to have bond dimension equal to one.')
    else:
        relunc_fomd = 0.1*imprecision
        l2df = [0]*n
        lpdf = [0]*n
        fomd = np.array([])
        iter_fomd = 0
        while True:
            tensors = [np.conj(a0[n-1]),c2d[n-1],a0[n-1]]
            legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
            l2df[n-2] = ncon(tensors,legs)
            l2df[n-2] = l2df[n-2][:,:,:,0,0,0]
            tensors = [np.conj(a0[n-1]),cpd[n-1],a0[n-1]]
            legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
            lpdf[n-2] = ncon(tensors,legs)
            lpdf[n-2] = lpdf[n-2][:,:,:,0,0,0]
            for x in range(n-2,0,-1):
                tensors = [np.conj(a0[x]),c2d[x],a0[x],l2df[x]]
                legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5]]
                l2df[x-1] = ncon(tensors,legs)
                tensors = [np.conj(a0[x]),cpd[x],a0[x],lpdf[x]]
                legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5]]
                lpdf[x-1] = ncon(tensors,legs)
            bdpsi1,bdpsi2,d = np.shape(a0[0])
            tensors = [c2d[0],l2df[0]]
            legs = [[-7,1,-3,-6],[-2,1,-5]]
            l2d = ncon(tensors,legs)
            l2d = np.reshape(l2d,(bdpsi1*bdpsi2*d,bdpsi1*bdpsi2*d),order='F')
            tensors = [cpd[0],lpdf[0]]
            legs = [[-7,1,-3,-6],[-2,1,-5]]
            lpd = ncon(tensors,legs)
            lpd = np.reshape(lpd,(bdpsi1*bdpsi2*d,bdpsi1*bdpsi2*d),order='F')
            eiginput = 2*lpd-l2d
            eiginput = (eiginput+np.conj(eiginput).T)/2
            fomdval,a0v = np.linalg.eig(eiginput)
            position = np.argmax(np.real(fomdval))
            a0v = np.reshape(a0v[:,position],-1,order='F')
            a0v = a0v/np.sqrt(np.abs(np.conj(a0v) @ a0v))
            a0[0] = np.reshape(a0v,(bdpsi1,bdpsi2,d),order='F')
            fomd = np.append(fomd,np.real(fomdval[position]))
            a0[0] = np.moveaxis(a0[0],2,0)
            a0[0] = np.reshape(a0[0],(d*bdpsi1,bdpsi2),order='F')
            u,s,vh = np.linalg.svd(a0[0],full_matrices=False)
            a0[0] = np.reshape(u,(d,bdpsi1,np.shape(s)[0]),order='F')
            a0[0] = np.moveaxis(a0[0],0,2)
            tensors = [np.diag(s) @ vh,a0[1]]
            legs = [[-1,1],[1,-2,-3]]
            a0[1] = ncon(tensors,legs)
            tensors = [np.conj(a0[0]),c2d[0],a0[0]]
            legs = [[-4,-1,1],[-5,-2,1,2],[-6,-3,2]]
            l2dc = ncon(tensors,legs)
            l2dc = l2dc[:,:,:,0,0,0]
            tensors = [np.conj(a0[0]),cpd[0],a0[0]]
            legs = [[-4,-1,1],[-5,-2,1,2],[-6,-3,2]]
            lpdc = ncon(tensors,legs)
            lpdc = lpdc[:,:,:,0,0,0]
            for x in range(1,n-1):
                bdpsi1,bdpsi2,d = np.shape(a0[x])
                tensors = [l2dc,c2d[x],l2df[x]]
                legs = [[-1,1,-4],[1,2,-3,-6],[-2,2,-5]]
                l2d = ncon(tensors,legs)
                l2d = np.reshape(l2d,(bdpsi1*bdpsi2*d,bdpsi1*bdpsi2*d),order='F')
                tensors = [lpdc,cpd[x],lpdf[x]]
                legs = [[-1,1,-4],[1,2,-3,-6],[-2,2,-5]]
                lpd = ncon(tensors,legs)
                lpd = np.reshape(lpd,(bdpsi1*bdpsi2*d,bdpsi1*bdpsi2*d),order='F')
                eiginput = 2*lpd-l2d
                eiginput = (eiginput+np.conj(eiginput).T)/2
                fomdval,a0v = np.linalg.eig(eiginput)
                position = np.argmax(np.real(fomdval))
                a0v = np.reshape(a0v[:,position],-1,order='F')
                a0v = a0v/np.sqrt(np.abs(np.conj(a0v) @ a0v))
                a0[x] = np.reshape(a0v,(bdpsi1,bdpsi2,d),order='F')
                fomd = np.append(fomd,np.real(fomdval[position]))
                a0[x] = np.moveaxis(a0[x],2,0)
                a0[x] = np.reshape(a0[x],(d*bdpsi1,bdpsi2),order='F')
                u,s,vh = np.linalg.svd(a0[x],full_matrices=False)
                a0[x] = np.reshape(u,(d,bdpsi1,np.shape(s)[0]),order='F')
                a0[x] = np.moveaxis(a0[x],0,2)
                tensors = [np.diag(s) @ vh,a0[x+1]]
                legs = [[-1,1],[1,-2,-3]]
                a0[x+1] = ncon(tensors,legs)
                tensors = [l2dc,np.conj(a0[x]),c2d[x],a0[x]]
                legs = [[3,4,5],[3,-1,1],[4,-2,1,2],[5,-3,2]]
                l2dc = ncon(tensors,legs)
                tensors = [lpdc,np.conj(a0[x]),cpd[x],a0[x]]
                legs = [[3,4,5],[3,-1,1],[4,-2,1,2],[5,-3,2]]
                lpdc = ncon(tensors,legs)
            bdpsi1,bdpsi2,d = np.shape(a0[n-1])
            tensors = [l2dc,c2d[n-1]]
            legs = [[-1,1,-4],[1,-7,-3,-6]]
            l2d = ncon(tensors,legs)
            l2d = np.reshape(l2d,(bdpsi1*bdpsi2*d,bdpsi1*bdpsi2*d),order='F')
            tensors = [lpdc,cpd[n-1]]
            legs = [[-1,1,-4],[1,-7,-3,-6]]
            lpd = ncon(tensors,legs)
            lpd = np.reshape(lpd,(bdpsi1*bdpsi2*d,bdpsi1*bdpsi2*d),order='F')
            eiginput = 2*lpd-l2d
            eiginput = (eiginput+np.conj(eiginput).T)/2
            fomdval,a0v = np.linalg.eig(eiginput)
            position = np.argmax(np.real(fomdval))
            a0v = np.reshape(a0v[:,position],-1,order='F')
            a0v = a0v/np.sqrt(np.abs(np.conj(a0v) @ a0v))
            a0[n-1] = np.reshape(a0v,(bdpsi1,bdpsi2,d),order='F')
            fomd = np.append(fomd,np.real(fomdval[position]))
            iter_fomd += 1
            for x in range(n-1,0,-1):
                bdpsi1,bdpsi2,d = np.shape(a0[x])
                a0[x] = np.moveaxis(a0[x],2,1)
                a0[x] = np.reshape(a0[x],(bdpsi1,d*bdpsi2),order='F')
                u,s,vh = np.linalg.svd(a0[x],full_matrices=False)
                a0[x] = np.reshape(vh,(np.shape(s)[0],d,bdpsi2),order='F')
                a0[x] = np.moveaxis(a0[x],1,2)
                tensors = [a0[x-1],u @ np.diag(s)]
                legs = [[-1,1,-3],[1,-2]]
                a0[x-1] = ncon(tensors,legs)
            if iter_fomd >= 2 and all(fomd[-2*n:] > 0) and np.std(fomd[-2*n:])/np.mean(fomd[-2*n:]) <= relunc_fomd:
                break
        fomdval = fomd[-1]
    return fomdval,a0


def fin_FoMD_PBC_optm(c2d,cpd,a0,imprecision=10**-2):
    """
    Optimization of FoMD over MPS for initial wave function. Function for finite size systems with PBC.
    
    Parameters:
      c2d: MPO for square of dual of SLD, expected ndarray of a shape (bd,bd,d,d,n)
      cpd: MPO for dual of generalized derivative of SLD, expected ndarray of a shape (bd,bd,d,d,n)
      a0: MPS for initial wave function, expected ndarray of a shape (bd,bd,d,n)
      imprecision: expected imprecision of the end results, default value is 10**-2
    
    Returns:
      fomdval: optimal value of FoMD
      a0: optimal MPS for initial wave function
    """
    n = np.shape(c2d)[4]
    d = np.shape(c2d)[2]
    bdl2d = np.shape(c2d)[0]
    bdlpd = np.shape(cpd)[0]
    bdpsi = np.shape(a0)[0]
    tol_fomd = 0.1*imprecision/n**2
    if n == 1:
        tensors = [c2d[:,:,:,:,0],np.eye(bdpsi),np.eye(bdpsi)]
        legs = [[1,1,-3,-6],[-2,-1],[-5,-4]]
        l2d = ncon(tensors,legs)
        l2d = np.reshape(l2d,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
        tensors = [cpd[:,:,:,:,0],np.eye(bdpsi),np.eye(bdpsi)]
        legs = [[1,1,-3,-6],[-2,-1],[-5,-4]]
        lpd = ncon(tensors,legs)
        lpd = np.reshape(lpd,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
        tensors = [np.eye(bdpsi),np.eye(bdpsi)]
        legs = [[-2,-1],[-4,-3]]
        psinorm = ncon(tensors,legs)
        psinorm = np.reshape(psinorm,(bdpsi*bdpsi,bdpsi*bdpsi),order='F')
        psinorm = (psinorm+np.conj(psinorm).T)/2
        psinormpinv = np.linalg.pinv(psinorm,tol_fomd,hermitian=True)
        psinormpinv = (psinormpinv+np.conj(psinormpinv).T)/2
        psinormpinv = np.kron(np.eye(d),psinormpinv)
        eiginput = 2*lpd-l2d
        eiginput = (eiginput+np.conj(eiginput).T)/2
        eiginput = psinormpinv @ eiginput
        fomdval,a0v = np.linalg.eig(eiginput)
        position = np.argmax(np.real(fomdval))
        a0v = np.reshape(a0v[:,position],-1,order='F')
        a0v = a0v/np.sqrt(np.abs(np.conj(a0v) @ np.kron(np.eye(d),psinorm) @ a0v))
        a0[:,:,:,0] = np.reshape(a0v,(bdpsi,bdpsi,d),order='F')
        fomdval = np.real(fomdval[position])
    else:
        relunc_fomd = 0.1*imprecision
        l2df = np.zeros((bdpsi,bdl2d,bdpsi,bdpsi,bdl2d,bdpsi,n-1),dtype=complex)
        lpdf = np.zeros((bdpsi,bdlpd,bdpsi,bdpsi,bdlpd,bdpsi,n-1),dtype=complex)
        psinormf = np.zeros((bdpsi,bdpsi,bdpsi,bdpsi,n-1),dtype=complex)
        fomd = np.array([])
        iter_fomd = 0
        while True:
            tensors = [np.conj(a0[:,:,:,n-1]),c2d[:,:,:,:,n-1],a0[:,:,:,n-1]]
            legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
            l2df[:,:,:,:,:,:,n-2] = ncon(tensors,legs)
            tensors = [np.conj(a0[:,:,:,n-1]),cpd[:,:,:,:,n-1],a0[:,:,:,n-1]]
            legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
            lpdf[:,:,:,:,:,:,n-2] = ncon(tensors,legs)
            tensors = [np.conj(a0[:,:,:,n-1]),a0[:,:,:,n-1]]
            legs = [[-1,-3,1],[-2,-4,1]]
            psinormf[:,:,:,:,n-2] = ncon(tensors,legs)
            for x in range(n-2,0,-1):
                tensors = [np.conj(a0[:,:,:,x]),c2d[:,:,:,:,x],a0[:,:,:,x],l2df[:,:,:,:,:,:,x]]
                legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5,-4,-5,-6]]
                l2df[:,:,:,:,:,:,x-1] = ncon(tensors,legs)
                tensors = [np.conj(a0[:,:,:,x]),cpd[:,:,:,:,x],a0[:,:,:,x],lpdf[:,:,:,:,:,:,x]]
                legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5,-4,-5,-6]]
                lpdf[:,:,:,:,:,:,x-1] = ncon(tensors,legs)
                tensors = [np.conj(a0[:,:,:,x]),a0[:,:,:,x],psinormf[:,:,:,:,x]]
                legs = [[-1,2,1],[-2,3,1],[2,3,-3,-4]]
                psinormf[:,:,:,:,x-1] = ncon(tensors,legs)
            tensors = [c2d[:,:,:,:,0],l2df[:,:,:,:,:,:,0]]
            legs = [[2,1,-3,-6],[-2,1,-5,-1,2,-4]]
            l2d = ncon(tensors,legs)
            l2d = np.reshape(l2d,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
            tensors = [cpd[:,:,:,:,0],lpdf[:,:,:,:,:,:,0]]
            legs = [[2,1,-3,-6],[-2,1,-5,-1,2,-4]]
            lpd = ncon(tensors,legs)
            lpd = np.reshape(lpd,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
            tensors = [psinormf[:,:,:,:,0]]
            legs = [[-2,-4,-1,-3]]
            psinorm = ncon(tensors,legs)
            psinorm = np.reshape(psinorm,(bdpsi*bdpsi,bdpsi*bdpsi),order='F')
            psinorm = (psinorm+np.conj(psinorm).T)/2
            psinormpinv = np.linalg.pinv(psinorm,tol_fomd,hermitian=True)
            psinormpinv = (psinormpinv+np.conj(psinormpinv).T)/2
            psinormpinv = np.kron(np.eye(d),psinormpinv)
            eiginput = 2*lpd-l2d
            eiginput = (eiginput+np.conj(eiginput).T)/2
            eiginput = psinormpinv @ eiginput
            fomdval,a0v = np.linalg.eig(eiginput)
            position = np.argmax(np.real(fomdval))
            a0v = np.reshape(a0v[:,position],-1,order='F')
            a0v = a0v/np.sqrt(np.abs(np.conj(a0v) @ np.kron(np.eye(d),psinorm) @ a0v))
            a0[:,:,:,0] = np.reshape(a0v,(bdpsi,bdpsi,d),order='F')
            fomd = np.append(fomd,np.real(fomdval[position]))
            tensors = [np.conj(a0[:,:,:,0]),c2d[:,:,:,:,0],a0[:,:,:,0]]
            legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
            l2dc = ncon(tensors,legs)
            tensors = [np.conj(a0[:,:,:,0]),cpd[:,:,:,:,0],a0[:,:,:,0]]
            legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
            lpdc = ncon(tensors,legs)
            tensors = [np.conj(a0[:,:,:,0]),a0[:,:,:,0]]
            legs = [[-1,-3,1],[-2,-4,1]]
            psinormc = ncon(tensors,legs)
            for x in range(1,n-1):
                tensors = [l2dc,c2d[:,:,:,:,x],l2df[:,:,:,:,:,:,x]]
                legs = [[3,4,5,-1,1,-4],[1,2,-3,-6],[-2,2,-5,3,4,5]]
                l2d = ncon(tensors,legs)
                l2d = np.reshape(l2d,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
                tensors = [lpdc,cpd[:,:,:,:,x],lpdf[:,:,:,:,:,:,x]]
                legs = [[3,4,5,-1,1,-4],[1,2,-3,-6],[-2,2,-5,3,4,5]]
                lpd = ncon(tensors,legs)
                lpd = np.reshape(lpd,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
                tensors = [psinormc,psinormf[:,:,:,:,x]]
                legs = [[1,2,-1,-3],[-2,-4,1,2]]
                psinorm = ncon(tensors,legs)
                psinorm = np.reshape(psinorm,(bdpsi*bdpsi,bdpsi*bdpsi),order='F')
                psinorm = (psinorm+np.conj(psinorm).T)/2
                psinormpinv = np.linalg.pinv(psinorm,tol_fomd,hermitian=True)
                psinormpinv = (psinormpinv+np.conj(psinormpinv).T)/2
                psinormpinv = np.kron(np.eye(d),psinormpinv)
                eiginput = 2*lpd-l2d
                eiginput = (eiginput+np.conj(eiginput).T)/2
                eiginput = psinormpinv @ eiginput
                fomdval,a0v = np.linalg.eig(eiginput)
                position = np.argmax(np.real(fomdval))
                a0v = np.reshape(a0v[:,position],-1,order='F')
                a0v = a0v/np.sqrt(np.abs(np.conj(a0v) @ np.kron(np.eye(d),psinorm) @ a0v))
                a0[:,:,:,x] = np.reshape(a0v,(bdpsi,bdpsi,d),order='F')
                fomd = np.append(fomd,np.real(fomdval[position]))
                tensors = [l2dc,np.conj(a0[:,:,:,x]),c2d[:,:,:,:,x],a0[:,:,:,x]]
                legs = [[-1,-2,-3,3,4,5],[3,-4,1],[4,-5,1,2],[5,-6,2]]
                l2dc = ncon(tensors,legs)
                tensors = [lpdc,np.conj(a0[:,:,:,x]),cpd[:,:,:,:,x],a0[:,:,:,x]]
                legs = [[-1,-2,-3,3,4,5],[3,-4,1],[4,-5,1,2],[5,-6,2]]
                lpdc = ncon(tensors,legs)
                tensors = [psinormc,np.conj(a0[:,:,:,x]),a0[:,:,:,x]]
                legs = [[-1,-2,2,3],[2,-3,1],[3,-4,1]]
                psinormc = ncon(tensors,legs)
            tensors = [l2dc,c2d[:,:,:,:,n-1]]
            legs = [[-2,2,-5,-1,1,-4],[1,2,-3,-6]]
            l2d = ncon(tensors,legs)
            l2d = np.reshape(l2d,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
            tensors = [lpdc,cpd[:,:,:,:,n-1]]
            legs = [[-2,2,-5,-1,1,-4],[1,2,-3,-6]]
            lpd = ncon(tensors,legs)
            lpd = np.reshape(lpd,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
            tensors = [psinormc]
            legs = [[-2,-4,-1,-3]]
            psinorm = ncon(tensors,legs)
            psinorm = np.reshape(psinorm,(bdpsi*bdpsi,bdpsi*bdpsi),order='F')
            psinorm = (psinorm+np.conj(psinorm).T)/2
            psinormpinv = np.linalg.pinv(psinorm,tol_fomd,hermitian=True)
            psinormpinv = (psinormpinv+np.conj(psinormpinv).T)/2
            psinormpinv = np.kron(np.eye(d),psinormpinv)
            eiginput = 2*lpd-l2d
            eiginput = (eiginput+np.conj(eiginput).T)/2
            eiginput = psinormpinv @ eiginput
            fomdval,a0v = np.linalg.eig(eiginput)
            position = np.argmax(np.real(fomdval))
            a0v = np.reshape(a0v[:,position],-1,order='F')
            a0v = a0v/np.sqrt(np.abs(np.conj(a0v) @ np.kron(np.eye(d),psinorm) @ a0v))
            a0[:,:,:,n-1] = np.reshape(a0v,(bdpsi,bdpsi,d),order='F')
            fomd = np.append(fomd,np.real(fomdval[position]))
            iter_fomd += 1
            if iter_fomd >= 2 and all(fomd[-2*n:] > 0) and np.std(fomd[-2*n:])/np.mean(fomd[-2*n:]) <= relunc_fomd:
                break
        fomdval = fomd[-1]
    return fomdval,a0


def fin_FoM_OBC_val(a,b,c):
    """
    Calculate the value of FoM. Function for finite size systems with OBC.
    
    Parameters:
      a: MPO for density matrix, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
      b: MPO for generalized derivative of density matrix, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
      c: MPO for SLD, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
    
    Returns:
      fomval: value of FoM
    """
    n = len(c)
    if n == 1:
        if np.shape(a[0])[0] == 1 and np.shape(b[0])[0] == 1 and np.shape(c[0])[0] == 1:
            tensors = [c[0][0,0,:,:],b[0][0:,0,:,:]]
            legs = [[1,2],[2,1]]
            l1 = ncon(tensors,legs)
            tensors = [c[0][0,0,:,:],[0][0,0,:,:],[0][0,0,:,:]]
            legs = [[1,2],[2,3],[3,1]]
            l2 = ncon(tensors,legs)
            fomval = 2*l1-l2
        else:
            warnings.warn('Tensor networks with OBC and length one have to have bond dimension equal to one.')
    else:
        tensors = [c[n-1],b[n-1]]
        legs = [[-1,-3,1,2],[-2,-4,2,1]]
        l1 = ncon(tensors,legs)
        l1 = l1[:,:,0,0]
        tensors = [c[n-1],a[n-1],c[n-1]]
        legs = [[-1,-4,1,2],[-2,-5,2,3],[-3,-6,3,1]]
        l2 = ncon(tensors,legs)
        l2 = l2[:,:,:,0,0,0]
        for x in range(n-2,0,-1):
            tensors = [c[x],b[x],l1]
            legs = [[-1,3,1,2],[-2,4,2,1],[3,4]]
            l1 = ncon(tensors,legs)
            tensors = [c[x],a[x],c[x],l2]
            legs = [[-1,4,1,2],[-2,5,2,3],[-3,6,3,1],[4,5,6]]
            l2 = ncon(tensors,legs)
        tensors = [c[0],b[0],l1]
        legs = [[-1,3,1,2],[-2,4,2,1],[3,4]]
        l1 = ncon(tensors,legs)
        l1 = float(l1)
        tensors = [c[0],a[0],c[0],l2]
        legs = [[-1,4,1,2],[-2,5,2,3],[-3,6,3,1],[4,5,6]]
        l2 = ncon(tensors,legs)
        l2 = float(l2)
        fomval = 2*l1-l2
    return fomval


def fin_FoM_PBC_val(a,b,c):
    """
    Calculate the value of FoM. Function for finite size systems with PBC.
    
    Parameters:
      a: MPO for a density matrix, expected ndarray of a shape (bd,bd,d,d,n)
      b: MPO for generalized derivative of a density matrix, expected ndarray of a shape (bd,bd,d,d,n)
      c: MPO for the SLD, expected ndarray of a shape (bd,bd,d,d,n)
    
    Returns:
      fomval: value of FoM
    """
    n = np.shape(a)[4]
    if n == 1:
        tensors = [c[:,:,:,:,0],b[:,:,:,:,0]]
        legs = [[3,3,1,2],[4,4,2,1]]
        l1 = ncon(tensors,legs)
        tensors = [c[:,:,:,:,0],a[:,:,:,:,0],c[:,:,:,:,0]]
        legs = [[4,4,1,2],[5,5,2,3],[6,6,3,1]]
        l2 = ncon(tensors,legs)
        fomval = 2*l1-l2
    else:
        tensors = [c[:,:,:,:,n-1],b[:,:,:,:,n-1]]
        legs = [[-1,-3,1,2],[-2,-4,2,1]]
        l1 = ncon(tensors,legs)
        tensors = [c[:,:,:,:,n-1],a[:,:,:,:,n-1],c[:,:,:,:,n-1]]
        legs = [[-1,-4,1,2],[-2,-5,2,3],[-3,-6,3,1]]
        l2 = ncon(tensors,legs)
        for x in range(n-2,0,-1):
            tensors = [c[:,:,:,:,x],b[:,:,:,:,x],l1]
            legs = [[-1,3,1,2],[-2,4,2,1],[3,4,-3,-4]]
            l1 = ncon(tensors,legs)
            tensors = [c[:,:,:,:,x],a[:,:,:,:,x],c[:,:,:,:,x],l2]
            legs = [[-1,4,1,2],[-2,5,2,3],[-3,6,3,1],[4,5,6,-4,-5,-6]]
            l2 = ncon(tensors,legs)
        tensors = [c[:,:,:,:,0],b[:,:,:,:,0],l1]
        legs = [[5,3,1,2],[6,4,2,1],[3,4,5,6]]
        l1 = ncon(tensors,legs)
        tensors = [c[:,:,:,:,0],a[:,:,:,:,0],c[:,:,:,:,0],l2]
        legs = [[7,4,1,2],[8,5,2,3],[9,6,3,1],[4,5,6,7,8,9]]
        l2 = ncon(tensors,legs)
        fomval = 2*l1-l2
    return fomval


def fin_FoMD_OBC_val(c2d,cpd,a0):
    """
    Calculate value of FoMD. Function for finite size systems with OBC.
    
    Parameters:
      c2d: MPO for square of dual of SLD, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
      cpd: MPO for dual of generalized derivative of SLD, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
      a0: MPS for initial wave function, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
    
    Returns:
      fomdval: value of FoMD
    """
    n = len(a0)
    if n == 1:
        if np.shape(c2d[0])[0] == 1 and np.shape(cpd[0])[0] == 1 and np.shape(a0[0])[0] == 1:
            tensors = [np.conj(a0[0][0,0,:]),c2d[0][0,0,:,:],a0[0][0,0,:]]
            legs = [[1],[1,2],[2]]
            l2d = ncon(tensors,legs)
            tensors = [np.conj(a0[0][0,0,:]),cpd[0][0,0,:,:],a0[0][0,0,:]]
            legs = [[1],[1,2],[2]]
            lpd = ncon(tensors,legs)
            fomdval = 2*lpd-l2d
        else:
            warnings.warn('Tensor networks with OBC and length one have to have bond dimension equal to one.')
    else:
        tensors = [np.conj(a0[n-1]),c2d[n-1],a0[n-1]]
        legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
        l2d = ncon(tensors,legs)
        l2d = l2d[:,:,:,0,0,0]
        tensors = [np.conj(a0[n-1]),cpd[n-1],a0[n-1]]
        legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
        lpd = ncon(tensors,legs)
        lpd = lpd[:,:,:,0,0,0]
        for x in range(n-2,0,-1):
            tensors = [np.conj(a0[x]),c2d[x],a0[x],l2d]
            legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5]]
            l2d = ncon(tensors,legs)
            tensors = [np.conj(a0[x]),cpd[x],a0[x],lpd]
            legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5]]
            lpd = ncon(tensors,legs)
        tensors = [np.conj(a0[0]),c2d[0],a0[0],l2d]
        legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5]]
        l2d = ncon(tensors,legs)
        l2d = float(l2d)
        tensors = [np.conj(a0[0]),cpd[0],a0[0],lpd]
        legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5]]
        lpd = ncon(tensors,legs)
        lpd = float(lpd)
        fomdval = 2*lpd-l2d
    return fomdval


def fin_FoMD_PBC_val(c2d,cpd,a0):
    """
    Calculate the value of FoMD. Function for finite size systems with PBC.
    
    Parameters:
      c2d: MPO for square of dual of the SLD, expected ndarray of a shape (bd,bd,d,d,n)
      cpd: MPO for dual of generalized derivative of the SLD, expected ndarray of a shape (bd,bd,d,d,n)
      a0: MPS for the initial wave function, expected ndarray of a shape (bd,bd,d,n)
    
    Returns:
      fomdval: value of FoMD
    """
    n = np.shape(c2d)[4]
    if n == 1:
        tensors = [np.conj(a0[:,:,:,0]),c2d[:,:,:,:,0],a0[:,:,:,0]]
        legs = [[3,3,1],[4,4,1,2],[5,5,2]]
        l2d = ncon(tensors,legs)
        tensors = [np.conj(a0[:,:,:,0]),cpd[:,:,:,:,0],a0[:,:,:,0]]
        legs = [[3,3,1],[4,4,1,2],[5,5,2]]
        lpd = ncon(tensors,legs)
        fomdval = 2*lpd-l2d
    else:
        tensors = [np.conj(a0[:,:,:,n-1]),c2d[:,:,:,:,n-1],a0[:,:,:,n-1]]
        legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
        l2d = ncon(tensors,legs)
        tensors = [np.conj(a0[:,:,:,n-1]),cpd[:,:,:,:,n-1],a0[:,:,:,n-1]]
        legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
        lpd = ncon(tensors,legs)
        for x in range(n-2,0,-1):
            tensors = [np.conj(a0[:,:,:,x]),c2d[:,:,:,:,x],a0[:,:,:,x],l2d]
            legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5,-4,-5,-6]]
            l2d = ncon(tensors,legs)
            tensors = [np.conj(a0[:,:,:,x]),cpd[:,:,:,:,x],a0[:,:,:,x],lpd]
            legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5,-4,-5,-6]]
            lpd = ncon(tensors,legs)
        tensors = [np.conj(a0[:,:,:,0]),c2d[:,:,:,:,0],a0[:,:,:,0],l2d]
        legs = [[6,3,1],[7,4,1,2],[8,5,2],[3,4,5,6,7,8]]
        l2d = ncon(tensors,legs)
        tensors = [np.conj(a0[:,:,:,0]),cpd[:,:,:,:,0],a0[:,:,:,0],lpd]
        legs = [[6,3,1],[7,4,1,2],[8,5,2],[3,4,5,6,7,8]]
        lpd = ncon(tensors,legs)
        fomdval = 2*lpd-l2d
    return fomdval


#################################################################
# 1.2.2 Problems with discrete approximation of the derivative. #
#################################################################


def fin2_FoM_FoMD_optbd(n,d,bc,ch,chp,epsilon,cini=None,a0ini=None,imprecision=10**-2,bdlmax=100,alwaysbdlmax=False,lherm=True,bdpsimax=100,alwaysbdpsimax=False):
    """
    Iterative optimization of FoM/FoMD over SLD MPO and initial wave function MPS and also a check of convergence with increasing bond dimensions. Function for finite size systems. Version with two channels separated by epsilon.
    
    Parameters:
      n: number of sites in TN
      d: dimension of the local Hilbert space (dimension of the physical index)
      bc: boundary conditions, 'O' for OBC, 'P' for PBC
      ch: MPO for a quantum channel at the value of estimated parameter phi=phi_0, expected list of length n of ndarrays of a shape (bd,bd,d**2,d**2) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d**2,d**2,n) for PBC
      chp: MPO for a quantum channel at the value of estimated parameter phi=phi_0+epsilon, expected list of length n of ndarrays of a shape (bd,bd,d**2,d**2) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d**2,d**2,n) for PBC
      epsilon: value of a separation between estimated parameters encoded in ch and chp, float
      cini: initial MPO for the SLD, expected list of length n of ndarrays of a shape (bd,bd,d,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,d,n) for PBC
      a0ini: initial MPS for the initial wave function, expected list of length n of ndarrays of a shape (bd,bd,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,n) for PBC
      imprecision: expected imprecision of the end results, default value is 10**-2
      bdlmax: maximal value of bd for SLD MPO, default value is 100
      alwaysbdlmax: boolean value, True if the maximal value of bd for SLD MPO has to be reached, otherwise False (default value)
      lherm: boolean value, True (default value) when Hermitian gauge is imposed on SLD MPO, otherwise False
      bdpsimax: maximal value of bd for the initial wave function MPS, default value is 100
      alwaysbdpsimax: boolean value, True if the maximal value of bd for initial wave function MPS has to be reached, otherwise False (default value)
    
    Returns:
      result: optimal value of FoM/FoMD
      resultm: matrix describing FoM/FoMD as a function of bd of respectively SLD MPO [rows] and the initial wave function MPS [columns]
      c: optimal MPO for SLD
      a0: optimal MPS for initial wave function
    """
    while True:
        if a0ini is None:
            bdpsi = 1
            a0 = np.zeros(d,dtype=complex)
            for i in range(d):
                a0[i] = np.sqrt(math.comb(d-1,i))*2**(-(d-1)/2) # prod
                # a0[i] = np.sqrt(2/(d+1))*np.sin((1+i)*np.pi/(d+1)) # sine
            if bc == 'O':
                a0 = a0[np.newaxis,np.newaxis,:]
                a0 = [a0]*n
            elif bc == 'P':
                a0 = a0[np.newaxis,np.newaxis,:,np.newaxis]
                a0 = np.tile(a0,(1,1,1,n))
        else:
            a0 = a0ini
            if bc == 'O':
                bdpsi = max([np.shape(a0[i])[0] for i in range(n)])
                a0 = [a0[i].astype(complex) for i in range(n)]
            elif bc == 'P':
                bdpsi = np.shape(a0)[0]
                a0 = a0.astype(complex)
        if cini is None:
            bdl = 1
            rng = np.random.default_rng()
            if bc == 'O':
                c = [0]*n
                c[0] = (rng.random((1,bdl,d,d)) + 1j*rng.random((1,bdl,d,d)))/bdl
                c[0] = (c[0] + np.conj(np.moveaxis(c[0],2,3)))/2
                for x in range(1,n-1):
                    c[x] = (rng.random((bdl,bdl,d,d)) + 1j*rng.random((bdl,bdl,d,d)))/bdl
                    c[x] = (c[x] + np.conj(np.moveaxis(c[x],2,3)))/2
                c[n-1] = (rng.random((bdl,1,d,d)) + 1j*rng.random((bdl,1,d,d)))/bdl
                c[n-1] = (c[n-1] + np.conj(np.moveaxis(c[n-1],2,3)))/2
            elif  bc == 'P':
                c = (rng.random((bdl,bdl,d,d,n)) + 1j*rng.random((bdl,bdl,d,d,n)))/bdl
                c = (c + np.conj(np.moveaxis(c,2,3)))/2
        else:
            c = cini
            if bc == 'O':
                bdl = max([np.shape(c[i])[0] for i in range(n)])
                c = [c[i].astype(complex) for i in range(n)]
            elif bc == 'P':
                bdl = np.shape(c)[0]
                c = c.astype(complex)
        resultm = np.zeros((bdlmax,bdpsimax),dtype=float)
        resultm[bdl-1,bdpsi-1],c,a0 = fin2_FoM_FoMD_optm(n,d,bc,c,a0,ch,chp,epsilon,imprecision,lherm)
        if bc == 'O' and n == 1:
            resultm = resultm[0:bdl,0:bdpsi]
            result = resultm[bdl-1,bdpsi-1]
            return result,resultm,c,a0
        factorv = np.array([0.5,0.25,0.1,1,0.01])
        problem = False
        while True:
            while True:
                if bdpsi == bdpsimax:
                    break
                else:
                    a0old = a0
                    bdpsi += 1
                    i = 0
                    while True:
                        a0 = fin_enlarge_bdpsi(a0,factorv[i])
                        resultm[bdl-1,bdpsi-1],cnew,a0new = fin2_FoM_FoMD_optm(n,d,bc,c,a0,ch,chp,epsilon,imprecision,lherm)
                        if resultm[bdl-1,bdpsi-1] >= resultm[bdl-1,bdpsi-2]:
                            break
                        i += 1
                        if i == np.size(factorv):
                            problem = True
                            break
                    if problem:
                        break
                    if not(alwaysbdpsimax) and resultm[bdl-1,bdpsi-1] < (1+imprecision)*resultm[bdl-1,bdpsi-2]:
                        bdpsi += -1
                        a0 = a0old
                        a0copy = a0new
                        ccopy = cnew
                        break
                    else:
                        a0 = a0new
                        c = cnew
            if problem:
                break
            if bdl == bdlmax:
                if bdpsi == bdpsimax:
                    resultm = resultm[0:bdl,0:bdpsi]
                    result = resultm[bdl-1,bdpsi-1]
                else:
                    a0 = a0copy
                    c = ccopy
                    resultm = resultm[0:bdl,0:bdpsi+1]
                    result = resultm[bdl-1,bdpsi]
                break
            else:
                bdl += 1
                i = 0
                while True:
                    c = fin_enlarge_bdl(c,factorv[i])
                    resultm[bdl-1,bdpsi-1],cnew,a0new = fin2_FoM_FoMD_optm(n,d,bc,c,a0,ch,chp,epsilon,imprecision,lherm)
                    if resultm[bdl-1,bdpsi-1] >= resultm[bdl-2,bdpsi-1]:
                        a0 = a0new
                        c = cnew
                        break
                    i += 1
                    if i == np.size(factorv):
                        problem = True
                        break
                if problem:
                    break
                if not(alwaysbdlmax) and resultm[bdl-1,bdpsi-1] < (1+imprecision)*resultm[bdl-2,bdpsi-1]:
                    if bdpsi == bdpsimax:
                        resultm = resultm[0:bdl,0:bdpsi]
                        result = resultm[bdl-1,bdpsi-1]
                    else:
                        if resultm[bdl-1,bdpsi-1] < resultm[bdl-2,bdpsi]:
                            a0 = a0copy
                            c = ccopy
                            resultm = resultm[0:bdl,0:bdpsi+1]
                            bdl += -1
                            bdpsi += 1
                            result = resultm[bdl-1,bdpsi-1]
                        else:
                            resultm = resultm[0:bdl,0:bdpsi+1]
                            result = resultm[bdl-1,bdpsi-1]
                    break
        if not(problem):
            break
    return result,resultm,c,a0


def fin2_FoM_optbd(n,d,bc,a,b,epsilon,cini=None,imprecision=10**-2,bdlmax=100,alwaysbdlmax=False,lherm=True):
    """
    Optimization of FoM over SLD MPO and also check of convergence in bond dimension. Function for finite size systems. Version with two states separated by epsilon.
    
    Parameters:
      n: number of sites in TN
      d: dimension of local Hilbert space (dimension of physical index)
      bc: boundary conditions, 'O' for OBC, 'P' for PBC
      a: MPO for the density matrix at the value of estimated parameter phi=phi_0, expected list of length n of ndarrays of a shape (bd,bd,d,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,d,n) for PBC
      b: MPO for the density matrix at the value of estimated parameter phi=phi_0+epsilon, expected list of length n of ndarrays of a shape (bd,bd,d,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,d,n) for PBC
      epsilon: value of a separation between estimated parameters encoded in a and b, float
      cini: initial MPO for SLD, expected list of length n of ndarrays of a shape (bd,bd,d,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,d,n) for PBC
      imprecision: expected imprecision of the end results, default value is 10**-2
      bdlmax: maximal value of bd for SLD MPO, default value is 100
      alwaysbdlmax: boolean value, True if maximal value of bd for SLD MPO have to be reached, otherwise False (default value)
      lherm: boolean value, True (default value) when Hermitian gauge is imposed on SLD MPO, otherwise False
    
    Returns:
      result: optimal value of FoM
      resultv: vector describing FoM as a function of bd of the SLD MPO
      c: optimal MPO for SLD
    """
    while True:
        if cini is None:
            bdl = 1
            rng = np.random.default_rng()
            if bc == 'O':
                c = [0]*n
                c[0] = (rng.random((1,bdl,d,d)) + 1j*rng.random((1,bdl,d,d)))/bdl
                c[0] = (c[0] + np.conj(np.moveaxis(c[0],2,3)))/2
                for x in range(1,n-1):
                    c[x] = (rng.random((bdl,bdl,d,d)) + 1j*rng.random((bdl,bdl,d,d)))/bdl
                    c[x] = (c[x] + np.conj(np.moveaxis(c[x],2,3)))/2
                c[n-1] = (rng.random((bdl,1,d,d)) + 1j*rng.random((bdl,1,d,d)))/bdl
                c[n-1] = (c[n-1] + np.conj(np.moveaxis(c[n-1],2,3)))/2
            elif  bc == 'P':
                c = (rng.random((bdl,bdl,d,d,n)) + 1j*rng.random((bdl,bdl,d,d,n)))/bdl
                c = (c + np.conj(np.moveaxis(c,2,3)))/2
        else:
            c = cini
            if bc == 'O':
                bdl = max([np.shape(c[i])[0] for i in range(n)])
                c = [c[i].astype(complex) for i in range(n)]
            elif bc == 'P':
                bdl = np.shape(c)[0]
                c = c.astype(complex)
        resultv = np.zeros(bdlmax,dtype=float)
        if bc == 'O':
            resultv[bdl-1],c = fin2_FoM_OBC_optm(a,b,epsilon,c,imprecision,lherm)
            if n == 1:
                resultv = resultv[0:bdl]
                result = resultv[bdl-1]
                return result,resultv,c
        elif  bc == 'P':
            resultv[bdl-1],c = fin2_FoM_PBC_optm(a,b,epsilon,c,imprecision,lherm)
        factorv = np.array([0.5,0.25,0.1,1,0.01])
        problem = False
        while True:
            if bdl == bdlmax:
                resultv = resultv[0:bdl]
                result = resultv[bdl-1]
                break
            else:
                bdl += 1
                i = 0
                while True:
                    c = fin_enlarge_bdl(c,factorv[i])
                    if bc == 'O':
                        resultv[bdl-1],cnew = fin2_FoM_OBC_optm(a,b,epsilon,c,imprecision,lherm)
                    elif  bc == 'P':
                        resultv[bdl-1],cnew = fin2_FoM_PBC_optm(a,b,epsilon,c,imprecision,lherm)
                    if resultv[bdl-1] >= resultv[bdl-2]:
                        c = cnew
                        break
                    i += 1
                    if i == np.size(factorv):
                        problem = True
                        break
                if problem:
                    break
                if not(alwaysbdlmax) and resultv[bdl-1] < (1+imprecision)*resultv[bdl-2]:
                    resultv = resultv[0:bdl]
                    result = resultv[bdl-1]
                    break
        if not(problem):
            break
    return result,resultv,c


def fin2_FoMD_optbd(n,d,bc,c2d,cd,cpd,epsilon,a0ini=None,imprecision=10**-2,bdpsimax=100,alwaysbdpsimax=False):
    """
    Optimization of FoMD over initial wave function MPS and also check of convergence when increasing the bond dimension. Function for finite size systems. Version with two dual SLDs separated by epsilon.
    
    Parameters:
      n: number of sites in TN
      d: dimension of local Hilbert space (dimension of physical index)
      bc: boundary conditions, 'O' for OBC, 'P' for PBC
      c2d: MPO for square of dual of SLD at the value of estimated parameter phi=-phi_0, expected list of length n of ndarrays of a shape (bd,bd,d,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,d,n) for PBC
      cd: MPO for dual of SLD at the value of estimated parameter phi=-phi_0, expected list of length n of ndarrays of a shape (bd,bd,d,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,d,n) for PBC
      cpd: MPO for dual of SLD at the value of estimated parameter phi=-(phi_0+epsilon), expected list of length n of ndarrays of a shape (bd,bd,d,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,d,n) for PBC
      epsilon: value of a separation between estimated parameters encoded in cd and cpd, float
      a0ini: initial MPS for initial wave function, expected list of length n of ndarrays of a shape (bd,bd,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,n) for PBC
      imprecision: expected imprecision of the end results, default value is 10**-2
      bdpsimax: maximal value of bd for initial wave function MPS, default value is 100
      alwaysbdpsimax: boolean value, True if maximal value of bd for initial wave function MPS have to be reached, otherwise False (default value)
    
    Returns:
      result: optimal value of FoMD
      resultv: vector describing FoMD in function of bd of initial wave function MPS
      a0: optimal MPS for initial wave function
    """
    while True:
        if a0ini is None:
            bdpsi = 1
            a0 = np.zeros(d,dtype=complex)
            for i in range(d):
                a0[i] = np.sqrt(math.comb(d-1,i))*2**(-(d-1)/2) # prod
                # a0[i] = np.sqrt(2/(d+1))*np.sin((1+i)*np.pi/(d+1)) # sine
            if bc == 'O':
                a0 = a0[np.newaxis,np.newaxis,:]
                a0 = [a0]*n
            elif bc == 'P':
                a0 = a0[np.newaxis,np.newaxis,:,np.newaxis]
                a0 = np.tile(a0,(1,1,1,n))
        else:
            a0 = a0ini
            if bc == 'O':
                bdpsi = max([np.shape(a0[i])[0] for i in range(n)])
                a0 = [a0[i].astype(complex) for i in range(n)]
            elif bc == 'P':
                bdpsi = np.shape(a0)[0]
                a0 = a0.astype(complex)
        resultv = np.zeros(bdpsimax,dtype=float)
        if bc == 'O':
            resultv[bdpsi-1],a0 = fin2_FoMD_OBC_optm(c2d,cd,cpd,epsilon,a0,imprecision)
            if n == 1:
                resultv = resultv[0:bdpsi]
                result = resultv[bdpsi-1]
                return result,resultv,a0
        elif  bc == 'P':
            resultv[bdpsi-1],a0 = fin2_FoMD_PBC_optm(c2d,cd,cpd,epsilon,a0,imprecision)
        factorv = np.array([0.5,0.25,0.1,1,0.01])
        problem = False
        while True:
            if bdpsi == bdpsimax:
                resultv = resultv[0:bdpsi]
                result = resultv[bdpsi-1]
                break
            else:
                bdpsi += 1
                i = 0
                while True:
                    a0 = fin_enlarge_bdpsi(a0,factorv[i])
                    if bc == 'O':
                        resultv[bdpsi-1],a0new = fin2_FoMD_OBC_optm(c2d,cd,cpd,epsilon,a0,imprecision)
                    elif  bc == 'P':
                        resultv[bdpsi-1],a0new = fin2_FoMD_PBC_optm(c2d,cd,cpd,epsilon,a0,imprecision)
                    if resultv[bdpsi-1] >= resultv[bdpsi-2]:
                        a0 = a0new
                        break
                    i += 1
                    if i == np.size(factorv):
                        problem = True
                        break
                if problem:
                    break
                if not(alwaysbdpsimax) and resultv[bdpsi-1] < (1+imprecision)*resultv[bdpsi-2]:
                    resultv = resultv[0:bdpsi]
                    result = resultv[bdpsi-1]
                    break
        if not(problem):
            break
    return result,resultv,a0


def fin2_FoM_FoMD_optm(n,d,bc,c,a0,ch,chp,epsilon,imprecision=10**-2,lherm=True):
    """
    Iterative optimization of FoM/FoMD over SLD MPO and initial wave function MPS. Function for finite size systems. Version with two channels separated by epsilon.
    
    Parameters:
      n: number of sites in TN
      d: dimension of local Hilbert space (dimension of physical index)
      bc: boundary conditions, 'O' for OBC, 'P' for PBC
      c: MPO for SLD, expected list of length n of ndarrays of a shape (bd,bd,d,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,d,n) for PBC
      a0: MPS for initial wave function, expected list of length n of ndarrays of a shape (bd,bd,d) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d,n) for PBC
      ch: MPO for quantum channel at the value of estimated parameter phi=phi_0, expected list of length n of ndarrays of a shape (bd,bd,d**2,d**2) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d**2,d**2,n) for PBC
      chp: MPO for quantum channel at the value of estimated parameter phi=phi_0+epsilon, expected list of length n of ndarrays of a shape (bd,bd,d**2,d**2) for OBC (bd can vary between sites), or ndarray of a shape (bd,bd,d**2,d**2,n) for PBC
      epsilon: value of a separation between estimated parameters encoded in ch and chp, float
      imprecision: expected imprecision of the end results, default value is 10**-2
      lherm: boolean value, True (default value) when Hermitian gauge is imposed on SLD MPO, otherwise False
    
    Returns:
      fval: optimal value of FoM/FoMD
      c: optimal MPO for SLD
      a0: optimal MPS for initial wave function
    """
    relunc_f = 0.1*imprecision
    if bc == 'O':
        chd = [0]*n
        chpd = [0]*n
        for x in range(n):
            chd[x] = np.conj(np.moveaxis(ch[x],2,3))
            chpd[x] = np.conj(np.moveaxis(chp[x],2,3))
    elif bc == 'P':
        chd = np.conj(np.moveaxis(ch,2,3))
        chpd = np.conj(np.moveaxis(chp,2,3))
    f = np.array([])
    iter_f = 0
    while True:
        a0_dm = wave_function_to_density_matrix(a0)
        a = channel_acting_on_operator(ch,a0_dm)
        b = channel_acting_on_operator(chp,a0_dm)
        if bc == 'O':
            fom,c = fin2_FoM_OBC_optm(a,b,epsilon,c,imprecision,lherm)
        elif bc == 'P':
            fom,c = fin2_FoM_PBC_optm(a,b,epsilon,c,imprecision,lherm)
        f = np.append(f,fom)
        if iter_f >= 2 and np.std(f[-4:])/np.mean(f[-4:]) <= relunc_f:
            break
        if bc == 'O':
            c2 = [0]*n
            for x in range(n):
                bdl1 = np.shape(c[x])[0]
                bdl2 = np.shape(c[x])[1]
                c2[x] = np.zeros((bdl1**2,bdl2**2,d,d),dtype=complex)
                for nx in range(d):
                    for nxp in range(d):
                        for nxpp in range(d):
                            c2[x][:,:,nx,nxp] = c2[x][:,:,nx,nxp]+np.kron(c[x][:,:,nx,nxpp],c[x][:,:,nxpp,nxp])
        elif bc == 'P':
            bdl = np.shape(c)[0]
            c2 = np.zeros((bdl**2,bdl**2,d,d,n),dtype=complex)
            for nx in range(d):
                for nxp in range(d):
                    for nxpp in range(d):
                        for x in range(n):
                            c2[:,:,nx,nxp,x] = c2[:,:,nx,nxp,x]+np.kron(c[:,:,nx,nxpp,x],c[:,:,nxpp,nxp,x])
        c2d = channel_acting_on_operator(chd,c2)
        cd = channel_acting_on_operator(chd,c)
        cpd = channel_acting_on_operator(chpd,c)
        if bc == 'O':
            fomd,a0 = fin2_FoMD_OBC_optm(c2d,cd,cpd,epsilon,a0,imprecision)
        elif bc == 'P':
            fomd,a0 = fin2_FoMD_PBC_optm(c2d,cd,cpd,epsilon,a0,imprecision)
        f = np.append(f,fomd)
        iter_f += 1
    fval = f[-1]
    return fval,c,a0


def fin2_FoM_OBC_optm(a,b,epsilon,c,imprecision=10**-2,lherm=True):
    """
    Optimization of FoM over MPO for SLD. Function for finite size systems with OBC. Version with two states separated by epsilon.
    
    Parameters:
      a: MPO for the density matrix at the value of estimated parameter phi=phi_0, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
      b: MPO for the density matrix at the value of estimated parameter phi=phi_0+epsilon, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
      epsilon: value of a separation between estimated parameters encoded in a and b, float
      c: MPO for the SLD, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
      imprecision: expected imprecision of the end results, default value is 10**-2
      lherm: boolean value, True (default value) when Hermitian gauge is imposed on SLD MPO, otherwise False
    
    Returns:
      fomval: optimal value of FoM
      c: optimal MPO for SLD
    """
    n = len(c)
    tol_fom = 0.1*imprecision/n**2
    if n == 1:
        if np.shape(a[0])[0] == 1 and np.shape(b[0])[0] == 1 and np.shape(c[0])[0] == 1:
            d = np.shape(c[0])[2]
            tensors = [b[0][0,0,:,:]]
            legs = [[-2,-1]]
            l1 = ncon(tensors,legs)
            l1 = np.reshape(l1,-1,order='F')
            tensors = [a[0][0,0,:,:]]
            legs = [[-2,-1]]
            l1_0 = ncon(tensors,legs)
            l1_0 = np.reshape(l1_0,-1,order='F')
            tensors = [a[0][0,0,:,:],np.eye(d)]
            legs = [[-2,-3],[-4,-1]]
            l2 = ncon(tensors,legs)
            l2 = np.reshape(l2,(d*d,d*d),order='F')
            dl2 = l2+l2.T
            dl1 = 2*(l1-l1_0)/epsilon
            dl2pinv = np.linalg.pinv(dl2,tol_fom)
            dl2pinv = (dl2pinv+dl2pinv.T)/2
            cv = dl2pinv @ dl1
            c[0][0,0,:,:] = np.reshape(cv,(d,d),order='F')
            if lherm:
                c[0] = (c[0]+np.conj(np.moveaxis(c[0],2,3)))/2
                cv = np.reshape(c[0],-1,order='F')
            fomval = np.real(2*cv @ (l1-l1_0)/epsilon - cv @ l2 @ cv)
        else:
            warnings.warn('Tensor networks with OBC and length one have to have bond dimension equal to one.')
    else:
        relunc_fom = 0.1*imprecision
        l1f = [0]*n
        l1_0f = [0]*n
        l2f = [0]*n
        fom = np.array([])
        iter_fom = 0
        while True:
            tensors = [c[n-1],b[n-1]]
            legs = [[-1,-3,1,2],[-2,-4,2,1]]
            l1f[n-2] = ncon(tensors,legs)
            l1f[n-2] = l1f[n-2][:,:,0,0]
            tensors = [c[n-1],a[n-1]]
            legs = [[-1,-3,1,2],[-2,-4,2,1]]
            l1_0f[n-2] = ncon(tensors,legs)
            l1_0f[n-2] = l1_0f[n-2][:,:,0,0]
            tensors = [c[n-1],a[n-1],c[n-1]]
            legs = [[-1,-4,1,2],[-2,-5,2,3],[-3,-6,3,1]]
            l2f[n-2] = ncon(tensors,legs)
            l2f[n-2] = l2f[n-2][:,:,:,0,0,0]
            for x in range(n-2,0,-1):
                tensors = [c[x],b[x],l1f[x]]
                legs = [[-1,3,1,2],[-2,4,2,1],[3,4]]
                l1f[x-1] = ncon(tensors,legs)
                tensors = [c[x],a[x],l1_0f[x]]
                legs = [[-1,3,1,2],[-2,4,2,1],[3,4]]
                l1_0f[x-1] = ncon(tensors,legs)
                tensors = [c[x],a[x],c[x],l2f[x]]
                legs = [[-1,4,1,2],[-2,5,2,3],[-3,6,3,1],[4,5,6]]
                l2f[x-1] = ncon(tensors,legs)
            bdl1,bdl2,d,d = np.shape(c[0])
            tensors = [b[0],l1f[0]]
            legs = [[-5,1,-4,-3],[-2,1]]
            l1 = ncon(tensors,legs)
            l1 = np.reshape(l1,-1,order='F')
            tensors = [a[0],l1_0f[0]]
            legs = [[-5,1,-4,-3],[-2,1]]
            l1_0 = ncon(tensors,legs)
            l1_0 = np.reshape(l1_0,-1,order='F')
            tensors = [a[0],np.eye(d),l2f[0]]
            legs = [[-9,1,-4,-7],[-8,-3],[-2,1,-6]]
            l2 = ncon(tensors,legs)
            l2 = np.reshape(l2,(bdl1*bdl2*d*d,bdl1*bdl2*d*d),order='F')
            dl2 = l2+l2.T
            dl1 = 2*(l1-l1_0)/epsilon
            dl2pinv = np.linalg.pinv(dl2,tol_fom)
            dl2pinv = (dl2pinv+dl2pinv.T)/2
            cv = dl2pinv @ dl1
            c[0] = np.reshape(cv,(bdl1,bdl2,d,d),order='F')
            if lherm:
                c[0] = (c[0]+np.conj(np.moveaxis(c[0],2,3)))/2
                cv = np.reshape(c[0],-1,order='F')
            fom = np.append(fom,np.real(2*cv @ (l1-l1_0)/epsilon - cv @ l2 @ cv))
            tensors = [c[0],b[0]]
            legs = [[-3,-1,1,2],[-4,-2,2,1]]
            l1c = ncon(tensors,legs)
            l1c = l1c[:,:,0,0]
            tensors = [c[0],a[0]]
            legs = [[-3,-1,1,2],[-4,-2,2,1]]
            l1_0c = ncon(tensors,legs)
            l1_0c = l1_0c[:,:,0,0]
            tensors = [c[0],a[0],c[0]]
            legs = [[-4,-1,1,2],[-5,-2,2,3],[-6,-3,3,1]]
            l2c = ncon(tensors,legs)
            l2c = l2c[:,:,:,0,0,0]
            for x in range(1,n-1):
                bdl1,bdl2,d,d = np.shape(c[x])
                tensors = [l1c,b[x],l1f[x]]
                legs = [[-1,1],[1,2,-4,-3],[-2,2]]
                l1 = ncon(tensors,legs)
                l1 = np.reshape(l1,-1,order='F')
                tensors = [l1_0c,a[x],l1_0f[x]]
                legs = [[-1,1],[1,2,-4,-3],[-2,2]]
                l1_0 = ncon(tensors,legs)
                l1_0 = np.reshape(l1_0,-1,order='F')
                tensors = [l2c,a[x],np.eye(d),l2f[x]]
                legs = [[-1,1,-5],[1,2,-4,-7],[-8,-3],[-2,2,-6]]
                l2 = ncon(tensors,legs)
                l2 = np.reshape(l2,(bdl1*bdl2*d*d,bdl1*bdl2*d*d),order='F')
                dl2 = l2+l2.T
                dl1 = 2*(l1-l1_0)/epsilon
                dl2pinv = np.linalg.pinv(dl2,tol_fom)
                dl2pinv = (dl2pinv+dl2pinv.T)/2
                cv = dl2pinv @ dl1
                c[x] = np.reshape(cv,(bdl1,bdl2,d,d),order='F')
                if lherm:
                    c[x] = (c[x]+np.conj(np.moveaxis(c[x],2,3)))/2
                    cv = np.reshape(c[x],-1,order='F')
                fom = np.append(fom,np.real(2*cv @ (l1-l1_0)/epsilon - cv @ l2 @ cv))
                tensors = [l1c,c[x],b[x]]
                legs = [[3,4],[3,-1,1,2],[4,-2,2,1]]
                l1c = ncon(tensors,legs)
                tensors = [l1_0c,c[x],a[x]]
                legs = [[3,4],[3,-1,1,2],[4,-2,2,1]]
                l1_0c = ncon(tensors,legs)
                tensors = [l2c,c[x],a[x],c[x]]
                legs = [[4,5,6],[4,-1,1,2],[5,-2,2,3],[6,-3,3,1]]
                l2c = ncon(tensors,legs)
            bdl1,bdl2,d,d = np.shape(c[n-1])
            tensors = [l1c,b[n-1]]
            legs = [[-1,1],[1,-5,-4,-3]]
            l1 = ncon(tensors,legs)
            l1 = np.reshape(l1,-1,order='F')
            tensors = [l1_0c,a[n-1]]
            legs = [[-1,1],[1,-5,-4,-3]]
            l1_0 = ncon(tensors,legs)
            l1_0 = np.reshape(l1_0,-1,order='F')
            tensors = [l2c,a[n-1],np.eye(d)]
            legs = [[-1,1,-5],[1,-9,-4,-7],[-8,-3]]
            l2 = ncon(tensors,legs)
            l2 = np.reshape(l2,(bdl1*bdl2*d*d,bdl1*bdl2*d*d),order='F')
            dl2 = l2+l2.T
            dl1 = 2*(l1-l1_0)/epsilon
            dl2pinv = np.linalg.pinv(dl2,tol_fom)
            dl2pinv = (dl2pinv+dl2pinv.T)/2
            cv = dl2pinv @ dl1
            c[n-1] = np.reshape(cv,(bdl1,bdl2,d,d),order='F')
            if lherm:
                c[n-1] = (c[n-1]+np.conj(np.moveaxis(c[n-1],2,3)))/2
                cv = np.reshape(c[n-1],-1,order='F')
            fom = np.append(fom,np.real(2*cv @ (l1-l1_0)/epsilon - cv @ l2 @ cv))
            iter_fom += 1
            if iter_fom >= 2 and all(fom[-2*n:] > 0) and np.std(fom[-2*n:])/np.mean(fom[-2*n:]) <= relunc_fom:
                break
        fomval = fom[-1]
    return fomval,c


def fin2_FoM_PBC_optm(a,b,epsilon,c,imprecision=10**-2,lherm=True):
    """
    Optimization of FoM over MPO for SLD. Function for finite size systems with PBC. Version with two states separated by epsilon.
    
    Parameters:
      a: MPO for the density matrix at the value of estimated parameter phi=phi_0, expected ndarray of a shape (bd,bd,d,d,n)
      b: MPO for the density matrix at the value of estimated parameter phi=phi_0+epsilon, expected ndarray of a shape (bd,bd,d,d,n)
      epsilon: value of a separation between estimated parameters encoded in a and b, float
      c: MPO for the SLD, expected ndarray of a shape (bd,bd,d,d,n)
      imprecision: expected imprecision of the end results, default value is 10**-2
      lherm: boolean value, True (default value) when Hermitian gauge is imposed on SLD MPO, otherwise False
    
    Returns:
      fomval: optimal value of FoM
      c: optimal MPO for SLD
    """
    n = np.shape(a)[4]
    d = np.shape(a)[2]
    bdr = np.shape(a)[0]
    bdrp = np.shape(b)[0]
    bdl = np.shape(c)[0]
    tol_fom = 0.1*imprecision/n**2
    if n == 1:
        tensors = [b[:,:,:,:,0],np.eye(bdl)]
        legs = [[1,1,-4,-3],[-2,-1]]
        l1 = ncon(tensors,legs)
        l1 = np.reshape(l1,-1,order='F')
        tensors = [a[:,:,:,:,0],np.eye(bdl)]
        legs = [[1,1,-4,-3],[-2,-1]]
        l1_0 = ncon(tensors,legs)
        l1_0 = np.reshape(l1_0,-1,order='F')
        tensors = [a[:,:,:,:,0],np.eye(d),np.eye(bdl),np.eye(bdl)]
        legs = [[1,1,-4,-7],[-8,-3],[-2,-1],[-6,-5]]
        l2 = ncon(tensors,legs)
        l2 = np.reshape(l2,(bdl*bdl*d*d,bdl*bdl*d*d),order='F')
        dl2 = l2+l2.T
        dl1 = 2*(l1-l1_0)/epsilon
        dl2pinv = np.linalg.pinv(dl2,tol_fom)
        dl2pinv = (dl2pinv+dl2pinv.T)/2
        cv = dl2pinv @ dl1
        c[:,:,:,:,0] = np.reshape(cv,(bdl,bdl,d,d),order='F')
        if lherm:
            c[:,:,:,:,0] = (c[:,:,:,:,0]+np.conj(np.moveaxis(c[:,:,:,:,0],2,3)))/2
            cv = np.reshape(c[:,:,:,:,0],-1,order='F')
        fomval = np.real(2*cv @ (l1-l1_0)/epsilon - cv @ l2 @ cv)
    else:
        relunc_fom = 0.1*imprecision
        l1f = np.zeros((bdl,bdrp,bdl,bdrp,n-1),dtype=complex)
        l1_0f = np.zeros((bdl,bdrp,bdl,bdrp,n-1),dtype=complex)
        l2f = np.zeros((bdl,bdr,bdl,bdl,bdr,bdl,n-1),dtype=complex)
        fom = np.array([])
        iter_fom = 0
        while True:
            tensors = [c[:,:,:,:,n-1],b[:,:,:,:,n-1]]
            legs = [[-1,-3,1,2],[-2,-4,2,1]]
            l1f[:,:,:,:,n-2] = ncon(tensors,legs)
            tensors = [c[:,:,:,:,n-1],a[:,:,:,:,n-1]]
            legs = [[-1,-3,1,2],[-2,-4,2,1]]
            l1_0f[:,:,:,:,n-2] = ncon(tensors,legs)
            tensors = [c[:,:,:,:,n-1],a[:,:,:,:,n-1],c[:,:,:,:,n-1]]
            legs = [[-1,-4,1,2],[-2,-5,2,3],[-3,-6,3,1]]
            l2f[:,:,:,:,:,:,n-2] = ncon(tensors,legs)
            for x in range(n-2,0,-1):
                tensors = [c[:,:,:,:,x],b[:,:,:,:,x],l1f[:,:,:,:,x]]
                legs = [[-1,3,1,2],[-2,4,2,1],[3,4,-3,-4]]
                l1f[:,:,:,:,x-1] = ncon(tensors,legs)
                tensors = [c[:,:,:,:,x],a[:,:,:,:,x],l1_0f[:,:,:,:,x]]
                legs = [[-1,3,1,2],[-2,4,2,1],[3,4,-3,-4]]
                l1_0f[:,:,:,:,x-1] = ncon(tensors,legs)
                tensors = [c[:,:,:,:,x],a[:,:,:,:,x],c[:,:,:,:,x],l2f[:,:,:,:,:,:,x]]
                legs = [[-1,4,1,2],[-2,5,2,3],[-3,6,3,1],[4,5,6,-4,-5,-6]]
                l2f[:,:,:,:,:,:,x-1] = ncon(tensors,legs)
            tensors = [b[:,:,:,:,0],l1f[:,:,:,:,0]]
            legs = [[2,1,-4,-3],[-2,1,-1,2]]
            l1 = ncon(tensors,legs)
            l1 = np.reshape(l1,-1,order='F')
            tensors = [a[:,:,:,:,0],l1_0f[:,:,:,:,0]]
            legs = [[2,1,-4,-3],[-2,1,-1,2]]
            l1_0 = ncon(tensors,legs)
            l1_0 = np.reshape(l1_0,-1,order='F')
            tensors = [a[:,:,:,:,0],np.eye(d),l2f[:,:,:,:,:,:,0]]
            legs = [[2,1,-4,-7],[-8,-3],[-2,1,-6,-1,2,-5]]
            l2 = ncon(tensors,legs)
            l2 = np.reshape(l2,(bdl*bdl*d*d,bdl*bdl*d*d),order='F')
            dl2 = l2+l2.T
            dl1 = 2*(l1-l1_0)/epsilon
            dl2pinv = np.linalg.pinv(dl2,tol_fom)
            dl2pinv = (dl2pinv+dl2pinv.T)/2
            cv = dl2pinv @ dl1
            c[:,:,:,:,0] = np.reshape(cv,(bdl,bdl,d,d),order='F')
            if lherm:
                c[:,:,:,:,0] = (c[:,:,:,:,0]+np.conj(np.moveaxis(c[:,:,:,:,0],2,3)))/2
                cv = np.reshape(c[:,:,:,:,0],-1,order='F')
            fom = np.append(fom,np.real(2*cv @ (l1-l1_0)/epsilon - cv @ l2 @ cv))
            tensors = [c[:,:,:,:,0],b[:,:,:,:,0]]
            legs = [[-1,-3,1,2],[-2,-4,2,1]]
            l1c = ncon(tensors,legs)
            tensors = [c[:,:,:,:,0],a[:,:,:,:,0]]
            legs = [[-1,-3,1,2],[-2,-4,2,1]]
            l1_0c = ncon(tensors,legs)
            tensors = [c[:,:,:,:,0],a[:,:,:,:,0],c[:,:,:,:,0]]
            legs = [[-1,-4,1,2],[-2,-5,2,3],[-3,-6,3,1]]
            l2c = ncon(tensors,legs)
            for x in range(1,n-1):
                tensors = [l1c,b[:,:,:,:,x],l1f[:,:,:,:,x]]
                legs = [[3,4,-1,1],[1,2,-4,-3],[-2,2,3,4]]
                l1 = ncon(tensors,legs)
                l1 = np.reshape(l1,-1,order='F')
                tensors = [l1_0c,a[:,:,:,:,x],l1_0f[:,:,:,:,x]]
                legs = [[3,4,-1,1],[1,2,-4,-3],[-2,2,3,4]]
                l1_0 = ncon(tensors,legs)
                l1_0 = np.reshape(l1_0,-1,order='F')
                tensors = [l2c,a[:,:,:,:,x],np.eye(d),l2f[:,:,:,:,:,:,x]]
                legs = [[3,4,5,-1,1,-5],[1,2,-4,-7],[-8,-3],[-2,2,-6,3,4,5]]
                l2 = ncon(tensors,legs)
                l2 = np.reshape(l2,(bdl*bdl*d*d,bdl*bdl*d*d),order='F')
                dl2 = l2+l2.T
                dl1 = 2*(l1-l1_0)/epsilon
                dl2pinv = np.linalg.pinv(dl2,tol_fom)
                dl2pinv = (dl2pinv+dl2pinv.T)/2
                cv = dl2pinv @ dl1
                c[:,:,:,:,x] = np.reshape(cv,(bdl,bdl,d,d),order='F')
                if lherm:
                    c[:,:,:,:,x] = (c[:,:,:,:,x]+np.conj(np.moveaxis(c[:,:,:,:,x],2,3)))/2
                    cv = np.reshape(c[:,:,:,:,x],-1,order='F')
                fom = np.append(fom,np.real(2*cv @ (l1-l1_0)/epsilon - cv @ l2 @ cv))
                tensors = [l1c,c[:,:,:,:,x],b[:,:,:,:,x]]
                legs = [[-1,-2,3,4],[3,-3,1,2],[4,-4,2,1]]
                l1c = ncon(tensors,legs)
                tensors = [l1_0c,c[:,:,:,:,x],a[:,:,:,:,x]]
                legs = [[-1,-2,3,4],[3,-3,1,2],[4,-4,2,1]]
                l1_0c = ncon(tensors,legs)
                tensors = [l2c,c[:,:,:,:,x],a[:,:,:,:,x],c[:,:,:,:,x]]
                legs = [[-1,-2,-3,4,5,6],[4,-4,1,2],[5,-5,2,3],[6,-6,3,1]]
                l2c = ncon(tensors,legs)
            tensors = [l1c,b[:,:,:,:,n-1]]
            legs = [[-2,2,-1,1],[1,2,-4,-3]]
            l1 = ncon(tensors,legs)
            l1 = np.reshape(l1,-1,order='F')
            tensors = [l1_0c,a[:,:,:,:,n-1]]
            legs = [[-2,2,-1,1],[1,2,-4,-3]]
            l1_0 = ncon(tensors,legs)
            l1_0 = np.reshape(l1_0,-1,order='F')
            tensors = [l2c,a[:,:,:,:,n-1],np.eye(d)]
            legs = [[-2,2,-6,-1,1,-5],[1,2,-4,-7],[-8,-3]]
            l2 = ncon(tensors,legs)
            l2 = np.reshape(l2,(bdl*bdl*d*d,bdl*bdl*d*d),order='F')
            dl2 = l2+l2.T
            dl1 = 2*(l1-l1_0)/epsilon
            dl2pinv = np.linalg.pinv(dl2,tol_fom)
            dl2pinv = (dl2pinv+dl2pinv.T)/2
            cv = dl2pinv @ dl1
            c[:,:,:,:,n-1] = np.reshape(cv,(bdl,bdl,d,d),order='F')
            if lherm:
                c[:,:,:,:,n-1] = (c[:,:,:,:,n-1]+np.conj(np.moveaxis(c[:,:,:,:,n-1],2,3)))/2
                cv = np.reshape(c[:,:,:,:,n-1],-1,order='F')
            fom = np.append(fom,np.real(2*cv @ (l1-l1_0)/epsilon - cv @ l2 @ cv))
            iter_fom += 1
            if iter_fom >= 2 and all(fom[-2*n:] > 0) and np.std(fom[-2*n:])/np.mean(fom[-2*n:]) <= relunc_fom:
                break
        fomval = fom[-1]
    return fomval,c


def fin2_FoMD_OBC_optm(c2d,cd,cpd,epsilon,a0,imprecision=10**-2):
    """
    Optimization of FoMD over MPS for initial wave function. Function for finite size systems with OBC. Version with two dual SLDs separated by epsilon.
    
    Parameters:
      c2d: MPO for the square of the dual of the SLD at the value of estimated parameter phi=-phi_0, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
      cd: MPO for the dual of the SLD at the value of estimated parameter phi=-phi_0, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
      cpd: MPO for the dual of the SLD at the value of estimated parameter phi=-(phi_0+epsilon), expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
      epsilon: value of a separation between estimated parameters encoded in cd and cpd, float
      a0: MPS for the initial wave function, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
      imprecision: expected imprecision of the end results, default value is 10**-2
    
    Returns:
      fomdval: optimal value of FoMD
      a0: optimal MPS for the initial wave function
    """
    n = len(a0)
    if n == 1:
        if np.shape(c2d[0])[0] == 1 and np.shape(cpd[0])[0] == 1 and np.shape(a0[0])[0] == 1:
            d = np.shape(a0[0])[2]
            tensors = [c2d[0][0,0,:,:]]
            legs = [[-1,-2]]
            l2d = ncon(tensors,legs)
            l2d = np.reshape(l2d,(d,d),order='F')
            tensors = [cpd[0][0,0,:,:]]
            legs = [[-1,-2]]
            lpd = ncon(tensors,legs)
            lpd = np.reshape(lpd,(d,d),order='F')
            tensors = [cd[0][0,0,:,:]]
            legs = [[-1,-2]]
            ld = ncon(tensors,legs)
            ld = np.reshape(ld,(d,d),order='F')
            eiginput = 2*(lpd-ld)/epsilon-l2d
            eiginput = (eiginput+np.conj(eiginput).T)/2
            fomdval,a0v = np.linalg.eig(eiginput)
            position = np.argmax(np.real(fomdval))
            a0v = np.reshape(a0v[:,position],-1,order='F')
            a0v = a0v/np.sqrt(np.abs(np.conj(a0v) @ a0v))
            a0[0][0,0,:] = np.reshape(a0v,(d),order='F')
            fomdval = np.real(fomdval[position])
        else:
            warnings.warn('Tensor networks with OBC and length one have to have bond dimension equal to one.')
    else:
        relunc_fomd = 0.1*imprecision
        l2df = [0]*n
        lpdf = [0]*n
        ldf = [0]*n
        fomd = np.array([])
        iter_fomd = 0
        while True:
            tensors = [np.conj(a0[n-1]),c2d[n-1],a0[n-1]]
            legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
            l2df[n-2] = ncon(tensors,legs)
            l2df[n-2] = l2df[n-2][:,:,:,0,0,0]
            tensors = [np.conj(a0[n-1]),cpd[n-1],a0[n-1]]
            legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
            lpdf[n-2] = ncon(tensors,legs)
            lpdf[n-2] = lpdf[n-2][:,:,:,0,0,0]
            tensors = [np.conj(a0[n-1]),cd[n-1],a0[n-1]]
            legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
            ldf[n-2] = ncon(tensors,legs)
            ldf[n-2] = ldf[n-2][:,:,:,0,0,0]
            for x in range(n-2,0,-1):
                tensors = [np.conj(a0[x]),c2d[x],a0[x],l2df[x]]
                legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5]]
                l2df[x-1] = ncon(tensors,legs)
                tensors = [np.conj(a0[x]),cpd[x],a0[x],lpdf[x]]
                legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5]]
                lpdf[x-1] = ncon(tensors,legs)
                tensors = [np.conj(a0[x]),cd[x],a0[x],ldf[x]]
                legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5]]
                ldf[x-1] = ncon(tensors,legs)
            bdpsi1,bdpsi2,d = np.shape(a0[0])
            tensors = [c2d[0],l2df[0]]
            legs = [[-7,1,-3,-6],[-2,1,-5]]
            l2d = ncon(tensors,legs)
            l2d = np.reshape(l2d,(bdpsi1*bdpsi2*d,bdpsi1*bdpsi2*d),order='F')
            tensors = [cpd[0],lpdf[0]]
            legs = [[-7,1,-3,-6],[-2,1,-5]]
            lpd = ncon(tensors,legs)
            lpd = np.reshape(lpd,(bdpsi1*bdpsi2*d,bdpsi1*bdpsi2*d),order='F')
            tensors = [cd[0],ldf[0]]
            legs = [[-7,1,-3,-6],[-2,1,-5]]
            ld = ncon(tensors,legs)
            ld = np.reshape(ld,(bdpsi1*bdpsi2*d,bdpsi1*bdpsi2*d),order='F')
            eiginput = 2*(lpd-ld)/epsilon-l2d
            eiginput = (eiginput+np.conj(eiginput).T)/2
            fomdval,a0v = np.linalg.eig(eiginput)
            position = np.argmax(np.real(fomdval))
            a0v = np.reshape(a0v[:,position],-1,order='F')
            a0v = a0v/np.sqrt(np.abs(np.conj(a0v) @ a0v))
            a0[0] = np.reshape(a0v,(bdpsi1,bdpsi2,d),order='F')
            fomd = np.append(fomd,np.real(fomdval[position]))
            a0[0] = np.moveaxis(a0[0],2,0)
            a0[0] = np.reshape(a0[0],(d*bdpsi1,bdpsi2),order='F')
            u,s,vh = np.linalg.svd(a0[0],full_matrices=False)
            a0[0] = np.reshape(u,(d,bdpsi1,np.shape(s)[0]),order='F')
            a0[0] = np.moveaxis(a0[0],0,2)
            tensors = [np.diag(s) @ vh,a0[1]]
            legs = [[-1,1],[1,-2,-3]]
            a0[1] = ncon(tensors,legs)
            tensors = [np.conj(a0[0]),c2d[0],a0[0]]
            legs = [[-4,-1,1],[-5,-2,1,2],[-6,-3,2]]
            l2dc = ncon(tensors,legs)
            l2dc = l2dc[:,:,:,0,0,0]
            tensors = [np.conj(a0[0]),cpd[0],a0[0]]
            legs = [[-4,-1,1],[-5,-2,1,2],[-6,-3,2]]
            lpdc = ncon(tensors,legs)
            lpdc = lpdc[:,:,:,0,0,0]
            tensors = [np.conj(a0[0]),cd[0],a0[0]]
            legs = [[-4,-1,1],[-5,-2,1,2],[-6,-3,2]]
            ldc = ncon(tensors,legs)
            ldc = ldc[:,:,:,0,0,0]
            for x in range(1,n-1):
                bdpsi1,bdpsi2,d = np.shape(a0[x])
                tensors = [l2dc,c2d[x],l2df[x]]
                legs = [[-1,1,-4],[1,2,-3,-6],[-2,2,-5]]
                l2d = ncon(tensors,legs)
                l2d = np.reshape(l2d,(bdpsi1*bdpsi2*d,bdpsi1*bdpsi2*d),order='F')
                tensors = [lpdc,cpd[x],lpdf[x]]
                legs = [[-1,1,-4],[1,2,-3,-6],[-2,2,-5]]
                lpd = ncon(tensors,legs)
                lpd = np.reshape(lpd,(bdpsi1*bdpsi2*d,bdpsi1*bdpsi2*d),order='F')
                tensors = [ldc,cd[x],ldf[x]]
                legs = [[-1,1,-4],[1,2,-3,-6],[-2,2,-5]]
                ld = ncon(tensors,legs)
                ld = np.reshape(ld,(bdpsi1*bdpsi2*d,bdpsi1*bdpsi2*d),order='F')
                eiginput = 2*(lpd-ld)/epsilon-l2d
                eiginput = (eiginput+np.conj(eiginput).T)/2
                fomdval,a0v = np.linalg.eig(eiginput)
                position = np.argmax(np.real(fomdval))
                a0v = np.reshape(a0v[:,position],-1,order='F')
                a0v = a0v/np.sqrt(np.abs(np.conj(a0v) @ a0v))
                a0[x] = np.reshape(a0v,(bdpsi1,bdpsi2,d),order='F')
                fomd = np.append(fomd,np.real(fomdval[position]))
                a0[x] = np.moveaxis(a0[x],2,0)
                a0[x] = np.reshape(a0[x],(d*bdpsi1,bdpsi2),order='F')
                u,s,vh = np.linalg.svd(a0[x],full_matrices=False)
                a0[x] = np.reshape(u,(d,bdpsi1,np.shape(s)[0]),order='F')
                a0[x] = np.moveaxis(a0[x],0,2)
                tensors = [np.diag(s) @ vh,a0[x+1]]
                legs = [[-1,1],[1,-2,-3]]
                a0[x+1] = ncon(tensors,legs)
                tensors = [l2dc,np.conj(a0[x]),c2d[x],a0[x]]
                legs = [[3,4,5],[3,-1,1],[4,-2,1,2],[5,-3,2]]
                l2dc = ncon(tensors,legs)
                tensors = [lpdc,np.conj(a0[x]),cpd[x],a0[x]]
                legs = [[3,4,5],[3,-1,1],[4,-2,1,2],[5,-3,2]]
                lpdc = ncon(tensors,legs)
                tensors = [ldc,np.conj(a0[x]),cd[x],a0[x]]
                legs = [[3,4,5],[3,-1,1],[4,-2,1,2],[5,-3,2]]
                ldc = ncon(tensors,legs)
            bdpsi1,bdpsi2,d = np.shape(a0[n-1])
            tensors = [l2dc,c2d[n-1]]
            legs = [[-1,1,-4],[1,-7,-3,-6]]
            l2d = ncon(tensors,legs)
            l2d = np.reshape(l2d,(bdpsi1*bdpsi2*d,bdpsi1*bdpsi2*d),order='F')
            tensors = [lpdc,cpd[n-1]]
            legs = [[-1,1,-4],[1,-7,-3,-6]]
            lpd = ncon(tensors,legs)
            lpd = np.reshape(lpd,(bdpsi1*bdpsi2*d,bdpsi1*bdpsi2*d),order='F')
            tensors = [ldc,cd[n-1]]
            legs = [[-1,1,-4],[1,-7,-3,-6]]
            ld = ncon(tensors,legs)
            ld = np.reshape(ld,(bdpsi1*bdpsi2*d,bdpsi1*bdpsi2*d),order='F')
            eiginput = 2*(lpd-ld)/epsilon-l2d
            eiginput = (eiginput+np.conj(eiginput).T)/2
            fomdval,a0v = np.linalg.eig(eiginput)
            position = np.argmax(np.real(fomdval))
            a0v = np.reshape(a0v[:,position],-1,order='F')
            a0v = a0v/np.sqrt(np.abs(np.conj(a0v) @ a0v))
            a0[n-1] = np.reshape(a0v,(bdpsi1,bdpsi2,d),order='F')
            fomd = np.append(fomd,np.real(fomdval[position]))
            iter_fomd += 1
            for x in range(n-1,0,-1):
                bdpsi1,bdpsi2,d = np.shape(a0[x])
                a0[x] = np.moveaxis(a0[x],2,1)
                a0[x] = np.reshape(a0[x],(bdpsi1,d*bdpsi2),order='F')
                u,s,vh = np.linalg.svd(a0[x],full_matrices=False)
                a0[x] = np.reshape(vh,(np.shape(s)[0],d,bdpsi2),order='F')
                a0[x] = np.moveaxis(a0[x],1,2)
                tensors = [a0[x-1],u @ np.diag(s)]
                legs = [[-1,1,-3],[1,-2]]
                a0[x-1] = ncon(tensors,legs)
            if iter_fomd >= 2 and all(fomd[-2*n:] > 0) and np.std(fomd[-2*n:])/np.mean(fomd[-2*n:]) <= relunc_fomd:
                break
        fomdval = fomd[-1]
    return fomdval,a0


def fin2_FoMD_PBC_optm(c2d,cd,cpd,epsilon,a0,imprecision=10**-2):
    """
    Optimization of FoMD over MPS for initial wave function. Function for finite size systems with PBC. Version with two dual SLDs separated by epsilon.
    
    Parameters:
      c2d: MPO for square of dual of SLD at the value of estimated parameter phi=-phi_0, expected ndarray of a shape (bd,bd,d,d,n)
      cd: MPO for dual of SLD at the value of estimated parameter phi=-phi_0, expected ndarray of a shape (bd,bd,d,d,n)
      cpd: MPO for dual of SLD at the value of estimated parameter phi=-(phi_0+epsilon), expected ndarray of a shape (bd,bd,d,d,n)
      epsilon: value of a separation between estimated parameters encoded in cd and cpd, float
      a0: MPS for initial wave function, expected ndarray of a shape (bd,bd,d,n)
      imprecision: expected imprecision of the end results, default value is 10**-2
    
    Returns:
      fomdval: optimal value of FoMD
      a0: optimal MPS for initial wave function
    """
    n = np.shape(c2d)[4]
    d = np.shape(c2d)[2]
    bdl2d = np.shape(c2d)[0]
    bdlpd = np.shape(cpd)[0]
    bdld = np.shape(cd)[0]
    bdpsi = np.shape(a0)[0]
    tol_fomd = 0.1*imprecision/n**2
    if n == 1:
        tensors = [c2d[:,:,:,:,0],np.eye(bdpsi),np.eye(bdpsi)]
        legs = [[1,1,-3,-6],[-2,-1],[-5,-4]]
        l2d = ncon(tensors,legs)
        l2d = np.reshape(l2d,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
        tensors = [cpd[:,:,:,:,0],np.eye(bdpsi),np.eye(bdpsi)]
        legs = [[1,1,-3,-6],[-2,-1],[-5,-4]]
        lpd = ncon(tensors,legs)
        lpd = np.reshape(lpd,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
        tensors = [cd[:,:,:,:,0],np.eye(bdpsi),np.eye(bdpsi)]
        legs = [[1,1,-3,-6],[-2,-1],[-5,-4]]
        ld = ncon(tensors,legs)
        ld = np.reshape(ld,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
        tensors = [np.eye(bdpsi),np.eye(bdpsi)]
        legs = [[-2,-1],[-4,-3]]
        psinorm = ncon(tensors,legs)
        psinorm = np.reshape(psinorm,(bdpsi*bdpsi,bdpsi*bdpsi),order='F')
        psinorm = (psinorm+np.conj(psinorm).T)/2
        psinormpinv = np.linalg.pinv(psinorm,tol_fomd,hermitian=True)
        psinormpinv = (psinormpinv+np.conj(psinormpinv).T)/2
        psinormpinv = np.kron(np.eye(d),psinormpinv)
        eiginput = 2*(lpd-ld)/epsilon-l2d
        eiginput = (eiginput+np.conj(eiginput).T)/2
        eiginput = psinormpinv @ eiginput
        fomdval,a0v = np.linalg.eig(eiginput)
        position = np.argmax(np.real(fomdval))
        a0v = np.reshape(a0v[:,position],-1,order='F')
        a0v = a0v/np.sqrt(np.abs(np.conj(a0v) @ np.kron(np.eye(d),psinorm) @ a0v))
        a0[:,:,:,0] = np.reshape(a0v,(bdpsi,bdpsi,d),order='F')
        fomdval = np.real(fomdval[position])
    else:
        relunc_fomd = 0.1*imprecision
        l2df = np.zeros((bdpsi,bdl2d,bdpsi,bdpsi,bdl2d,bdpsi,n-1),dtype=complex)
        lpdf = np.zeros((bdpsi,bdlpd,bdpsi,bdpsi,bdlpd,bdpsi,n-1),dtype=complex)
        ldf = np.zeros((bdpsi,bdld,bdpsi,bdpsi,bdld,bdpsi,n-1),dtype=complex)
        psinormf = np.zeros((bdpsi,bdpsi,bdpsi,bdpsi,n-1),dtype=complex)
        fomd = np.array([])
        iter_fomd = 0
        while True:
            tensors = [np.conj(a0[:,:,:,n-1]),c2d[:,:,:,:,n-1],a0[:,:,:,n-1]]
            legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
            l2df[:,:,:,:,:,:,n-2] = ncon(tensors,legs)
            tensors = [np.conj(a0[:,:,:,n-1]),cpd[:,:,:,:,n-1],a0[:,:,:,n-1]]
            legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
            lpdf[:,:,:,:,:,:,n-2] = ncon(tensors,legs)
            tensors = [np.conj(a0[:,:,:,n-1]),cd[:,:,:,:,n-1],a0[:,:,:,n-1]]
            legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
            ldf[:,:,:,:,:,:,n-2] = ncon(tensors,legs)
            tensors = [np.conj(a0[:,:,:,n-1]),a0[:,:,:,n-1]]
            legs = [[-1,-3,1],[-2,-4,1]]
            psinormf[:,:,:,:,n-2] = ncon(tensors,legs)
            for x in range(n-2,0,-1):
                tensors = [np.conj(a0[:,:,:,x]),c2d[:,:,:,:,x],a0[:,:,:,x],l2df[:,:,:,:,:,:,x]]
                legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5,-4,-5,-6]]
                l2df[:,:,:,:,:,:,x-1] = ncon(tensors,legs)
                tensors = [np.conj(a0[:,:,:,x]),cpd[:,:,:,:,x],a0[:,:,:,x],lpdf[:,:,:,:,:,:,x]]
                legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5,-4,-5,-6]]
                lpdf[:,:,:,:,:,:,x-1] = ncon(tensors,legs)
                tensors = [np.conj(a0[:,:,:,x]),cd[:,:,:,:,x],a0[:,:,:,x],ldf[:,:,:,:,:,:,x]]
                legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5,-4,-5,-6]]
                ldf[:,:,:,:,:,:,x-1] = ncon(tensors,legs)
                tensors = [np.conj(a0[:,:,:,x]),a0[:,:,:,x],psinormf[:,:,:,:,x]]
                legs = [[-1,2,1],[-2,3,1],[2,3,-3,-4]]
                psinormf[:,:,:,:,x-1] = ncon(tensors,legs)
            tensors = [c2d[:,:,:,:,0],l2df[:,:,:,:,:,:,0]]
            legs = [[2,1,-3,-6],[-2,1,-5,-1,2,-4]]
            l2d = ncon(tensors,legs)
            l2d = np.reshape(l2d,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
            tensors = [cpd[:,:,:,:,0],lpdf[:,:,:,:,:,:,0]]
            legs = [[2,1,-3,-6],[-2,1,-5,-1,2,-4]]
            lpd = ncon(tensors,legs)
            lpd = np.reshape(lpd,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
            tensors = [cd[:,:,:,:,0],ldf[:,:,:,:,:,:,0]]
            legs = [[2,1,-3,-6],[-2,1,-5,-1,2,-4]]
            ld = ncon(tensors,legs)
            ld = np.reshape(ld,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
            tensors = [psinormf[:,:,:,:,0]]
            legs = [[-2,-4,-1,-3]]
            psinorm = ncon(tensors,legs)
            psinorm = np.reshape(psinorm,(bdpsi*bdpsi,bdpsi*bdpsi),order='F')
            psinorm = (psinorm+np.conj(psinorm).T)/2
            psinormpinv = np.linalg.pinv(psinorm,tol_fomd,hermitian=True)
            psinormpinv = (psinormpinv+np.conj(psinormpinv).T)/2
            psinormpinv = np.kron(np.eye(d),psinormpinv)
            eiginput = 2*(lpd-ld)/epsilon-l2d
            eiginput = (eiginput+np.conj(eiginput).T)/2
            eiginput = psinormpinv @ eiginput
            fomdval,a0v = np.linalg.eig(eiginput)
            position = np.argmax(np.real(fomdval))
            a0v = np.reshape(a0v[:,position],-1,order='F')
            a0v = a0v/np.sqrt(np.abs(np.conj(a0v) @ np.kron(np.eye(d),psinorm) @ a0v))
            a0[:,:,:,0] = np.reshape(a0v,(bdpsi,bdpsi,d),order='F')
            fomd = np.append(fomd,np.real(fomdval[position]))
            tensors = [np.conj(a0[:,:,:,0]),c2d[:,:,:,:,0],a0[:,:,:,0]]
            legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
            l2dc = ncon(tensors,legs)
            tensors = [np.conj(a0[:,:,:,0]),cpd[:,:,:,:,0],a0[:,:,:,0]]
            legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
            lpdc = ncon(tensors,legs)
            tensors = [np.conj(a0[:,:,:,0]),cd[:,:,:,:,0],a0[:,:,:,0]]
            legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
            ldc = ncon(tensors,legs)
            tensors = [np.conj(a0[:,:,:,0]),a0[:,:,:,0]]
            legs = [[-1,-3,1],[-2,-4,1]]
            psinormc = ncon(tensors,legs)
            for x in range(1,n-1):
                tensors = [l2dc,c2d[:,:,:,:,x],l2df[:,:,:,:,:,:,x]]
                legs = [[3,4,5,-1,1,-4],[1,2,-3,-6],[-2,2,-5,3,4,5]]
                l2d = ncon(tensors,legs)
                l2d = np.reshape(l2d,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
                tensors = [lpdc,cpd[:,:,:,:,x],lpdf[:,:,:,:,:,:,x]]
                legs = [[3,4,5,-1,1,-4],[1,2,-3,-6],[-2,2,-5,3,4,5]]
                lpd = ncon(tensors,legs)
                lpd = np.reshape(lpd,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
                tensors = [ldc,cd[:,:,:,:,x],ldf[:,:,:,:,:,:,x]]
                legs = [[3,4,5,-1,1,-4],[1,2,-3,-6],[-2,2,-5,3,4,5]]
                ld = ncon(tensors,legs)
                ld = np.reshape(ld,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
                tensors = [psinormc,psinormf[:,:,:,:,x]]
                legs = [[1,2,-1,-3],[-2,-4,1,2]]
                psinorm = ncon(tensors,legs)
                psinorm = np.reshape(psinorm,(bdpsi*bdpsi,bdpsi*bdpsi),order='F')
                psinorm = (psinorm+np.conj(psinorm).T)/2
                psinormpinv = np.linalg.pinv(psinorm,tol_fomd,hermitian=True)
                psinormpinv = (psinormpinv+np.conj(psinormpinv).T)/2
                psinormpinv = np.kron(np.eye(d),psinormpinv)
                eiginput = 2*(lpd-ld)/epsilon-l2d
                eiginput = (eiginput+np.conj(eiginput).T)/2
                eiginput = psinormpinv @ eiginput
                fomdval,a0v = np.linalg.eig(eiginput)
                position = np.argmax(np.real(fomdval))
                a0v = np.reshape(a0v[:,position],-1,order='F')
                a0v = a0v/np.sqrt(np.abs(np.conj(a0v) @ np.kron(np.eye(d),psinorm) @ a0v))
                a0[:,:,:,x] = np.reshape(a0v,(bdpsi,bdpsi,d),order='F')
                fomd = np.append(fomd,np.real(fomdval[position]))
                tensors = [l2dc,np.conj(a0[:,:,:,x]),c2d[:,:,:,:,x],a0[:,:,:,x]]
                legs = [[-1,-2,-3,3,4,5],[3,-4,1],[4,-5,1,2],[5,-6,2]]
                l2dc = ncon(tensors,legs)
                tensors = [lpdc,np.conj(a0[:,:,:,x]),cpd[:,:,:,:,x],a0[:,:,:,x]]
                legs = [[-1,-2,-3,3,4,5],[3,-4,1],[4,-5,1,2],[5,-6,2]]
                lpdc = ncon(tensors,legs)
                tensors = [ldc,np.conj(a0[:,:,:,x]),cd[:,:,:,:,x],a0[:,:,:,x]]
                legs = [[-1,-2,-3,3,4,5],[3,-4,1],[4,-5,1,2],[5,-6,2]]
                ldc = ncon(tensors,legs)
                tensors = [psinormc,np.conj(a0[:,:,:,x]),a0[:,:,:,x]]
                legs = [[-1,-2,2,3],[2,-3,1],[3,-4,1]]
                psinormc = ncon(tensors,legs)
            tensors = [l2dc,c2d[:,:,:,:,n-1]]
            legs = [[-2,2,-5,-1,1,-4],[1,2,-3,-6]]
            l2d = ncon(tensors,legs)
            l2d = np.reshape(l2d,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
            tensors = [lpdc,cpd[:,:,:,:,n-1]]
            legs = [[-2,2,-5,-1,1,-4],[1,2,-3,-6]]
            lpd = ncon(tensors,legs)
            lpd = np.reshape(lpd,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
            tensors = [ldc,cd[:,:,:,:,n-1]]
            legs = [[-2,2,-5,-1,1,-4],[1,2,-3,-6]]
            ld = ncon(tensors,legs)
            ld = np.reshape(ld,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
            tensors = [psinormc]
            legs = [[-2,-4,-1,-3]]
            psinorm = ncon(tensors,legs)
            psinorm = np.reshape(psinorm,(bdpsi*bdpsi,bdpsi*bdpsi),order='F')
            psinorm = (psinorm+np.conj(psinorm).T)/2
            psinormpinv = np.linalg.pinv(psinorm,tol_fomd,hermitian=True)
            psinormpinv = (psinormpinv+np.conj(psinormpinv).T)/2
            psinormpinv = np.kron(np.eye(d),psinormpinv)
            eiginput = 2*(lpd-ld)/epsilon-l2d
            eiginput = (eiginput+np.conj(eiginput).T)/2
            eiginput = psinormpinv @ eiginput
            fomdval,a0v = np.linalg.eig(eiginput)
            position = np.argmax(np.real(fomdval))
            a0v = np.reshape(a0v[:,position],-1,order='F')
            a0v = a0v/np.sqrt(np.abs(np.conj(a0v) @ np.kron(np.eye(d),psinorm) @ a0v))
            a0[:,:,:,n-1] = np.reshape(a0v,(bdpsi,bdpsi,d),order='F')
            fomd = np.append(fomd,np.real(fomdval[position]))
            iter_fomd += 1
            if iter_fomd >= 2 and all(fomd[-2*n:] > 0) and np.std(fomd[-2*n:])/np.mean(fomd[-2*n:]) <= relunc_fomd:
                break
        fomdval = fomd[-1]
    return fomdval,a0


def fin2_FoM_OBC_val(a,b,epsilon,c):
    """
    Calculate value of FoM. Function for finite size systems with OBC. Version with two states separated by epsilon.
    
    Parameters:
      a: MPO for density matrix at the value of estimated parameter phi=phi_0, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
      b: MPO for density matrix at the value of estimated parameter phi=phi_0+epsilon, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
      epsilon: value of a separation between estimated parameters encoded in a and b, float
      c: MPO for SLD, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
    
    Returns:
      fomval: value of FoM
    """
    n = len(c)
    if n == 1:
        if np.shape(a[0])[0] == 1 and np.shape(b[0])[0] == 1 and np.shape(c[0])[0] == 1:
            tensors = [c[0][0,0,:,:],b[0][0,0,:,:]]
            legs = [[1,2],[2,1]]
            l1 = ncon(tensors,legs)
            tensors = [c[0][0,0,:,:],a[0][0,0,:,:]]
            legs = [[1,2],[2,1]]
            l1_0 = ncon(tensors,legs)
            tensors = [c[0][0,0,:,:],a[0][0,0,:,:],c[0][0,0,:,:]]
            legs = [[1,2],[2,3],[3,1]]
            l2 = ncon(tensors,legs)
            fomval = 2*(l1-l1_0)/epsilon-l2
        else:
            warnings.warn('Tensor networks with OBC and length one have to have bond dimension equal to one.')
    else:
        tensors = [c[n-1],b[n-1]]
        legs = [[-1,-3,1,2],[-2,-4,2,1]]
        l1 = ncon(tensors,legs)
        l1 = l1[:,:,0,0]
        tensors = [c[n-1],a[n-1]]
        legs = [[-1,-3,1,2],[-2,-4,2,1]]
        l1_0 = ncon(tensors,legs)
        l1_0 = l1_0[:,:,0,0]
        tensors = [c[n-1],a[n-1],c[n-1]]
        legs = [[-1,-4,1,2],[-2,-5,2,3],[-3,-6,3,1]]
        l2 = ncon(tensors,legs)
        l2 = l2[:,:,:,0,0,0]
        for x in range(n-2,0,-1):
            tensors = [c[x],b[x],l1]
            legs = [[-1,3,1,2],[-2,4,2,1],[3,4]]
            l1 = ncon(tensors,legs)
            tensors = [c[x],a[x],l1_0]
            legs = [[-1,3,1,2],[-2,4,2,1],[3,4]]
            l1_0 = ncon(tensors,legs)
            tensors = [c[x],a[x],c[x],l2]
            legs = [[-1,4,1,2],[-2,5,2,3],[-3,6,3,1],[4,5,6]]
            l2 = ncon(tensors,legs)
        tensors = [c[0],b[0],l1]
        legs = [[-1,3,1,2],[-2,4,2,1],[3,4]]
        l1 = ncon(tensors,legs)
        l1 = float(l1)
        tensors = [c[0],a[0],l1_0]
        legs = [[-1,3,1,2],[-2,4,2,1],[3,4]]
        l1_0 = ncon(tensors,legs)
        l1_0 = float(l1_0)
        tensors = [c[0],a[0],c[0],l2]
        legs = [[-1,4,1,2],[-2,5,2,3],[-3,6,3,1],[4,5,6]]
        l2 = ncon(tensors,legs)
        l2 = float(l2)
        fomval = 2*(l1-l1_0)/epsilon-l2
    return fomval


def fin2_FoM_PBC_val(a,b,epsilon,c):
    """
    Calculate value of FoM. Function for finite size systems with PBC. Version with two states separated by epsilon.
    
    Parameters:
      a: MPO for density matrix at the value of estimated parameter phi=phi_0, expected ndarray of a shape (bd,bd,d,d,n)
      b: MPO for density matrix at the value of estimated parameter phi=phi_0+epsilon, expected ndarray of a shape (bd,bd,d,d,n)
      epsilon: value of a separation between estimated parameters encoded in a and b, float
      c: MPO for SLD, expected ndarray of a shape (bd,bd,d,d,n)
    
    Returns:
      fomval: value of FoM
    """
    n = np.shape(a)[4]
    if n == 1:
        tensors = [c[:,:,:,:,0],b[:,:,:,:,0]]
        legs = [[3,3,1,2],[4,4,2,1]]
        l1 = ncon(tensors,legs)
        tensors = [c[:,:,:,:,0],a[:,:,:,:,0]]
        legs = [[3,3,1,2],[4,4,2,1]]
        l1_0 = ncon(tensors,legs)
        tensors = [c[:,:,:,:,0],a[:,:,:,:,0],c[:,:,:,:,0]]
        legs = [[4,4,1,2],[5,5,2,3],[6,6,3,1]]
        l2 = ncon(tensors,legs)
        fomval = 2*(l1-l1_0)/epsilon-l2
    else:
        tensors = [c[:,:,:,:,n-1],b[:,:,:,:,n-1]]
        legs = [[-1,-3,1,2],[-2,-4,2,1]]
        l1 = ncon(tensors,legs)
        tensors = [c[:,:,:,:,n-1],a[:,:,:,:,n-1]]
        legs = [[-1,-3,1,2],[-2,-4,2,1]]
        l1_0 = ncon(tensors,legs)
        tensors = [c[:,:,:,:,n-1],a[:,:,:,:,n-1],c[:,:,:,:,n-1]]
        legs = [[-1,-4,1,2],[-2,-5,2,3],[-3,-6,3,1]]
        l2 = ncon(tensors,legs)
        for x in range(n-2,0,-1):
            tensors = [c[:,:,:,:,x],b[:,:,:,:,x],l1]
            legs = [[-1,3,1,2],[-2,4,2,1],[3,4,-3,-4]]
            l1 = ncon(tensors,legs)
            tensors = [c[:,:,:,:,x],a[:,:,:,:,x],l1_0]
            legs = [[-1,3,1,2],[-2,4,2,1],[3,4,-3,-4]]
            l1_0 = ncon(tensors,legs)
            tensors = [c[:,:,:,:,x],a[:,:,:,:,x],c[:,:,:,:,x],l2]
            legs = [[-1,4,1,2],[-2,5,2,3],[-3,6,3,1],[4,5,6,-4,-5,-6]]
            l2 = ncon(tensors,legs)
        tensors = [c[:,:,:,:,0],b[:,:,:,:,0],l1]
        legs = [[5,3,1,2],[6,4,2,1],[3,4,5,6]]
        l1 = ncon(tensors,legs)
        tensors = [c[:,:,:,:,0],a[:,:,:,:,0],l1_0]
        legs = [[5,3,1,2],[6,4,2,1],[3,4,5,6]]
        l1_0 = ncon(tensors,legs)
        tensors = [c[:,:,:,:,0],a[:,:,:,:,0],c[:,:,:,:,0],l2]
        legs = [[7,4,1,2],[8,5,2,3],[9,6,3,1],[4,5,6,7,8,9]]
        l2 = ncon(tensors,legs)
        fomval = 2*(l1-l1_0)/epsilon-l2
    return fomval


def fin2_FoMD_OBC_val(c2d,cd,cpd,epsilon,a0):
    """
    Calculate value of FoMD. Function for finite size systems with OBC. Version with two dual SLDs separated by epsilon.
    
    Parameters:
      c2d: MPO for square of dual of SLD at the value of estimated parameter phi=-phi_0, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
      cd: MPO for dual of SLD at the value of estimated parameter phi=-phi_0, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
      cpd: MPO for dual of SLD at the value of estimated parameter phi=-(phi_0+epsilon), expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
      epsilon: value of a separation between estimated parameters encoded in cd and cpd, float
      a0: MPS for initial wave function, expected list of length n of ndarrays of a shape (bd,bd,d,d) (bd can vary between sites)
    
    Returns:
      fomdval: value of FoMD
    """
    n = len(a0)
    if n == 1:
        if np.shape(c2d[0])[0] == 1 and np.shape(cpd[0])[0] == 1 and np.shape(a0[0])[0] == 1:
            tensors = [np.conj(a0[0][0,0,:]),c2d[0][0,0,:,:],a0[0][0,0,:]]
            legs = [[1],[1,2],[2]]
            l2d = ncon(tensors,legs)
            tensors = [np.conj(a0[0][0,0,:]),cpd[0][0,0,:,:],a0[0][0,0,:]]
            legs = [[1],[1,2],[2]]
            lpd = ncon(tensors,legs)
            tensors = [np.conj(a0[0][0,0,:]),cd[0][0,0,:,:],a0[0][0,0,:]]
            legs = [[1],[1,2],[2]]
            ld = ncon(tensors,legs)
            fomdval = 2*(lpd-ld)/epsilon-l2d
        else:
            warnings.warn('Tensor networks with OBC and length one have to have bond dimension equal to one.')
    else:
        tensors = [np.conj(a0[n-1]),c2d[n-1],a0[n-1]]
        legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
        l2d = ncon(tensors,legs)
        l2d = l2d[:,:,:,0,0,0]
        tensors = [np.conj(a0[n-1]),cpd[n-1],a0[n-1]]
        legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
        lpd = ncon(tensors,legs)
        lpd = lpd[:,:,:,0,0,0]
        tensors = [np.conj(a0[n-1]),cd[n-1],a0[n-1]]
        legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
        ld = ncon(tensors,legs)
        ld = ld[:,:,:,0,0,0]
        for x in range(n-2,0,-1):
            tensors = [np.conj(a0[x]),c2d[x],a0[x],l2d]
            legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5]]
            l2d = ncon(tensors,legs)
            tensors = [np.conj(a0[x]),cpd[x],a0[x],lpd]
            legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5]]
            lpd = ncon(tensors,legs)
            tensors = [np.conj(a0[x]),cd[x],a0[x],ld]
            legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5]]
            ld = ncon(tensors,legs)
        tensors = [np.conj(a0[0]),c2d[0],a0[0],l2d]
        legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5]]
        l2d = ncon(tensors,legs)
        l2d = float(l2d)
        tensors = [np.conj(a0[0]),cpd[0],a0[0],lpd]
        legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5]]
        lpd = ncon(tensors,legs)
        lpd = float(lpd)
        tensors = [np.conj(a0[0]),cd[0],a0[0],ld]
        legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5]]
        ld = ncon(tensors,legs)
        ld = float(ld)
        fomdval = 2*(lpd-ld)/epsilon-l2d
    return fomdval


def fin2_FoMD_PBC_val(c2d,cd,cpd,epsilon,a0):
    """
    Calculate value of FoMD. Function for finite size systems with PBC. Version with two dual SLDs separated by epsilon.
    
    Parameters:
      c2d: MPO for square of dual of SLD at the value of estimated parameter phi=-phi_0, expected ndarray of a shape (bd,bd,d,d,n)
      cd: MPO for dual of SLD at the value of estimated parameter phi=-phi_0, expected ndarray of a shape (bd,bd,d,d,n)
      cpd: MPO for dual of SLD at the value of estimated parameter phi=-(phi_0+epsilon), expected ndarray of a shape (bd,bd,d,d,n)
      epsilon: value of a separation between estimated parameters encoded in cd and cpd, float
      a0: MPS for initial wave function, expected ndarray of a shape (bd,bd,d,n)
    
    Returns:
      fomdval: value of FoMD
    """
    n = np.shape(c2d)[4]
    if n == 1:
        tensors = [np.conj(a0[:,:,:,0]),c2d[:,:,:,:,0],a0[:,:,:,0]]
        legs = [[3,3,1],[4,4,1,2],[5,5,2]]
        l2d = ncon(tensors,legs)
        tensors = [np.conj(a0[:,:,:,0]),cpd[:,:,:,:,0],a0[:,:,:,0]]
        legs = [[3,3,1],[4,4,1,2],[5,5,2]]
        lpd = ncon(tensors,legs)
        tensors = [np.conj(a0[:,:,:,0]),cd[:,:,:,:,0],a0[:,:,:,0]]
        legs = [[3,3,1],[4,4,1,2],[5,5,2]]
        ld = ncon(tensors,legs)
        fomdval = 2*(lpd-ld)/epsilon-l2d
    else:
        tensors = [np.conj(a0[:,:,:,n-1]),c2d[:,:,:,:,n-1],a0[:,:,:,n-1]]
        legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
        l2d = ncon(tensors,legs)
        tensors = [np.conj(a0[:,:,:,n-1]),cpd[:,:,:,:,n-1],a0[:,:,:,n-1]]
        legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
        lpd = ncon(tensors,legs)
        tensors = [np.conj(a0[:,:,:,n-1]),cd[:,:,:,:,n-1],a0[:,:,:,n-1]]
        legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
        ld = ncon(tensors,legs)
        for x in range(n-2,0,-1):
            tensors = [np.conj(a0[:,:,:,x]),c2d[:,:,:,:,x],a0[:,:,:,x],l2d]
            legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5,-4,-5,-6]]
            l2d = ncon(tensors,legs)
            tensors = [np.conj(a0[:,:,:,x]),cpd[:,:,:,:,x],a0[:,:,:,x],lpd]
            legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5,-4,-5,-6]]
            lpd = ncon(tensors,legs)
            tensors = [np.conj(a0[:,:,:,x]),cd[:,:,:,:,x],a0[:,:,:,x],ld]
            legs = [[-1,3,1],[-2,4,1,2],[-3,5,2],[3,4,5,-4,-5,-6]]
            ld = ncon(tensors,legs)
        tensors = [np.conj(a0[:,:,:,0]),c2d[:,:,:,:,0],a0[:,:,:,0],l2d]
        legs = [[6,3,1],[7,4,1,2],[8,5,2],[3,4,5,6,7,8]]
        l2d = ncon(tensors,legs)
        tensors = [np.conj(a0[:,:,:,0]),cpd[:,:,:,:,0],a0[:,:,:,0],lpd]
        legs = [[6,3,1],[7,4,1,2],[8,5,2],[3,4,5,6,7,8]]
        lpd = ncon(tensors,legs)
        tensors = [np.conj(a0[:,:,:,0]),cd[:,:,:,:,0],a0[:,:,:,0],ld]
        legs = [[6,3,1],[7,4,1,2],[8,5,2],[3,4,5,6,7,8]]
        ld = ncon(tensors,legs)
        fomdval = 2*(lpd-ld)/epsilon-l2d
    return fomdval


##########################################
#                                        #
#                                        #
# 2 Functions for infinite size systems. #
#                                        #
#                                        #
##########################################


#############################
#                           #
# 2.1 High level functions. #
#                           #
#############################


def inf(so_before_list, h, so_after_list, L_ini=None, psi0_ini=None, imprecision=10**-2, D_L_max=100, D_L_max_forced=False, L_herm=True, D_psi0_max=100, D_psi0_max_forced=False):
    """
    Optimization of the lim_{N --> infinity} QFI/N over operator tilde{L} (in iMPO representation) and wave function psi0 (in iMPS representation) and check of convergence in their bond dimensions. Function for infinite size systems.
    
    User has to provide information about the dynamics by specifying quantum channel. It is assumed that quantum channel is translationally invariant and is build from layers of quantum operations.
    User has to provide one defining for each layer operation as a local superoperator. Those local superoperator have to be input in order of their action on the system.
    Parameter encoding is a stand out quantum operation. It is assumed that parameter encoding acts only once and is unitary so the user have to provide only its generator h.
    Generator h have to be diagonal in computational basis, or in other words it is assumed that local superoperators are expressed in the eigenbasis of h.
    
    Parameters:
      so_before_list: list of ndarrays of a shape (d**(2*k),d**(2*k)) where k describes on how many sites particular local superoperator acts
        List of local superoperators (in order) which act before unitary parameter encoding.
      h: ndarray of a shape (d,d)
        Generator of unitary parameter encoding. Dimension d is the dimension of local Hilbert space (dimension of physical index).
        Generator h have to be diagonal in computational basis, or in other words it is assumed that local superoperators are expressed in the eigenbasis of h.
      so_after_list: list of ndarrays of a shape (d**(2*k),d**(2*k)) where k describes on how many sites particular local superoperator acts
        List of local superoperators (in order) which act after unitary parameter encoding.
      L_ini: ndarray of a shape (D_L,D_L,d,d), optional
        Initial iMPO for tilde{L}.
      psi0_ini: ndarray of a shape (D_psi0,D_psi0,d), optional
        Initial iMPS for psi0.
      imprecision: float, optional
        Expected relative imprecision of the end results.
      D_L_max: integer, optional
        Maximal value of D_L (D_L is bond dimension for iMPO representing tilde{L}).
      D_L_max_forced: bool, optional
        True if D_L_max have to be reached, otherwise False.
      L_herm: bool, optional
        True if Hermitian gauge have to be imposed on iMPO representing tilde{L}, otherwise False.
      D_psi0_max: integer, optional
        Maximal value of D_psi0 (D_psi0 is bond dimension for iMPS representing psi0).
      D_psi0_max_forced: bool, optional
        True if D_psi0_max have to be reached, otherwise False.
    
    Returns:
      result: float
        Optimal value of figure of merit.
      result_m: ndarray
        Matrix describing figure of merit in function of bond dimensions of respectively tilde{L} [rows] and psi0 [columns].
      L: ndarray of a shape (D_L,D_L,d,d)
        Optimal tilde{L} in iMPO representation.
      psi0: ndarray of a shape (D_psi0,D_psi0,d)
        Optimal psi0 in iMPS representation.
    """
    if np.linalg.norm(h - np.diag(np.diag(h))) > 10**-10:
        warnings.warn('Generator h have to be diagonal in computational basis, or in other words it is assumed that local superoperators are expressed in the eigenbasis of h.')
    d = np.shape(h)[0]
    epsilon = 10**-4
    aux = np.kron(h, np.eye(d)) - np.kron(np.eye(d), h)
    z = np.diag(np.exp(-1j * np.diag(aux) * epsilon))
    ch = inf_create_channel(d, so_before_list + so_after_list)
    ch2 = inf_create_channel(d, so_before_list + [z] + so_after_list)
    result, result_m, L, psi0 = inf_gen(d, ch, ch2, epsilon, inf_L_symfun, inf_psi0_symfun, L_ini, psi0_ini, imprecision, D_L_max, D_L_max_forced, L_herm, D_psi0_max, D_psi0_max_forced)
    return result, result_m, L, psi0


def inf_gen(d, ch, ch2, epsilon, symfun_L, symfun_psi0, L_ini=None, psi0_ini=None, imprecision=10**-2, D_L_max=100, D_L_max_forced=False, L_herm=True, D_psi0_max=100, D_psi0_max_forced=False):
    """
    Optimization of the figure of merit (usually interpreted as lim_{N --> infinity} QFI/N) over operator tilde{L} (in iMPO representation) and wave function psi0 (in iMPS representation) and check of convergence in their bond dimensions. Function for infinite size systems.
    
    User has to provide information about the dynamics by specifying two channels separated by small parameter epsilon as superoperators in iMPO representation.
    By definition this infinite approach assumes translation invariance of the problem, other than that there are no constraints on the structure of the channel but the complexity of calculations highly depends on channel's bond dimension.
    
    Parameters:
      d: integer
        Dimension of local Hilbert space (dimension of physical index).
      ch: ndarray of a shape (D_ch,D_ch,d**2,d**2)
        Quantum channel as superoperator in iMPO representation.
      ch2: ndarray of a shape (D_ch2,D_ch2,d**2,d**2)
        Quantum channel as superoperator in iMPO representation for the value of estimated parameter shifted by epsilon in relation to ch.
      epsilon: float
        Value of a separation between estimated parameters encoded in ch and ch2.
      symfun_L: function
        Function which symmetrize iMPO for tilde{L} after each step of otimization (the most simple one would be lambda x: x).
        Choosing good function is key factor for successful optimization in infinite approach.
        TNQMetro package features inf_L_symfun function which performs well in dephasing type problems.
      symfun_psi0: function
        Function which symmetrize iMPS for psi0 after each step of otimization (the most simple one would be lambda x: x).
        Choosing good function is key factor for successful optimization in infinite approach.
        TNQMetro package features inf_psi0_symfun function which performs well in dephasing type problems.
      L_ini: ndarray of a shape (D_L,D_L,d,d), optional
        Initial iMPO for tilde{L}.
      psi0_ini: ndarray of a shape (D_psi0,D_psi0,d), optional
        Initial iMPS for psi0.
      imprecision: float, optional
        Expected relative imprecision of the end results.
      D_L_max: integer, optional
        Maximal value of D_L (D_L is bond dimension for iMPO representing tilde{L}).
      D_L_max_forced: bool, optional
        True if D_L_max have to be reached, otherwise False.
      L_herm: bool, optional
        True if Hermitian gauge have to be imposed on iMPO representing tilde{L}, otherwise False.
      D_psi0_max: integer, optional
        Maximal value of D_psi0 (D_psi0 is bond dimension for iMPS representing psi0).
      D_psi0_max_forced: bool, optional
        True if D_psi0_max have to be reached, otherwise False.
    
    Returns:
      result: float
        Optimal value of figure of merit.
      result_m: ndarray
        Matrix describing figure of merit in function of bond dimensions of respectively tilde{L} [rows] and psi0 [columns].
      L: ndarray of a shape (D_L,D_L,d,d)
        Optimal tilde{L} in iMPO representation.
      psi0: ndarray of a shape (D_psi0,D_psi0,d)
        Optimal psi0 in iMPS representation.
    """
    result, result_m, L, psi0 = inf_FoM_FoMD_optbd(d, ch, ch2, epsilon, symfun_L, symfun_psi0, L_ini, psi0_ini, imprecision, D_L_max, D_L_max_forced, L_herm, D_psi0_max, D_psi0_max_forced)
    return result, result_m, L, psi0


def inf_state(so_before_list, h, so_after_list, rho0, L_ini=None, imprecision=10**-2, D_L_max=100, D_L_max_forced=False, L_herm=True):
    """
    Optimization of the lim_{N --> infinity} QFI/N over operator tilde{L} (in iMPO representation) and check of convergence in its bond dimension. Function for infinite size systems and fixed state of the system.
    
    User has to provide information about the dynamics by specifying quantum channel. It is assumed that quantum channel is translationally invariant and is build from layers of quantum operations.
    User has to provide one defining for each layer operation as a local superoperator. Those local superoperator have to be input in order of their action on the system.
    Parameter encoding is a stand out quantum operation. It is assumed that parameter encoding acts only once and is unitary so the user have to provide only its generator h.
    Generator h have to be diagonal in computational basis, or in other words it is assumed that local superoperators are expressed in the eigenbasis of h.
    
    Parameters:
      so_before_list: list of ndarrays of a shape (d**(2*k),d**(2*k)) where k describes on how many sites particular local superoperator acts
        List of local superoperators (in order) which act before unitary parameter encoding.
      h: ndarray of a shape (d,d)
        Generator of unitary parameter encoding. Dimension d is the dimension of local Hilbert space (dimension of physical index).
        Generator h have to be diagonal in computational basis, or in other words it is assumed that local superoperators are expressed in the eigenbasis of h.
      so_after_list: list of ndarrays of a shape (d**(2*k),d**(2*k)) where k describes on how many sites particular local superoperator acts
        List of local superoperators (in order) which act after unitary parameter encoding.
      rho0: ndarray of a shape (D_rho0,D_rho0,d,d)
        Density matrix describing initial state of the system in iMPO representation.
      L_ini: ndarray of a shape (D_L,D_L,d,d), optional
        Initial iMPO for tilde{L}.
      imprecision: float, optional
        Expected relative imprecision of the end results.
      D_L_max: integer, optional
        Maximal value of D_L (D_L is bond dimension for iMPO representing tilde{L}).
      D_L_max_forced: bool, optional
        True if D_L_max have to be reached, otherwise False.
      L_herm: bool, optional
        True if Hermitian gauge have to be imposed on iMPO representing tilde{L}, otherwise False.
    
    Returns:
      result: float
        Optimal value of figure of merit.
      result_v: ndarray
        Vector describing figure of merit in function of bond dimensions of tilde{L}.
      L: ndarray of a shape (D_L,D_L,d,d)
        Optimal tilde{L} in iMPO representation.
    """
    if np.linalg.norm(h - np.diag(np.diag(h))) > 10**-10:
        warnings.warn('Generator h have to be diagonal in computational basis, or in other words it is assumed that local superoperators are expressed in the eigenbasis of h.')
    d = np.shape(h)[0]
    epsilon = 10**-4
    aux = np.kron(h, np.eye(d)) - np.kron(np.eye(d), h)
    z = np.diag(np.exp(-1j * np.diag(aux) * epsilon))
    ch = inf_create_channel(d, so_before_list + so_after_list)
    ch2 = inf_create_channel(d, so_before_list + [z] + so_after_list)
    rho = channel_acting_on_operator(ch, rho0)
    rho2 = channel_acting_on_operator(ch2, rho0)
    result, result_v, L = inf_state_gen(d, rho, rho2, epsilon, inf_L_symfun, L_ini, imprecision, D_L_max, D_L_max_forced, L_herm)
    return result, result_v, L


def inf_state_gen(d, rho, rho2, epsilon, symfun_L, L_ini=None, imprecision=10**-2, D_L_max=100, D_L_max_forced=False, L_herm=True):
    """
    Optimization of the figure of merit (usually interpreted as lim_{N --> infinity} QFI/N) over operator tilde{L} (in iMPO representation) and check of convergence in its bond dimension. Function for infinite size systems and fixed state of the system.
    
    User has to provide information about the dynamics by specifying two channels separated by small parameter epsilon as superoperators in iMPO representation.
    By definition this infinite approach assumes translation invariance of the problem, other than that there are no constraints on the structure of the channel but the complexity of calculations highly depends on channel's bond dimension.
    
    Parameters:
      d: integer
        Dimension of local Hilbert space (dimension of physical index).
      rho: ndarray of a shape (D_rho,D_rho,d,d)
        Density matrix at the output of quantum channel in iMPO representation.
      rho2: ndarray of a shape (D_rho2,D_rho2,d,d)
        Density matrix at the output of quantum channel in iMPO representation for the value of estimated parameter shifted by epsilon in relation to rho.
      epsilon: float
        Value of a separation between estimated parameters encoded in rho and rho2.
      symfun_L: function
        Function which symmetrize iMPO for tilde{L} after each step of otimization (the most simple one would be lambda x: x).
        Choosing good function is key factor for successful optimization in infinite approach.
        TNQMetro package features inf_L_symfun function which performs well in dephasing type problems.
      L_ini: ndarray of a shape (D_L,D_L,d,d), optional
        Initial iMPO for tilde{L}.
      imprecision: float, optional
        Expected relative imprecision of the end results.
      D_L_max: integer, optional
        Maximal value of D_L (D_L is bond dimension for iMPO representing tilde{L}).
      D_L_max_forced: bool, optional
        True if D_L_max have to be reached, otherwise False.
      L_herm: bool, optional
        True if Hermitian gauge have to be imposed on iMPO representing tilde{L}, otherwise False.
    
    Returns:
      result: float
        Optimal value of figure of merit.
      result_v: ndarray
        Vector describing figure of merit in function of bond dimensions of tilde{L}.
      L: ndarray of a shape (D_L,D_L,d,d)
        Optimal tilde{L} in iMPO representation.
    """
    result, result_v, L = inf_FoM_optbd(d, rho, rho2, epsilon, symfun_L, L_ini, imprecision, D_L_max, D_L_max_forced, L_herm)
    return result, result_v, L


def inf_L_symfun(l):
    """
    Symmetrize function for iMPO representing tilde{L} which performs well in dephasing type problems.
    
    Parameters:
      l: ndarray of a shape (D_L,D_L,d,d)
        iMPO for tilde{L}.
    
    Returns:
      l: ndarray of a shape (D_L,D_L,d,d)
        Symmetrize iMPO for tilde{L}.
    """
    bdl = np.shape(l)[0]
    d = np.shape(l)[2]
    if bdl == 1:
        l = np.reshape(l,(d,d),order='F')
        lmd = np.mean(np.diag(l))
        l = np.imag(l)
        l = (l+np.rot90(l,2).T)/2
        l = lmd*np.eye(d)+1j*l
        l = np.reshape(l,(bdl,bdl,d,d),order='F')
    else:
        for nx in range(d):
            l[:,:,nx,nx] = np.zeros((bdl,bdl),dtype=complex)
            l[0,0,nx,nx] = 1
    return l


def inf_psi0_symfun(p):
    """
    Symmetrize function for iMPS representing psi0 which performs well in dephasing type problems.
    
    Parameters:
      p: ndarray of a shape (D_psi0,D_psi0,d)
        iMPS for psi0.
    
    Returns:
      p: ndarray of a shape (D_psi0,D_psi0,d)
        Symmetrize iMPS for psi0.
    """
    p = (p+np.conj(np.moveaxis(p,0,1)))/2
    p = (p+np.moveaxis(np.flip(p,2),0,1))/2
    p = (p+np.moveaxis(np.rot90(p,2),0,1))/2
    return p


############################
#                          #
# 2.2 Low level functions. #
#                          #
############################


def inf_create_channel(d, so_list, tol=10**-10):
    """
    Creates iMPO for superoperator describing translationally invariant quantum channel from list of local superoperators. Function for infinite size systems.
    
    Local superoperators acting on more then 4 neighbour sites are not currently supported.
    
    Parameters:
      d: integer
        Dimension of local Hilbert space (dimension of physical index).
      so_list: list of ndarrays of a shape (d**(2*k),d**(2*k)) where k describes on how many sites particular local superoperator acts
        List of local superoperators in order of their action on the system.
        Local superoperators acting on more then 4 neighbour sites are not currently supported.
      tol: float, optional
        Factor which after multiplication by the highest singular value give cutoff on singular values.
    
    Returns:
      ch: ndarray of a shape (D_ch,D_ch,d**2,d**2)
        Quantum channel as superoperator in iMPO representation.
    """
    if so_list == []:
        ch = np.eye(d**2,dtype=complex)
        ch = ch[np.newaxis,np.newaxis,:,:]
        return ch
    for i in range(len(so_list)):
        so = so_list[i]
        k = int(math.log(np.shape(so)[0],d**2))
        if np.linalg.norm(so-np.diag(np.diag(so))) < 10**-10:
            so = np.diag(so)
            if k == 1:
                bdchi = 1
                chi = np.zeros((bdchi,bdchi,d**2,d**2),dtype=complex)
                for nx in range(d**2):
                    chi[:,:,nx,nx] = so[nx]
            elif k == 2:
                so = np.reshape(so,(d**2,d**2),order='F')
                u,s,vh = np.linalg.svd(so)
                s = s[s > s[0]*tol]
                bdchi = np.shape(s)[0]
                u = u[:,:bdchi]
                vh = vh[:bdchi,:]
                us = u @ np.diag(np.sqrt(s))
                sv = np.diag(np.sqrt(s)) @ vh
                chi = np.zeros((bdchi,bdchi,d**2,d**2),dtype=complex)
                for nx in range(d**2):
                    chi[:,:,nx,nx] = np.outer(sv[:,nx],us[nx,:])
            elif k == 3:
                so = np.reshape(so,(d**2,d**4),order='F')
                u1,s1,vh1 = np.linalg.svd(so,full_matrices=False)
                s1 = s1[s1 > s1[0]*tol]
                bdchi1 = np.shape(s1)[0]
                u1 = u1[:,:bdchi1]
                vh1 = vh1[:bdchi1,:]
                us1 = u1 @ np.diag(np.sqrt(s1))
                sv1 = np.diag(np.sqrt(s1)) @ vh1
                sv1 = np.reshape(sv1,(bdchi1*d**2,d**2),order='F')
                u2,s2,vh2 = np.linalg.svd(sv1,full_matrices=False)
                s2 = s2[s2 > s2[0]*tol]
                bdchi2 = np.shape(s2)[0]
                u2 = u2[:,:bdchi2]
                vh2 = vh2[:bdchi2,:]
                us2 = u2 @ np.diag(np.sqrt(s2))
                us2 = np.reshape(us2,(bdchi1,d**2,bdchi2),order='F')
                sv2 = np.diag(np.sqrt(s2)) @ vh2
                bdchi = bdchi2*bdchi1
                chi = np.zeros((bdchi,bdchi,d**2,d**2),dtype=complex)
                for nx in range(d**2):
                    tensors = [sv2[:,nx],us2[:,nx,:],us1[nx,:]]
                    legs = [[-1],[-2,-3],[-4]]
                    chi[:,:,nx,nx] = np.reshape(ncon(tensors,legs),(bdchi,bdchi),order='F')
            elif k == 4:
                so = np.reshape(so,(d**2,d**6),order='F')
                u1,s1,vh1 = np.linalg.svd(so,full_matrices=False)
                s1 = s1[s1 > s1[0]*tol]
                bdchi1 = np.shape(s1)[0]
                u1 = u1[:,:bdchi1]
                vh1 = vh1[:bdchi1,:]
                us1 = u1 @ np.diag(np.sqrt(s1))
                sv1 = np.diag(np.sqrt(s1)) @ vh1
                sv1 = np.reshape(sv1,(bdchi1*d**2,d**4),order='F')
                u2,s2,vh2 = np.linalg.svd(sv1,full_matrices=False)
                s2 = s2[s2 > s2[0]*tol]
                bdchi2 = np.shape(s2)[0]
                u2 = u2[:,:bdchi2]
                vh2 = vh2[:bdchi2,:]
                us2 = u2 @ np.diag(np.sqrt(s2))
                us2 = np.reshape(us2,(bdchi1,d**2,bdchi2),order='F')
                sv2 = np.diag(np.sqrt(s2)) @ vh2
                sv2 = np.reshape(sv2,(bdchi2*d**2,d**2),order='F')
                u3,s3,vh3 = np.linalg.svd(sv2,full_matrices=False)
                s3 = s3[s3 > s3[0]*tol]
                bdchi3 = np.shape(s3)[0]
                u3 = u3[:,:bdchi3]
                vh3 = vh3[:bdchi3,:]
                us3 = u3 @ np.diag(np.sqrt(s3))
                us3 = np.reshape(us3,(bdchi2,d**2,bdchi3),order='F')
                sv3 = np.diag(np.sqrt(s3)) @ vh3
                bdchi = bdchi3*bdchi2*bdchi1
                chi = np.zeros((bdchi,bdchi,d**2,d**2),dtype=complex)
                for nx in range(d**2):
                    tensors = [sv3[:,nx],us3[:,nx,:],us2[:,nx,:],us1[nx,:]]
                    legs = [[-1],[-2,-4],[-3,-5],[-6]]
                    chi[:,:,nx,nx] = np.reshape(ncon(tensors,legs),(bdchi,bdchi),order='F')
            else:
                warnings.warn('Local noise superoperators acting on more then 4 neighbour sites are not currently supported.')
        else:
            if k == 1:
                bdchi = 1
                chi = so[np.newaxis,np.newaxis,:,:]
            elif k == 2:
                u,s,vh = np.linalg.svd(so)
                s = s[s > s[0]*tol]
                bdchi = np.shape(s)[0]
                u = u[:,:bdchi]
                vh = vh[:bdchi,:]
                us = u @ np.diag(np.sqrt(s))
                sv = np.diag(np.sqrt(s)) @ vh
                us = np.reshape(us,(d**2,d**2,bdchi),order='F')
                sv = np.reshape(sv,(bdchi,d**2,d**2),order='F')
                tensors = [sv,us]
                legs = [[-1,-3,1],[1,-4,-2]]
                chi = ncon(tensors,legs)
            elif k == 3:
                so = np.reshape(so,(d**4,d**8),order='F')
                u1,s1,vh1 = np.linalg.svd(so,full_matrices=False)
                s1 = s1[s1 > s1[0]*tol]
                bdchi1 = np.shape(s1)[0]
                u1 = u1[:,:bdchi1]
                vh1 = vh1[:bdchi1,:]
                us1 = u1 @ np.diag(np.sqrt(s1))
                sv1 = np.diag(np.sqrt(s1)) @ vh1
                us1 = np.reshape(us1,(d**2,d**2,bdchi1),order='F')
                sv1 = np.reshape(sv1,(bdchi1*d**4,d**4),order='F')
                u2,s2,vh2 = np.linalg.svd(sv1,full_matrices=False)
                s2 = s2[s2 > s2[0]*tol]
                bdchi2 = np.shape(s2)[0]
                u2 = u2[:,:bdchi2]
                vh2 = vh2[:bdchi2,:]
                us2 = u2 @ np.diag(np.sqrt(s2))
                us2 = np.reshape(us2,(bdchi1,d**2,d**2,bdchi2),order='F')
                sv2 = np.diag(np.sqrt(s2)) @ vh2
                sv2 = np.reshape(sv2,(bdchi2,d**2,d**2),order='F')
                tensors = [sv2,us2,us1]
                legs = [[-1,-5,1],[-2,1,2,-3],[2,-6,-4]]
                chi = ncon(tensors,legs)
                bdchi = bdchi2*bdchi1
                chi = np.reshape(chi,(bdchi,bdchi,d**2,d**2),order='F')
            elif k == 4:
                so = np.reshape(so,(d**4,d**12),order='F')
                u1,s1,vh1 = np.linalg.svd(so,full_matrices=False)
                s1 = s1[s1 > s1[0]*tol]
                bdchi1 = np.shape(s1)[0]
                u1 = u1[:,:bdchi1]
                vh1 = vh1[:bdchi1,:]
                us1 = u1 @ np.diag(np.sqrt(s1))
                sv1 = np.diag(np.sqrt(s1)) @ vh1
                us1 = np.reshape(us1,(d**2,d**2,bdchi1),order='F')
                sv1 = np.reshape(sv1,(bdchi1*d**4,d**8),order='F')
                u2,s2,vh2 = np.linalg.svd(sv1,full_matrices=False)
                s2 = s2[s2 > s2[0]*tol]
                bdchi2 = np.shape(s2)[0]
                u2 = u2[:,:bdchi2]
                vh2 = vh2[:bdchi2,:]
                us2 = u2 @ np.diag(np.sqrt(s2))
                us2 = np.reshape(us2,(bdchi1,d**2,d**2,bdchi2),order='F')
                sv2 = np.diag(np.sqrt(s2)) @ vh2
                sv2 = np.reshape(sv2,(bdchi2*d**4,d**4),order='F')
                u3,s3,vh3 = np.linalg.svd(sv2,full_matrices=False)
                s3 = s3[s3 > s3[0]*tol]
                bdchi3 = np.shape(s3)[0]
                u3 = u3[:,:bdchi3]
                vh3 = vh3[:bdchi3,:]
                us3 = u3 @ np.diag(np.sqrt(s3))
                us3 = np.reshape(us3,(bdchi2,d**2,d**2,bdchi3),order='F')
                sv3 = np.diag(np.sqrt(s3)) @ vh3
                sv3 = np.reshape(sv3,(bdchi3,d**2,d**2),order='F')
                tensors = [sv3,us3,us2,us1]
                legs = [[-1,-7,1],[-2,1,2,-4],[-3,2,3,-5],[3,-8,-6]]
                chi = ncon(tensors,legs)
                bdchi = bdchi3*bdchi2*bdchi1
                chi = np.reshape(chi,(bdchi,bdchi,d**2,d**2),order='F')
            else:
                warnings.warn('Local noise superoperators acting on more then 4 neighbour sites are not currently supported.')
        if i == 0:
            bdch = bdchi
            ch = chi
        else:
            bdch = bdchi*bdch
            tensors = [chi,ch]
            legs = [[-1,-3,-5,1],[-2,-4,1,-6]]
            ch = ncon(tensors,legs)
            ch = np.reshape(ch,(bdch,bdch,d**2,d**2),order='F')
    return ch


def inf_L_normalization(l):
    """
    Normalize (shifted) SLD iMPO.
    
    Parameters:
      l: (shifted) SLD iMPO, expected ndarray of a shape (bd,bd,d,d)
    
    Returns:
      l: normalized (shifted) SLD iMPO
    """
    d = np.shape(l)[2]
    tensors = [l]
    legs = [[-1,-2,1,1]]
    tm = ncon(tensors,legs)
    ltr = np.linalg.eigvals(tm)
    ltr = ltr[np.argmax(np.abs(ltr))]
    ltr = np.real(ltr)
    l = d*l/ltr
    return l


def inf_psi0_normalization(p):
    """
    Normalize wave function iMPS.
    
    Parameters:
      p: wave function iMPS, expected ndarray of a shape (bd,bd,d)
    
    Returns:
      p: normalized wave function iMPS
    """
    bdp = np.shape(p)[0]
    tensors = [np.conj(p),p]
    legs = [[-1,-3,1],[-2,-4,1]]
    tm = ncon(tensors,legs)
    tm = np.reshape(tm,(bdp*bdp,bdp*bdp),order='F')
    tm = (tm+np.conj(tm).T)/2
    tmval = np.linalg.eigvalsh(tm)
    pnorm = np.sqrt(tmval[-1])
    p = p/pnorm
    return p


def inf_enlarge_bdl(cold,factor,symfun):
    """
    Enlarge bond dimension of (shifted) SLD iMPO. Function for infinite size systems.
    
    Parameters:
      cold: (shifted) SLD iMPO, expected ndarray of a shape (bd,bd,d,d)
      factor: factor which determine on average relation between old and newly added values of (shifted) SLD iMPO
      symfun: symmetrize function
    
    Returns:
      c: (shifted) SLD iMPO with bd += 1
    """
    d = np.shape(cold)[2]
    bdl = np.shape(cold)[0]+1
    rng = np.random.default_rng()
    c = np.zeros((bdl,bdl,d,d),dtype=complex)
    for nx in range(d):
        for nxp in range(d):
            meanrecold = np.sum(np.abs(np.real(cold[:,:,nx,nxp])))/(bdl-1)**2
            meanimcold = np.sum(np.abs(np.imag(cold[:,:,nx,nxp])))/(bdl-1)**2
            c[:,:,nx,nxp] = (meanrecold*rng.random((bdl,bdl))+1j*meanimcold*rng.random((bdl,bdl)))*factor
    c = (c + np.conj(np.moveaxis(c,2,3)))/2
    c[0:bdl-1,0:bdl-1,:,:] = cold
    c = symfun(c)
    c = inf_L_normalization(c)
    return c


def inf_enlarge_bdpsi(a0old,ratio,symfund):
    """
    Enlarge bond dimension of wave function iMPS. Function for infinite size systems.
    
    Parameters:
      a0old: wave function iMPS, expected ndarray of a shape (bd,bd,d)
      ratio: factor which determine on average relation between last and next to last values of diagonals of wave function iMPS
      symfund: symmetrize function
    
    Returns:
      a0: wave function iMPS with bd += 1
    """
    d = np.shape(a0old)[2]
    bdpsi = np.shape(a0old)[0]+1
    a0 = np.zeros((bdpsi,bdpsi,d),dtype=complex)
    for i in range(d):
        if i <= np.ceil(d/2)-1:
            a0oldihalf = np.triu(np.rot90(a0old[:,:,i],-1))
            a0[0:bdpsi-1,1:bdpsi,i] = a0oldihalf
            a0[:,:,i] = a0[:,:,i]+a0[:,:,i].T
            a0[:,:,i] = a0[:,:,i]+np.diag(np.concatenate(([0],np.diag(a0[:,:,i],2),[0])))
            a0[:,:,i] = np.rot90(a0[:,:,i],1)
            a0[0,-1,i] = ratio*(1+1j)*np.abs(a0[0,-2,i])
            a0[-1,0,i] = np.conj(a0[0,-1,i])
            if i == np.ceil(d/2)-1 and np.mod(d,2) == 1:
                a0[:,:,i] = (a0[:,:,i]+a0[:,:,i].T)/2
        else:
            a0[:,:,i] = a0[:,:,d-1-i].T
    a0 = symfund(a0)
    a0 = inf_psi0_normalization(a0)
    return a0


def inf_FoM_FoMD_optbd(d,ch,chp,epsilon,symfun,symfund,cini=None,a0ini=None,imprecision=10**-2,bdlmax=100,alwaysbdlmax=False,lherm=True,bdpsimax=100,alwaysbdpsimax=False):
    """
    Iterative optimization of FoM/FoMD over (shifted) SLD iMPO and initial wave function iMPS and also check of convergence in bond dimensions. Function for infinite size systems.
    
    Parameters:
      d: dimension of local Hilbert space (dimension of physical index)
      ch: iMPO for quantum channel at the value of estimated parameter phi=phi_0, expected ndarray of a shape (bd,bd,d**2,d**2)
      chp: iMPO for quantum channel at the value of estimated parameter phi=phi_0+epsilon, expected ndarray of a shape (bd,bd,d**2,d**2)
      epsilon: value of a separation between estimated parameters encoded in ch and chp, float
      symfun: symmetrize function for iMPO for (shifted) SLD
      symfund: symmetrize function for iMPS for initial wave function
      cini: initial iMPO for (shifted) SLD, expected TN of a shape (bd,bd,d,d)
      a0ini: initial iMPS for initial wave function, expected TN of a shape (bd,bd,d)
      imprecision: expected imprecision of the end results, default value is 10**-2
      bdlmax: maximal value of bd for (shifted) SLD iMPO, default value is 100
      alwaysbdlmax: boolean value, True if maximal value of bd for (shifted) SLD iMPO have to be reached, otherwise False (default value)
      lherm: boolean value, True (default value) when Hermitian gauge is imposed on SLD iMPO, otherwise False
      bdpsimax: maximal value of bd for iMPS for initial wave function, default value is 100
      alwaysbdpsimax: boolean value, True if maximal value of bd for iMPS for initial wave function have to be reached, otherwise False (default value)
    
    Returns:
      result: optimal value of FoM/FoMD
      resultm: matrix describing FoM/FoMD in function of bd of respectively (shifted) SLD iMPO [rows] and initial wave function iMPS [columns]
      c: optimal iMPO for (shifted) SLD
      a0: optimal iMPS for initial wave function
    """
    while True:
        if a0ini is None:
            bdpsi = 1
            a0 = np.zeros(d,dtype=complex)
            for i in range(d):
                a0[i] = np.sqrt(math.comb(d-1,i))*2**(-(d-1)/2) # prod
                # a0[i] = np.sqrt(2/(d+1))*np.sin((1+i)*np.pi/(d+1)) # sine
            a0 = a0[np.newaxis,np.newaxis,:]
        else:
            a0 = a0ini
            bdpsi = np.shape(a0)[0]
            a0 = a0.astype(complex)
        if cini is None:
            bdl = 1
            c = np.triu(np.ones((d,d))-np.eye(d))
            c = 1j*epsilon*c
            c = c+np.conj(c).T
            c = np.eye(d)+c
            c = np.reshape(c,(bdl,bdl,d,d),order='F')
        else:
            c = cini
            bdl = np.shape(c)[0]
            c = c.astype(complex)
        resultm = np.zeros((bdlmax,bdpsimax),dtype=float)
        resultm[bdl-1,bdpsi-1],c,a0 = inf_FoM_FoMD_optm(c,a0,ch,chp,epsilon,symfun,symfund,imprecision,lherm)
        ratiov = np.array([10**-3,10**-2.5,10**-2])
        factorv = np.array([0.5,0.25,0.1,1,0.01])
        problem = False
        while True:
            while True:
                if bdpsi == bdpsimax:
                    break
                else:
                    a0old = a0
                    bdpsi += 1
                    i = 0
                    while True:
                        a0 = inf_enlarge_bdpsi(a0,ratiov[i],symfund)
                        resultm[bdl-1,bdpsi-1],cnew,a0new = inf_FoM_FoMD_optm(c,a0,ch,chp,epsilon,symfun,symfund,imprecision,lherm)
                        if resultm[bdl-1,bdpsi-1] >= resultm[bdl-1,bdpsi-2]:
                            break
                        i += 1
                        if i == np.size(ratiov):
                            problem = True
                            break
                    if problem:
                        break
                    if not(alwaysbdpsimax) and resultm[bdl-1,bdpsi-1] < (1+imprecision)*resultm[bdl-1,bdpsi-2]:
                        bdpsi += -1
                        a0 = a0old
                        a0copy = a0new
                        ccopy = cnew
                        break
                    else:
                        a0 = a0new
                        c = cnew
            if problem:
                break
            if bdl == bdlmax:
                if bdpsi == bdpsimax:
                    resultm = resultm[0:bdl,0:bdpsi]
                    result = resultm[bdl-1,bdpsi-1]
                else:
                    a0 = a0copy
                    c = ccopy
                    resultm = resultm[0:bdl,0:bdpsi+1]
                    result = resultm[bdl-1,bdpsi]
                break
            else:
                bdl += 1
                i = 0
                while True:
                    c = inf_enlarge_bdl(c,factorv[i],symfun)
                    resultm[bdl-1,bdpsi-1],cnew,a0new = inf_FoM_FoMD_optm(c,a0,ch,chp,epsilon,symfun,symfund,imprecision,lherm)
                    if resultm[bdl-1,bdpsi-1] >= resultm[bdl-2,bdpsi-1]:
                        a0 = a0new
                        c = cnew
                        break
                    i += 1
                    if i == np.size(factorv):
                        problem = True
                        break
                if problem:
                    break
                if not(alwaysbdlmax) and resultm[bdl-1,bdpsi-1] < (1+imprecision)*resultm[bdl-2,bdpsi-1]:
                    if bdpsi == bdpsimax:
                        resultm = resultm[0:bdl,0:bdpsi]
                        result = resultm[bdl-1,bdpsi-1]
                    else:
                        if resultm[bdl-1,bdpsi-1] < resultm[bdl-2,bdpsi]:
                            a0 = a0copy
                            c = ccopy
                            resultm = resultm[0:bdl,0:bdpsi+1]
                            bdl += -1
                            bdpsi += 1
                            result = resultm[bdl-1,bdpsi-1]
                        else:
                            resultm = resultm[0:bdl,0:bdpsi+1]
                            result = resultm[bdl-1,bdpsi-1]
                    break
        if not(problem):
            break
    return result,resultm,c,a0


def inf_FoM_optbd(d,a,b,epsilon,symfun,cini=None,imprecision=10**-2,bdlmax=100,alwaysbdlmax=False,lherm=True):
    """
    Optimization of FoM over (shifted) SLD iMPO and also check of convergence in bond dimension. Function for infinite size systems.
    
    Parameters:
      d: dimension of local Hilbert space (dimension of physical index)
      a: iMPO for density matrix at the value of estimated parameter phi=phi_0, expected ndarray of a shape (bd,bd,d,d)
      b: iMPO for density matrix at the value of estimated parameter phi=phi_0+epsilon, expected ndarray of a shape (bd,bd,d,d)
      epsilon: value of a separation between estimated parameters encoded in a and b, float
      symfun: symmetrize function for iMPO for (shifted) SLD
      cini: initial iMPO for (shifted) SLD, expected TN of a shape (bd,bd,d,d)
      imprecision: expected imprecision of the end results, default value is 10**-2
      bdlmax: maximal value of bd for (shifted) SLD iMPO, default value is 100
      alwaysbdlmax: boolean value, True if maximal value of bd for (shifted) SLD iMPO have to be reached, otherwise False (default value)
      lherm: boolean value, True (default value) when Hermitian gauge is imposed on SLD iMPO, otherwise False
    
    Returns:
      result: optimal value of FoM
      resultv: matrix describing FoM in function of bd of (shifted) SLD iMPO
      c: optimal iMPO for (shifted) SLD
    """
    while True:
        if cini is None:
            bdl = 1
            c = np.triu(np.ones((d,d))-np.eye(d))
            c = 1j*epsilon*c
            c = c+np.conj(c).T
            c = np.eye(d)+c
            c = np.reshape(c,(bdl,bdl,d,d),order='F')
        else:
            c = cini
            bdl = np.shape(c)[0]
            c = c.astype(complex)
        resultv = np.zeros(bdlmax,dtype=float)
        resultv[bdl-1],c = inf_FoM_optm_glob(a,b,c,epsilon,symfun,imprecision,lherm)
        factorv = np.array([0.5,0.25,0.1,1,0.01])
        problem = False
        while True:
            if bdl == bdlmax:
                resultv = resultv[0:bdl]
                result = resultv[bdl-1]
                break
            else:
                bdl += 1
                i = 0
                while True:
                    c = inf_enlarge_bdl(c,factorv[i],symfun)
                    resultv[bdl-1],cnew = inf_FoM_optm_glob(a,b,c,epsilon,symfun,imprecision,lherm)
                    if resultv[bdl-1] >= resultv[bdl-2]:
                        c = cnew
                        break
                    i += 1
                    if i == np.size(factorv):
                        problem = True
                        break
                if problem:
                    break
                if not(alwaysbdlmax) and resultv[bdl-1] < (1+imprecision)*resultv[bdl-2]:
                    resultv = resultv[0:bdl]
                    result = resultv[bdl-1]
                    break
        if not(problem):
            break
    return result,resultv,c


def inf_FoMD_optbd(d,c2d,cpd,epsilon,symfund,a0ini=None,imprecision=10**-2,bdpsimax=100,alwaysbdpsimax=False):
    """
    Optimization of FoMD over initial wave function iMPS and also check of convergence in bond dimension. Function for infinite size systems.
    
    Parameters:
      d: dimension of local Hilbert space (dimension of physical index)
      c2d: iMPO for square of dual of (shifted) SLD at the value of estimated parameter phi=-phi_0, expected ndarray of a shape (bd,bd,d,d)
      cpd: iMPO for dual of (shifted) SLD at the value of estimated parameter phi=-(phi_0+epsilon), expected ndarray of a shape (bd,bd,d,d)
      epsilon: value of a separation between estimated parameters encoded in c2d and cpd, float
      symfund: symmetrize function for iMPS for initial wave function
      a0ini: initial iMPS for initial wave function, expected TN of a shape (bd,bd,d)
      imprecision: expected imprecision of the end results, default value is 10**-2
      bdpsimax: maximal value of bd for iMPS for initial wave function, default value is 100
      alwaysbdpsimax: boolean value, True if maximal value of bd for iMPS for initial wave function have to be reached, otherwise False (default value)
    
    Returns:
      result: optimal value of FoMD
      resultv: matrix describing FoMD in function of bd of initial wave function iMPS
      a0: optimal iMPS for initial wave function
    """
    while True:
        if a0ini is None:
            bdpsi = 1
            a0 = np.zeros(d,dtype=complex)
            for i in range(d):
                a0[i] = np.sqrt(math.comb(d-1,i))*2**(-(d-1)/2) # prod
                # a0[i] = np.sqrt(2/(d+1))*np.sin((1+i)*np.pi/(d+1)) # sine
            a0 = a0[np.newaxis,np.newaxis,:]
        else:
            a0 = a0ini
            bdpsi = np.shape(a0)[0]
            a0 = a0.astype(complex)
        resultv = np.zeros(bdpsimax,dtype=float)
        resultv[bdpsi-1],a0 = inf_FoMD_optm_glob(c2d,cpd,a0,epsilon,symfund,imprecision)
        ratiov = np.array([10**-3,10**-2.5,10**-2])
        problem = False
        while True:
            if bdpsi == bdpsimax:
                resultv = resultv[0:bdpsi]
                result = resultv[bdpsi-1]
                break
            else:
                bdpsi += 1
                i = 0
                while True:
                    a0 = inf_enlarge_bdpsi(a0,ratiov[i],symfund)
                    resultv[bdpsi-1],a0new = inf_FoMD_optm_glob(c2d,cpd,a0,epsilon,symfund,imprecision)
                    if resultv[bdpsi-1] >= resultv[bdpsi-2]:
                        a0 = a0new
                        break
                    i += 1
                    if i == np.size(ratiov):
                        problem = True
                        break
                if problem:
                    break
                if not(alwaysbdpsimax) and resultv[bdpsi-1] < (1+imprecision)*resultv[bdpsi-2]:
                    resultv = resultv[0:bdpsi]
                    result = resultv[bdpsi-1]
                    break
        if not(problem):
            break
    return result,resultv,a0


def inf_FoM_FoMD_optm(c,a0,ch,chp,epsilon,symfun,symfund,imprecision=10**-2,lherm=True):
    """
    Iterative optimization of FoM/FoMD over (shifted) SLD iMPO and initial wave function iMPS. Function for infinite size systems.
    
    Parameters:
      c: iMPO for (shifted) SLD, expected ndarray of a shape (bd,bd,d,d)
      a0: iMPS for initial wave function, expected ndarray of a shape (bd,bd,d)
      ch: iMPO for quantum channel at the value of estimated parameter phi=phi_0, expected ndarray of a shape (bd,bd,d**2,d**2)
      chp: iMPO for quantum channel at the value of estimated parameter phi=phi_0+epsilon, expected ndarray of a shape (bd,bd,d**2,d**2)
      epsilon: value of a separation between estimated parameters encoded in ch and chp, float
      symfun: symmetrize function for iMPO for (shifted) SLD
      symfund: symmetrize function for iMPS for initial wave function
      imprecision: expected imprecision of the end results, default value is 10**-2
      lherm: boolean value, True (default value) when Hermitian gauge is imposed on SLD iMPO, otherwise False
    
    Returns:
      fval: optimal value of FoM/FoMD
      c: optimal iMPO for (shifted) SLD
      a0: optimal iMPS for initial wave function
    """
    d = np.shape(c)[2]
    bdl = np.shape(c)[0]
    relunc_f = 0.1*imprecision
    chd = np.conj(np.moveaxis(ch,2,3))
    chpd = np.conj(np.moveaxis(chp,2,3))
    f = np.array([])
    iter_f = 0
    while True:
        a0_dm = wave_function_to_density_matrix(a0)
        a = channel_acting_on_operator(ch,a0_dm)
        b = channel_acting_on_operator(chp,a0_dm)
        fom,c = inf_FoM_optm_glob(a,b,c,epsilon,symfun,imprecision,lherm)
        f = np.append(f,fom)
        if iter_f >= 2 and np.std(f[-4:])/np.mean(f[-4:]) <= relunc_f:
            break
        c2 = np.zeros((bdl**2,bdl**2,d,d),dtype=complex)
        for nx in range(d):
            for nxp in range(d):
                for nxpp in range(d):
                    c2[:,:,nx,nxp] = c2[:,:,nx,nxp]+np.kron(c[:,:,nx,nxpp],c[:,:,nxpp,nxp])
        c2d = channel_acting_on_operator(chd,c2)
        cpd = channel_acting_on_operator(chpd,c)
        fomd,a0 = inf_FoMD_optm_glob(c2d,cpd,a0,epsilon,symfund,imprecision)
        f = np.append(f,fomd)
        iter_f += 1
    fval = f[-1]
    return fval,c,a0


def inf_FoM_optm_glob(a,b,c,epsilon,symfun,imprecision=10**-2,lherm=True):
    """
    Optimization of FoM over iMPO for (shifted) SLD. Function for infinite size systems.
    
    Parameters:
      a: iMPO for density matrix at the value of estimated parameter phi=phi_0, expected ndarray of a shape (bd,bd,d,d)
      b: iMPO for density matrix at the value of estimated parameter phi=phi_0+epsilon, expected ndarray of a shape (bd,bd,d,d)
      c: iMPO for (shifted) SLD, expected ndarray of a shape (bd,bd,d,d)
      epsilon: value of a separation between estimated parameters encoded in a and b, float
      symfun: symmetrize function for iMPO for (shifted) SLD
      imprecision: expected imprecision of the end results, default value is 10**-2
      lherm: boolean value, True (default value) when Hermitian gauge is imposed on SLD iMPO, otherwise False
    Returns:
      fomval: optimal value of FoM
      c: optimal iMPO for (shifted) SLD
    """
    def inf_FoM_optm_glob_mix(m,c_old,opt_flag=True,c_old_locopt=None):
        """
        Nested function for mixing localy optimal iMPO for (shifted) SLD with its initial form according to parameter m.
        
        Parameters:
          m: mixing parameter
          c_old: initial iMPO for (shifted) SLD, expected ndarray of a shape (bd,bd,d,d)
          opt_flag: boolean value, True (default value) if calculating localy optimal iMPO is necessary, otherwise False
          c_old_locopt: localy optimal iMPO for (shifted) SLD used when opt_flag=False, default value is None
        Returns:
          fomvalf: value of FoM
          c_newf: iMPO for (shifted) SLD after mixing
          c_locoptf: localy optimal iMPO for (shifted) SLD before mixing
        """
        if opt_flag:
            c_locoptf = inf_FoM_optm_loc(a,b,c_old,epsilon,imprecision,lherm)
            c_locoptf = symfun(c_locoptf)
            c_locoptf = inf_L_normalization(c_locoptf)
        else:
            c_locoptf = c_old_locopt
        c_newf = c_locoptf*np.sin(m*np.pi)-c_old*np.cos(m*np.pi)
        c_newf = symfun(c_newf)
        c_newf = inf_L_normalization(c_newf)
        fomvalf = inf_FoM_val(a,b,c_newf,epsilon)
        return fomvalf,c_newf,c_locoptf
    
    step_ini = 10**-1
    step_tiny = 10**-10
    relunc_fom = 0.1*imprecision
    fom = np.array([])
    fom_1 = inf_FoM_val(a,b,c,epsilon)
    fom_05,c_05 = inf_FoM_optm_glob_mix(1/2,c)[:2]
    if fom_05 > fom_1:
        c = c_05
        fom = np.append(fom,fom_05)
    else:
        fom = np.append(fom,fom_1)
    del c_05
    fom_tinylean = inf_FoM_optm_glob_mix(1+step_tiny,c)[0]
    if fom_tinylean > fom[0]:
        step = step_ini
    else:
        step = -step_ini
    opt_flag = True
    c_locopt = None
    iter_fom = 1
    while True:
        fomval,c_new,c_locopt = inf_FoM_optm_glob_mix(1+step,c,opt_flag,c_locopt)
        if fomval > fom[-1]:
            c = c_new
            fom = np.append(fom,fomval)
            opt_flag = True
            iter_fom += 1
        else:
            step = step/2
            opt_flag = False
        if np.abs(step) < step_tiny or (iter_fom >= 4 and all(fom[-4:] > 0) and np.std(fom[-4:])/np.mean(fom[-4:]) <= relunc_fom):
            break
    fomval = fom[-1]
    return fomval,c


def inf_FoMD_optm_glob(c2d,cpd,a0,epsilon,symfund,imprecision=10**-2):
    """
    Optimization of FoMD over iMPS for initial wave function. Function for infinite size systems.
    
    Parameters:
      c2d: iMPO for square of dual of (shifted) SLD at the value of estimated parameter phi=-phi_0, expected ndarray of a shape (bd,bd,d,d)
      cpd: iMPO for dual of (shifted) SLD at the value of estimated parameter phi=-(phi_0+epsilon), expected ndarray of a shape (bd,bd,d,d)
      a0: iMPS for initial wave function, expected ndarray of a shape (bd,bd,d)
      epsilon: value of a separation between estimated parameters encoded in c2d and cpd, float
      symfund: symmetrize function for iMPS for initial wave function
      imprecision: expected imprecision of the end results, default value is 10**-2
    
    Returns:
      fomdval: optimal value of FoMD
      a0: optimal iMPS for initial wave function
    """
    def inf_FoMD_optm_glob_mix(m,a0_old,opt_flag=True,a0_old_locopt=None):
        """
        Nested function for mixing localy optimal iMPS for initial wave function with its initial form according to parameter m.
        
        Parameters:
          m: mixing parameter
          a0_old: initial iMPS for initial wave function, expected ndarray of a shape (bd,bd,d)
          opt_flag: boolean value, True (default value) if calculating localy optimal iMPS is necessary, otherwise False
          a0_old_locopt: localy optimal iMPS for initial wave function used when opt_flag=False, default value is None
        
        Returns:
          fomdvalf: value of FoMD
          a0_newf: iMPS for initial wave function after mixing
          a0_locoptf: localy optimal iMPS for initial wave function before mixing
        """
        if opt_flag:
            a0_locoptf = inf_FoMD_optm_loc(c2d,cpd,a0_old,epsilon,imprecision)
            a0_locoptf = symfund(a0_locoptf)
            a0_locoptf = inf_psi0_normalization(a0_locoptf)
        else:
            a0_locoptf = a0_old_locopt
        a0_newf = a0_locoptf*np.sin(m*np.pi)-a0_old*np.cos(m*np.pi)
        a0_newf = symfund(a0_newf)
        a0_newf = inf_psi0_normalization(a0_newf)
        fomdvalf = inf_FoMD_val(c2d,cpd,a0_newf,epsilon)
        return fomdvalf,a0_newf,a0_locoptf
    
    step_ini = 10**-1
    step_tiny = 10**-10
    relunc_fomd = 0.1*imprecision
    fomd = np.array([])
    fomd_1 = inf_FoMD_val(c2d,cpd,a0,epsilon)
    fomd_05,a0_05 = inf_FoMD_optm_glob_mix(1/2,a0)[:2]
    if fomd_05 > fomd_1:
        a0 = a0_05
        fomd = np.append(fomd,fomd_05)
    else:
        fomd = np.append(fomd,fomd_1)
    del a0_05
    fomd_tinylean = inf_FoMD_optm_glob_mix(1+step_tiny,a0)[0]
    if fomd_tinylean > fomd[0]:
        step = step_ini
    else:
        step = -step_ini
    opt_flag = True
    a0_locopt = None
    iter_fomd = 1
    while True:
        fomdval,a0_new,a0_locopt = inf_FoMD_optm_glob_mix(1+step,a0,opt_flag,a0_locopt)
        if fomdval > fomd[-1]:
            a0 = a0_new
            fomd = np.append(fomd,fomdval)
            opt_flag = True
            iter_fomd += 1
        else:
            step = step/2
            opt_flag = False
        if np.abs(step) < step_tiny or (iter_fomd >= 4 and all(fomd[-4:] > 0) and np.std(fomd[-4:])/np.mean(fomd[-4:]) <= relunc_fomd):
            break
    fomdval = fomd[-1]
    return fomdval,a0


def inf_FoM_optm_loc(a,b,c,epsilon,imprecision=10**-2,lherm=True):
    """
    Calculate localy optimal iMPO for (shifted) SLD. Function for infinite size systems.
    
    Parameters:
      a: iMPO for density matrix at the value of estimated parameter phi=phi_0, expected ndarray of a shape (bd,bd,d,d)
      b: iMPO for density matrix at the value of estimated parameter phi=phi_0+epsilon, expected ndarray of a shape (bd,bd,d,d)
      c: iMPO for (shifted) SLD, expected ndarray of a shape (bd,bd,d,d)
      epsilon: value of a separation between estimated parameters encoded in a and b, float
      imprecision: expected imprecision of the end results, default value is 10**-2
      lherm: boolean value, True (default value) when Hermitian gauge is imposed on SLD iMPO, otherwise False
    
    Returns:
      c: localy optimal iMPO for (shifted) SLD
    """
    d = np.shape(a)[2]
    bdr = np.shape(a)[0]
    bdrp = np.shape(b)[0]
    bdl = np.shape(c)[0]
    tol_fom = imprecision*epsilon**2
    tensors = [c,b]
    legs = [[-1,-3,1,2],[-2,-4,2,1]]
    tm = ncon(tensors,legs)
    tm = np.reshape(tm,(bdl*bdrp,bdl*bdrp),order='F')
    tmval,tmvr = np.linalg.eig(tm)
    tmvr = tmvr[:,np.argmax(np.abs(tmval))]
    tmval,tmvl = np.linalg.eig(tm.T)
    tmvl = tmvl[:,np.argmax(np.abs(tmval))]
    tmvnorm = np.sqrt(tmvl @ tmvr)
    l1r = np.reshape(tmvr/tmvnorm,(bdl,bdrp),order='F')
    l1l = np.reshape(tmvl/tmvnorm,(bdl,bdrp),order='F')
    tensors = [c,a,c]
    legs = [[-1,-4,1,2],[-2,-5,2,3],[-3,-6,3,1]]
    tm = ncon(tensors,legs)
    tm = np.reshape(tm,(bdl*bdr*bdl,bdl*bdr*bdl),order='F')
    tmval,tmvr = np.linalg.eig(tm)
    tmvr = tmvr[:,np.argmax(np.abs(tmval))]
    tmval,tmvl = np.linalg.eig(tm.T)
    tmvl = tmvl[:,np.argmax(np.abs(tmval))]
    tmvnorm = np.sqrt(tmvl @ tmvr)
    l2r = np.reshape(tmvr/tmvnorm,(bdl,bdr,bdl),order='F')
    l2l = np.reshape(tmvl/tmvnorm,(bdl,bdr,bdl),order='F')
    tensors = [l1l,b,l1r]
    legs = [[-1,2],[2,1,-4,-3],[-2,1]]
    l1 = ncon(tensors,legs)
    l1 = np.reshape(l1,-1,order='F')
    tensors = [l2l,a,np.eye(d),l2r]
    legs = [[-1,2,-5],[2,1,-4,-7],[-8,-3],[-2,1,-6]]
    l2 = ncon(tensors,legs)
    l2 = np.reshape(l2,(bdl*bdl*d*d,bdl*bdl*d*d),order='F')
    dl2 = l2+l2.T
    dl1 = 2*l1
    dl2pinv = np.linalg.pinv(dl2,tol_fom)
    dl2pinv = (dl2pinv+dl2pinv.T)/2
    cv = dl2pinv @ dl1
    c = np.reshape(cv,(bdl,bdl,d,d),order='F')
    if lherm:
        c = (c+np.conj(np.moveaxis(c,2,3)))/2
    return c


def inf_FoMD_optm_loc(c2d,cpd,a0,epsilon,imprecision=10**-2):
    """
    Calculate localy optimal iMPS for initial wave function. Function for infinite size systems.
    
    Parameters:
      c2d: iMPO for square of dual of (shifted) SLD at the value of estimated parameter phi=-phi_0, expected ndarray of a shape (bd,bd,d,d)
      cpd: iMPO for dual of (shifted) SLD at the value of estimated parameter phi=-(phi_0+epsilon), expected ndarray of a shape (bd,bd,d,d)
      a0: iMPS for initial wave function, expected ndarray of a shape (bd,bd,d)
      epsilon: value of a separation between estimated parameters encoded in c2d and cpd, float
      imprecision: expected imprecision of the end results, default value is 10**-2
    
    Returns:
      a0: localy optimal iMPS for initial wave function
    """
    d = np.shape(c2d)[2]
    bdl2d = np.shape(c2d)[0]
    bdlpd = np.shape(cpd)[0]
    bdpsi = np.shape(a0)[0]
    tol_fomd = imprecision*epsilon**2
    tensors = [np.conj(a0),cpd,a0]
    legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
    tm = ncon(tensors,legs)
    tm = np.reshape(tm,(bdpsi*bdlpd*bdpsi,bdpsi*bdlpd*bdpsi),order='F')
    tmval,tmvr = np.linalg.eig(tm)
    tmvr = tmvr[:,np.argmax(np.abs(tmval))]
    tmval,tmvl = np.linalg.eig(tm.T)
    tmvl = tmvl[:,np.argmax(np.abs(tmval))]
    tmvnorm = np.sqrt(tmvl @ tmvr)
    lpdr = np.reshape(tmvr/tmvnorm,(bdpsi,bdlpd,bdpsi),order='F')
    lpdl = np.reshape(tmvl/tmvnorm,(bdpsi,bdlpd,bdpsi),order='F')
    tensors = [np.conj(a0),c2d,a0]
    legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
    tm = ncon(tensors,legs)
    tm = np.reshape(tm,(bdpsi*bdl2d*bdpsi,bdpsi*bdl2d*bdpsi),order='F')
    tmval,tmvr = np.linalg.eig(tm)
    tmvr = tmvr[:,np.argmax(np.abs(tmval))]
    tmval,tmvl = np.linalg.eig(tm.T)
    tmvl = tmvl[:,np.argmax(np.abs(tmval))]
    tmvnorm = np.sqrt(tmvl @ tmvr)
    l2dr = np.reshape(tmvr/tmvnorm,(bdpsi,bdl2d,bdpsi),order='F')
    l2dl = np.reshape(tmvl/tmvnorm,(bdpsi,bdl2d,bdpsi),order='F')
    tensors = [np.conj(a0),a0]
    legs = [[-1,-3,1],[-2,-4,1]]
    tm = ncon(tensors,legs)
    tm = np.reshape(tm,(bdpsi*bdpsi,bdpsi*bdpsi),order='F')
    tm = (tm+np.conj(tm).T)/2
    tmval,tmv = np.linalg.eigh(tm)
    tmv = tmv[:,-1]
    psinorm = np.reshape(tmv,(bdpsi,bdpsi),order='F')
    psinorm = (psinorm+np.conj(psinorm).T)/2
    tensors = [lpdl,cpd,lpdr]
    legs = [[-1,2,-4],[2,1,-3,-6],[-2,1,-5]]
    lpd = ncon(tensors,legs)
    lpd = np.reshape(lpd,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
    tensors = [l2dl,c2d,l2dr]
    legs = [[-1,2,-4],[2,1,-3,-6],[-2,1,-5]]
    l2d = ncon(tensors,legs)
    l2d = np.reshape(l2d,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
    psinormpinv = np.linalg.pinv(psinorm,tol_fomd,hermitian=True)
    psinormpinv = (psinormpinv+np.conj(psinormpinv).T)/2
    tensors = [np.conj(psinormpinv),np.eye(d),psinormpinv]
    legs = [[-1,-4],[-3,-6],[-2,-5]]
    psinormpinv = ncon(tensors,legs)
    psinormpinv = np.reshape(psinormpinv,(bdpsi*bdpsi*d,bdpsi*bdpsi*d),order='F')
    eiginput = 2*lpd-l2d
    eiginput = (eiginput+np.conj(eiginput).T)/2
    eiginput = psinormpinv @ eiginput
    fomdval,a0v = np.linalg.eig(eiginput)
    position = np.argmax(np.real(fomdval))
    a0v = np.reshape(a0v[:,position],-1,order='F')
    a0 = np.reshape(a0v,(bdpsi,bdpsi,d),order='F')
    return a0


def inf_FoM_val(a,b,c,epsilon):
    """
    Calculate value of FoM. Function for infinite size systems.
    
    Parameters:
      a: iMPO for density matrix at the value of estimated parameter phi=phi_0, expected ndarray of a shape (bd,bd,d,d)
      b: iMPO for density matrix at the value of estimated parameter phi=phi_0+epsilon, expected ndarray of a shape (bd,bd,d,d)
      c: iMPO for (shifted) SLD, expected ndarray of a shape (bd,bd,d,d)
      epsilon: value of a separation between estimated parameters encoded in a and b, float
    
    Returns:
      fomval: value of FoM
    """
    bdr = np.shape(a)[0]
    bdrp = np.shape(b)[0]
    bdl = np.shape(c)[0]
    tensors = [c,b]
    legs = [[-1,-3,1,2],[-2,-4,2,1]]
    tm = ncon(tensors,legs)
    tm = np.reshape(tm,(bdl*bdrp,bdl*bdrp),order='F')
    l1 = np.linalg.eigvals(tm)
    l1 = l1[np.argmax(np.abs(l1))]
    tensors = [c,a,c]
    legs = [[-1,-4,1,2],[-2,-5,2,3],[-3,-6,3,1]]
    tm = ncon(tensors,legs)
    tm = np.reshape(tm,(bdl*bdr*bdl,bdl*bdr*bdl),order='F')
    l2 = np.linalg.eigvals(tm)
    l2 = l2[np.argmax(np.abs(l2))]
    fomval = np.real((2*l1-l2-1)/epsilon**2)
    return fomval


def inf_FoMD_val(c2d,cpd,a0,epsilon):
    """
    Calculate value of FoMD. Function for infinite size systems.
    
    Parameters:
      c2d: iMPO for square of dual of (shifted) SLD at the value of estimated parameter phi=-phi_0, expected ndarray of a shape (bd,bd,d,d)
      cpd: iMPO for dual of (shifted) SLD at the value of estimated parameter phi=-(phi_0+epsilon), expected ndarray of a shape (bd,bd,d,d)
      a0: iMPS for initial wave function, expected ndarray of a shape (bd,bd,d)
      epsilon: value of a separation between estimated parameters encoded in c2d and cpd, float
    
    Returns:
      fomdval: value of FoMD
    """
    bdl2d = np.shape(c2d)[0]
    bdlpd = np.shape(cpd)[0]
    bdpsi = np.shape(a0)[0]
    tensors = [np.conj(a0),cpd,a0]
    legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
    tm = ncon(tensors,legs)
    tm = np.reshape(tm,(bdpsi*bdlpd*bdpsi,bdpsi*bdlpd*bdpsi),order='F')
    lpd = np.linalg.eigvals(tm)
    lpd = lpd[np.argmax(np.abs(lpd))]
    tensors = [np.conj(a0),c2d,a0]
    legs = [[-1,-4,1],[-2,-5,1,2],[-3,-6,2]]
    tm = ncon(tensors,legs)
    tm = np.reshape(tm,(bdpsi*bdl2d*bdpsi,bdpsi*bdl2d*bdpsi),order='F')
    l2d = np.linalg.eigvals(tm)
    l2d = l2d[np.argmax(np.abs(l2d))]
    fomdval = np.real((2*lpd-l2d-1)/epsilon**2)
    return fomdval


##########################
#                        #
#                        #
# 3 Auxiliary functions. #
#                        #
#                        #
##########################


def channel_acting_on_operator(ch, o):
    """
    Creates MPO/iMPO for operator o after the evolution through quantum channel ch.
    
    Parameters:
      ch: list of length N of ndarrays of a shape (Dl_ch,Dr_ch,d**2,d**2) for OBC (Dl_ch, Dr_ch can vary between sites) or ndarray of a shape (D_ch,D_ch,d**2,d**2,N) for PBC or ndarray of a shape (D_ch,D_ch,d**2,d**2) for infinite approach
        Quantum channel as superoperator in MPO/iMPO representation.
      o: list of length N of ndarrays of a shape (Dl_o,Dr_o,d,d) for OBC (Dl_o,Dr_o can vary between sites) or ndarray of a shape (D_o,D_o,d,d,N) for PBC or ndarray of a shape (D_o,D_o,d,d) for infinite approach
        Operator in MPO/iMPO representation.
    
    Returns:
      ch_o: list of length N of ndarrays of a shape (Dl_ch*Dl_o,Dr_ch*Dr_o,d,d) for OBC (Dl_ch*Dl_o, Dr_ch*Dr_o can vary between sites) or ndarray of a shape (D_ch*D_o,D_ch*D_o,d,d,N) for PBC or ndarray of a shape (D_ch*D_o,D_ch*D_o,d,d) for infinite approach
        Operator after the evolution through quantum channel in MPO/iMPO representation.
    """
    if type(o) is list and type(ch) is list:
        if len(o) != len(ch):
            warnings.warn('Tensor networks representing channel and operator have to be of the same length.')
        n = len(o)
        ch_o = [0]*n
        for x in range(n):
            d = np.shape(o[x])[2]
            bdo1 = np.shape(o[x])[0]
            bdo2 = np.shape(o[x])[1]
            bdch1 = np.shape(ch[x])[0]
            bdch2 = np.shape(ch[x])[1]
            o[x] = np.reshape(o[x],(bdo1,bdo2,d**2),order='F')
            ch_o[x] = np.zeros((bdo1*bdch1,bdo2*bdch2,d**2),dtype=complex)
            for nx in range(d**2):
                for nxp in range(d**2):
                    ch_o[x][:,:,nx] = ch_o[x][:,:,nx]+np.kron(ch[x][:,:,nx,nxp],o[x][:,:,nxp])
            ch_o[x] = np.reshape(ch_o[x],(bdo1*bdch1,bdo2*bdch2,d,d),order='F')
            o[x] = np.reshape(o[x],(bdo1,bdo2,d,d),order='F')
    elif type(o) is np.ndarray and type(ch) is np.ndarray:
        if np.ndim(o) != np.ndim(ch) or (np.ndim(o) == 5 and np.shape(o)[4] != np.shape(ch)[4]):
            warnings.warn('Tensor networks representing channel and operator have to be of the same length.')
        d = np.shape(o)[2]
        bdo = np.shape(o)[0]
        bdch = np.shape(ch)[0]
        if np.ndim(o) == 4:
            o = np.reshape(o,(bdo,bdo,d**2),order='F')
            ch_o = np.zeros((bdo*bdch,bdo*bdch,d**2),dtype=complex)
            for nx in range(d**2):
                for nxp in range(d**2):
                    ch_o[:,:,nx] = ch_o[:,:,nx]+np.kron(ch[:,:,nx,nxp],o[:,:,nxp])
            ch_o = np.reshape(ch_o,(bdo*bdch,bdo*bdch,d,d),order='F')
            o = np.reshape(o,(bdo,bdo,d,d),order='F')
        elif np.ndim(o) == 5:
            n = np.shape(o)[4]
            o = np.reshape(o,(bdo,bdo,d**2,n),order='F')
            ch_o = np.zeros((bdo*bdch,bdo*bdch,d**2,n),dtype=complex)
            for nx in range(d**2):
                for nxp in range(d**2):
                    for x in range(n):
                        ch_o[:,:,nx,x] = ch_o[:,:,nx,x]+np.kron(ch[:,:,nx,nxp,x],o[:,:,nxp,x])
            ch_o = np.reshape(ch_o,(bdo*bdch,bdo*bdch,d,d,n),order='F')
            o = np.reshape(o,(bdo,bdo,d,d,n),order='F')
    else:
        warnings.warn('Channel and operator have to be of the same type (list or numpy.ndarray).')
    return ch_o


def wave_function_to_density_matrix(p):
    """
    Creates density matrix in MPO/iMPO representation from wave function in MPS/iMPS representation.
    
    Parameters:
      p: list of length N of ndarrays of a shape (Dl_p,Dr_p,d) for OBC (Dl_p, Dr_p can vary between sites) or ndarray of a shape (D_p,D_p,d,N) for PBC or ndarray of a shape (D_p,D_p,d) for infinite approach
        Wave function in MPS/iMPS representation.
    
    Returns:
      r: list of length N of ndarrays of a shape (Dl_r,Dr_r,d,d) for OBC (Dl_r, Dr_r can vary between sites) or ndarray of a shape (D_r,D_r,d,d,N) for PBC or ndarray of a shape (D_r,D_r,d,d) for infinite approach
        Density matrix in MPO/iMPO representation.
    """
    if type(p) is list:
        n = len(p)
        r = [0]*n
        for x in range(n):
            d = np.shape(p[x])[2]
            bdp1 = np.shape(p[x])[0]
            bdp2 = np.shape(p[x])[1]
            r[x] = np.zeros((bdp1**2,bdp2**2,d,d),dtype=complex)
            for nx in range(d):
                for nxp in range(d):
                    r[x][:,:,nx,nxp] = np.kron(p[x][:,:,nx],np.conj(p[x][:,:,nxp]))
    elif type(p) is np.ndarray:
        d = np.shape(p)[2]
        bdp = np.shape(p)[0]
        if np.ndim(p) == 3:
            r = np.zeros((bdp**2,bdp**2,d,d),dtype=complex)
            for nx in range(d):
                for nxp in range(d):
                    r[:,:,nx,nxp] = np.kron(p[:,:,nx],np.conj(p[:,:,nxp]))
        elif np.ndim(p) == 4:
            n = np.shape(p)[3]
            r = np.zeros((bdp**2,bdp**2,d,d,n),dtype=complex)
            for nx in range(d):
                for nxp in range(d):
                    for x in range(n):
                        r[:,:,nx,nxp,x] = np.kron(p[:,:,nx,x],np.conj(p[:,:,nxp,x]))
    return r


def Kraus_to_superoperator(kraus_list):
    """
    Creates superoperator from the list of Kraus operators.
    
    This function is designed to creates local superoperators from the list of local Kraus operators so dk = d**k where d is dimension of local Hilbert space (dimension of physical index) and k is number of sites on which local Kraus operators acts.
    In this framework Kraus operators have to be square.
    
    Parameters:
      kraus_list: list of ndarrays of a shape (dk,dk) where dk is dimension of a Kraus operator
        List of Kraus operators.
    
    Returns:
      so: ndarray of a shape (dk**2,dk**2)
        Superoperator.
    """
    if np.shape(kraus_list[0])[0] != np.shape(kraus_list[0])[1]:
        warnings.warn('In this framework Kraus operators have to be square.')
    dk = np.shape(kraus_list[0])[0]
    dynamicalmatrix = np.zeros((dk**2,dk**2),dtype=complex)
    for kraus in kraus_list:
        krausvec = np.reshape(kraus,-1,order='F')
        dynamicalmatrix = dynamicalmatrix + np.outer(krausvec,np.conj(krausvec))
    # Proper dynamical matrix would have also 1/dk factor.
    so = np.reshape(np.moveaxis(np.reshape(dynamicalmatrix,(dk,dk,dk,dk),order='F'),1,2),(dk**2,dk**2),order='F')
    return so


def fullHilb(N, so_before_list, h, so_after_list, BC='O', imprecision=10**-2):
    """
    Optimization of the QFI over operator L (in full Hilbert space) and wave function psi0 (in full Hilbert space).
    
    Function designed to be complementary to fin() so it has the same inputs.
    User has to provide information about the dynamics by specifying quantum channel. It is assumed that quantum channel is translationally invariant and is build from layers of quantum operations.
    User has to provide one defining for each layer operation as a local superoperator. Those local superoperator have to be input in order of their action on the system.
    Parameter encoding is a stand out quantum operation. It is assumed that parameter encoding acts only once and is unitary so the user have to provide only its generator h.
    Generator h have to be diagonal in computational basis, or in other words it is assumed that local superoperators are expressed in the eigenbasis of h.
    
    Parameters:
      N: integer
        Number of sites in the chain of tensors (usually number of particles).
      so_before_list: list of ndarrays of a shape (d**(2*k),d**(2*k)) where k describes on how many sites particular local superoperator acts
        List of local superoperators (in order) which act before unitary parameter encoding.
      h: ndarray of a shape (d,d)
        Generator of unitary parameter encoding. Dimension d is the dimension of local Hilbert space (dimension of physical index).
        Generator h have to be diagonal in computational basis, or in other words it is assumed that local superoperators are expressed in the eigenbasis of h.
      so_after_list: list of ndarrays of a shape (d**(2*k),d**(2*k)) where k describes on how many sites particular local superoperator acts
        List of local superoperators (in order) which act after unitary parameter encoding.
      BC: 'O' or 'P'
        Boundary conditions, 'O' for OBC, 'P' for PBC.
      imprecision: float, optional
        Expected relative imprecision of the end results.
    
    Returns:
      result: float
        Optimal value of figure of merit.
      L: ndarray of a shape (d**N,d**N)
        Optimal L in full Hilbert space.
      psi0: ndarray of a shape (d**N,)
        Optimal psi0 in full Hilbert space.
    """
    if np.linalg.norm(h - np.diag(np.diag(h))) > 10**-10:
        warnings.warn('Generator h have to be diagonal in computational basis, or in other words we assume that local superoperators are expressed in the eigenbasis of h.')
    d = np.shape(h)[0]
    ch = fin_create_channel(N, d, BC, so_before_list + so_after_list)
    ch2 = fin_create_channel_derivative(N, d, BC, so_before_list, h, so_after_list)
    ch_fH = MPO_to_fullHilb_superoperator(ch)
    ch2_fH = MPO_to_fullHilb_superoperator(ch2)
    result, L, psi0 = fullHilb_FoM_FoMD_opt(N, d, ch_fH, ch2_fH, imprecision)
    return result, L, psi0


def fullHilb_FoM_FoMD_opt(n,d,ch,chp,imprecision=10**-2):
    """
    Iterative optimization of FoM/FoMD over SLD and initial wave function using standard full Hilbert space description.
    
    Parameters:
      n: number of particles
      d: dimension of local Hilbert space
      ch: superoperator for quantum channel describing decoherence, expected ndarray of a shape (d**n,d**n)
      chp: superoperator for generalized derivative of quantum channel describing decoherence, expected ndarray of a shape (d**n,d**n)
      imprecision: expected imprecision of the end results, default value is 10**-2
    
    Returns:
      result: optimal value of FoM/FoMD
      l: optimal SLD
      psi0: optimal initial wave function
    """
    relunc_f = 0.1*imprecision
    chd = np.conj(ch).T
    chpd = np.conj(chp).T
    psi0 = np.zeros(d,dtype=complex)
    for i in range(d):
        psi0[i] = np.sqrt(math.comb(d-1,i))*2**(-(d-1)/2) # prod
        # psi0[i] = np.sqrt(2/(d+1))*np.sin((1+i)*np.pi/(d+1)) # sine
    psi0_0 = np.copy(psi0)
    for x in range(n-1):
        psi0 = np.kron(psi0,psi0_0)
    rho0 = np.outer(psi0,np.conj(psi0))
    rho0vec = np.reshape(rho0,-1,order='F')
    f = np.array([])
    iter_f = 0
    while True:
        rhovec = ch @ rho0vec
        rho = np.reshape(rhovec,(d**n,d**n),order='F')
        rhopvec = chp @ rho0vec
        rhop = np.reshape(rhopvec,(d**n,d**n),order='F')
        rho = (rho + np.conj(rho).T)/2
        rhoeigval,rhoeigvec = np.linalg.eigh(rho)
        lpart1 = np.zeros((d**n,d**n),dtype=complex)
        for nt in range(d**n):
            for ntp in range(d**n):
                if np.abs(rhoeigval[nt]+rhoeigval[ntp]) > 10**-10:
                    lpart1[nt,ntp] = 1/(rhoeigval[nt]+rhoeigval[ntp])
                else:
                    lpart1[nt,ntp] = 0
        lpart2 = np.conj(rhoeigvec).T @ rhop @ rhoeigvec
        l = rhoeigvec @ (2*lpart1*lpart2) @ np.conj(rhoeigvec).T
        fom = np.real(np.trace(rhop @ l))
        f = np.append(f,fom)
        if iter_f >= 4 and np.std(f[-4:])/np.mean(f[-4:]) <= relunc_f:
            break
        lvec = np.reshape(l,-1,order='F')
        l2vec = np.reshape(l @ l,-1,order='F')
        l2dvec = chd @ l2vec
        l2d = np.reshape(l2dvec,(d**n,d**n),order='F')
        lpdvec = chpd @ lvec
        lpd = np.reshape(lpdvec,(d**n,d**n),order='F')
        eiginput = 2*lpd-l2d
        eiginput = (eiginput+np.conj(eiginput).T)/2
        eigval,eigvec = np.linalg.eigh(eiginput)
        fomd = eigval[-1]
        f = np.append(f,fomd)
        psi0 = eigvec[:,-1]
        rho0 = np.outer(psi0,np.conj(psi0))
        rho0vec = np.reshape(rho0,-1,order='F')
        iter_f += 1
    result = f[-1]
    return result,l,psi0


def fullHilb_FoM_val(rho,rhop):
    """
    Calculate value of FoM using standard full Hilbert space description.
    
    Parameters:
      rho: density matrix
      rhop: generalized derivative of density matrix
    
    Returns:
      fomval: value of FoM
    """
    dn = np.shape(rho)[0]
    rhoeigval,rhoeigvec = np.linalg.eigh(rho)
    lpart1 = np.zeros((dn,dn),dtype=complex)
    for nt in range(dn):
        for ntp in range(dn):
            if np.abs(rhoeigval[nt]+rhoeigval[ntp]) > 10**-10:
                lpart1[nt,ntp] = 1/(rhoeigval[nt]+rhoeigval[ntp])
            else:
                lpart1[nt,ntp] = 0
    lpart2 = np.conj(rhoeigvec).T @ rhop @ rhoeigvec
    l = rhoeigvec @ (2*lpart1*lpart2) @ np.conj(rhoeigvec).T
    fomval = np.real(np.trace(rhop @ l))
    return fomval


def fullHilb_FoMD_val(l2d,lpd):
    """
    Calculate value of FoMD using standard full Hilbert space description.
    
    Parameters:
      l2d: square of dual of SLD
      lpd: dual of generalized derivative of SLD
    
    Returns:
      fomdval: value of FoMD
    """
    eiginput = 2*lpd-l2d
    eiginput = (eiginput+np.conj(eiginput).T)/2
    eigval,eigvec = np.linalg.eigh(eiginput)
    fomdval = eigval[-1]
    return fomdval


def MPS_to_fullHilb_wave_function(a):
    """
    Creates wave function in full Hilbert space from its MPS description.
    
    Parameters:
      a: list of length N of ndarrays of a shape (Dl_a,Dr_a,d) for OBC (Dl_a, Dr_a can vary between sites) or ndarray of a shape (D_a,D_a,d,N) for PBC
        MPS.
    
    Returns:
      b: ndarray of a shape (d**N,)
        Wave function in full Hilbert space.
    """
    if type(a) is list:
        bc = 'O'
        n = len(a)
        d = np.shape(a[0])[2]
    elif type(a) is np.ndarray:
        bc = 'P'
        n = np.shape(a)[3]
        d = np.shape(a)[2]
    b = np.zeros(d**n,dtype=complex)
    nt = 0
    for ntc in itertools.product(np.arange(d,dtype=int),repeat=n):
        if bc == 'O':
            if n == 1:
                if np.shape(a[0])[0] == 1:
                    b[nt] = a[0][0,0,ntc[0]]
                else:
                    warnings.warn('Tensor networks with OBC and length one have to have bond dimension equal to one.')
            else:
                aux = a[0][:,:,ntc[0]]
                for x in range(1,n):
                    aux = aux @ a[x][:,:,ntc[x]]
                b[nt] = aux
        elif bc == 'P':
            aux = a[:,:,ntc[0],0]
            for x in range(1,n):
                aux = aux @ a[:,:,ntc[x],x]
            b[nt] = np.trace(aux)
        nt += 1
    return b


def MPO_to_fullHilb_operator(a):
    """
    Creates operator in full Hilbert space from its MPO description.
    
    Parameters:
      a: list of length N of ndarrays of a shape (Dl_a,Dr_a,d,d) for OBC (Dl_a, Dr_a can vary between sites) or ndarray of a shape (D_a,D_a,d,d,N) for PBC
        MPO.
    
    Returns:
      b: ndarray of a shape (d**N,d**N)
        Operator in full Hilbert space.
    """
    if type(a) is list:
        bc = 'O'
        n = len(a)
        d = np.shape(a[0])[2]
    elif type(a) is np.ndarray:
        bc = 'P'
        n = np.shape(a)[4]
        d = np.shape(a)[2]
    b = np.zeros((d**n,d**n),dtype=complex)
    nt = 0
    for ntc in itertools.product(np.arange(d,dtype=int),repeat=n):
        ntp = 0
        for ntpc in itertools.product(np.arange(d,dtype=int),repeat=n):
            if bc == 'O':
                if n == 1:
                    if np.shape(a[0])[0] == 1:
                        b[nt,ntp] = a[0][0,0,ntc[0],ntpc[0]]
                    else:
                        warnings.warn('Tensor networks with OBC and length one have to have bond dimension equal to one.')
                else:
                    aux = a[0][:,:,ntc[0],ntpc[0]]
                    for x in range(1,n):
                        aux = aux @ a[x][:,:,ntc[x],ntpc[x]]
                    b[nt,ntp] = aux
            elif bc == 'P':
                aux = a[:,:,ntc[0],ntpc[0],0]
                for x in range(1,n):
                    aux = aux @ a[:,:,ntc[x],ntpc[x],x]
                b[nt,ntp] = np.trace(aux)
            ntp += 1
        nt += 1
    return b


def MPO_to_fullHilb_superoperator(a):
    """
    Creates a superoperator in the full Hilbert space from its MPO description.
    
    Parameters:
      a: list of length N of ndarrays of a shape (Dl_a,Dr_a,d**2,d**2) for OBC (Dl_a, Dr_a can vary between sites) or ndarray of a shape (D_a,D_a,d**2,d**2,N) for PBC
        MPO.
    
    Returns:
      b: ndarray of a shape (d**(2*N),d**(2*N))
        Superoperator in full Hilbert space.
    """
    if type(a) is list:
        bc = 'O'
        n = len(a)
        d2 = np.shape(a[0])[2]
    elif type(a) is np.ndarray:
        bc = 'P'
        n = np.shape(a)[4]
        d2 = np.shape(a)[2]
    d = int(round(np.sqrt(d2)))
    indexlist = []
    for x in itertools.product(np.arange(d,dtype=int),repeat=n):
        for y in itertools.product(np.arange(d,dtype=int),repeat=n):
            helplist = []
            for z in range(n):
                helplist.append(d*x[z]+y[z])
            indexlist.append(helplist)
    b = np.zeros((d2**n,d2**n),dtype=complex)
    for x in range(d2**n):
        for y in range(d2**n):
            if bc == 'O':
                if n == 1:
                    if np.shape(a[0])[0] == 1:
                        b[x,y] = a[0][0,0,indexlist[x][0],indexlist[y][0]]
                    else:
                        warnings.warn('Tensor networks with OBC and length one have to have bond dimension equal to one.')
                else:
                    aux = a[0][:,:,indexlist[x][0],indexlist[y][0]]
                    for z in range(1,n):
                        aux = aux @ a[z][:,:,indexlist[x][z],indexlist[y][z]]
                    b[x,y] = aux
            elif bc == 'P':
                aux = a[:,:,indexlist[x][0],indexlist[y][0],0]
                for z in range(1,n):
                    aux = aux @ a[:,:,indexlist[x][z],indexlist[y][z],z]
                b[x,y] = np.trace(aux)
    return b