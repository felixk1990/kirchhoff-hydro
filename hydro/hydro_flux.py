# @Author:  Felix Kramer
# @Date:   2021-06-03T11:02:57+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-06-03T11:03:48+02:00
# @License: MIT



import sys
import networkx as nx
import numpy as np
import scipy as sy
import scipy.integrate as si
import scipy.spatial as sp
import scipy.optimize as sc
import os.path as op
import os
import pickle
import scipy.linalg as lina
import random as rd
import init_integration
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)
import multiprocessing as mp

class flux_network:

    def __init__(self):
        self.beta=0.
        self.mode_boundary='absorbing_boundary'
        self.dict_in={}
        self.dict_out={}
        self.dict_edges={}
        self.dict_node_out={}
        self.dict_node_in={}

    def initialize(self, K):

        self.N=len(K.G.nodes)
        self.M=len(K.G.edges)
        self.c=np.zeros(self.N)
        self.list_e=list(K.G.edges())
        self.list_n=list(K.G.nodes())

        self.roots=self.find_roots(K.G)
        self.sinks=self.find_sinks(K.G)
        self.nodes_sinks=[K.G.nodes[sink]['label'] for sink in self.sinks]
        self.nodes_roots=[K.G.nodes[source]['label'] for source in self.roots]

        if self.mode_boundary=='absorbing_boundary':
            self.idx_eff=[i for i in range(self.N) if i not in self.nodes_sinks]
        elif self.mode_boundary=='mixed_boundary':
            self.idx_not_sinks=[i for i in range(self.N) if i not in self.nodes_sinks]
            self.idx_not_roots=[i for i in range(self.N) if i not in self.nodes_roots]

        for i,n in  enumerate(self.list_n):
            self.dict_in[n]=[]
            self.dict_out[n]=[]
            self.dict_node_out[n]=np.where(K.B[i,:]>0)[0]
            self.dict_node_in[n]=np.where(K.B[i,:]<0)[0]

        for j,e in  enumerate(self.list_e):

            alpha=e[1]
            omega=e[0]
            if K.B[alpha,j] > 0.:
                self.dict_edges[e]=[alpha,omega]
                self.dict_in[omega].append(alpha)
                self.dict_out[alpha].append(omega)

            elif K.B[alpha,j] < 0.:
                self.dict_edges[e]=[omega,alpha]
                self.dict_in[alpha].append(omega)
                self.dict_out[omega].append(alpha)

            else:
                print('and I say...whats going on? I say heyayayayayaaaaaaa...')

    def find_roots(self,G):
        roots=[]
        for n in G.nodes():
            if G.nodes[n]['source']>0:
                roots.append(n)
        return roots

    def find_sinks(self,G):
        sinks=[]
        for n in G.nodes():
            if G.nodes[n]['source']<0:
                sinks.append(n)
        return sinks

    def alpha_omega(self,G,j):

        for e in G.edges():

            if j == G.edges[e]['label']:
                # print('edge'+str(e))
                alpha=e[1]
                omega=e[0]

        return alpha,omega

    def calc_flows_pressures(self,K):

        OP=np.dot(K.B,np.dot(np.diag(K.C),K.BT))
        P,RES,RG,si=np.linalg.lstsq(OP,K.J,rcond=None)
        dP=np.dot(K.BT,P)
        Q=np.dot(np.diag(K.C),dP)
        K.dV=dP
        K.Q=Q

        return Q, dP, P

    def calc_flows_pressures_noise(self,graph_matrices):

        C_aux,K=graph_matrices
        OP=np.dot(K.B,np.dot(np.diag(C_aux),K.BT))
        P,RES,RG,si=np.linalg.lstsq(OP,K.J,rcond=None)
        dP=np.dot(K.BT,P)
        Q=np.multiply(C_aux,dP)

        return [Q,P,dP]

    def calc_flux_mixed_boundary_conditions(self,K):

        idx_sources=list(K.G.graph['sources'])
        idx_potential=list(K.G.graph['potentials'])
        D=np.dot(K.B,np.dot(np.diag(K.C),K.BT))

        # initial V,S are zero vectors with exception of given bopundary coordinates
        b=np.subtract(K.J,np.dot(D,K.V))
        L=D[:,:]
        n=len(L[0,:])
        for j in idx_potential:
                L[:,j]=np.zeros(n)
                L[j,j]=-1.
        X,RES,RG,si=np.linalg.lstsq(L,b,rcond=None)

        P=np.array(K.V[:])
        P[idx_sources]=X[idx_sources]
        S=np.array(K.J[:])
        S[idx_potential]=X[idx_potential]
        K.J=S[:]
        dP=np.dot(K.BT,P)
        Q=np.dot(np.diag(K.C),dP)

        return Q,dP,P

    def calc_PE(self,K):

        R_sq=np.power(K.R,2)
        V=np.divide(K.Q,R_sq*np.pi)
        return np.multiply(V,K.l/K.D)

class complex_flux_uptake_network(flux_network,object):

    def __init__(self):
        super(complex_flux_uptake_network,self).__init__()
        self.alpha=0.
        self.gamma=0.

    def calc_surface_transport_S(self, K,D):

        S=D*np.pi*np.divide(np.multiply(self.alpha,K.R),np.absolute(K.Q))

        return S

    def calc_uptake_rate_beta(self,PE,S):

        ones=np.ones(len(PE))
        A1=48.*ones
        A2=np.power(np.divide(self.alpha,S),2)
        A=np.divide(24*PE,np.add(A1,A2))

        B1=np.divide(S,PE)*8.
        B2=np.divide(np.power(self.alpha,2),np.multiply(PE,S)*6.)
        B=np.sqrt(np.add(ones,np.add(B1,B2)))

        beta=np.multiply(A,np.subtract(B,ones))

        return beta

    def calc_flux_orientations(self,K):

        G=K.G
        dict_incoming,dict_outcoming,dict_mem_n,dict_mem_nodes,dict_mem_e,dict_incoming_noroot,dict_incoming_noroot_e={},{},{},{},{},{},{}
        BQ=np.zeros((len(K.B[:,0]),len(K.B[0,:])))

        idx_e=[G.edges[e]['label'] for e in self.list_e]
        for n in self.list_n:

            idx_n=G.nodes[n]['label']
            BQ[idx_n,:]=np.multiply(K.B[idx_n,:],K.Q)
            b=BQ[idx_n,:]
            subhit1_list=[]
            subhit2_list=[]
            subhit3_list=[]

            dict_outcoming[n]=np.where( b > 0)[0]
            hit_list=np.where( b < 0)[0]
            dict_incoming[n]=hit_list

            for idx in hit_list:
                e=list_e[idx]

                if e[0]!=n:
                    subhit1_list.append(e[0])
                    dict_mem_e[e]=G.nodes[e[0]]['label']
                    if e[0]!=self.root:
                        subhit2_list.append(e[0])
                        subhit3_list.append(idx)
                else:
                    subhit1_list.append(e[1])
                    dict_mem_e[e]=G.nodes[e[1]]['label']
                    if e[1]!=self.root:
                        subhit2_list.append(e[1])
                        subhit3_list.append(idx)

            dict_mem_nodes[n]=subhit1_list
            dict_incoming_noroot[n]=subhit2_list
            dict_incoming_noroot_e[n]=subhit3_list
            dict_mem_n[n]=[(G.nodes[m]['label']) for m in subhit1_list]

        return dict_incoming,dict_outcoming,dict_mem_n,dict_mem_nodes,dict_mem_e,dict_incoming_noroot,dict_incoming_noroot_e,BQ

    def concentrate_love_repeat(self,K,F,*args):

        G=K.G
        dict_fluxes,push_list_nodes,master_list,dict_incoming,dict_outcoming,AQ,beta,PE=args
        nodes_left_undetermined=True
        dict_E,dict_idx_E,dict_incoming_noroot={},{},{}
        for n in self.list_n:

            dict_E[n]=self.list_e
            dict_idx_E[n]=[G.edges[e]['label'] for e in self.list_e ]
            dict_incoming_noroot[n]=[]

        while(nodes_left_undetermined):

            push_list_cache=[]
            push_list_cache_idx=[]
            for n in push_list_nodes:

                if sorted(dict_fluxes[n]) == sorted(dict_incoming[n]):
                    idx_n=G.nodes[n]['label']
                    if len(dict_outcoming[n])!=0:

                        self.c[idx_n]=np.divide(np.sum(F[dict_incoming[n]]),np.sum(AQ[dict_outcoming[n]]))
                        master_list.append(idx_n)
                        for idx_e in dict_outcoming[n]:
                            dict_fluxes[n].append(idx_e)
                            F[idx_e]=F[idx_e]*self.c[idx_n]
                        for i,e in enumerate(dict_E[n]):
                            for m in e:
                                idx_n=G.nodes[m]['label']
                                if (idx_n not in master_list) :
                                    dict_fluxes[m].append(dict_idx_E[n][i])
                                    if (idx_n not in push_list_cache_idx) :
                                        push_list_cache.append(m)
                                        push_list_cache_idx.append(idx_n)
                    else:
                        master_list.append(idx_n)

                else:
                    push_list_cache.append(n)

            push_list_nodes=push_list_cache

            if len(master_list)==self.N:
                nodes_left_undetermined=False

        return F

    def calc_nodal_concentrations(self, K, PE,beta,c0):

        AQ=np.absolute(np.multiply(np.add(np.ones(self.M),np.divide(beta,PE)),K.Q))
        F=np.multiply(AQ,np.exp(-beta))

        n_idx=G.nodes[self.root]['label']
        self.c[n_idx]=c0

        master_list=[n_idx]
        E=G.edges(self.root)
        push_list_nodes=[]
        dict_incoming,dict_outcoming,dict_mem_n,dict_mem_nodes,dict_mem_e,dict_incoming_noroot,dict_incoming_noroot_e,BQ=self.calc_flux_orientations(K)
        dict_fluxes={}

        for n in self.list_n:
            dict_fluxes[n]=[]
        for e in E:
            idx_e=G.edges[e]['label']
            if BQ[n_idx,idx_e]>0:
                idx_e=G.edges[e]['label']
                F[idx_e]=F[idx_e]*c0

                for n in e:

                    idx_n=G.nodes[n]['label']
                    if idx_n not in master_list:
                        push_list_nodes.append(n)
                        dict_fluxes[n].append(idx_e)

        F=self.concentrate_love_repeat(F,dict_fluxes,push_list_nodes,master_list,dict_incoming,dict_outcoming,AQ,beta,PE)

        return F,dict_incoming,dict_outcoming,dict_mem_n,dict_mem_nodes,dict_mem_e,dict_incoming_noroot,dict_incoming_noroot_e

    def recursive_topo(self,topo,topo_nodes,topo_edges,dict_incoming_noroot,dict_incoming_noroot_e,i,G):

        for n in topo[i]:
            topo_edges[i+1]=[idx for idx in dict_incoming_noroot_e[n]]
            topo_nodes[i+1]=[G.nodes[v]['label'] for v in dict_incoming_noroot[n]]
            topo[i+1]=[v for v in dict_incoming_noroot[n]]

        if len(topo[i+1])!=0:
            topo,topo_nodes,topo_edges=recursive_topo(topo,topo_nodes,topo_edges,dict_incoming_noroot,dict_incoming_noroot_e,i+1,G)

        return topo,topo_nodes,topo_edges

    def calc_absorption(self,J,beta,S,PE):

        ones=np.ones(len(beta))

        A=np.divide(J,np.add(ones,np.divide(beta,PE)))
        b=np.divide(np.power(self.alpha,2),np.multiply(S,PE))
        B=np.divide(np.subtract(2.*np.divide(S,beta),b/12.),np.add(ones,b/4.))
        C=np.subtract(ones,np.exp(-beta))

        phi=np.multiply(np.multiply(A,B),C)

        return phi

    def calc_PE_S_jacobian(self,flux_par,abs_par,G):

        L,F,R_sq=flux_par
        PE,S,beta,diff_R=abs_par
        ones=np.ones(self.M)
        INV=lina.pinv(np.dot(K.B,np.dot(np.diag(K.C),K.BT)))
        D=np.dot(np.dot(K.BT,INV),K.B)
        I=np.identity(M)

        SGN=np.ones(len(K.Q))
        SGN[np.where(K.Q<0.)[0]]=-1.
        Q=np.absolute(K.Q)
        f1= 2.*np.multiply(np.divide(PE,K.R),SGN)
        f2= np.multiply( np.divide(S,K.R)* (-1),SGN)
        f3= 4.* np.multiply(np.divide(Q,K.R),SGN)

        J_PE, J_S, J_Q= np.zeros((self.M,self.M)),np.zeros((self.M,self.M)),np.zeros((self.M,self.M))
        for i,c in enumerate(K.C):
            J_PE[i,:]= f1[i] * np.multiply(np.subtract( I[i,:], 2.* c * np.multiply( D[:,i], R_sq/R_sq[i] ) ),SGN)
            J_S[i,:]= f2[i] * np.multiply(np.subtract( 3.*I[i,:], 4.* c * np.multiply( D[:,i] , np.multiply( np.power(np.divide(Q[i],Q),2), np.power( np.divide( K.R,K.R[i]), 5 ) ) ) ),SGN)

            J_Q[i,:]= f3[i] * np.multiply(np.subtract( I[i,:], c*np.multiply( D[:,i], np.multiply( np.divide(L[i],L) , np.power( R_sq/R_sq[i] , 2 ) ) ) ),SGN)

        return J_PE,J_S,J_Q

    def calc_PE_beta_jacobian(self,K,flux_par,abs_par,J_PE,J_S):

        G=K.G
        L,F,R_sq=flux_par
        PE,S,beta,diff_R=abs_par
        J_beta=np.zeros((self.M,self.M))
        ones=np.ones(self.M)
        I=np.identity(self.M)

        f1=np.divide( np.power(self.alpha,2) ,6.*np.multiply(S,PE) )
        f2=np.power(np.divide(self.alpha,S),2)

        x=np.sqrt( np.add( np.add(ones,8.*np.divide(S,PE)),f1) )
        y=np.add( ones*48. , f2)

        B1= np.multiply(np.reciprocal(PE),np.subtract(beta,96.*np.divide(S,np.multiply(x,y))))
        B2= np.divide(np.add( np.multiply(np.divide(beta,S),f2) , np.reciprocal(x)*48. ),y)*2.
        B3= np.divide(np.multiply(f2,S),np.multiply(x,y))*2.

        for i,r in enumerate(K.R):

            J_beta[i,:] = np.add(J_beta[i,:],np.multiply( J_PE[i,:] , B1 ))
            J_beta[i,:] = np.add(J_beta[i,:],np.multiply( J_S[i,:] , B2 ))
            J_beta[i,:] = np.add(J_beta[i,:],np.multiply( I[i,:], B3/r))

        return J_beta

    def calc_flux_matrices(self,K,beta,PE,J_Q,J_beta,J_PE,dict_incoming,dict_outcoming):

        G=K.G
        Q=np.absolute(K.Q)
        idx_n_list=np.zeros(self.N,dtype=np.int8)
        idx_sink=G.nodes[self.sink]['label']
        idx_root=G.nodes[self.root]['label']

        # define tensor templates
        f1=np.add(np.ones(self.M),np.divide(beta,PE))
        f2=np.exp(-beta)
        flux_out= -np.ones(self.N)
        flux_in=np.multiply(f1,Q)
        for i,n in enumerate(self.list_n):
            idx_n_list[i]=G.nodes[n]['label']
            if idx_n_list[i] != idx_sink:
                flux_out[idx_n_list[i]]=np.sum(flux_in[dict_outcoming[n]])

        flux_in=np.multiply(f2,flux_in)
        A=np.outer(np.reciprocal(flux_out),flux_in)
        J_A=np.zeros((self.N,self.M,self.M))

        # define auxiliary varibles & lists, fix and sort signs
        X,Y,Z=J_A[:,:,:],J_A[:,:,:],J_A[:,:,:]
        # calculate auxiliary derivates
        DF_IN=np.zeros((self.M,self.M))
        DF_OUT=np.zeros((self.M,self.N))
        for j,e in enumerate(list_e):
            A1=np.multiply(J_Q[j,:],f1)
            A2=np.multiply(J_beta[j,:],np.multiply(Q,np.subtract(np.reciprocal(PE),f1)))
            A3=np.multiply(J_PE[j,:],np.divide(np.multiply(beta,Q),np.power(PE,2)))
            DF_IN[j,:]=np.multiply(f2,np.subtract(np.add(A1,A2),A3))

        for i,n in enumerate(self.list_n):
            idx_n=G.nodes[n]['label']
            idx_out=dict_outcoming[n]

            f3= np.divide(K.Q[idx_out],np.power(PE[idx_out],2))
            if idx_n != idx_sink and idx_n !=idx_root:
                for j,e in enumerate(self.list_e):
                    DF_OUT[j,idx_n]=np.sum( np.add( np.multiply( J_Q[j,idx_out],f1[idx_out] ), np.multiply( f3, np.subtract( np.multiply( J_beta[j,idx_out],PE[idx_out] ), np.multiply( beta[idx_out],J_PE[j,idx_out] ) ) ) ) )

            elif idx_n==idx_sink:
                A[idx_n,:]=0.
        # calculate jacobian
        for i,e1 in enumerate(self.list_e):
            idx_e1=G.edges[e1]['label']
            for j,e2 in enumerate(self.list_e):
                idx_e2=G.edges[e2]['label']
                J_A[:,idx_e1,idx_e2]= np.subtract(  DF_IN[idx_e1,idx_e2]* np.reciprocal(flux_out) , flux_in[idx_e2]* np.divide(DF_OUT[idx_e1,:],np.power(flux_out,2)))
        J_A[idx_root,:,:]=np.zeros((M,M))
        J_A[idx_sink,:,:]=np.zeros((M,M))

        return A,J_A

    def calc_flux_jacobian(self,K,flux_par,abs_par,dicts_par,J_PE,J_S,J_Q,J_beta):

        G=K.G
        L,F,R_sq=flux_par
        PE,S,beta,diff_R=abs_par
        dict_incoming,dict_outcoming,dict_mem_n,dict_mem_nodes,dict_mem_e,dict_incoming_noroot,dict_incoming_noroot_e=dicts_par

        # calc concentration jacobian
        J_C=np.zeros((self.M,N))

        idx_e_list=np.zeros(self.M,dtype=np.int8)
        idx_n_list=np.zeros(self.N,dtype=np.int8)
        Q=np.absolute(K.Q)
        A,J_A=self.calc_flux_matrices(K,beta,PE,J_Q,J_beta,J_PE,dict_incoming,dict_outcoming)
        idx_root=G.nodes[self.root]['label']
        idx_sink=G.nodes[self.sink]['label']
        SUM_J_AC=np.zeros((self.N,self.M))
        for i,n in enumerate(self.list_n):
            idx_n_list[i]=G.nodes[n]['label']
            if idx_n_list[i] != idx_root and idx_n_list[i]!=idx_sink:
                for j,e in enumerate(self.list_e):
                    idx_e_list[j]=( G.edges[e]['label'] )
                    SUM_J_AC[idx_n_list[i],idx_e_list[j]]=np.sum( np.multiply( J_A[ idx_n_list[i] , idx_e_list[j], dict_incoming[n]], self.c[dict_mem_n[n]]))

        for i,n in enumerate(self.list_n):
            idx_n=idx_n_list[i]
            if idx_n != idx_root and idx_n != idx_sink:
                idx_in=dict_incoming_noroot_e[n]
                J_C[:,idx_n]=SUM_J_AC[idx_n,:]

                if len(dict_incoming_noroot[n])!=0:
                    for j,e in enumerate(self.list_e):
                        SUM_JC=SUM_J_AC[:,idx_e_list[j]]
                        for k,m in enumerate(dict_incoming_noroot[n]):
                            J_C[idx_e_list[j],idx_n]=np.add( J_C[idx_e_list[j],idx_n],self.calc_iterative_increment(idx_root,dict_incoming_noroot_e,dict_incoming_noroot,A[idx_n,idx_in[k]],A,SUM_JC,m,n,G.nodes[m]['label']) )

        # calc flux jacobian
        J_F=np.zeros((self.M,self.M))
        identity=np.identity(self.M)
        A1=np.zeros(self.M)
        C1=np.zeros(self.M)
        for i,e1 in enumerate(self.list_e):
            for j,e2 in enumerate(self.list_e):
                idx_e=G.edges[e2]['label']
                A1[idx_e]=np.multiply(J_C[i,dict_mem_e[e2]],Q[idx_e])
                C1[idx_e]=self.c[dict_mem_e[e2]]
            J_F[i,:]=np.add( J_F[i,:], A1 )
            J_F[i,:]=np.add( J_F[i,:], np.multiply(J_Q[i,:], C1) )

        return J_F

    def calc_iterative_increment(self,idx_root,dict_incoming_noroot_e,dict_incoming_noroot,A_aux,A,SUM_JC,m_in,n,idx_in):

        increment=A_aux*SUM_JC[idx_in]
        for k,m in enumerate(dict_incoming_noroot[m_in]):
            increment=np.add(increment,self.calc_iterative_increment(idx_root,dict_incoming_noroot_e,dict_incoming_noroot,A_aux*A[idx_in,dict_incoming_noroot_e[m_in][k]],A,SUM_JC,m,m_in,G.nodes[m]['label']))

        return increment

    def calc_coefficient_jacobian(self,K,flux_par,abs_par,J_PE,J_S,J_beta):

        G=K.G
        L,F,R_sq=flux_par
        PE,S,beta,diff_R=abs_par

        J_COEFF=np.zeros((self.M,self.M))
        I=np.identity(self.M)
        ones=np.ones(self.M)

        f1=np.divide(np.power(self.alpha,2),np.multiply(PE,S))
        f2_a=np.add(ones,f1/4.)
        f2=np.reciprocal(f2_a)
        f3=np.divide(S,beta)
        B1=2.*np.multiply(np.reciprocal(beta),f2)
        B2=np.multiply(B1,f3)
        B3=np.multiply(np.multiply(f1,np.power(f2,2)),np.add(ones,6.*f3))/12.
        for i,r in enumerate(K.R):

            J_COEFF[i,:]=np.add(J_COEFF[i,:],np.multiply(J_S[i,:],B1))
            J_COEFF[i,:]=np.subtract(J_COEFF[i,:],np.multiply(J_beta[i,:],B2))
            J_COEFF[i,:]=np.subtract(J_COEFF[i,:],np.multiply(I[i,:],B3/r))

        return J_COEFF
    # @profile
    def calc_absorption_jacobian(flux_par,abs_par,dicts_par,G):

        L,F,R_sq=flux_par
        PE,S,beta,diff_R=abs_par

        J_PE,J_S,J_Q=self.calc_PE_S_jacobian(K,flux_par,abs_par)
        J_beta=self.calc_PE_beta_jacobian(K,flux_par,abs_par,J_PE,J_S)
        J_c=self.calc_flux_jacobian(K,flux_par,abs_par,dicts_par,J_PE,J_S,J_Q,J_beta)
        # J_c=calc_flux_jacobian_topological(flux_par,abs_par,dicts_par,G,J_PE,J_S,J_Q,J_beta)
        f1=np.divide(np.power(self.alpha,2),np.multiply(PE,S))
        J_coeff=calc_coefficient_jacobian(K,flux_par,abs_par,J_PE,J_S,J_beta)
        ones=np.ones(self.M)

        exp=np.exp(-beta)
        F=np.divide(F,np.multiply(exp,np.add(ones,np.divide(beta,PE))))
        coeff1=np.subtract(ones,exp)
        coeff2=np.divide(np.subtract(2.*np.divide(S,beta),f1/12.),np.add(ones,f1/4.))
        phi_jacobian=np.zeros((self.M,self.M))

        proxy1=np.zeros((self.M,self.M))
        proxy2=np.zeros((self.M,self.M))

        B1=np.multiply(coeff2,coeff1)
        B2=np.multiply(F,coeff1)
        B3=np.multiply(np.multiply(coeff2,F),exp)

        for i,R in enumerate(K.R):

            phi_jacobian[i,:]=np.add(phi_jacobian[i,:],np.multiply(J_c[i,:],B1))
            phi_jacobian[i,:]=np.add(phi_jacobian[i,:],np.multiply(J_coeff[i,:],B2))
            phi_jacobian[i,:]=np.add(phi_jacobian[i,:],np.multiply(J_beta[i,:],B3))

        return phi_jacobian

    def supply_pattern(self,K,mode):

        phi0=np.zeros(len(K.C))
        if 'random' in mode:
            phi0=np.random.random(len(K.C))
        if 'constant' in mode:
            phi0=np.ones(len(K.C))
        if 'gradient' in mode:
            dist={}
            for j,e in enumerate(K.G.edges()):
                d=np.linalg.norm(np.add(K.G.nodes[e[0]]['pos'],K.G.nodes[e[1]]['pos']))*0.5
                phi0[j]=1./d

        return phi0

class simple_flux_uptake_network(flux_network,object):

    def __init__(self):
        super(simple_flux_uptake_network,self).__init__()
        self.alpha=0.
        self.gamma=0.

    def update_stationary_operator(self,K):

        Q,dP,P=self.calc_flows_pressures(K)
        K.PE=self.calc_PE(K)

        A=np.pi*np.power(K.R,2)*(K.D/K.l)
        x,z,sinh_x,cosh_x,coth_x,e_up,e_down=self.compute_flux_pars(K)

        f1= np.multiply(z,A)
        f2= np.multiply(A,np.multiply(x,coth_x))*0.5
        f3= np.divide(np.multiply(A,x),sinh_x)*0.5

        self.B_eff=np.zeros((self.N,self.N))

        for i,n in enumerate(K.G.nodes()):
            self.B_eff[i,i]= np.sum(  np.add( np.multiply(K.B[i,:],f1),np.multiply(np.absolute(K.B[i,:]),f2))  )
            self.B_eff[i,self.dict_in[n]]= -np.multiply( e_up[self.dict_node_in[n]],f3[self.dict_node_in[n]] )
            self.B_eff[i,self.dict_out[n]]= -np.multiply( e_down[self.dict_node_out[n]],f3[self.dict_node_out[n]] )

    def solve_inlet_peak(self,K):

        B_new=np.delete(np.delete(self.B_eff,self.idx_sink,axis=0),self.idx_source,axis=1)
        b=self.B_eff[self.idx_not_sinks,:]
        S=np.subtract( K.J_C[self.idx_not_sinks], b[:,self.nodes_root]*K.C0 )

        A=np.linalg.inv(B_new)
        c=np.dot(A,S)

        idx=0
        for i,n in enumerate(K.G.nodes()):
            if i in self.nodes_roots:
                K.G.nodes[n]['concentrations']=K.C0
            else:
                K.G.nodes[n]['concentrations']=c[idx]
                idx+=1

        return c,B_new,K


    def solve_absorbing_boundary(self,K):

        B_new=self.B_eff[self.idx_eff,:]
        B_new=B_new[:,self.idx_eff]
        S=K.J_C[self.idx_eff]
        c=np.dot(np.linalg.inv(B_new),S)

        # idx=0
        C=np.zeros(self.N)
        C[self.idx_eff]=c[:]
        for i,n in enumerate(self.list_n):
             K.G.nodes[n]['concentrations']=C[i]
            # if i in self.nodes_sinks:
            #     K.G.nodes[n]['concentrations']=0.
            # else:
            #     K.G.nodes[n]['concentrations']=c[idx]
            #     idx+=1
            # if i in self.idx_eff:
            #
            # else:
            #      K.G.nodes[n]['concentrations']=0.

        return c,B_new,K

    def calc_profile_concentration(self,K):

        self.update_stationary_operator(K)

        # use absorbing boundaries + reduced equation system
        if self.mode_boundary=='absorbing_boundary':
            c,B_new,K=self.solve_absorbing_boundary(K)

        # use inlet delta peak + reduced equation system
        elif self.mode_boundary=='mixed_boundary':
            c,B_new,K=self.solve_inlet_peak(K)

        return c,B_new,K

    def calc_stationary_concentration(self,K):

        c,B_new,K=self.calc_profile_concentration(K)

        # set containers
        A=np.multiply(K.R,K.R)*np.pi*(K.D/K.l)
        J_a,J_b=np.zeros(self.M),np.zeros(self.M)
        phi=np.zeros(self.M)
        ones=np.ones(self.M)
        c_a,c_b=np.ones(self.M),np.ones(self.M)

        # calc coefficients
        for j,e in enumerate(self.list_e):
            a,b=self.dicts[0][e]
            c_a[j]=K.G.nodes[a]['concentrations']
            c_b[j]=K.G.nodes[b]['concentrations']

        K.PE=calc_PE(Q,K)
        x,z,sinh_x,cosh_x,coth_x,e_up,e_down=self.compute_flux_pars(K)

        f1= np.divide(x,sinh_x)*0.5
        f1_up=np.multiply( f1,e_up )
        f1_down=np.multiply( f1,e_down )

        F1=np.add( np.subtract( np.multiply(x,coth_x)*0.5 , f1_up), z)
        F2=np.subtract( np.subtract( np.multiply(x,coth_x)*0.5 , f1_down), z)

        f2= np.add( z, np.multiply(x,coth_x)*0.5 )
        f3= np.subtract( z ,np.multiply(x,coth_x)*0.5 )

        # calc edgewise absorption
        phi=np.add(np.multiply(c_a,F1) ,np.multiply( c_b,F2 ))
        phi=np.multiply( phi, A)

        J_a=np.multiply(A, np.subtract( np.multiply(f2,c_a) , np.multiply(f1_down,c_b )) )
        J_b=np.multiply(A, np.add( np.multiply(f3,c_b), np.multiply(f1_up,c_a )) )

        return c,J_a,J_b,phi

    def calc_absorption(self,R, K):

        # set containers
        phi=np.zeros(self.M)
        ones=np.ones(self.M)
        c_a,c_b=np.ones(self.M),np.ones(self.M)
        # calc coefficients
        for j,e in enumerate(self.list_e):
            a,b=self.dict_edges[e]
            c_a[j]=K.G.nodes[a]['concentrations']
            c_b[j]=K.G.nodes[b]['concentrations']

        K.PE=calc_PE(K)
        x,z,sinh_x,cosh_x,coth_x,e_up,e_down=self.compute_flux_pars(K)

        f1= np.divide(x,sinh_x)*0.5
        F1=np.add( np.subtract( np.multiply(x,coth_x)*0.5 , np.multiply( f1,e_up )), z)
        F2=np.add( np.subtract( np.multiply(x,coth_x)*0.5 , np.multiply( f1,e_down )), -z)

        # calc edgewise absorption
        phi=np.add(np.multiply(c_a,F1) ,np.multiply( c_b,F2 ))
        A=np.pi*np.multiply(R,R)*(K.D/K.l)

        return np.multiply( A, phi )

    def calc_flux_jacobian(self,R,*args):

        # unzip parameters
        L,K= args

        # init containers
        I=np.identity(self.M)
        J_PE, J_Q= np.zeros((self.M,self.M)),np.zeros((self.M,self.M))

        # set coefficients
        f1= 2.*np.divide(K.PE,R)
        f2= 4.* np.divide(K.Q,R)
        R_sq=np.power(R,2)
        INV=lina.pinv(np.dot(K.B,np.dot(np.diag(K.C),K.BT)))
        D=np.dot(np.dot(K.BT,INV),K.B)

        # calc jacobian
        for i,c in enumerate(K.C):
            J_PE[i,:]= f1[i] * np.subtract( I[i,:], 2.* c * np.multiply( D[:,i], R_sq/R_sq[i] ) )
            J_Q[i,:]= f2[i] * np.subtract( I[i,:], c*np.multiply( D[:,i], np.multiply( np.divide(L[i],L) , np.power( R_sq/R_sq[i] , 2 ) ) ) )

        return J_PE,J_Q

    def calc_concentration_jacobian(self, R,*args ):

        # unzip
        J_PE,c,K=args
        # set containers
        ones=np.ones(self.M)
        J_C=np.zeros((self.M,self.N))

        # set coefficients
        A=np.pi*np.multiply(R,R)*(K.D/K.l)
        x,z,sinh_x,cosh_x,coth_x,e_up,e_down=self.compute_flux_pars(K)
        f1= np.multiply(z,A)
        f2= np.multiply(np.multiply(x,coth_x),A)*0.5
        f3= np.divide(np.multiply(A,x),sinh_x)*0.5

        j_coth_x=np.power(np.divide(coth_x,cosh_x),2)
        f2= np.multiply(x,coth_x)*0.5
        f4=np.subtract( np.multiply( np.divide(z,x), coth_x) ,  np.multiply( z,j_coth_x )*0.5 )

        f_up=np.divide(np.multiply( e_up ,x ), sinh_x )*0.5
        f_down=np.divide( np.multiply( e_down ,x ), sinh_x )*0.5
        f5=np.divide( np.multiply(A, e_up), sinh_x )
        f6=np.divide( np.multiply(A, e_down), sinh_x )

        J_f_up=-np.multiply( f5, np.subtract( np.add( np.divide(z,x), x*0.25 ), np.multiply(z,coth_x)*0.5 ))
        J_f_down=-np.multiply( f6, np.subtract( np.subtract( np.divide(z,x), x*0.25 ), np.multiply(z,coth_x)*0.5 ))

        inv_B,c= self.calc_inv_B( K ,c)
        for j,e in enumerate(self.list_e):
            JB_eff=np.zeros((self.N,self.N))
            J_A=np.zeros((self.M,self.M))
            J_A[j,j]=2.*np.pi*R[j]*(K.D/K.l)
            for i,n in enumerate(self.list_n):

                b=K.B[i,:]
                JB_eff[i,i]=  np.sum( np.multiply( J_A, np.add( np.multiply(b,z), np.multiply(np.absolute(b),f2))  ) )+np.sum( np.multiply( J_PE[j,:], np.multiply(A, np.add( b*0.5, np.multiply(np.absolute(b),f4) ) ) ))
                JB_eff[i,self.dict_out[n]]= np.subtract( np.multiply( J_PE[j,self.dict_node_out[n]] , J_f_down[self.dict_node_out[n]] ) , np.multiply( J_A[j,self.dict_node_out[n]], f_down[self.dict_node_out[n]] ))
                JB_eff[i,self.dict_in[n]]=  np.subtract( np.multiply( J_PE[j,self.dict_node_in[n]] , J_f_up[self.dict_node_in[n]] ) , np.multiply( J_A[j,self.dict_node_in[n]], f_up[self.dict_node_in[n]] ))

            self.evaluate_jacobian(self,j,J_C,JB_eff,inv_B,c)

        return J_C

    def calc_absorption_jacobian(self,R,K):

        # set containers
        ones=np.ones(self.M)
        L=ones*K.l
        J_phi= np.zeros((self.M,self.M))
        phi=np.zeros(self.M)
        c_a,c_b,c_n=np.zeros(self.M),np.zeros(self.M),np.zeros(self.N)
        alphas,omegas=[],[]

        # calc coefficients
        for j,e in enumerate(self.list_e):
            a,b=self.dict_edges[e]
            c_a[j]=K.G.nodes[a]['concentrations']
            c_b[j]=K.G.nodes[b]['concentrations']
            alphas.append(a)
            omegas.append(b)
        for i,n in enumerate(self.list_n):
            c_n[i]=K.G.nodes[n]['concentrations']

        x,z,sinh_x,cosh_x,coth_x,e_up,e_down=self.compute_flux_pars(K)

        f1= 0.5*np.divide(x,sinh_x)
        F1=np.add( np.subtract( 0.5*np.multiply(x,coth_x) , np.multiply( f1,e_up )), z )
        F2=np.subtract( np.subtract( 0.5*np.multiply(x,coth_x) , np.multiply( f1,e_down )), z)

        f2_up=np.subtract( np.multiply( np.divide(PE,x), np.subtract( cosh_x, e_up )), np.divide(z,sinh_x))
        f3_up=np.add( np.multiply( e_up, np.subtract( np.multiply(coth_x,z), 0.5*x ) ) , sinh_x)
        f2_down=np.subtract(np.multiply( np.divide(PE,x), np.subtract( cosh_x, e_down )), np.divide(z,sinh_x))
        f3_down=np.subtract( np.multiply( e_down, np.add( np.multiply(coth_x,z), 0.5*x  ) ), sinh_x )

        F3= 0.5*np.divide( np.add(f2_up, f3_up) , sinh_x)
        F4= 0.5*np.divide( np.add(f2_down, f3_down) , sinh_x)
        phi=np.add( np.multiply(c_a,F1) ,np.multiply(c_b,F2 ) )

        # calc jacobian
        J_PE,J_Q= self.calc_flux_jacobian(R,L,K)
        A=np.pi*np.multiply(R,R)*(K.D/K.l)
        J_A=2.*np.pi*np.diag(R)*(K.D/K.l)
        J_C=self.calc_concentration_jacobian( R,J_PE,c_n,K)

        qa=np.multiply(A,c_a)
        qb=np.multiply(A,c_b)
        q1=np.multiply( A, F1 )
        q2=np.multiply( A, F2 )

        for j,e in enumerate(self.list_e):
            J_phi[j,:]=np.add(J_phi[j,:],np.multiply( J_A[j,:], phi))

            J_phi[j,:]=np.add(J_phi[j,:], np.multiply( J_C[j,alphas], q1 ))
            J_phi[j,:]=np.add(J_phi[j,:], np.multiply( J_C[j,omegas], q2 ))

            J_phi[j,:]=np.add(J_phi[j,:],np.multiply( J_PE[j,:], np.multiply(qa,F3)))
            J_phi[j,:]=np.add(J_phi[j,:],np.multiply( J_PE[j,:], np.multiply(qb,F4)))

        return J_phi

    def compute_flux_pars(self,K):

        x=np.sqrt( np.add( np.power(K.PE,2),K.beta ) )
        z=PE*0.5
        sinh_x=np.sinh(x*0.5)
        cosh_x=np.cosh(x*0.5)
        coth_x=np.reciprocal(np.tanh(x*0.5))
        e_up=np.exp(z)
        e_down=np.exp(-z)

        return x,z,sinh_x,cosh_x,coth_x,e_up,e_down

# overflow handling
class simple_flux_uptake_network_OFH(simple_flux_uptake_network,object):

    def __init__(self):
        super(simple_flux_uptake_network_OFH,self).__init__()
        self.crit_pe=50.


    def initialize_broken_link(self, K):

        self.initialize(K)

        # x=int(K.percentage_broken*nx.number_of_edges(K.G))
        broken_sets=[]
        num_sets=50000
        K.AUX=nx.Graph(K.G)
        for i in range(num_sets):
            # cond,idx=self.generate_coherent_closure(K.AUX,x)
            cond,idx=self.generate_coherent_closure(K)
            if cond:
                broken_sets.append(idx)

        K.broken_sets=broken_sets
        print(len(K.broken_sets))
        if len(K.broken_sets)==0:
            sys.exit('nothing broken here... srsly check initialize_broken_link() though')
        # shear_sq,dV_sq, F_sq, avg_phi = self.calc_sq_flow_broken_link(K)
        # shear_sq,dV_sq, F_sq = self.calc_sq_flow_broken_link(K)
        diss,dV_sq,F_sq,R = self.calc_sq_flow_broken_link(K)
        K.dV_sq=dV_sq[:]
        K.F=np.zeros(len(K.R))

    def update_stationary_operator(self,K):

        Q,dP,P=self.calc_flows_pressures(K)
        K.PE=self.calc_PE(K)

        A=np.pi*np.power(K.R,2)*(K.D/K.l)
        x,z,e_up_sinh_x,e_down_sinh_x,coth_x,idx_pack=self.compute_flux_pars(K)

        f1= np.multiply(z,A)
        f2= np.multiply(A,np.multiply(x,coth_x))*0.5

        f3= np.multiply(np.multiply(A,x),e_up_sinh_x)*0.5
        f4= np.multiply(np.multiply(A,x),e_down_sinh_x)*0.5

        self.B_eff=np.zeros((self.N,self.N))

        for i,n in enumerate(K.G.nodes()):
            self.B_eff[i,i]= np.sum(  np.add( np.multiply(K.B[i,:],f1),np.multiply(np.absolute(K.B[i,:]),f2))  )
            self.B_eff[i,self.dict_in[n]]= -f3[self.dict_node_in[n]]
            self.B_eff[i,self.dict_out[n]]= -f4[self.dict_node_out[n]]

    def update_stationary_operator_noise(self,K,R,flow_observables):

        K.Q=flow_observables[0]
        R_sq=np.power(R,2)
        V=np.divide(K.Q,R_sq*np.pi)
        K.PE=np.multiply(V,K.l/K.D)

        A=np.pi*R_sq*(K.D/K.l)
        x,z,e_up_sinh_x,e_down_sinh_x,coth_x,idx_pack = self.compute_flux_pars(K)

        f1= np.multiply(z,A)
        f2= np.multiply(A,np.multiply(x,coth_x))*0.5

        f3= np.multiply(np.multiply(A,x),e_up_sinh_x)*0.5
        f4= np.multiply(np.multiply(A,x),e_down_sinh_x)*0.5

        self.B_eff=np.zeros((self.N,self.N))

        for i,n in enumerate(K.G.nodes()):
            self.B_eff[i,i]= np.sum(  np.add( np.multiply(K.B[i,:],f1),np.multiply(np.absolute(K.B[i,:]),f2))  )
            self.B_eff[i,self.dict_in[n]]= -f3[self.dict_node_in[n]]
            self.B_eff[i,self.dict_out[n]]= -f4[self.dict_node_out[n]]

    def generate_coherent_closure_deterministic(self,H,x):

        idx=rd.sample(range(len(self.list_e)),x)
        for e in idx:
            H.remove_edge(*self.list_e[e])
        cond=nx.is_connected(H)

        for e in idx:
            H.add_edge(*self.list_e[e])

        return cond,idx

    def generate_coherent_closure(self,K):

        prob=np.random.sample(self.M)
        idx=np.where(   prob <= K.percentage_broken )[0]

        for e in idx:
            K.AUX.remove_edge(*self.list_e[e])
        cond=nx.is_connected(K.AUX)

        for e in idx:
            K.AUX.add_edge(*self.list_e[e])

        return cond,idx

    def break_link(self,K,idx):

        C_aux=np.array(K.C[:])
        C_aux[idx]=np.power(10.,-20)

        return C_aux

    def calc_sq_flow_broken_link(self,K):

        # block p percent of the edges per realization

        idx=rd.choices(K.broken_sets,k=K.num_iteration)
        C_broken_ensemble=[self.break_link(K,i) for i in idx]
        graph_matrices=[[C_broken_ensemble[i],K] for i in range(K.num_iteration)]

        # calc the flow landscapes for each realization
        # pool = mp.Pool(processes=4)
        # with mp.Pool(processes=4) as pool:
        #     flow_observables=list(pool.map(self.calc_flows_pressures_noise,graph_matrices))

        flow_observables=list(map(self.calc_flows_pressures_noise,graph_matrices))

        # calc ensemble averages
        F_sq=np.power([fo[0] for fo in flow_observables],2)
        dV_sq=np.power([fo[2] for fo in flow_observables],2)
        R_sq=[np.sqrt(C_broken_ensemble[i]/K.k)  for i in range(K.num_iteration)]
        R_cb=[np.power(C_broken_ensemble[i]/K.k,0.75)  for i in range(K.num_iteration)]
        R=[np.power(C_broken_ensemble[i]/K.k,0.25)  for i in range(K.num_iteration)]
        # PHI=list(map( self.calc_absorption_noise , graph_matrices, flow_observables ) )
        # PHI=[ self.calc_absorption_noise(K,fo) for  fo in flow_observables  ]

        # avg_shear_sq=np.sum(np.multiply(dV_sq,R_sq),axis=0)/float(K.num_iteration)
        avg_diss=np.sum(np.multiply(dV_sq,R_cb),axis=0)/float(K.num_iteration)
        avg_R=np.mean(R,axis=0)
        avg_dV_sq=np.mean(dV_sq,axis=0)
        avg_F_sq= np.mean(F_sq,axis=0)
        # avg_PHI= np.mean(PHI,axis=0)

        # return avg_shear_sq,avg_dV_sq,avg_F_sq,avg_PHI
        # return avg_shear_sq,avg_dV_sq,avg_F_sq
        return avg_diss,avg_dV_sq,avg_F_sq,avg_R

    def calc_sq_flow_noise(self,K):

        # block p percent of the edges per realization

        idx=rd.choices(K.broken_sets,k=K.num_iteration)
        C_broken_ensemble=[self.break_link(K,i) for i in idx]
        graph_matrices=[[C_broken_ensemble[i],K] for i in range(K.num_iteration)]

        # calc the flow landscapes for each realization
        # pool = mp.Pool(processes=4)
        # with mp.Pool(processes=4) as pool:
        #     flow_observables=list(pool.map(self.calc_flows_pressures_noise,graph_matrices))

        flow_observables=list(map(self.calc_flows_pressures_noise,graph_matrices))

        # calc ensemble averages
        F_sq=np.power([fo[0] for fo in flow_observables],2)
        dV_sq=np.power([fo[2] for fo in flow_observables],2)
        R_sq=[np.sqrt(C_broken_ensemble[i]/K.k)  for i in range(K.num_iteration)]
        PHI=list(map( self.calc_absorption_noise , graph_matrices, flow_observables ) )
        # PHI=[ self.calc_absorption_noise(K,fo) for  fo in flow_observables  ]

        avg_shear_sq=np.sum(np.multiply(dV_sq,R_sq),axis=0)/float(K.num_iteration)
        avg_dV_sq=np.mean(dV_sq,axis=0)
        avg_F_sq= np.mean(F_sq,axis=0)
        avg_PHI= np.mean(PHI,axis=0)

        return avg_shear_sq,avg_dV_sq,avg_F_sq,avg_PHI
        # return avg_shear_sq,avg_dV_sq,avg_F_sq

    def calc_absorption_noise(self, graph_matrices, flow_observables):

        C_aux,K=graph_matrices
        R=np.power(C_aux/K.k,0.25)
        self.update_stationary_operator_noise(K, R ,flow_observables)

        # use absorbing boundaries + reduced equation system
        if self.mode_boundary=='absorbing_boundary':
            c,B_new,K=self.solve_absorbing_boundary(K)

        # use inlet delta peak + reduced equation system
        elif self.mode_boundary=='mixed_boundary':
            c,B_new,K=self.solve_inlet_peak(K)

        return self.calc_absorption(R, K)

    def calc_absorption(self,R, K):

        # set containers
        c_a,c_b=np.ones(self.M),np.ones(self.M)
        # calc coefficients
        for j,e in enumerate(self.list_e):
            a,b=self.dict_edges[e]
            c_a[j]=K.G.nodes[a]['concentrations']
            c_b[j]=K.G.nodes[b]['concentrations']

        x,z,e_up_sinh_x,e_down_sinh_x,coth_x,idx_pack=self.compute_flux_pars(K)

        f1_up= np.multiply(x,e_up_sinh_x)*0.5
        f1_down= np.multiply(x,e_down_sinh_x)*0.5
        F1=np.add( np.subtract( np.multiply(x,coth_x)*0.5 , f1_up), z)
        F2=np.add( np.subtract( np.multiply(x,coth_x)*0.5 , f1_down), -z)
        # calc edgewise absorption
        phi=np.add(np.multiply(c_a,F1) ,np.multiply( c_b,F2 ))
        A=np.pi*np.multiply(R,R)*(K.D/K.l)

        return np.multiply( A, phi )

    def calc_coefficients( self, R, *args  ):

        # unzip
        J_PE,c,K=args
        dict_coeff={}

        A=np.pi*np.multiply(R,R)*(K.D/K.l)
        x,z,e_up_sinh_x,e_down_sinh_x,coth_x,idx_pack=self.compute_flux_pars(K)

        f1= np.multiply(z,A)
        f2= np.multiply(np.multiply(x,coth_x),A)*0.5

        f3= np.multiply(np.multiply(A,x),e_up_sinh_x)*0.5
        f4= np.multiply(np.multiply(A,x),e_down_sinh_x)*0.5

        j_coth_x=np.zeros(self.M)
        idx_lower=idx_pack[0]

        # subcritical
        j_coth_x[idx_lower]=np.power(np.divide(coth_x[idx_lower],np.cosh(x[idx_lower])),2)
        # overcritical
        # j_coth_x[idx_over]=0.

        f2= np.multiply(x,coth_x)*0.5
        f4=np.subtract( np.multiply( np.divide(z,x), coth_x) ,  np.multiply( z,j_coth_x )*0.5 )

        f_up=np.multiply( e_up_sinh_x ,x )*0.5
        f_down=np.multiply( e_down_sinh_x ,x )*0.5
        f5= np.multiply(A, e_up_sinh_x )
        f6= np.multiply(A, e_down_sinh_x)

        J_f_up=-np.multiply( f5, np.subtract( np.add( np.divide(z,x), x*0.25 ), np.multiply(z,coth_x)*0.5 ))
        J_f_down=-np.multiply( f6, np.subtract( np.subtract( np.divide(z,x), x*0.25 ), np.multiply(z,coth_x)*0.5 ))
        dict_coeff['f_up']=f_up
        dict_coeff['f_down']=f_down
        dict_coeff['J_f_up']=J_f_up
        dict_coeff['J_f_down']=J_f_down

        flux_sum_1=np.array([ np.add( np.multiply(K.B[i,:],z), np.multiply(np.absolute(K.B[i,:]),f2)) for i,n in enumerate(self.list_n)])
        flux_sum_2=np.array([ np.multiply(A, np.add( K.B[i,:]*0.5, np.multiply(np.absolute(K.B[i,:]),f4) ) ) for i,n in enumerate(self.list_n)])
        dict_coeff['flux_sum_1']=flux_sum_1
        dict_coeff['flux_sum_2']=flux_sum_2

        return dict_coeff

    def calc_incidence_jacobian_diag(self, flux_sum_1,flux_sum_2,pars ):

        J_diag,J_PE_j,J_A,j=pars
        JB_eff=  J_diag * flux_sum_1[j] + np.sum( np.multiply( J_PE_j, flux_sum_2  ))

        return JB_eff

    def calc_incidence_jacobian_dev(self,JB_eff,dict_coeff,pars):

        J_diag,J_PE_j,J_A,j=pars
        for i,n in enumerate(self.list_n):

            JB_eff[i,self.dict_out[n]]= np.subtract( np.multiply( J_PE_j[self.dict_node_out[n]] , dict_coeff['J_f_down'][self.dict_node_out[n]] ), np.multiply( J_A[self.dict_node_out[n]], dict_coeff['f_down'][self.dict_node_out[n]] ))

            JB_eff[i,self.dict_in[n]]=  np.subtract( np.multiply( J_PE_j[self.dict_node_in[n]] , dict_coeff['J_f_up'][self.dict_node_in[n]] ) , np.multiply( J_A[self.dict_node_in[n]], dict_coeff['f_up'][self.dict_node_in[n]] ))

    def calc_inv_B(self, K ,c):

        # inlet peak
        if self.mode_boundary=='mixed_boundary':

            B_new=self.B_eff[self.idx_not_sinks,:]
            B_new=B_new[:,self.idx_not_roots]
            c=c[self.idx_not_sinks]

        # absorbing boundary
        elif self.mode_boundary=='absorbing_boundary':

            B_new=self.B_eff[self.idx_eff,:]
            B_new=B_new[:,self.idx_eff]
            c=c[self.idx_eff]

        inv_B=np.linalg.inv(B_new)

        return inv_B, c

    def evaluate_jacobian(self,j,J_C,JB_eff,inv_B,c):

        # absorbing boundary
        if self.mode_boundary=='absorbing_boundary':

            JB_new=JB_eff[self.idx_eff,:]
            JB_new=JB_new[:,self.idx_eff]
            J_C[j,self.idx_eff]=-np.dot(inv_B, np.dot( JB_new, c ))

        # inlet peak
        elif self.mode_boundary=='mixed_boundary':

            JB_new=JB_eff[self.idx_not_sinks,:]
            JB_new=JB_new[:,self.idx_not_roots]
            J_C[j,self.idx_not_sinks]=-np.dot(inv_B, np.dot( JB_new, c ))

    def calc_concentration_jacobian(self, R,*args ):

        # unzip
        J_PE,c,K=args
        # set coefficients
        dict_coeff=self.calc_coefficients( R,J_PE,c,K )
        inv_B, c = self.calc_inv_B(K ,c)

        J_C=np.zeros((self.M,self.N))
        J_diag=R*2.*np.pi*(K.D/K.l)
        J_A=np.zeros(self.M)
        for j,e in enumerate(K.G.edges()):
            J_A[j-1]=0.
            J_A[j]=J_diag[j]
            J_PE_j=J_PE[j,:]
            pars=[  [J_diag[j],J_PE_j,J_A,j] for i in range(self.N) ]

            JB_eff=np.diag( list( map(self.calc_incidence_jacobian_diag, dict_coeff['flux_sum_1'], dict_coeff['flux_sum_2'],pars) ) )
            self.calc_incidence_jacobian_dev(JB_eff,dict_coeff,pars[0])
            self.evaluate_jacobian(j,J_C,JB_eff,inv_B,c)

        return J_C

    def calc_absorption_jacobian(self,R, K):

        # set containers
        ones=np.ones(self.M)
        L=ones*K.l
        J_phi= np.zeros((self.M,self.M))
        phi=np.zeros(self.M)
        c_a,c_b,c_n=np.zeros(self.M),np.zeros(self.M),np.zeros(self.N)
        alphas,omegas=[],[]
        # calc coefficients
        for j,e in enumerate(self.list_e):
            a,b=self.dict_edges[e]
            c_a[j]=K.G.nodes[a]['concentrations']
            c_b[j]=K.G.nodes[b]['concentrations']
            alphas.append(a)
            omegas.append(b)
        for i,n in enumerate(self.list_n):
            c_n[i]=K.G.nodes[n]['concentrations']
        #
        # PE=self.calc_PE(K)
        x,z,e_up_sinh_x,e_down_sinh_x,coth_x,idx_pack=self.compute_flux_pars(K)

        f1_up= 0.5*np.multiply(x,e_up_sinh_x)
        f1_down= 0.5*np.multiply(x,e_down_sinh_x)

        F1=np.add( np.subtract( 0.5*np.multiply(x,coth_x) ,  f1_up ), z )
        F2=np.subtract( np.subtract( 0.5*np.multiply(x,coth_x) , f1_down ), z)
        F3,F4=self.calc_absorption_jacobian_coefficients(idx_pack, x,z,e_up_sinh_x,e_down_sinh_x,coth_x)

        phi=np.add( np.multiply(c_a,F1) ,np.multiply(c_b,F2 ) )

        # calc jacobian
        J_PE,J_Q= self.calc_flux_jacobian(R,L,K)
        A=np.pi*np.multiply(R,R)*(K.D/K.l)
        J_A=2.*np.pi*np.diag(R)*(K.D/K.l)
        J_C=self.calc_concentration_jacobian( R,J_PE,c_n,K )

        qa=np.multiply(A,c_a)
        qb=np.multiply(A,c_b)
        q1=np.multiply( A, F1 )
        q2=np.multiply( A, F2 )

        for j,e in enumerate(self.list_e):

            J_phi[j,:]=np.add(J_phi[j,:],np.multiply( J_A[j,:], phi))

            J_phi[j,:]=np.add(J_phi[j,:], np.multiply( J_C[j,alphas], q1 ))
            J_phi[j,:]=np.add(J_phi[j,:], np.multiply( J_C[j,omegas], q2 ))

            J_phi[j,:]=np.add(J_phi[j,:],np.multiply( J_PE[j,:], np.multiply(qa,F3)))
            J_phi[j,:]=np.add(J_phi[j,:],np.multiply( J_PE[j,:], np.multiply(qb,F4)))

        return J_phi

    def calc_absorption_jacobian_coefficients(self,idx_pack, *args):

        F3=np.zeros(self.M)
        F4=np.zeros(self.M)
        idx_lower=idx_pack[0]
        idx_over=idx_pack[1]
        x,z,e_up_sinh_x,e_down_sinh_x,coth_x=args
        # subcritical
        sinh_x=np.sinh(x[idx_lower]*0.5)
        cosh_x=np.cosh(x[idx_lower]*0.5)
        e_up=np.exp(z[idx_lower])
        e_down=np.exp(-z[idx_lower])

        f2_up=np.subtract( np.multiply( np.divide(2.*z[idx_lower],x[idx_lower]), np.subtract( cosh_x, e_up )), np.divide(z[idx_lower],sinh_x))
        f3_up=np.add( np.multiply( e_up, np.subtract( np.multiply(coth_x[idx_lower],z[idx_lower]), 0.5*x[idx_lower] ) ) , sinh_x)
        f2_down=np.subtract(np.multiply( np.divide(2.*z[idx_lower],x[idx_lower]), np.subtract( cosh_x, e_down )), np.divide(z[idx_lower],sinh_x))
        f3_down=np.subtract( np.multiply( e_down, np.add( np.multiply(coth_x[idx_lower],z[idx_lower]), 0.5*x[idx_lower]  ) ), sinh_x )

        F3[idx_lower]= 0.5*np.divide( np.add(f2_up, f3_up) , sinh_x)
        F4[idx_lower]= 0.5*np.divide( np.add(f2_down, f3_down) , sinh_x)


        # overcritical
        f2_up= np.multiply( np.divide(2.*z[idx_over],x[idx_over]), np.subtract( coth_x[idx_over], 2. ))
        f3_up=np.add( 2.* np.subtract( np.multiply(coth_x[idx_over],z[idx_over]), 0.5*x[idx_over] )  , 1.)

        f2_down=np.multiply( np.divide(2.*z[idx_over],x[idx_over]), coth_x[idx_over] )
        # f3_down=np.zeros(len(idx_over))

        F3[idx_over]= 0.5* np.add(f2_up, f3_up)
        F4[idx_over]= 0.5* f2_down

        return F3,F4

    def compute_flux_pars(self,K):

        x=np.sqrt( np.add( np.power(K.PE,2),K.beta ) )
        z=K.PE*0.5

        e_up_sinh_x=np.zeros(self.M)
        e_down_sinh_x=np.zeros(self.M)
        coth_x=np.zeros(self.M)
        # establish the use of converging limit expressions to prevent overflow error
        idx_lower=np.where(np.absolute(K.PE)<self.crit_pe)[0]
        idx_over_pos=np.where((np.absolute(K.PE)>=self.crit_pe) & (K.PE > 0))[0]
        idx_over_neg=np.where((np.absolute(K.PE)>=self.crit_pe) & (K.PE < 0))[0]
        idx_pack=[list(idx_lower),list(idx_over_pos)+list(idx_over_neg)]

        # subcritical pe
        e_up=np.exp(z[idx_lower])
        e_down=np.exp(-z[idx_lower])
        e_up_sinh_x[idx_lower]=e_up/np.sinh(x[idx_lower]*0.5)
        e_down_sinh_x[idx_lower]=e_down/np.sinh(x[idx_lower]*0.5)
        coth_x[idx_lower]=np.reciprocal(np.tanh(x[idx_lower]*0.5))

        # overcriticial, pe positive
        e_up_sinh_x[idx_over_pos]=2.
        e_down_sinh_x[idx_over_pos]=0.
        coth_x[idx_over_pos]=1.

        # overcriticial, pe negative
        e_up_sinh_x[idx_over_neg]=0.
        e_down_sinh_x[idx_over_neg]=2.
        coth_x[idx_over_neg]=1.

        return x,z,e_up_sinh_x,e_down_sinh_x,coth_x,idx_pack
