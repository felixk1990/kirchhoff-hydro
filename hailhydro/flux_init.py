# @Author:  Felix Kramer
# @Date:   2021-06-03T11:02:57+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-06-03T18:25:32+02:00
# @License: MIT

import networkx as nx
import numpy as np
import scipy.linalg as lina
from flow_init import *
from flow_random import *

class flux(flow, object):

    def __init__(self,*args):
        super(flux,self).__init__(args[0])

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

# class simple_flux_uptake_network(flux_network,object):
#
#     def __init__(self):
#         super(simple_flux_uptake_network,self).__init__()
#         self.alpha=0.
#         self.gamma=0.
#
#     def update_stationary_operator(self,K):
#
#         Q,dP,P=self.calc_flows_pressures(K)
#         K.PE=self.calc_PE(K)
#
#         A=np.pi*np.power(K.R,2)*(K.D/K.l)
#         x,z,sinh_x,cosh_x,coth_x,e_up,e_down=self.compute_flux_pars(K)
#
#         f1= np.multiply(z,A)
#         f2= np.multiply(A,np.multiply(x,coth_x))*0.5
#         f3= np.divide(np.multiply(A,x),sinh_x)*0.5
#
#         self.B_eff=np.zeros((self.N,self.N))
#
#         for i,n in enumerate(K.G.nodes()):
#             self.B_eff[i,i]= np.sum(  np.add( np.multiply(K.B[i,:],f1),np.multiply(np.absolute(K.B[i,:]),f2))  )
#             self.B_eff[i,self.dict_in[n]]= -np.multiply( e_up[self.dict_node_in[n]],f3[self.dict_node_in[n]] )
#             self.B_eff[i,self.dict_out[n]]= -np.multiply( e_down[self.dict_node_out[n]],f3[self.dict_node_out[n]] )
#
#     def solve_inlet_peak(self,K):
#
#         B_new=np.delete(np.delete(self.B_eff,self.idx_sink,axis=0),self.idx_source,axis=1)
#         b=self.B_eff[self.idx_not_sinks,:]
#         S=np.subtract( K.J_C[self.idx_not_sinks], b[:,self.nodes_root]*K.C0 )
#
#         A=np.linalg.inv(B_new)
#         c=np.dot(A,S)
#
#         idx=0
#         for i,n in enumerate(K.G.nodes()):
#             if i in self.nodes_roots:
#                 K.G.nodes[n]['concentrations']=K.C0
#             else:
#                 K.G.nodes[n]['concentrations']=c[idx]
#                 idx+=1
#
#         return c,B_new,K
#
#     def solve_absorbing_boundary(self,K):
#
#         B_new=self.B_eff[self.idx_eff,:]
#         B_new=B_new[:,self.idx_eff]
#         S=K.J_C[self.idx_eff]
#         c=np.dot(np.linalg.inv(B_new),S)
#
#         # idx=0
#         C=np.zeros(self.N)
#         C[self.idx_eff]=c[:]
#         for i,n in enumerate(self.list_n):
#              K.G.nodes[n]['concentrations']=C[i]
#             # if i in self.nodes_sinks:
#             #     K.G.nodes[n]['concentrations']=0.
#             # else:
#             #     K.G.nodes[n]['concentrations']=c[idx]
#             #     idx+=1
#             # if i in self.idx_eff:
#             #
#             # else:
#             #      K.G.nodes[n]['concentrations']=0.
#
#         return c,B_new,K
#
#     def calc_profile_concentration(self,K):
#
#         self.update_stationary_operator(K)
#
#         # use absorbing boundaries + reduced equation system
#         if self.mode_boundary=='absorbing_boundary':
#             c,B_new,K=self.solve_absorbing_boundary(K)
#
#         # use inlet delta peak + reduced equation system
#         elif self.mode_boundary=='mixed_boundary':
#             c,B_new,K=self.solve_inlet_peak(K)
#
#         return c,B_new,K
#
#     def calc_stationary_concentration(self,K):
#
#         c,B_new,K=self.calc_profile_concentration(K)
#
#         # set containers
#         A=np.multiply(K.R,K.R)*np.pi*(K.D/K.l)
#         J_a,J_b=np.zeros(self.M),np.zeros(self.M)
#         phi=np.zeros(self.M)
#         ones=np.ones(self.M)
#         c_a,c_b=np.ones(self.M),np.ones(self.M)
#
#         # calc coefficients
#         for j,e in enumerate(self.list_e):
#             a,b=self.dicts[0][e]
#             c_a[j]=K.G.nodes[a]['concentrations']
#             c_b[j]=K.G.nodes[b]['concentrations']
#
#         K.PE=calc_PE(Q,K)
#         x,z,sinh_x,cosh_x,coth_x,e_up,e_down=self.compute_flux_pars(K)
#
#         f1= np.divide(x,sinh_x)*0.5
#         f1_up=np.multiply( f1,e_up )
#         f1_down=np.multiply( f1,e_down )
#
#         F1=np.add( np.subtract( np.multiply(x,coth_x)*0.5 , f1_up), z)
#         F2=np.subtract( np.subtract( np.multiply(x,coth_x)*0.5 , f1_down), z)
#
#         f2= np.add( z, np.multiply(x,coth_x)*0.5 )
#         f3= np.subtract( z ,np.multiply(x,coth_x)*0.5 )
#
#         # calc edgewise absorption
#         phi=np.add(np.multiply(c_a,F1) ,np.multiply( c_b,F2 ))
#         phi=np.multiply( phi, A)
#
#         J_a=np.multiply(A, np.subtract( np.multiply(f2,c_a) , np.multiply(f1_down,c_b )) )
#         J_b=np.multiply(A, np.add( np.multiply(f3,c_b), np.multiply(f1_up,c_a )) )
#
#         return c,J_a,J_b,phi
#
#     def calc_absorption(self,R, K):
#
#         # set containers
#         phi=np.zeros(self.M)
#         ones=np.ones(self.M)
#         c_a,c_b=np.ones(self.M),np.ones(self.M)
#         # calc coefficients
#         for j,e in enumerate(self.list_e):
#             a,b=self.dict_edges[e]
#             c_a[j]=K.G.nodes[a]['concentrations']
#             c_b[j]=K.G.nodes[b]['concentrations']
#
#         K.PE=calc_PE(K)
#         x,z,sinh_x,cosh_x,coth_x,e_up,e_down=self.compute_flux_pars(K)
#
#         f1= np.divide(x,sinh_x)*0.5
#         F1=np.add( np.subtract( np.multiply(x,coth_x)*0.5 , np.multiply( f1,e_up )), z)
#         F2=np.add( np.subtract( np.multiply(x,coth_x)*0.5 , np.multiply( f1,e_down )), -z)
#
#         # calc edgewise absorption
#         phi=np.add(np.multiply(c_a,F1) ,np.multiply( c_b,F2 ))
#         A=np.pi*np.multiply(R,R)*(K.D/K.l)
#
#         return np.multiply( A, phi )
#
#     def calc_flux_jacobian(self,R,*args):
#
#         # unzip parameters
#         L,K= args
#
#         # init containers
#         I=np.identity(self.M)
#         J_PE, J_Q= np.zeros((self.M,self.M)),np.zeros((self.M,self.M))
#
#         # set coefficients
#         f1= 2.*np.divide(K.PE,R)
#         f2= 4.* np.divide(K.Q,R)
#         R_sq=np.power(R,2)
#         INV=lina.pinv(np.dot(K.B,np.dot(np.diag(K.C),K.BT)))
#         D=np.dot(np.dot(K.BT,INV),K.B)
#
#         # calc jacobian
#         for i,c in enumerate(K.C):
#             J_PE[i,:]= f1[i] * np.subtract( I[i,:], 2.* c * np.multiply( D[:,i], R_sq/R_sq[i] ) )
#             J_Q[i,:]= f2[i] * np.subtract( I[i,:], c*np.multiply( D[:,i], np.multiply( np.divide(L[i],L) , np.power( R_sq/R_sq[i] , 2 ) ) ) )
#
#         return J_PE,J_Q
#
#     def calc_concentration_jacobian(self, R,*args ):
#
#         # unzip
#         J_PE,c,K=args
#         # set containers
#         ones=np.ones(self.M)
#         J_C=np.zeros((self.M,self.N))
#
#         # set coefficients
#         A=np.pi*np.multiply(R,R)*(K.D/K.l)
#         x,z,sinh_x,cosh_x,coth_x,e_up,e_down=self.compute_flux_pars(K)
#         f1= np.multiply(z,A)
#         f2= np.multiply(np.multiply(x,coth_x),A)*0.5
#         f3= np.divide(np.multiply(A,x),sinh_x)*0.5
#
#         j_coth_x=np.power(np.divide(coth_x,cosh_x),2)
#         f2= np.multiply(x,coth_x)*0.5
#         f4=np.subtract( np.multiply( np.divide(z,x), coth_x) ,  np.multiply( z,j_coth_x )*0.5 )
#
#         f_up=np.divide(np.multiply( e_up ,x ), sinh_x )*0.5
#         f_down=np.divide( np.multiply( e_down ,x ), sinh_x )*0.5
#         f5=np.divide( np.multiply(A, e_up), sinh_x )
#         f6=np.divide( np.multiply(A, e_down), sinh_x )
#
#         J_f_up=-np.multiply( f5, np.subtract( np.add( np.divide(z,x), x*0.25 ), np.multiply(z,coth_x)*0.5 ))
#         J_f_down=-np.multiply( f6, np.subtract( np.subtract( np.divide(z,x), x*0.25 ), np.multiply(z,coth_x)*0.5 ))
#
#         inv_B,c= self.calc_inv_B( K ,c)
#         for j,e in enumerate(self.list_e):
#             JB_eff=np.zeros((self.N,self.N))
#             J_A=np.zeros((self.M,self.M))
#             J_A[j,j]=2.*np.pi*R[j]*(K.D/K.l)
#             for i,n in enumerate(self.list_n):
#
#                 b=K.B[i,:]
#                 JB_eff[i,i]=  np.sum( np.multiply( J_A, np.add( np.multiply(b,z), np.multiply(np.absolute(b),f2))  ) )+np.sum( np.multiply( J_PE[j,:], np.multiply(A, np.add( b*0.5, np.multiply(np.absolute(b),f4) ) ) ))
#                 JB_eff[i,self.dict_out[n]]= np.subtract( np.multiply( J_PE[j,self.dict_node_out[n]] , J_f_down[self.dict_node_out[n]] ) , np.multiply( J_A[j,self.dict_node_out[n]], f_down[self.dict_node_out[n]] ))
#                 JB_eff[i,self.dict_in[n]]=  np.subtract( np.multiply( J_PE[j,self.dict_node_in[n]] , J_f_up[self.dict_node_in[n]] ) , np.multiply( J_A[j,self.dict_node_in[n]], f_up[self.dict_node_in[n]] ))
#
#             self.evaluate_jacobian(self,j,J_C,JB_eff,inv_B,c)
#
#         return J_C
#
#     def calc_absorption_jacobian(self,R,K):
#
#         # set containers
#         ones=np.ones(self.M)
#         L=ones*K.l
#         J_phi= np.zeros((self.M,self.M))
#         phi=np.zeros(self.M)
#         c_a,c_b,c_n=np.zeros(self.M),np.zeros(self.M),np.zeros(self.N)
#         alphas,omegas=[],[]
#
#         # calc coefficients
#         for j,e in enumerate(self.list_e):
#             a,b=self.dict_edges[e]
#             c_a[j]=K.G.nodes[a]['concentrations']
#             c_b[j]=K.G.nodes[b]['concentrations']
#             alphas.append(a)
#             omegas.append(b)
#         for i,n in enumerate(self.list_n):
#             c_n[i]=K.G.nodes[n]['concentrations']
#
#         x,z,sinh_x,cosh_x,coth_x,e_up,e_down=self.compute_flux_pars(K)
#
#         f1= 0.5*np.divide(x,sinh_x)
#         F1=np.add( np.subtract( 0.5*np.multiply(x,coth_x) , np.multiply( f1,e_up )), z )
#         F2=np.subtract( np.subtract( 0.5*np.multiply(x,coth_x) , np.multiply( f1,e_down )), z)
#
#         f2_up=np.subtract( np.multiply( np.divide(PE,x), np.subtract( cosh_x, e_up )), np.divide(z,sinh_x))
#         f3_up=np.add( np.multiply( e_up, np.subtract( np.multiply(coth_x,z), 0.5*x ) ) , sinh_x)
#         f2_down=np.subtract(np.multiply( np.divide(PE,x), np.subtract( cosh_x, e_down )), np.divide(z,sinh_x))
#         f3_down=np.subtract( np.multiply( e_down, np.add( np.multiply(coth_x,z), 0.5*x  ) ), sinh_x )
#
#         F3= 0.5*np.divide( np.add(f2_up, f3_up) , sinh_x)
#         F4= 0.5*np.divide( np.add(f2_down, f3_down) , sinh_x)
#         phi=np.add( np.multiply(c_a,F1) ,np.multiply(c_b,F2 ) )
#
#         # calc jacobian
#         J_PE,J_Q= self.calc_flux_jacobian(R,L,K)
#         A=np.pi*np.multiply(R,R)*(K.D/K.l)
#         J_A=2.*np.pi*np.diag(R)*(K.D/K.l)
#         J_C=self.calc_concentration_jacobian( R,J_PE,c_n,K)
#
#         qa=np.multiply(A,c_a)
#         qb=np.multiply(A,c_b)
#         q1=np.multiply( A, F1 )
#         q2=np.multiply( A, F2 )
#
#         for j,e in enumerate(self.list_e):
#             J_phi[j,:]=np.add(J_phi[j,:],np.multiply( J_A[j,:], phi))
#
#             J_phi[j,:]=np.add(J_phi[j,:], np.multiply( J_C[j,alphas], q1 ))
#             J_phi[j,:]=np.add(J_phi[j,:], np.multiply( J_C[j,omegas], q2 ))
#
#             J_phi[j,:]=np.add(J_phi[j,:],np.multiply( J_PE[j,:], np.multiply(qa,F3)))
#             J_phi[j,:]=np.add(J_phi[j,:],np.multiply( J_PE[j,:], np.multiply(qb,F4)))
#
#         return J_phi
#
#     def compute_flux_pars(self,K):
#
#         x=np.sqrt( np.add( np.power(K.PE,2),K.beta ) )
#         z=PE*0.5
#         sinh_x=np.sinh(x*0.5)
#         cosh_x=np.cosh(x*0.5)
#         coth_x=np.reciprocal(np.tanh(x*0.5))
#         e_up=np.exp(z)
#         e_down=np.exp(-z)
#
#         return x,z,sinh_x,cosh_x,coth_x,e_up,e_down
