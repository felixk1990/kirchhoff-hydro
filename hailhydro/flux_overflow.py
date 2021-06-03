# @Author:  Felix Kramer
# @Date:   2021-06-03T11:02:57+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-06-03T18:25:16+02:00
# @License: MIT



import sys
import networkx as nx
import numpy as np
import scipy.linalg as lina
import random as rd
from flux_init import *

class overflow(flux,object):

    def __init__(self,*args):
        super(overflow,self).__init__(args[0])
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
