# @Author:  Felix Kramer
# @Date:   2021-06-03T11:02:57+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-06-12T20:36:10+02:00
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

    def update_transport_matrix(self):

        ref_var=self.circuit.scale['diffusion']/self.circuit.scale['length']
        R_sq= self.calc_cross_section_from_conductivity(self.circuit.edge['conductivity'],self.circuit.scale['conductance'])
        A=self.calc_diff_flux( R_sq,ref_var )

        Q=self.calc_flow(self.circuit.edge['conductivity'],self.circuit.node['source'])
        V=self.calc_velocity_from_flowrate(Q,R_sq)
        self.circuit.edge['peclet']=self.calc_peclet(V,1./ref_var)
        self.circuit.edge['flow_rate']=Q

        x,z,e_up_sinh_x,e_down_sinh_x,coth_x,idx_pack=self.compute_flux_pars()

        f1= np.multiply(z,A)
        f2= np.multiply(A,np.multiply(x,coth_x))*0.5

        f3= np.multiply(np.multiply(A,x),e_up_sinh_x)*0.5
        f4= np.multiply(np.multiply(A,x),e_down_sinh_x)*0.5

        self.B_eff=np.zeros((self.N,self.N))

        for i,n in enumerate(self.circuit.list_graph_nodes):

            self.B_eff[i,i]= np.sum(  np.add( np.multiply(self.B[i,:],f1),np.multiply(np.absolute(self.B[i,:]),f2))  )
            self.B_eff[i,self.dict_in[n]]= -f3[self.dict_node_in[n]]
            self.B_eff[i,self.dict_out[n]]= -f4[self.dict_node_out[n]]

    def calc_absorption(self,R_sq):

        # set containers
        c_a,c_b=np.ones(self.M),np.ones(self.M)
        # calc coefficients
        for j,e in enumerate(self.circuit.list_graph_edges):
            a,b=self.dict_edges[e]
            c_a[j]=self.circuit.G.nodes[a]['concentrations']
            c_b[j]=self.circuit.G.nodes[b]['concentrations']

        x,z,e_up_sinh_x,e_down_sinh_x,coth_x,idx_pack=self.compute_flux_pars()

        f1_up= np.multiply(x,e_up_sinh_x)*0.5
        f1_down= np.multiply(x,e_down_sinh_x)*0.5
        F1=np.add( np.subtract( np.multiply(x,coth_x)*0.5 , f1_up), z)
        F2=np.add( np.subtract( np.multiply(x,coth_x)*0.5 , f1_down), -z)

        # calc edgewise absorption
        phi=np.add(np.multiply(c_a,F1) ,np.multiply( c_b,F2 ))
        ref_var=self.circuit.scale['diffusion']/self.circuit.scale['length']
        A=self.calc_diff_flux( R_sq,ref_var )

        return np.multiply( A, phi )

    def calc_coefficients( self, R, *args  ):

        # unzip
        J_PE,c,K=args
        dict_coeff={}

        ref_var=self.circuit.scale['diffusion']/self.circuit.scale['length']
        R_sq=np.multiply(R,R)
        A=self.calc_diff_flux( R_sq,ref_var )

        x,z,e_up_sinh_x,e_down_sinh_x,coth_x,idx_pack=self.compute_flux_pars()

        f1= np.multiply(z,A)
        f2= np.multiply(np.multiply(x,coth_x),A)*0.5

        f3= np.multiply(np.multiply(A,x),e_up_sinh_x)*0.5
        f4= np.multiply(np.multiply(A,x),e_down_sinh_x)*0.5

        j_coth_x=np.zeros(self.M)
        idx_lower=idx_pack[0]
        j_coth_x[idx_lower]=np.power(np.divide(coth_x[idx_lower],np.cosh(x[idx_lower])),2)

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

        flux_sum_1=np.array([ np.add( np.multiply(self.B[i,:],z), np.multiply(np.absolute(self.B[i,:]),f2)) for i,n in enumerate(self.circuit.list_graph_nodes)])
        flux_sum_2=np.array([ np.multiply(A, np.add( self.B[i,:]*0.5, np.multiply(np.absolute(self.B[i,:]),f4) ) ) for i,n in enumerate(self.circuit.list_graph_nodes)])
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

    def calc_inv_B(self ,c):

        B_new=self.B_eff[self.idx_eff,:]
        B_new=B_new[:,self.idx_eff]
        c=c[self.idx_eff]
        inv_B=np.linalg.inv(B_new)

        return inv_B, c

    def evaluate_jacobian(self,j,J_C,JB_eff,inv_B,c):

        JB_new=JB_eff[self.idx_eff,:]
        JB_new=JB_new[:,self.idx_eff]
        J_C[j,self.idx_eff]=-np.dot(inv_B, np.dot( JB_new, c ))

    def calc_concentration_jacobian(self, R,*args ):

        # unzip
        J_PE,c=args
        # set coefficients
        dict_coeff=self.calc_coefficients( R,J_PE,c )
        inv_B, c = self.calc_inv_B(c)

        ref_var=self.circuit.scale['diffusion']/self.circuit.scale['length']
        J_C=np.zeros((self.M,self.N))
        J_diag=R*2.*np.pi*ref_var

        J_A=np.zeros(self.M)
        for j,e in enumerate(self.circuit.list_graph_edges):

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
        J_phi= np.zeros((self.M,self.M))
        phi=np.zeros(self.M)
        c_a,c_b,c_n=np.zeros(self.M),np.zeros(self.M),np.zeros(self.N)
        alphas,omegas=[],[]

        # calc coefficients
        for j,e in enumerate(self.circuit.list_graph_edges):
            a,b=self.dict_edges[e]
            c_a[j]=self.circuit.G.nodes[a]['concentrations']
            c_b[j]=self.circuit.G.nodes[b]['concentrations']
            alphas.append(a)
            omegas.append(b)

        for i,n in enumerate(self.circuit.list_graph_nodes):
            c_n[i]=K.G.nodes[n]['concentrations']

        x,z,e_up_sinh_x,e_down_sinh_x,coth_x,idx_pack=self.compute_flux_pars()

        f1_up= 0.5*np.multiply(x,e_up_sinh_x)
        f1_down= 0.5*np.multiply(x,e_down_sinh_x)

        F1=np.add( np.subtract( 0.5*np.multiply(x,coth_x) ,  f1_up ), z )
        F2=np.subtract( np.subtract( 0.5*np.multiply(x,coth_x) , f1_down ), z)
        F3,F4=self.calc_absorption_jacobian_coefficients(idx_pack, x,z,e_up_sinh_x,e_down_sinh_x,coth_x)

        # calc current absorption
        phi=np.add( np.multiply(c_a,F1) ,np.multiply(c_b,F2 ) )

        # calc jacobian components
        ref_var=self.circuit.scale['diffusion']/self.circuit.scale['length']
        R_sq=np.multiply(R,R)
        A=self.calc_diff_flux( R_sq,ref_var )

        J_PE,J_Q= self.calc_flux_jacobian(R)
        J_A=2.*np.pi*np.diag(R)*ref_var
        J_C=self.calc_concentration_jacobian( R,J_PE,c_n,K )

        qa=np.multiply(A,c_a)
        qb=np.multiply(A,c_b)
        q1=np.multiply( A, F1 )
        q2=np.multiply( A, F2 )

        for j,e in enumerate(self.circuit.list_graph_edges):

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

    def compute_flux_pars(self):

        x=np.sqrt( np.add( np.power(self.circuit.edges['peclet'],2),self.circuit.edges['absorption'] ) )
        z=self.circuit.edges['peclet']*0.5

        e_up_sinh_x=np.zeros(self.M)
        e_down_sinh_x=np.zeros(self.M)
        coth_x=np.zeros(self.M)

        # establish the use of converging limit expressions to prevent overflow error
        idx_lower=np.where(np.absolute(self.circuit.edges['peclet'])<self.crit_pe)[0]
        idx_over_pos=np.where((np.absolute(self.circuit.edges['peclet'])>=self.crit_pe) & (self.circuit.edges['peclet'] > 0))[0]
        idx_over_neg=np.where((np.absolute(self.circuit.edges['peclet'])>=self.crit_pe) & (self.circuit.edges['peclet'] < 0))[0]
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

    def solve_absorbing_boundary(self):

        # reduce transport matrix by cutting row,col corresponding to absorbing boundary
        B_new=self.B_eff[self.idx_eff,:]
        B_new=B_new[:,self.idx_eff]
        S=self.circuit.nodes['solute'][self.idx_eff]
        concentration=np.zeros(self.N)

        # solving inverse flux problem for absorbing boundaries
        concentration_reduced=np.dot(np.linalg.inv(B_new),S)
        concentration[self.idx_eff]=concentration_reduced[:]

        # export solution
        for i,n in enumerate(self.circuit.list_graph_nodes):
             self.circuit.G.nodes[n]['concentrations']=concentration[i]

        return concentration_reduced ,B_new

    def calc_profile_concentration(self):

        self.update_transport_matrix()
        c,B_new=self.solve_absorbing_boundary()

        return c,B_new

    def calc_flux_jacobian(self,R,*args):

        # init containers
        I=np.identity(self.M)
        J_PE, J_Q= np.zeros((self.M,self.M)),np.zeros((self.M,self.M))

        # set coefficients
        f1= 2.*np.divide(self.circuit.edge['peclet'],R)
        f2= 4.* np.divide(self.circuit.edge['flow_rate'],R)
        R_sq=np.power(R,2)
        INV=lina.pinv(np.dot(self.B,np.dot(np.diag(self.circuit.edge['conductivity']),self.BT)))
        D=np.dot(np.dot(self.BT,INV),self.B)

        # calc jacobian
        for i,c in enumerate(self.circuit.edge['conductivity']):
            J_PE[i,:]= f1[i] * np.subtract( I[i,:], 2.* c * np.multiply( D[:,i], R_sq/R_sq[i] ) )
            J_Q[i,:]= f2[i] * np.subtract( I[i,:], c*np.multiply( D[:,i], np.multiply( np.divide(self.circuit.edge['length'][i],self.circuit.edge['length']) , np.power( R_sq/R_sq[i] , 2 ) ) ) )

        return J_PE,J_Q
