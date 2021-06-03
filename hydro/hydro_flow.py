# @Author:  Felix Kramer
# @Date:   2021-06-03T11:02:33+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-06-03T11:51:17+02:00
# @License: MIT

import numpy as np
import scipy.linalg as lina
import sys
import scipy as sy
import scipy.integrate as si
import scipy.optimize as sc
import random as rd
import networkx as nx
import kirchhoff

# take an initiliazed kirchoff network and start computing flows
class hyrdro(circuit,object):

    def __init__(self):

        super(hydro,self).__init__()
        self.B,self.BT=self.get_incidence_matrices()

    def find_roots(self,G):

        roots=[n for n in self.list_graph_nodes if G.nodes[n]['source']>0]

        return roots

    def find_sinks(self,G):

        sinks=[n for n in self.list_graph_nodes if G.nodes[n]['source']<0]

        return sinks

    def alpha_omega(self,G,j):

        labels=nx.get_edge_attributes(G,'label')
        for e,label in labels.items():
            if label==j:
                alpha=e[1]
                omega=e[0]

        return alpha,omega

    def calc_pressure(self,*args):

        conduct,source=args

        OP=np.dot(self.B,np.dot(np.diag(conduct),self.BT))
        P,RES,RG,si=np.linalg.lstsq(OP,source,rcond=None)
        dP=np.dot(self.BT,P)

        return dP, P

    def calc_flow_from_pressure(self,*args):

        conduct,dP=args
        Q=np.dot(np.diag(conduct),dP)

        return dP, P

    def calc_flow(self,*args):

        conduct,source=args

        dP,P=self.calc_pressure(args)
        Q=np.dot(np.diag(conduct),dP)

        return Q

    def calc_sq_flow(self,*args):

        dP,P=self.calc_pressure(args)
        Q=self.calc_flow_from_pressure(args[0],dP)

        p_sq=np.multiply(dP,dP)
        q_sq=np.multiply(Q,Q)

        return p_sq, q_sq

class hydro_rand(hyrdro,object):

    def __init__(self):

        super( hydro_rand,self).__init__()

    def setup_random_fluctuations(self,N, mean, variance):

        self.mu=mean
        self.var=variance

        self.G=np.identity(N)
        for n in range(N):
            for m in range(N):
                h=0.
                if n==0 and m==0:
                    h+=(N-1)
                elif n==m and n!=0:
                    h+=1.
                elif n==0 and m!=0:
                    h-=1
                elif m==0 and n!=0:
                    h-1

                self.G[n,m]=h
        self.H=np.identity(N)
        for n in range(N):

            for m in range(N):
                h=0.
                if n==0 and m==0:
                    h+=(1-N)*(1-N)
                elif n!=0 and m!=0:
                    h+=1.
                elif n==0 and m!=0:
                    h+=(1-N)
                elif m==0 and n!=0:
                    h+=(1-N)

                self.H[n,m]=h

    def setup_random_fluctuations_reduced(self,K, mean, std):

        x=np.where(K.J > 0)[0][0]

        N=len(K.J)
        idx=np.where(x!=range(N))[0]
        L0=np.ones((N,N))
        L0[idx,:]=0.
        L0[:,idx]=0.

        L1=np.identity(N)
        L1[x,x]=0.

        L2=np.zeros((N,N))
        L2[idx,:]=1.-N
        L2[:,idx]=1.-N
        L2[x,x]=(N-1)**2

        alpha=mean
        beta=std
        f_beta=1+beta/(N-1)

        self.Z = (L0 + beta * L1 + f_beta * L2)*alpha

    def setup_random_fluctuations_multisink(self,K ):

        num_n=nx.number_of_nodes(K.G)
        x=np.where(K.J > 0)[0]
        idx=np.where(K.J < 0)[0]
        N=len(idx)
        M=len(x)

        U=np.zeros((num_n,num_n))
        V=np.zeros((num_n,num_n))

        m_sq=float(M*M)
        NM=num_n*num_n/float(m_sq)
        Nm=(N/m_sq)+2./M

        for i in range(num_n):
            for j in range(num_n)[i:]:
                delta=0.
                sum_delta=0.
                sum_delta_sq=0.

                if i==j:
                    delta=1.

                if (i in x):
                    sum_delta=1.

                if (j in x):
                    sum_delta=1.

                if (i in x) and (j in x):
                    sum_delta_sq=1.
                    sum_delta=2.

                U[i,j]= ( m_sq - num_n*sum_delta + NM*sum_delta_sq )
                V[i,j]= ( ( Nm + delta )*sum_delta_sq - (1.+M*delta)*sum_delta + m_sq*delta)

                U[j,i]=U[i,j]
                V[j,i]=V[i,j]

        self.Z = np.add(U,np.multiply(self.noise,V))

    def setup_random_fluctuations_effective(self,K):

        self.x=np.where(K.J > 0)[0][0]
        L0=np.ones((self.N,self.N))
        L0[self.x,:]=0.
        L0[:,self.x]=0.

        L1=np.identity(self.N)
        L1[self.x,self.x]=0.

        L2=np.zeros((self.N,self.N))
        L2[self.x,:]=1.-self.N
        L2[:,self.x]=1.-self.N
        L2[self.x,self.x]=(self.N-1)**2

        f_noise=1+self.noise/(self.N-1)

        self.Z = np.add(np.add(L0 , self.noise * L1) ,f_noise * L2)

    def setup_random_fluctuations_terminals(self,K):

        x=np.where(K.J > 0)[0][0]
        y=np.where(K.J < 0)[0][0]

        L0=np.ones((self.N,self.N))

        L0[:,y]=(self.fraction+1.)*(self.N-2)
        L0[y,:]=(self.fraction+1.)*(self.N-2)
        L0[x,:]=-(self.fraction+1.)*(self.N-2)
        L0[:,x]=-(self.fraction+1.)*(self.N-2)
        L0[x,y]-=(2.+self.fraction*(self.fraction+1.)*((self.N-2)**2))
        L0[y,x]-=(2.+self.fraction*(self.fraction+1.)*((self.N-2)**2))
        L0[x,x]=((self.fraction+1.)*(self.N-2))**2
        L0[y,y]=self.fraction**2

        L1=np.identity(self.N)
        L1[x,:]=-1.
        L1[:,x]=-1.
        L1[x,x]=self.N
        L1[y,y]=0.

        self.Z = np.add(L0 ,self.noise * L1 )

    def calc_sq_flow(self,C,B,BT):

        OP=np.dot(B,np.dot(np.diag(C),BT))
        inverse=lina.pinv(OP)
        D=np.dot(BT,inverse)
        DT=np.transpose(D)
        A=np.dot(D,self.Z)
        V=np.dot(A,DT)
        dV_sq=np.diag(V)
        F_sq=np.multiply(np.multiply(C,C),dV_sq)

        return dV_sq,F_sq

    def calc_sq_flow_random(self,C,B,BT):

        OP=np.dot(np.dot(B,C),BT)
        MP=lina.pinv(OP)
        D=np.dot(C,np.dot(BT,MP))
        DT=np.transpose(D)
        # print(D)
        var_matrix=np.dot(np.dot(D,self.G),DT)
        mean_matrix=np.dot(np.dot(D,self.H),DT)
        # print(D)
        # print(self.H)
        var_flow=np.diag(var_matrix)
        mean_flow=np.diag(mean_matrix)

        # print(var_flow)
        # print(mean_flow)
        F_sq= np.add(self.var*var_flow , self.mu*self.mu*mean_flow)
        # print(F_sq)
        return F_sq

    def calc_sq_flow_random_reduced(self,C,B,BT):

        OP=np.dot(np.dot(B,C),BT)
        MP=lina.pinv(OP)
        D=np.dot(C,np.dot(BT,MP))
        DT=np.transpose(D)

        F_sq=np.diag(np.dot(D,np.dot(self.Z,DT)))

        return F_sq


class hydro_reroute(hyrdro,object):

    def __init__(self):

        super( hydro_rand,self).__init__()

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
