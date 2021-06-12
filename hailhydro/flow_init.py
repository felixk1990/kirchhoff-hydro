# @Author:  Felix Kramer
# @Date:   2021-06-03T11:02:33+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-06-12T16:30:50+02:00
# @License: MIT

import numpy as np
import networkx as nx

# take an initiliazed circuit and start computing flows
def initialize_flow_on_circuit(circuit):

    flow_landscape=flow(circuit)

    return flow_landscape


class flow():

    def __init__(self,circuit):
        
        self.circuit=circuit
        self.B,self.BT=self.circuit.get_incidence_matrices()

    def find_roots(self,G):

        roots=[n for n in self.circuit.list_graph_nodes if G.nodes[n]['source']>0]

        return roots

    def find_sinks(self,G):

        sinks=[n for n in self.circuit.list_graph_nodes if G.nodes[n]['source']<0]

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

        return Q

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
