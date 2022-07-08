# @Author: Felix Kramer <kramer>
# @Date:   08-03-2022
# @Email:  felixuwekramer@proton.me
# @Last modified by:   kramer
# @Last modified time: 08-07-2022


import numpy as np
import networkx as nx
from kirchhoff.circuit_init import Circuit
from kirchhoff.circuit_flow import FlowCircuit
from dataclasses import dataclass, field


@dataclass
class Flow():
    """
    The flow class defines variables and methods for computing Hagen-Poiseuille
    flows on kirchhoff networks.

    To be used in conjunction with 'kirchhoff' and 'goflow' in order to
    simulate flow-driven network morphogenesis.

    Attributes:
        constr (networkx.Graph):\n
            A networkx graph or circuit to initilize a flow on.
        pars_source (dict):\n
            The boundary conditions (Neumann) determining the in/outlfow of
            fluid accross the network.
        pars_plexus (dict):\n
            The initial plexus, edge values of  conductivity, the flow is to
            be calculated on.

    Methods:
        init_flow():\n
            Initialize flow variables, boundaries and handle constructor
            exceptions.
        set_boundaries():\n
            Explicitly set Neumann-boudaries and initial plexus as defined via
            'pars_source/plexus' parameters. Set internal output varaibles and
            incidence information.
        find_roots(G):\n
            Given a networkx graph, return all source-nodes (needs the nodal
            'source' attribute set).
        find_sinks(G):\n
            Given a networkx graph, return all sink-nodes (needs the nodal
            'source' attribute set).
        alpha_omega(G, j):\n
            Return the start (alpha) and end(omega) node of an edge, for any
            given networkx graph with edge labeling j.
        calc_pressure(conduct, source):\n
            Compute the pressure landscape, considering the current parameter
            and plexus condition.
        calc_flow_from_pressure(conduct, dP):\n
            Compute the flow landscape, considering the current parameter
            and plexus condition.
        calc_flow(conduct, source):\n
            Compute the flow landscape, considering the current parameter
            and plexus condition.
        calc_sq_flow(sconduct, source):\n
            Compute the squared pressure/flow landscape, considering the
            current parameter and plexus condition.
        calc_cross_section_from_conductivity(conductivity, conductance):\n
            Compute the squared radii values from the current conductivity
            matrix and conductance value.
        calc_conductivity_from_cross_section(R_sq, conductance):\n
            Compute the conductivity matrix from the current squared radii
            values and conductance value.
        calc_configuration_flow():\n
            Compute the pressure/flow landscape, considering the current
            parameter and plexus condition.

    """

    constr: nx.Graph = field(repr=False, init=True)
    pars_source: dict = field(default_factory=dict, repr=False)
    pars_plexus: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):

        self.init_flow()

    def init_flow(self):

        """
        Initialize flow variables, boundaries and handle constructor
        exceptions.

        Raises:
            Exception:\n
                Warning! Non-networkx type given for initialization, no
                internal circuit established.

        """

        self.info: str = 'unknown'

        if isinstance(self.constr, nx.Graph):

            self.circuit = FlowCircuit(self.constr)

        elif isinstance(self.constr, FlowCircuit):

            self.circuit = self.constr

        elif isinstance(self.constr, Circuit):

            self.circuit = FlowCircuit(self.constr.G)

        else:
            raise Exception(
                '''
                Warning! Non-networkx type given for initialization, no
                internal circuit established.
                '''
                )

        self.set_boundaries()

    def set_boundaries(self):

        """
        Explicitly set Neumann-boudaries and initial plexus as defined via
        'pars_source/plexus' parameters. Set internal output varaibles and
        incidence information.

        """

        par1 = self.circuit.graph['source_mode']
        par2 = self.circuit.graph['plexus_mode']

        if par1 == '' or par2 == '':
            self.circuit.set_source_landscape(**self.pars_source)
            self.circuit.set_plexus_landscape(**self.pars_plexus)

        self.info = self.circuit.info
        self.B, self.BT = self.circuit.get_incidence_matrices()

    def find_roots(self, G):

        """
        Given a networkx graph, return all source-nodes (needs the nodal
        'source' attribute set).

        Args:
            G (networkx.Graph):\n
                A networkx graph.

        Returns:
            list:\n
                A list of root/source nodes of the given graph.

        """

        roots = [n for n in G.nodes() if G.nodes[n]['source'] > 0]

        return roots

    def find_sinks(self, G):
        """
        Given a networkx graph, return all sink-nodes (needs the nodal
        'source' attribute set).


        Args:
            G (networkx.Graph):\n
                A networkx graph.

        Returns:
            list:\n
                A list of outlet/sink nodes of the given graph.

        """

        list_n = self.circuit.list_graph_nodes
        sinks = [n for n in list_n if G.nodes[n]['source'] < 0]

        return sinks

    def alpha_omega(self, G, j):
        """
        Return the start (alpha) and end(omega) node of an edge, for any given
        networkx graph with edge labeling j.

        Args:
            G (networkx.Graph):\n
                A networkx graph.
            j (int):\n
                An existent edge label.

        Returns:
            node: The 'alpha' node of edge (labeld j)
            node: The 'omega' node of edge (labeld j)
        """

        labels = nx.get_edge_attributes(G, 'label')
        for e, label in labels.items():
            if label == j:
                alpha = e[1]
                omega = e[0]

        return alpha, omega

    def calc_pressure(self, conduct, source):
        """
        Compute the pressure landscape, considering the current parameter
        and plexus condition.

        Args:
            conduct (array):\n
                The network's edge conductivity matrix.
            source (array):\n
                The nodal source vector.

        Returns:
            ndarray: Edge-vector of pressure-differences.
            ndarray: Node-vector of pressures levels.

        """

        OP = np.dot(self.B, np.dot(np.diag(conduct), self.BT))
        P, RES, RG, si = np.linalg.lstsq(OP, source, rcond=None)
        dP = np.dot(self.BT, P)

        return dP,  P

    def calc_flow_from_pressure(self, conduct, dP):
        """
        Compute the flow landscape, considering the current parameter
        and plexus condition.

        Args:
            conduct (array):\n
                The network's edge conductivity matrix.
            dP (array):\n
                Edge-vector of pressure-differences.

        Returns:
            ndarray: Edge-vector of directed flow rates.
        """

        Q = np.dot(np.diag(conduct), dP)

        return Q

    def calc_flow(self, conduct, source):
        """
        Compute the flow landscape, considering the current parameter
        and plexus condition.

        Args:
            conduct (array):\n
                The network's edge conductivity matrix.
            source (array):\n
                The nodal source vector.

        Returns:
            ndarray: Edge-vector of directed flow rates.
        """

        dP, P = self.calc_pressure(conduct, source)
        Q = np.dot(np.diag(conduct), dP)

        return Q

    def calc_sq_flow(self, conduct, source):
        """
        Compute the squared pressure/flow landscape, considering the current
        parameter and plexus condition.

        Args:
            conduct (array):\n
                The network's edge conductivity matrix.
            source (array):\n
                The nodal source vector.

        Returns:
            ndarray: Edge-vector of squared flow rate values.
            ndarray: Edge-vector of squared pressure difference values.
        """

        dP, P = self.calc_pressure(conduct, source)
        Q = self.calc_flow_from_pressure(conduct, dP)

        p_sq = np.multiply(dP, dP)
        q_sq = np.multiply(Q, Q)

        return p_sq,  q_sq

    def calc_cross_section_from_conductivity(self, conductivity, conductance):
        """
        Compute the squared radii values from the current conductivity matrix
        and conductance value.

        Args:
            conductivity (array):\n
                The network's edge conductivity matrix.
            conductance (array):\n
                The graph's conductance unit.

        Returns:
            ndarray: Edge-vector of squared radii values.
        """

        R_sq = np.sqrt(conductivity/conductance)

        return R_sq

    def calc_conductivity_from_cross_section(self, R_sq, conductance):
        """
        Compute the conductivity matrix from the current squared radii values
        and conductance value.

        Args:
            R_sq (array):\n
                Edge-vector of squared radii values.
            conductance (array):\n
                The graph's conductance unit.

        Returns:
            ndarray: The network's edge conductivity matrix.
        """

        conductivity = np.power(R_sq, 2)*conductance

        return conductivity

    def calc_configuration_flow(self):
        """
        Compute the pressure/flow landscape, considering the current parameter
        and plexus condition.

        Returns:
            ndarray: Edge-vector of directed flow rates.
            ndarray: Edge-vector of pressure differences.
        """

        k = self.circuit.edges['conductivity']
        src = self.circuit.nodes['source']

        dP, P = self.calc_pressure(k, src)
        Q = np.dot(np.diag(k), dP)

        return Q, dP
