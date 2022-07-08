# @Author: Felix Kramer <kramer>
# @Date:   08-03-2022
# @Email:  felixuwekramer@proton.me
# @Last modified by:   kramer
# @Last modified time: 08-07-2022

import numpy as np
import scipy.linalg as lina
import random as rd
import networkx as nx
from hailhydro.flow_init import Flow
from dataclasses import dataclass, field
rand = {'mode': 'default', 'noise': 0.}
reroute = {'p_broken': 0, 'num_iter': 100}


@dataclass
class FlowRandom(Flow):
    """
    The random flow class defines variables and methods for computing
    randomized-averaged Hagen-Poiseuille flows on kirchhoff networks.

    To be used in conjunction with 'kirchhoff' and 'goflow' in order to
    simulate flow-driven network morphogenesis. The implementation is based on
    the models of Corson (PRL, 2010) and Hu-Cai(PRL, 2013), which consider
    random flow patterns to emerge from a variation in sink strengths.

    Attributes:
        constr (networkx.Graph):\n
            A networkx graph or circuit to initilize a flow on.
        pars_source (dict):\n
            The boundary conditions (Neumann) determining the in/outlfow of
            fluid accross the network.
        pars_plexus (dict):\n
            The initial plexus, edge values of  conductivity, the flow is to
            be calculated on.
        flow_setting (dict):\n
            The setting for randomized flow, i.e. noise levels and
            correlations.

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
        init_random():\n
            Initialize random-flow variables and boundaries.
        set_root_source_matrix(mean, variance):\n
            Set the (positive) parameters of deterministic and noisy flow
            levels.
         set_effective_source_matrix(noise):\n
            Set the effective fluctuation matrix with ggiven noise level.
        calc_sq_flow_effective(conduct):\n
            Compute the squared pressure/flow landscape, considering the
            current parameter and plexus condition (given the internal
            effective noise matrix was initialized).

        calc_sq_flow_root(conduct):\n
            Compute the squared pressure/flow landscape, considering the
            current parameter and plexus condition (given the internal sink
            mean and variance were initialized).

    """
    default = dict(repr=False, init=True)
    flow_setting: dict = field(default_factory=dict(), **default)

    def __post_init__(self):

        self.init_flow()
        self.init_random()

    def init_random(self):

        """
        Initialize random-flow variables and boundaries.

        Raises:
            Exception:\n
                Warning flow landscape not set!

        """

        try:
            if self.flow_setting['mode'] == 'default':
                self.set_effective_source_matrix(self.flow_setting['noise'])

            elif self.flow_setting['mode'] == 'root':
                self.mu_sq = self.flow_setting['mu_sq']
                self.var = self.flow_setting['var']
                self.set_multi_source_matrix(self.mu_sq, self.var)
        except Exception:
            raise Exception(
                '''
                Warning flow landscape not set!
                '''
                )

    # setup_random_fluctuations
    def set_root_source_matrix(self, mean, variance):
        """
        Set the (positive) parameters of deterministic and noisy flow levels.

        Args:
            mean (float):\n
                The squared mean value of sink strength.
            variance (float):\n
                The variance value of sink strength.

        """

        N = len(self.circuit.list_graph_nodes)

        self.matrix_mu = np.identity()
        for n in range(N):
            for m in range(N):
                h = 0.
                if n == 0 and m == 0:
                    h += (N-1)
                elif n == m and n != 0:
                    h += 1.
                elif n == 0 and m != 0:
                    h -= 1.
                elif m == 0 and n != 0:
                    h -= 1.

                self.matrix_mu[n, m] = h

        self.matrix_mu = self.matrix_mu*mean

        self.matrix_var = np.identity(N)
        for n in range(N):

            for m in range(N):
                h = 0.
                if n == 0 and m == 0:
                    h += (1-N)*(1-N)
                elif n != 0 and m != 0:
                    h += 1.
                elif n == 0 and m != 0:
                    h += (1-N)
                elif m == 0 and n != 0:
                    h += (1-N)

                self.matrix_var[n, m] = h

        self.matrix_var = self.matrix_var*variance

    # setup_random_fluctuations_multisink
    def set_effective_source_matrix(self, noise):
        """
        Set the effective fluctuation matrix with ggiven noise level.

        Args:
            noise (float):\n
                The ratio of squared mean and variance of sink strength.
        """
        self.noise = noise
        num_n = len(self.circuit.list_graph_nodes)
        x = np.where(self.circuit.nodes['source'] > 0)[0]
        # idx = np.where(self.circuit.nodes['source'] < 0)[0]
        # N = len(idx)
        N = num_n - len(x)
        M = len(x)

        U = np.zeros((num_n, num_n))
        V = np.zeros((num_n, num_n))

        m_sq = float(M*M)
        NM = num_n*num_n/float(m_sq)
        Nm = (N/m_sq)+2./M

        for i in range(num_n):
            for j in range(num_n)[i:]:
                delta = 0.
                sum_delta = 0.
                sum_delta_sq = 0.

                if i == j:
                    delta = 1.

                if (i in x):
                    sum_delta = 1.

                if (j in x):
                    sum_delta = 1.

                if (i in x) and (j in x):
                    sum_delta_sq = 1.
                    sum_delta = 2.

                U[i, j] = (m_sq - num_n*sum_delta + NM*sum_delta_sq)

                v1 = (Nm + delta)*sum_delta_sq
                v2 = (1.+M*delta)*sum_delta
                v3 = m_sq*delta
                V[i, j] = (v1 - v2 + v3)

                U[j, i] = U[i, j]
                V[j, i] = V[i, j]

        self.Z = np.add(U, np.multiply(self.noise, V))

    # calc_sq_flow
    def calc_sq_flow_effective(self, conduct):
        """
        Compute the squared pressure/flow landscape, considering the current
        parameter and plexus condition (given the internal effective noise
        matrix was initialized).

        Args:
            conduct (array):\n
                The network's edge conductivity matrix.

        Returns:
            ndarray: Edge-vector of squared pressure difference values.
            ndarray: Edge-vector of squared flow rate values.
        """

        OP = np.dot(self.B, np.dot(np.diag(conduct), self.BT))
        inverse = lina.pinv(OP)
        D = np.dot(self.BT, inverse)
        DT = np.transpose(D)

        A = np.dot(np.dot(D, self.Z), DT)
        dV_sq = np.diag(A)
        F_sq = np.multiply(np.multiply(conduct, conduct), dV_sq)

        return dV_sq, F_sq

    # calc_sq_flow_random
    def calc_sq_flow_root(self, conduct):
        """
        Compute the squared pressure/flow landscape, considering the current
        parameter and plexus condition (given the internal sink mean and
        variance were initialized).

        Args:
            conduct (array):\n
                The network's edge conductivity matrix.

        Returns:
            ndarray: Edge-vector of squared pressure difference values.
            ndarray: Edge-vector of squared flow rate values.
        """

        OP = np.dot(np.dot(self.B, conduct), self.BT)
        inverse = lina.pinv(OP)
        D = np.dot(self.BT, inverse)
        DT = np.transpose(D)

        var_matrix = np.dot(np.dot(D, self.matrix_mu), DT)
        mean_matrix = np.dot(np.dot(D, self.matrix_var), DT)

        var_flow = np.diag(var_matrix)
        mean_flow = np.diag(mean_matrix)

        dV_sq = np.add(var_flow, mean_flow)
        F_sq = np.multiply(np.multiply(conduct, conduct), dV_sq)

        return dV_sq, F_sq


@dataclass
class FlowReroute(Flow):
    """
    The random flow class defines variables and methods for computing
    randomized-averaged Hagen-Poiseuille flows on kirchhoff networks.

    To be used in conjunction with 'kirchhoff' and 'goflow' in order to
    simulate flow-driven network morphogenesis. The implementation is based on
    the models which assume flow variation to emerge from random edge failure
    (homogenous probability), e.g. see Katifori, PRL, 2010

    Attributes:
        constr (networkx.Graph):\n
            A networkx graph or circuit to initilize a flow on.
        pars_source (dict):\n
            The boundary conditions (Neumann) determining the in/outlfow of
            fluid accross the network.
        pars_plexus (dict):\n
            The initial plexus, edge values of  conductivity, the flow is to
            be calculated on.
        flow_setting (dict):\n
            The setting for randomized flow, i.e. noise levels and
            correlations.

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

        initialize_broken_link():\n
            Initialize random-flow variables and boundaries, pre-generate broken
            edge states for faster sampling.

        generate_coherent_closure_deterministic(H, x):\n
            Generate a pruned version of a plexus, where a number 'x' of
            randomly sampled edges are failing during each realizsation.

        generate_coherent_closure():\n
            Generate a pruned version of a plexus, where a random number of
            edges are failing during each realizsation(each indivually with
            probability p).
        break_links(idx, conduct):\n
            Break/Fail the edges in list 'idx', by setting the respective
            conductivity values to numerically close to zero.
        get_sets():\n
            Sample a random set of network edges.

        calc_random_radii(idx, conduct):\n
            Compute the edge radii values, correpsonding to a randomly pruned
            conductivity matrix.
        calc_sq_flow(idx, conduct):\n
            Compute the squared pressure/flow landscape, considering a
            sequence of pruned conductivity matrices.
        calc_flows_mapping(graph_matrices):\n
            Compute the pressure/flow landscape, considering a sequence of
            conductivity matrix realizations.
        calc_sq_flow_avg(conduct):\n
            Compute the average squared pressure/flow landscape, by internally
            generating a random sequence of pruned conductivity matrix
            realizations.
        get_broken_links_asarray(idx, conduct):\n
            Compute a sequence of pruned conductivity matrices based on the
            failed edge sets 'idx'.

    """

    default = dict(repr=False, init=True)

    flow_setting: dict = field(default_factory=dict, **default)

    def __post_init__(self):

        self.init_flow()
        self.num_iteration = self.flow_setting['num_iter']
        self.percentage_broken = self.flow_setting['p_broken']
        self.initialize_broken_link()

    def initialize_broken_link(self):
        """
        Initialize random-flow variables and boundaries, pre-generate broken
        edge states for faster sampling.

        """

        broken_sets = []
        self.num_sets = 50000
        self.AUX = nx.Graph(self.circuit.G)
        for i in range(self.num_sets):
            cond, idx = self.generate_coherent_closure()
            if cond:
                broken_sets.append(idx)

        self.broken_sets = broken_sets

        assert(len(self.broken_sets) != 0)

    def generate_coherent_closure_deterministic(self, H, x):
        """
        Generate a pruned version of a plexus, where a number 'x' of
        randomly sampled edges are failing during each realizsation.

        Args:
            H (networkx.Graph):\n
                A networkx graph.
            x (int):\n
                The pre-determined number of edge to fail.
        Returns:
            bool: Boolean on whether the pruned network is connected.
            list: The list of failed edges.
        """

        idx = rd.sample(range(len(self.circuit.list_graph_edges)), x)
        for e in idx:
            H.remove_edge(*self.circuit.list_graph_edges[e])
        cond = nx.is_connected(H)

        for e in idx:
            H.add_edge(*self.circuit.list_graph_edges[e])

        return cond, idx

    def generate_coherent_closure(self):
        """
        Generate a pruned version of a plexus, where a random number of
        edges are failing during each realizsation(each indivually with
        probability p).

        Returns:
            bool: Boolean on whether the pruned network is connected.
            list: The list of failed edges.
        """
        prob = np.random.sample(len(self.circuit.list_graph_edges))
        idx = np.where(prob <= self.percentage_broken)[0]

        for e in idx:
            self.AUX.remove_edge(*self.circuit.list_graph_edges[e])
        cond = nx.is_connected(self.AUX)

        for e in idx:
            self.AUX.add_edge(*self.circuit.list_graph_edges[e])

        return cond, idx

    def break_links(self, idx, conduct):
        """
        Break/Fail the edges in list 'idx', by setting the respective
        conductivity values to numerically close to zero.

        Args:
            idx (list):\n
                A list of edges to fail.
            conduct (array):\n
                The network's conductivity matrix.

        Returns:
            ndarray:\n
                The network's pruned conductivity matrix.

        """
        C_aux = np.array(conduct)
        C_aux[idx] = np.power(10., -20)

        return C_aux

    def get_sets(self):
        """
        Sample a random set of network edges.

        Returns:
            list:\n
                A list of randomly sampled edges.

        """

        idx = rd.choices(self.broken_sets, k=self.num_iteration)

        return idx

    def calc_random_radii(self, idx, conduct):
        """
        Compute the edge radii values, correpsonding to a randomly pruned
        conductivity matrix.

        Args:
            idx (list):\n
                The list of failed edges.
            conduct (array):\n
                The network's conductivity matrix.

        Returns:
            list:\n
                A list of radii values R, as well as radii functions R^2, R^3

        """
        graph_matrices = self.get_broken_links_asarray(idx, conduct)

        R, R_sq, R_cb = [], [], []
        for gm in graph_matrices:

            kernel = gm/self.circuit.scales['conductance']

            R.append(np.power(kernel, 0.25))
            R_sq.append(np.sqrt(kernel))
            R_cb.append(np.power(kernel, 0.75))

        return [R, R_sq, R_cb]

    def calc_sq_flow(self, idx, conduct):
        """
        Compute the squared pressure/flow landscape, considering a sequence of
        pruned conductivity matrices.

        Args:
            idx (list):\n
                The list of failed edges.
            conduct (array):\n
                The network's conductivity matrix.

        Returns:
            ndarray: Edge-vector of squared pressure difference values.
            ndarray: Edge-vector of squared flow rate values.
        """

        # block p percent of the edges per realization
        graph_matrices = self.get_broken_links_asarray(idx, conduct)
        flow_observables = list(map(self.calc_flows_mapping, graph_matrices))

        # calc ensemble averages
        q_sq = np.power([fo[0] for fo in flow_observables], 2)
        p_sq = np.power([fo[2] for fo in flow_observables], 2)

        return p_sq, q_sq

    def calc_flows_mapping(self, graph_matrices):
        """
        Compute the pressure/flow landscape, considering a sequence of
        conductivity matrix realizations.

        Args:
            graph_matrices (iterable):\n
                An iterable of conductivity matrix realizations.

        Returns:
            list: A list of edge/nodal-vectors of effective flow/pressure
            landscape realizations.

        """
        C_aux = graph_matrices
        dP, P = self.calc_pressure(C_aux, self.circuit.nodes['source'])
        Q = self.calc_flow_from_pressure(C_aux, dP)

        return [Q, P, dP]

    def calc_sq_flow_avg(self, conduct):
        """
        Compute the average squared pressure/flow landscape, by internally
        generating a random sequence of pruned conductivity matrix
        realizations.

        Args:
            conduct (array):\n
                The network's conductivity matrix.

        Returns:
            list: A list of average, squared flows, pressures as well as the
            average radii and dissipation values.
        """
        idx = rd.choices(self.broken_sets, k=self.num_iteration)

        p_sq, q_sq = self.calc_sq_flow(idx, conduct)
        R, R_sq, R_cb = self.calc_random_radii(idx, conduct)

        n = float(self.num_iteration)
        avg_diss = np.sum(np.multiply(p_sq, R_cb), axis=0)/n
        avg_R = np.mean(R, axis=0)
        avg_dP_sq = np.mean(p_sq, axis=0)
        avg_F_sq = np.mean(q_sq, axis=0)

        return [avg_dP_sq, avg_F_sq, avg_R, avg_diss]

    def get_broken_links_asarray(self, idx, conduct):
        """
        Compute a sequence of pruned conductivity matrices based on the failed
        edge sets 'idx'.

        Args:
            idx (list):\n
                The list of failed edge sets.
            conduct (array):\n
                The network's conductivity matrix.

        Returns:
            list: A list of pruned conductivity matrices.

        """
        graph_matrices = [self.break_links(i, conduct) for i in idx]

        return graph_matrices
