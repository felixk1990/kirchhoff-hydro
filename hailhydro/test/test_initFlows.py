import networkx as nx
import numpy as np
import hailhydro


from hailhydro.flow_init import Flow
from kirchhoff.circuit_flow import FlowCircuit
def test_flowContructors():

    print('\n test init from graph')
    n=3
    G=nx.grid_graph((n,n,1))
    F1 = Flow(G)
    print(F1)
    print(F1.circuit)
    print(F1.circuit.graph)

    print('\n test init from on the fly-constructed flow circuit')
    C = FlowCircuit(G)
    F2 = Flow(C)
    print(F2)
    print(F2.circuit)

    print('\n test init from prebuild flow circuit')
    import kirchhoff.circuit_flow as kfc
    K = kfc.initialize_flow_circuit_from_crystal('simple',3)
    F3 = Flow(K)
    print(F3)
    print(F3.circuit)

    try:
        print(Flow(0))
    except:
        print('whoops...not a network or circuit?')

    for f in [F1, F2, F3]:

        print(f.calc_configuration_flow())

from hailhydro.flux_init import Flux
from kirchhoff.circuit_flux import FluxCircuit
def test_fluxConstructors():


    print('\n test init from graph')
    n=3
    G=nx.grid_graph((n,n,1))
    F1 = Flux(G)
    print(F1)
    print(F1.circuit)

    print('\n test init from on the fly-constructed flux circuit')
    C = FluxCircuit(G)
    F2 = Flux(C)
    print(F2)
    print(F2.circuit)

    print('\n test init from prebuild flux circuit')
    import kirchhoff.circuit_flux as kfc
    K = kfc.initialize_flux_circuit_from_crystal('simple',3)
    F3 = Flux(K)
    print(F3)
    print(F3.circuit)

    try:
        print(Flux(0))
    except:
        print('whoops...not a network or circuit?')


from hailhydro.flux_overflow import Overflow

def test_overflowConstructor():

    print('\n test init from graph')
    n=3
    G=nx.grid_graph((n,n,1))
    F1 = Overflow(G, pars_source=dict(modeSRC='dipole_border'))
    print(F1)
    print(F1.circuit)

    print('\n test init from on the fly-constructed overflow circuit')
    C = FluxCircuit(G)
    F2 = Overflow(C, pars_source=dict(modeSRC='dipole_border'))
    print(F2)
    # print(F2.circuit.edges)

    print('\n test init from prebuild overflow circuit')
    import kirchhoff.circuit_flux as kfc
    K = kfc.initialize_flux_circuit_from_crystal('simple',3)
    F3 = Overflow(K, pars_source=dict(modeSRC='dipole_border'))
    print(F3)
    print(F3.circuit)

    try:
        print(Overflow(0))
    except:
        print('whoops...not a network or circuit?')

    for f in [F1, F2, F3]:

        m = len(f.circuit.G.edges())
        x = np.ones(m)
        f.update_transport_matrix(x)
        f.solve_absorbing_boundary()
        f.calc_absorption()
        jac = f.calc_absorption_jacobian()
        print(jac)

from hailhydro.flow_random import FlowRandom

def test_randomFlowConstructor():
    setting=dict(mode='default', noise=1.)
    print('\n test init from graph')
    n=3
    G=nx.grid_graph((n,n,1))
    F1 = FlowRandom(constr=G, flow_setting=setting)
    print(F1)
    print(F1.circuit)

    print('\n test init from on the fly-constructed flow circuit')
    C = FlowCircuit(G)
    F2 = FlowRandom(C, flow_setting=setting)
    print(F2)
    print(F2.circuit)

    print('\n test init from prebuild flow circuit')
    import kirchhoff.circuit_flow as kfc
    K = kfc.initialize_flow_circuit_from_crystal('simple',3)
    F3 = FlowRandom(K, flow_setting=setting)
    print(F3)
    print(F3.circuit)

    try:
        print(FlowRandom(0))
    except:
        print('whoops...not a network or circuit?')

    for f, circuit in zip([F1, F2, F3],[F1.circuit, C, K]):

        q = f.calc_sq_flow_effective(circuit.edges['conductivity'])
        print(q)


from hailhydro.flux_random_overflow import FluxRandom

def test_randomOverflow():

    print('\n test init from graph')
    n=3
    G=nx.grid_graph((n,n,1))
    F1 = Overflow(G)
    print(F1)
    print(F1.circuit)

    print('\n test init from on the fly-constructed overflow circuit')
    C = FluxCircuit(G)
    F2 = Overflow(C)
    print(F2)
    print(F2.circuit)

    print('\n test init from prebuild overflow circuit')
    import kirchhoff.circuit_flux as kfc
    K = kfc.initialize_flux_circuit_from_crystal('simple',3)
    F3 = Overflow(K)
    print(F3)
    print(F3.circuit)

    try:
        print(Flux(0))
    except:
        print('whoops...not a network or circuit?')

    for f in [F1, F2, F3]:

        m = len(f.circuit.G.edges())
        x = np.ones(m)
        f.update_transport_matrix(x)
        f.solve_absorbing_boundary()
        f.calc_absorption()
        jac = f.calc_absorption_jacobian()
        print(jac)

from hailhydro.flow_random import FlowReroute

def test_RerouteConstructors():

    src = dict(mode='root_multi')
    setting=dict(mode='default', num_iter=10, p_broken=.1)

    print('\n test init from graph')
    n=3
    G=nx.grid_graph((n,n,1))
    F1 = FlowReroute(constr=G, pars_source=src, flow_setting=setting)
    print(F1)
    print(F1.circuit)

    print('\n test init from on the fly-constructed flow circuit')
    C = FlowCircuit(G)
    F2 = FlowReroute(C, pars_source=src, flow_setting=setting)
    print(F2)
    print(F2.circuit)

    print('\n test init from prebuild flow circuit')
    import kirchhoff.circuit_flow as kfc
    K = kfc.initialize_flow_circuit_from_crystal('simple',3)
    F3 = FlowReroute(K, pars_source=src, flow_setting=setting)
    print(F3)
    print(F3.circuit)

    try:
        print(FlowRandom(0))
    except:
        print('whoops...not a network or circuit?')

    for f, circuit in zip([F1, F2, F3],[F1.circuit, C, K]):

        q = f.calc_sq_flow_avg(circuit.edges['conductivity'])
        print(q)
