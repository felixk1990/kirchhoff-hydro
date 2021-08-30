# hailhydro
hyrdrodynamics methods for kirchhoff graphs
##  Introduction
This module 'hailhydro' is part of a series of pyton packages encompassing a set of class and method implementations for a kirchhoff network datatype, in order to to calculate flow/flux on lumped parameter model circuits. The flow/flux objects are embedded in the kirchhoff networks, and can be altered independently from the underlying graph structure. This is meant for fast(er) and efficient computation in the follow-up module 'goflow' and dependend of 'kirchhoff'.


##  Installation
pip install hailhydro

##  Usage

Generally, just take a weighted networkx graph and create a kirchhoff circuit from it (giving it a defined spatialy embedding and conductivity structure)
```
import kirchhoff.circuit_flow as kfc
import hailhydro.flow_init as hf
circuit=kfc.initialize_flow_circuit_from_crystal('simple',3)
flow=hf.initialize_flow_on_circuit(circuit)

```
To set node and edge attributes ('source','potential' ,'conductivity','flow_rate') use the set_source_landscape(), set_plexus_landscape() methods of the kirchhoff class. Further offering non-trivial random flow patterns for complex net adapation models(see 'goflow')
```
import kirchhoff.circuit_flow as kfc
import hailhydro.flow_random as hfr

circuit1=kfc.initialize_flow_circuit_from_crystal('simple',3)
circuit1.set_source_landscape('root_multi',num_sources=1)
circuit1.set_plexus_landscape()
random_flow=hfr.initialize_random_flow_on_circuit(circuit1)

circuit2=kfc.initialize_flow_circuit_from_crystal('simple',3)
circuit2.set_source_landscape('root_multi',num_sources=1)
circuit2.set_plexus_landscape()
rerouting_flow=hfr.initialize_rerouting_flow_on_circuit(circuit2)

```
./notebook contains examples to play with in the form of jupyter notebooks
##  Requirements
``` pandas ```,``` networkx ```, ``` numpy ```,```plotly```, ```kirchhoff```

##  Gallery

## Acknowledgement
```hailhydro``` written by Felix Kramer
