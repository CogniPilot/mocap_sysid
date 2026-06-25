"""UDE residual model placeholder.

The previous implementation carried a hand-written Torch copy of the Sport Cub
grey-box equations. The grey-box plant is now owned by Modelica and compiled by
Rumoca, so that duplicate equation path was removed. Reintroduce this method by
building on Rumoca's generated JAX/CasADi targets rather than copying the plant
equations into Torch.
"""

raise ImportError("6DOF-UDE-NN needs a Rumoca-generated JAX/CasADi implementation")
