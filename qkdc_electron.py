import pennylane as qml
from jax import random
from jax import numpy as jnp

# embed pepper/state based on float array
def angleEmbed(input, pepper):
    qml.AngleEmbedding(pepper, range(input.size))

# induce superposition on all wires if no pepper is specified
def superPos(inputs):
    for q in range(inputs.size): 
        qml.Hadamard(wires=q)

# arbitrary rotations on x,y,z axis
def rotLoop(inputs):
    for w,q in enumerate(inputs): 
        phi = jnp.tan(q) * jnp.sin(q)
        qml.RX(phi, wires=w)
        qml.RY(phi, wires=w)
        qml.RZ(phi, wires=w)

# qutrit rotations on x,y,z axis
def qutritLoop(inputs):
    for w,q in enumerate(inputs):
        phi = jnp.tan(q) * jnp.sin(q)
        qml.TRX(phi, wires=w)
        qml.TRY(phi, wires=w)
        qml.TRZ(phi, wires=w)

# single excitation across wires
def singleX(inputs):
    qml.PauliX(wires=0)
    for w,q in enumerate(inputs):
        phi = jnp.tan(q) * jnp.sin(q)
        if w == 0:
            pass
        else:
            qml.SingleExcitation(phi, wires=[0, inputs.size-w])

# strong entanglement interaction across all wires
def strongTangle(inputs, key):
    shape = qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=inputs.size)
    weights = random.uniform(key, shape)
    qml.StronglyEntanglingLayers(weights=weights, wires=range(inputs.size), imprimitive=qml.ops.CZ)