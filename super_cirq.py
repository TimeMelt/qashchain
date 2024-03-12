import pennylane as qml
from jax import random, jit
import qkdc_electron as electron
from functools import partial
import qkdc_helper as helper

@partial(jit, static_argnames=['num_wires','device', 'shots'])
def qxHashCirq(input, num_wires, seed, pepper, device, shots=None):
    key = random.PRNGKey(seed)
    if device == 'default.qubit.jax':
        qdev = qml.device('default.qubit.jax', wires=num_wires, prng_key=key, shots=shots)
    elif device == 'qiskit.aer':
        backend = helper.chooseBackend(shots)
        qdev = qml.device(device, wires=num_wires, shots=shots, backend=backend)
    else:
        qdev = qml.device(device, wires=num_wires, shots=shots)

    @qml.qnode(qdev, interface="jax")
    def cirq(input, pepper, key):
        if pepper is None:
            electron.superPos(input)
        else:
            electron.angleEmbed(input,pepper)
        electron.rotLoop(input)
        electron.singleX(input)
        electron.strongTangle(input, key)
        electron.rotLoop(input)
        return [qml.var(qml.PauliZ(wires=i)) for i in range(num_wires)]
    
    return cirq(input, pepper, key)