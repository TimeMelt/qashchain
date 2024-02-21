import datetime
import qkdc_helper as helper
import super_cirq as s_cirq
import photonic_cirq as p_cirq
import jax.random as random
from jax import numpy as jnp

class Block:
    def __init__(self, index, timestamp, data, previous_hash, num_wires, seed, pepper, device, circuit):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.hash_block(num_wires, seed, pepper, device, circuit)

    def hash_block(self, num_wires, seed, pepper, device, circuit): #calculate hash for new block
        new_index = helper.convert2Arr(self.index)
        new_timestamp = str(self.timestamp)
        new_timestamp = new_timestamp.replace("-", "").replace(":", "").replace(".", "").replace(" ", "")
        new_timestamp = helper.createData(new_timestamp)
        new_prev_hash = helper.createData(self.previous_hash)
        if circuit == 'superconductor':
            index_hash = s_cirq.qxHashCirq(new_index, len(new_index), seed, pepper, device)
            index_hash = helper.processOutput(index_hash, 'hex')
            timestamp_hash = s_cirq.qxHashCirq(new_timestamp[:num_wires], num_wires, seed, pepper, device)
            timestamp_hash = helper.processOutput(timestamp_hash, 'hex')
            data_hash = s_cirq.qxHashCirq(self.data[:num_wires], num_wires, seed, pepper, device)
            data_hash = helper.processOutput(data_hash, 'hex')
            prev_hash_hashed = s_cirq.qxHashCirq(new_prev_hash[:num_wires], num_wires, seed, pepper, device)
            prev_hash_hashed = helper.processOutput(prev_hash_hashed, 'hex')
            return index_hash + timestamp_hash + data_hash + prev_hash_hashed
        elif circuit == 'photonic':
            index_hash = p_cirq.qxBerryCirq(new_index, len(new_index), pepper, device)
            index_hash = helper.processOutput(index_hash, 'hex')
            timestamp_hash = p_cirq.qxBerryCirq(new_timestamp[:num_wires], num_wires, pepper, device)
            timestamp_hash = helper.processOutput(timestamp_hash, 'hex')
            data_hash = p_cirq.qxBerryCirq(self.data[:num_wires], num_wires, pepper, device)
            data_hash = helper.processOutput(data_hash, 'hex')
            prev_hash_hashed = p_cirq.qxBerryCirq(new_prev_hash[:num_wires], num_wires, pepper, device)
            prev_hash_hashed = helper.processOutput(prev_hash_hashed, 'hex')
            return index_hash + timestamp_hash + data_hash + prev_hash_hashed
        else:
            print("circuit type provided is unsupported...")

class Blockchain:
    def __init__(self, chain, device, circuit):
        self.device = device
        self.circuit = circuit
        self.print_params()
        self.chain = self.load_chain(chain)

    def print_params(self): # print device and circuit being used
        print(f"device '{self.device}' is now set...")
        print(f"circuit '{self.circuit}' is now set...")
    
    def load_chain(self, chain): # load blockchain from .npy file or directly from list/tuple
        if chain is None:
            print("no chain file specified, blank chain is being created...")
            return []
        elif type(chain) is list:
            return chain
        elif type(chain) is tuple:
            return list(chain)
        else:
            try:
                print("chain file is loading...")
                out = jnp.load("blockchain.npy", allow_pickle=True)
                return list(out)
            except:
                print("error loading blockchain from file, blank chain is being created...")
                return []

    def save_chain(self): # save chain to .npy file
        jnp.save("blockchain.npy", self.chain, allow_pickle=True) 
    
    def create_genesis_block(self, num_wires, seed, pepper): # create starting block on new blockchain
        print("creating/adding genesis block...")
        data_string = helper.generateRandomString(num_wires)
        data_array = helper.createAndPad(data_string, 0)
        key = random.PRNGKey(seed)
        prev_hash_array = random.uniform(key, (num_wires,1), minval=0, maxval=1000, dtype="float64").flatten()
        if self.circuit == 'superconductor':
            prev_hash_hashed = s_cirq.qxHashCirq(prev_hash_array, num_wires, seed, pepper, self.device)
        else:  
            prev_hash_hashed = p_cirq.qxBerryCirq(prev_hash_array, num_wires, pepper, self.device)
        prev_hash_hashed = helper.processOutput(prev_hash_hashed, 'hex')
        genesis_block = Block(0, datetime.datetime.now(), data_array, prev_hash_hashed, num_wires, seed, pepper, self.device, self.circuit)
        self.chain.append(genesis_block)

    def add_block(self, data, num_wires, seed, pepper): # add new block to blockchain
        print(f"adding block {len(self.chain)}...")
        previous_block = self.chain[-1]
        new_block = Block(len(self.chain), datetime.datetime.now(), data, previous_block.hash, num_wires, seed, pepper, self.device, self.circuit)
        self.chain.append(new_block)

    def print_blocks(self): # print all blocks and their values from blockchain
        print("...")
        for block in self.chain:
            print(f"index: {block.index}")
            print(f"timestamp: {block.timestamp}")
            print(f"data: {block.data}")
            print(f"previous hash: {block.previous_hash}")
            print(f"current hash: {block.hash}")
            print("...")