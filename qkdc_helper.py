from jax import config
from jax.lib import xla_bridge
from jax import numpy as jnp
import struct, os, string, random
from binascii import b2a_base64

# switch on/off double precision float mode
def x64Switch(x64):
    config.update("jax_enable_x64", x64)
    if x64 is True:
        print("jax float64 = enabled")
    else:
        print("jax float64 = disabled")

# check whether cpu or gpu is used
def getBackend():
    print(f"platform = {xla_bridge.get_backend().platform}")

# process output of quantum circuits into proper hash
def processOutput(output, format):
    hex_params = {
        "unpack_long_long": '<Q',
        "pack_double": '<d',
    }
    output_alt = []
    for value in output: # process complex into hex
        val_alt = value*100
        if val_alt < 0:
            output_alt.append(hex(struct.unpack(hex_params['unpack_long_long'], struct.pack(hex_params['pack_double'], -val_alt))[0]))
        else:
            output_alt.append(hex(struct.unpack(hex_params['unpack_long_long'], struct.pack(hex_params['pack_double'], val_alt))[0]))
    output_string = ''.join(output_alt)
    output_string = output_string.replace('0x', "") # remove hex markers
    output_string = output_string.replace(output_string[:4], "", 1) # remove chars for increased hash security 
    if format == 'hex':
        pass
    elif format == 'base64':
        output_string = b2a_base64(bytes(output_string, 'utf-8')).decode('utf-8')# convert to base64
    else:
        print("invalid format...")
    return output_string

def setEnvVars(device):
    if device == "lightning.kokkos":
        os.environ["OMP_PROC_BIND"] ='spread'
        print("enviornment variables are now set!")
    else:
        print("kokkos is not being used, no enviornment variables need to be set...")

def generateRandomString(length):
  characters = string.ascii_letters + string.punctuation + string.digits 
  random_string = ''.join(random.choice(characters) for i in range(length))
  return random_string

def convert2String(input):
    out = []
    for ch in input:
        out.append(chr(int(ch)))
    output = ''.join(out)
    return output

def convert2Arr(input):
    output = jnp.array([])
    if hasattr(input, '__iter__'):
        for ch in input:
            output = jnp.append(output, ch)
    else:
       output = jnp.append(output, input)
    return output
        
# convert chars to unicode
def createData(text): 
    arr = jnp.array([])
    for ch in text:
        arr = jnp.append(arr, ord(ch))
    return arr

# pad data based on desired hash length
def padData(txt_arr, pad_count): 
    if pad_count == 0:
        return txt_arr
    else:
        new_arr = txt_arr
        for q in range(pad_count):
            new_arr = jnp.append(new_arr, txt_arr[q] * jnp.tan(txt_arr[q]))
        return new_arr

# convert to unicode and pad
def createAndPad(text, pad_count):
    arr = createData(text)
    arr1 = padData(arr, pad_count)
    return arr1