"""Guarda los pesos grandes del modelo ONNX en float16 (la mitad de tamaño) y
agrega un Cast a float32 para que el cálculo siga igual de preciso."""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
new_inits, casts = [], []
for init in g.initializer:
    arr = numpy_helper.to_array(init)
    if init.data_type == TensorProto.FLOAT and arr.size >= 1024:
        name16 = init.name + "__f16"
        new_inits.append(numpy_helper.from_array(arr.astype(np.float16), name16))
        casts.append(helper.make_node("Cast", [name16], [init.name], to=TensorProto.FLOAT, name=init.name + "__cast"))
    else:
        new_inits.append(init)
del g.initializer[:]
g.initializer.extend(new_inits)
nodes = list(g.node)
del g.node[:]
g.node.extend(casts + nodes)
onnx.save(m, dst)
