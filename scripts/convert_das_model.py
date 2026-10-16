"""Convert pinned DAS numeric graph; never deserialize saved Python bytecode."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from chatflypt.paths import sha256,dump

SOURCE=ROOT/'research/outward-signals/das-pretrained/pulse+sine_model.h5'
TARGET=ROOT/'data/derived/acoustic-readout/das-model.onnx'


def load_graph():
    import h5py
    manifest=json.loads((SOURCE.parent/'sources.json').read_text())
    expected=next(s['sha256'] for s in manifest if s['path']==SOURCE.name)
    if sha256(SOURCE)!=expected:raise ValueError('DAS source hash changed')
    with h5py.File(SOURCE,'r') as f:
        graph=json.loads(f.attrs['model_config'])['config']; weights={}
        for layer in graph['layers']:
            name=layer['name']
            if layer['class_name'] in ('Conv1D','Dense'):
                weights[name]=[np.array(f['model_weights'][name][name][k+':0'],dtype=np.float32) for k in ('kernel','bias')]
    return graph,weights


def inputs(layer):
    return [node[0] for node in layer['inbound_nodes'][0]]


def convert():
    import onnx
    from onnx import helper as h, numpy_helper as nh, TensorProto as T
    graph,weights=load_graph();nodes=[];constants=[];counts={}
    def const(name,x):
        constants.append(nh.from_array(np.asarray(x),name));return name
    eps=const('norm_epsilon',np.float32(1e-5))
    for layer in graph['layers']:
        kind=layer['class_name'];name=layer['name'];c=layer['config']
        counts[kind]=counts.get(kind,0)+1
        if kind=='InputLayer':continue
        src=inputs(layer);x=src[0]
        if kind=='Conv1D':
            if c['strides']!=[1] or c['padding']!='same' or c['data_format']!='channels_last' or c['activation']!='linear':
                raise ValueError('Unexpected convolution')
            kernel,bias=weights[name];pad=(len(kernel)-1)*c['dilation_rate'][0]
            nodes.extend([h.make_node('Transpose',[x],[name+'_nct'],perm=[0,2,1]),
                h.make_node('Conv',[name+'_nct',const(name+'_w',kernel.transpose(2,1,0)),const(name+'_b',bias)],
                    [name+'_conv'],dilations=c['dilation_rate'],pads=[pad//2,pad-pad//2]),
                h.make_node('Transpose',[name+'_conv'],[name],perm=[0,2,1])])
        elif kind=='Activation':
            act=c['activation']
            if act not in ('relu','softmax'):raise ValueError('Unexpected activation')
            nodes.append(h.make_node('Relu' if act=='relu' else 'Softmax',[x],[name],**({'axis':2} if act=='softmax' else {})))
        elif kind=='Lambda':
            if c['module']!='dss.tcn.tcn':raise ValueError('Unexpected Lambda module')
            # Published channel_normalization; never unmarshal the saved function.
            nodes.extend([h.make_node('Abs',[x],[name+'_abs']),
                h.make_node('ReduceMax',[name+'_abs'],[name+'_max'],axes=[2],keepdims=1),
                h.make_node('Add',[name+'_max',eps],[name+'_den']),
                h.make_node('Div',[x,name+'_den'],[name])])
        elif kind=='SpatialDropout1D':nodes.append(h.make_node('Identity',[x],[name]))
        elif kind=='Add':
            for i,y in enumerate(src[1:]):
                target=name if i==len(src)-2 else name+'_sum'+str(i)
                nodes.append(h.make_node('Add',[x,y],[target]));x=target
        elif kind=='Dense':
            if c['activation']!='linear':raise ValueError('Unexpected dense activation')
            kernel,bias=weights[name]
            nodes.extend([h.make_node('MatMul',[x,const(name+'_w',kernel)],[name+'_matmul']),
                          h.make_node('Add',[name+'_matmul',const(name+'_b',bias)],[name])])
        else:raise ValueError('Unexpected layer '+kind)
    model=h.make_model(h.make_graph(nodes,'DAS Dmel_single_v1.0',
        [h.make_tensor_value_info('input_1',T.FLOAT,[None,4096,1])],
        [h.make_tensor_value_info('activation_16',T.FLOAT,[None,4096,3])],constants),
        opset_imports=[h.make_opsetid('',13)],ir_version=8,producer_name='ChatFlyPT explicit DAS graph converter')
    onnx.checker.check_model(model);TARGET.parent.mkdir(parents=True,exist_ok=True);onnx.save(model,TARGET)
    receipt={'source_sha256':sha256(SOURCE),'onnx_sha256':sha256(TARGET),'layers':counts,
        'weights_retrained':False,'serialized_python_executed':False,'classes':['noise','sine','pulse'],
        'source':'https://github.com/janclemenslab/das-menagerie/releases/tag/Dmel_single_v1.0',
        'normalization_source':'https://github.com/janclemenslab/das/blob/master/src/das/models/tcn/tcn.py',
        'normalization':'x / (max(abs(x), channel axis, keepdims=True) + 1e-5)',
        'numeric_reference':'independent NumPy graph, not original TensorFlow runtime execution'}
    dump(TARGET.with_suffix('.json'),receipt)
    print(json.dumps(receipt,indent=2))


def numpy_reference(batch):
    """Independent direct cross-correlation; validation only, not serving."""
    graph,weights=load_graph();values={'input_1':np.array(batch,dtype=np.float32)}
    for layer in graph['layers']:
        kind=layer['class_name'];name=layer['name'];c=layer['config']
        if kind=='InputLayer':continue
        src=[values[s] for s in inputs(layer)];x=src[0]
        if kind=='Conv1D':
            kernel,bias=weights[name];d=c['dilation_rate'][0];padding=(len(kernel)-1)*d
            padded=np.pad(x,((0,0),(padding//2,padding-padding//2),(0,0)))
            out=np.broadcast_to(bias,(*x.shape[:2],len(bias))).copy()
            for i,w in enumerate(kernel):out+=padded[:,i*d:i*d+x.shape[1]]@w
        elif kind=='Activation':
            if c['activation']=='relu':out=np.maximum(x,0)
            else:
                out=np.exp(x-x.max(axis=2,keepdims=True));out/=out.sum(axis=2,keepdims=True)
        elif kind=='Lambda':out=x/(abs(x).max(axis=2,keepdims=True)+np.float32(1e-5))
        elif kind=='SpatialDropout1D':out=x
        elif kind=='Add':
            out=src[0].copy()
            for y in src[1:]:out+=y
        elif kind=='Dense':out=x@weights[name][0]+weights[name][1]
        else:raise ValueError(kind)
        values[name]=out
    return values[graph['output_layers'][0][0]]


if __name__=='__main__':convert()
