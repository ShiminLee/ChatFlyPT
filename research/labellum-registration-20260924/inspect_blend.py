"""Read Blender DNA/ID names without opening or changing the source model."""
import json
from pathlib import Path
import re
import struct

folder=Path(__file__).resolve().parent
source=folder/'neuromechfly_full_model.blend'
data=source.read_bytes()
assert data[:7]==b'BLENDER'
pointer=8 if data[7:8]==b'-' else 4
endian='<' if data[8:9]==b'v' else '>'
header=struct.Struct(endian+('4sIQII' if pointer==8 else '4sIIII'))
blocks=[];addresses={};offset=12
while offset+header.size<=len(data):
    code,size,address,sdna,count=header.unpack_from(data,offset)
    offset+=header.size
    blocks.append((code,sdna,count,data[offset:offset+size]))
    addresses[address]=blocks[-1]
    offset+=size
    if code==b'ENDB':break
dna=next(b[3] for b in blocks if b[0]==b'DNA1')
assert dna[:8]==b'SDNANAME'
i=8
def integer():
    global i
    value=struct.unpack_from(endian+'I',dna,i)[0];i+=4;return value
def strings(n):
    global i
    result=[]
    for _ in range(n):
        end=dna.index(0,i);result.append(dna[i:end].decode('utf8'));i=end+1
    i=(i+3)&~3
    return result
names=strings(integer());assert dna[i:i+4]==b'TYPE';i+=4
types=strings(integer());assert dna[i:i+4]==b'TLEN';i+=4
lengths=struct.unpack_from(endian+str(len(types))+'H',dna,i);i+=2*len(types);i=(i+3)&~3
assert dna[i:i+4]==b'STRC';i+=4
structs=[]
for _ in range(integer()):
    st,n=struct.unpack_from(endian+'HH',dna,i);i+=4;fields=[]
    for _ in range(n):
        typ,name=struct.unpack_from(endian+'HH',dna,i);i+=4
        fields.append((typ,name))
    structs.append((st,fields))
by_type={types[st]:(st,fields) for st,fields in structs}
def field_offset(st,name):
    pos=0
    for typ,ni in by_type[st][1]:
        field=names[ni]
        if field==name:return pos
        count=1
        for num in re.findall(r'\[(\d+)\]',field):count*=int(num)
        pos+=(pointer if '*' in field else lengths[typ])*count
    raise KeyError((st,name))
id_name_offset=field_offset('ID','name[66]')
result={}
for code,sdna,count,payload in blocks:
    if sdna>=len(structs):continue
    st=types[structs[sdna][0]]
    if st in ('Object','Mesh','Material','Collection','Armature','Text'):
        # These datablocks all begin with a Blender ID structure.
        assert names[by_type[st][1][0][1]]=='id'
        label=payload[id_name_offset:id_name_offset+66].split(b'\0',1)[0].decode('utf8','replace')
        result.setdefault(st,[]).append(label)
    elif st in ('bDeformGroup','bPoseChannel','Bone'):
        fields=by_type[st][1]
        name=next(names[n] for _,n in fields if names[n].startswith('name['))
        no=field_offset(st,name);nbytes=int(re.search(r'\[(\d+)\]',name)[1])
        label=payload[no:no+nbytes].split(b'\0',1)[0].decode('utf8','replace')
        result.setdefault(st,[]).append(label)
def ptr(payload,pos):return struct.unpack_from(endian+('Q' if pointer==8 else 'I'),payload,pos)[0]
def value(payload,st,name,fmt='i'):
    return struct.unpack_from(endian+fmt,payload,field_offset(st,name))[0]
groups=[]
for code,sdna,count,payload in blocks:
    if code!=b'OB\0\0':continue
    label=payload[id_name_offset:id_name_offset+66].split(b'\0',1)[0].decode('utf8','replace')
    group_address=ptr(payload,field_offset('Object','defbase'));group_names=[]
    while group_address:
        gp=addresses[group_address][3];go=field_offset('bDeformGroup','name[64]')
        group_names.append(gp[go:go+64].split(b'\0',1)[0].decode())
        group_address=ptr(gp,field_offset('bDeformGroup','*next'))
    if not any(w in ' '.join(group_names).lower() for w in ('labell','probosc')):continue
    mesh_addr=ptr(payload,field_offset('Object','*data'));mesh=addresses[mesh_addr][3]
    dvert_addr=ptr(mesh,field_offset('Mesh','*dvert'))
    vertex_count=value(mesh,'Mesh','totvert');weighted={n:[] for n in group_names}
    if dvert_addr:
        dvert=addresses[dvert_addr][3];stride=lengths[by_type['MDeformVert'][0]]
        for index in range(vertex_count):
            entry=dvert[index*stride:(index+1)*stride]
            nw=value(entry,'MDeformVert','totweight');wa=ptr(entry,field_offset('MDeformVert','*dw'))
            if not nw:continue
            weights=addresses[wa][3];ws=lengths[by_type['MDeformWeight'][0]]
            for k in range(nw):
                w=weights[k*ws:(k+1)*ws];gi=value(w,'MDeformWeight','def_nr');wt=value(w,'MDeformWeight','weight','f')
                weighted[group_names[gi]].append([index,wt])
    groups.append({'object':label,'groups':group_names,'vertex_count':vertex_count,'has_dvert':bool(dvert_addr),
                   'weights':weighted})
report={'source':source.name,'blender_header':data[:12].decode(),'reader_scope':'SDNA ID names and named mouth vertex-group membership; no visual anatomy inference',
        'datablock_names':result,'mouth_vertex_groups':groups,
        'block_types':sorted({types[structs[b[1]][0]] for b in blocks if b[1]<len(structs)})}
(folder/'original-blender-datablocks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'counts':{k:len(v) for k,v in result.items()},'mouth_matches':{k:[v for v in vals if any(w in v.lower() for w in ('rostr','haust','lab','probosc','mouth','taste','sensil'))] for k,vals in result.items()}},indent=2))
print(json.dumps([{**{k:v for k,v in g.items() if k!='weights'},'weight_counts':{k:len(v) for k,v in g['weights'].items()}} for g in groups],indent=2))
