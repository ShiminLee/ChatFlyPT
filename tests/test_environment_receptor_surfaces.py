"""Real collision measurements must retain receptor-surface specificity."""
import json
import os
import subprocess

from chatflypt.environment import sample_environment
from chatflypt.paths import ROOT, dump


def test_real_contact_surface_and_negative_controls():
    # A minimal mechanical model isolates segment identity and surface contact;
    # this is not a new fly morphology or a neural/behavior calibration trial.
    code=r"""
import json
import mujoco
from chatflypt.environment import measure_environment_contacts, sample_environment

channels={key:{'status':'candidate','side_ids':{'left':['1'],'right':['2'],'bilateral':['1','2']}}
          for key in ('foreleg_touch','midleg_touch','hindleg_touch','wing_touch','body_touch',
                      'sugar_specific','high_salt','bright')}
for key in ('leg_sweet_local','leg_appetitive_ascending'):
    channels[key]={'status':'candidate','source':'test-only anatomical routing catalogue',
        'body_part_ids':{p:{'left':[p+'L'],'right':[p+'R']} for p in ('foreleg','midleg','hindleg')}}
catalogue={'channels':channels}
rows=[]
for segment in ('rf_tarsus5','rf_tibia','rf_coxa','lm_trochanterfemur',
                'c_rostrum','c_haustellum','l_wing','r_wing','c_thorax','l_eye','c_abdomen12'):
    xml=f'''<mujoco><worldbody><geom name="env_food" type="plane" size="2 2 .1"/>
    <body><freejoint/><geom name="fly/{segment}" type="sphere" size=".1" margin=".2" gap=".1"/>
    </body></worldbody></mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    for z in (.15,.05):
        data.qpos[2]=z;mujoco.mj_forward(model,data)
        contacts=measure_environment_contacts(model,data)
        probe={'body_names':[],'frame':{'body_positions_mm':[]},'environment_contacts':contacts}
        for kind in ('sugar','high_salt','yeast','block'):
            env={'revision':1,'objects':[{'id':'food','type':kind,'strength':1.,
                'position_mm':[0,0,0],'size_mm':[1,1,1],'range_mm':1.,'yaw_deg':0.}]}
            sample=sample_environment(env,probe,catalogue)
            keys={i['channel'] for i in sample['items']}
            if z>.1:
                assert not contacts and not keys
            else:
                assert contacts and all(c['surface_contact'] and c['active_constraint'] for c in contacts)
                assert all('position_mm' in c for c in contacts)
                if segment.startswith('rf_'):
                    touch=[i for i in sample['items'] if i['channel']=='foreleg_touch']
                    assert len(touch)==1 and touch[0]['side']=='right'
                if segment=='lm_trochanterfemur':
                    assert any(i['channel']=='midleg_touch' and i['side']=='left' for i in sample['items'])
                sweet={'leg_sweet_local','leg_appetitive_ascending'}
                assert (keys & sweet)==(sweet if kind=='sugar' and segment=='rf_tarsus5' else set())
                assert 'sugar_specific' not in keys and 'high_salt' not in keys
                assert 'wing_touch' not in keys and 'body_touch' not in keys
                if segment.endswith('_wing'):
                    rejected=[x for x in sample['unmapped'] if x['channel']=='wing_touch']
                    assert rejected and rejected[0]['side']==('left' if segment[0]=='l' else 'right')
                if segment=='c_thorax':
                    assert any(x['channel']=='body_touch' and 'PDMN' in x['reason'] for x in sample['unmapped'])
                if segment in ('c_rostrum','c_haustellum') and kind in ('sugar','high_salt'):
                    assert any(x.get('body_part')=='mouth' and 'labellum' in x['reason'] for x in sample['unmapped'])
            rows.append({'segment':segment,'height':z,'kind':kind,'measured_contacts':contacts,
                         'items':sample['items'],'unmapped':sample['unmapped']})
print(json.dumps({'mujoco_version':mujoco.__version__,'rows':rows}))
"""
    python=ROOT/'.venv-body'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    result=subprocess.run([str(python),'-c',code],cwd=ROOT,capture_output=True,text=True,
                          encoding='utf8',timeout=30)
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    assert len(report['rows'])==88
    dump(ROOT/'reports/unified-pathway-20260924/receptor-surface-physics.json',report)


def test_missing_arista_never_falls_back_to_olfactory_site():
    probe={'body_names':['l_funiculus','r_funiculus'],
           'frame':{'body_positions_mm':[[0,0,0],[1,0,0]]},'environment_contacts':[]}
    catalogue={'channels':{k:{'status':'candidate','side_ids':{'left':['1'],'right':['2']}}
                           for k in ('warm','cold','yeast')}}
    for kind,channel in (('heat','warm'),('cold','cold')):
        obj={'id':'field','type':kind,'position_mm':[0,0,0],'size_mm':[1,1,1],
             'range_mm':.1,'strength':1.,'yaw_deg':0.}
        sample=sample_environment({'revision':1,'objects':[obj]},probe,catalogue)
        assert not sample['items']
        assert sum(row['channel']==channel and 'arista' in row['reason'] for row in sample['unmapped'])==2
        obj['type']='yeast'
        sample=sample_environment({'revision':1,'objects':[obj]},probe,catalogue)
        assert sample['items'][0]['channel']=='yeast' and sample['items'][0]['side']=='left'
        evidence=next(x for x in sample['evidence'] if x['mode']=='position_field')
        assert evidence['receptor_sites']['left']['body_name']=='l_funiculus'


def test_temperature_fields_combine_at_each_measured_arista_before_transduction():
    probe={'body_names':['l_arista','r_arista','l_funiculus','r_funiculus'],
           'frame':{'body_positions_mm':[[0,1,1],[0,-1,1],[9,9,9],[-9,-9,-9]]},
           'environment_contacts':[]}
    catalogue={'channels':{k:{'status':'candidate','side_ids':{'left':['1'],'right':['2']}}
                           for k in ('warm','cold')}}
    def source(oid,kind,pos,strength):
        return {'id':oid,'type':kind,'position_mm':pos,'size_mm':[.05,.05,.05],
                'range_mm':.1,'strength':strength,'yaw_deg':0.}
    hot=source('hot','heat',[0,1,1],1.)
    cold=source('cold','cold',[0,1,1],1.)
    env={'revision':1,'objects':[hot,cold]}
    sample=sample_environment(env,probe,catalogue)
    assert not sample['items']
    assert all(x['net_signed_field']==0 for x in sample['evidence'] if x['mode']=='net_temperature_field')
    cold['strength']=.25
    sample=sample_environment(env,probe,catalogue)
    assert [(x['channel'],x['side'],x['intensity']) for x in sample['items']]==[('warm','left',.75)]
    # Per-side receptor positions, not target sides/object IDs, determine the
    # simultaneous left-warm/right-cold result.
    cold['position_mm']=[0,-1,1]
    sample=sample_environment(env,probe,catalogue)
    assert {(x['channel'],x['side'],x['intensity']) for x in sample['items']}=={
        ('warm','left',1.),('cold','right',.25)}
    cold['position_mm']=[0,1,1];cold['strength']=2.
    sample=sample_environment(env,probe,catalogue)
    assert [(x['channel'],x['side'],x['intensity']) for x in sample['items']]==[('cold','left',1.)]
