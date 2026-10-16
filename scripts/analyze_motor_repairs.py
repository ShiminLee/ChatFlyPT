"""Audit measured joint changes, zero controls and explicit transmitter ablation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from chatflypt.paths import ROOT,dump,sha256
from scripts.test_motor_repairs import REPORT,RAW


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    paired=[]
    for folder in sorted((RAW/'replay').iterdir()):
        old=read(folder/'legacy/body-result.json');new=read(folder/'tibia/body-result.json')
        assert old['source_spikes_sha256']==new['source_spikes_sha256']
        a=np.array([f['joint_angles_rad'] for f in old['trajectory']]);b=np.array([f['joint_angles_rad'] for f in new['trajectory']])
        t=np.array([f['time_ms'] for f in new['trajectory']]);j=[i for i,n in enumerate(new['joint_names']) if 'tibia' in n and 'pitch' in n]
        delta=np.rad2deg(b[:,j]-a[:,j]);which=int(np.unravel_index(np.abs(delta).argmax(),delta.shape)[1])
        drive=read(folder/'tibia/behavior-drive.json')
        first_command=min(at for at,r in zip(drive['available_at_ms'],drive['joint_motor']['offsets_rad']) if any(r.values()))
        np.testing.assert_array_equal(a[t<=first_command],b[t<=first_command])
        assert np.abs(delta).max()>0
        paired.append({'case':folder.name,'max_tibia_difference_deg':float(np.abs(delta).max()),
            'joint':new['joint_names'][j[which]],'joint_motor_spikes':new['joint_motor_spikes'],
            'first_joint_command_ms':first_command,'no_motion_difference_before_command':True,
            'source_spikes_sha256':new['source_spikes_sha256'],
            'legacy_body':str((folder/'legacy/body-result.json').relative_to(ROOT)),
            'new_body':str((folder/'tibia/body-result.json').relative_to(ROOT))})
    closed=read(REPORT/'malecns-results.json')['records'];controls=[]
    for seed in (491701,491702):
        zero=read(RAW/'closed-loop'/f'malecns-{seed}-empty/turn-0/body-result.json')
        touch=read(RAW/'closed-loop'/f'malecns-{seed}-strong_touch/turn-0/body-result.json')
        assert zero['controls']['powered_steps']==touch['controls']['powered_steps']==0
        for field in ('body_positions_mm','body_quaternions_wxyz','joint_angles_rad'):
            np.testing.assert_array_equal([f[field] for f in zero['trajectory']],[f[field] for f in touch['trajectory']])
        controls.append({'seed':seed,'touch_identical_to_no_stimulus_body':True,
            'unmapped_active_cells':touch['motor_coverage']['unmapped_active_neurons'],
            'Fe_reductor_cells':[x for x in touch['motor_coverage']['unmapped'] if x['type']=='Fe reductor MN']})
    salt=read(REPORT/'salt-transmitter-ablation.json')
    for seed in (491701,491702):
        recs=[r for r in salt['records'] if r['seed']==seed]
        sugar=[r for r in recs if r['condition']=='mouth_sugar'];s=[r for r in recs if r['condition']=='mouth_high_salt']
        assert sugar[0]['MN9_spikes']==sugar[1]['MN9_spikes']>0
        assert s[0]['MN9_spikes']>0 and s[1]['MN9_spikes']==0
        for field in ('counts','spike_ids','spike_steps','state_record'):
            with np.load(ROOT/sugar[0]['neural_record']) as a,np.load(ROOT/sugar[1]['neural_record']) as b:
                np.testing.assert_array_equal(a[field],b[field])
    physical=list((RAW/'replay').glob('*/*/body-result.json'))+list((RAW/'closed-loop').glob('*/turn-0/body-result.json'))+list((RAW/'salt-sign').glob('*/body-result.json'))
    for path in physical:
        b=read(path);assert b['status']=='ok' and b['physics_run'] and b['summary']['finite_state'] and not b['summary']['physics_warnings']
    audit={'paired_replay':paired,'touch_controls':controls,'physical_runs':len(physical),
        'all_physics_finite_without_warnings':True,'closed_loop_runs':len(closed),
        'salt_hypothesis_default_enabled':False,'sugar_neural_response_unchanged_by_salt_sign':True,
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in
            [ROOT/'chatflypt/joint_motor.py',ROOT/'chatflypt/behavior_readout.py',ROOT/'chatflypt/behavior_body.py',
             ROOT/'chatflypt/body_feedback.py',ROOT/'vendor/badapple-fly/backend/body.py',ROOT/'vendor/badapple-fly/backend/evidence.py']}}
    dump(REPORT/'audit.json',audit)
    plot(paired,salt)
    print(json.dumps(audit,ensure_ascii=False,indent=2))


def plot(paired,salt):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ax,seed in zip(axes[:,0],(491701,491702)):
        p=next(r for r in paired if r['case']==f'{seed}-leg_sugar')
        for name,path,color in [('原控制器',p['legacy_body'],'#9b8775'),('新增胫节通道',p['new_body'],'#087f8c')]:
            r=read(ROOT/path);j=r['joint_names'].index(p['joint']);t=[f['time_ms'] for f in r['trajectory']]
            ax.plot(t,np.rad2deg([f['joint_angles_rad'][j] for f in r['trajectory']]),label=name,color=color,lw=1.8)
        ax.set(title=f'真实腿部糖接触记录 · 种子 {seed}\n{p["joint"]}（差异最大的胫节）',xlabel='仿真时间（ms）',ylabel='实际关节角（°）')
        ax.grid(alpha=.18);ax.legend(fontsize=9)
    ax=axes[0,1];labels=['口器糖','口器高盐','腿部糖'];conditions=['mouth_sugar','mouth_high_salt','leg_sugar'];x=np.arange(3)
    for k,seed in enumerate((491701,491702)):
        values=[next(r['max_tibia_difference_deg'] for r in paired if r['case']==f'{seed}-{c}') for c in conditions]
        bars=ax.bar(x+(k-.5)*.32,values,.32,label=f'种子 {seed}',color=('#087f8c','#e6a14a')[k]);ax.bar_label(bars,fmt='%.2f',fontsize=9,padding=3)
    ax.set_xticks(x,labels);ax.set(ylabel='新旧回放的最大胫节角差（°）',title='同一份脉冲记录，仅改变运动适配器');ax.set_ylim(0,16);ax.legend()
    ax=axes[1,1];labels=['糖 701','糖 702','高盐 701','高盐 702'];x=np.arange(4)
    for k,profile in enumerate(('official_prediction','LB3d_inhibitory_hypothesis')):
        values=[next(r['MN9_spikes'] for r in salt['records'] if r['condition']==c and r['seed']==seed and r['profile']==profile)
            for c in ('mouth_sugar','mouth_high_salt') for seed in (491701,491702)]
        bars=ax.bar(x+(k-.5)*.32,values,.32,label=['原递质预测','LB3d 抑制性假设'][k],color=('#9b8775','#087f8c')[k]);ax.bar_label(bars,fontsize=9,padding=3)
    ax.set_xticks(x,labels);ax.set(ylabel='MN9 口器伸展神经元脉冲数',title='固定感觉事件，检验高盐递质冲突\n候选模型实验，未启用为默认模型');ax.set_ylim(0,46);ax.legend(fontsize=9)
    fig.suptitle('ChatFlyPT · MuJoCo 运动修复实测\n500 ms / 条；未使用 LLM 行动控制；关节增益尚未生理标定',fontsize=15)
    fig.savefig(REPORT/'motor-repair.png',dpi=180);fig.savefig(REPORT/'motor-repair.svg');plt.close(fig)


if __name__=='__main__':main()
