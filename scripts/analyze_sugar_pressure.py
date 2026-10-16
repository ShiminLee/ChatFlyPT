"""Audit fixed exploratory records and plot measured responses, without refitting."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from chatflypt.paths import ROOT,dump,sha256
from chatflypt.brain import Trial
from chatflypt.state_readout import StateDecoder,OUT as BASE_OUT
from chatflypt.feedback_readout import FeedbackDecoder
from chatflypt.expression import expression_report
from scripts.try_sugar_pressure import REPORT,RAW,SEEDS


def main():
    plan=json.loads((REPORT/'protocol.json').read_text(encoding='utf8'))
    rows=[];readers={};details=[]
    for key in ['malecns','flywire']:
        source=json.loads((REPORT/f'{key}-results.json').read_text(encoding='utf8'))
        assert source['protocol_sha256']==sha256(REPORT/'protocol.json')
        adapter=BASE_OUT.parent/'feedback-readout'/('flywire-adapter.disabled-after-heat-fp.npz' if key=='flywire' else 'malecns-adapter.npz')
        ds={'malecns':'MaleCNS v1.0','flywire':'FlyWire v783'}[key]
        readers[ds]=FeedbackDecoder(StateDecoder(ds),adapter,BASE_OUT/f'{key}-decoder.npz')
        rows.extend(source['records'])
    expected={(c['id'],i) for c in plan['cases'] for i,_ in enumerate(c['messages'])}
    assert {(r['case'],r['turn']) for r in rows}==expected and len(rows)==28
    for r in rows:
        raw=ROOT/r['raw_record'];assert sha256(raw)==r['raw_sha256']
        body=json.loads((ROOT/r['body_record']).read_text(encoding='utf8'))
        feedback=json.loads((ROOT/r['feedback_record']).read_text(encoding='utf8'))
        assert body['status']=='ok' and body['summary']['finite_state'] and not body['summary']['physics_warnings']
        assert not feedback['llm_called'] and not feedback['goal_controller'] and feedback['sample_count']==5
        assert all(len(s['vision']['ommatidia'])==2 and all(len(eye)==721 for eye in s['vision']['ommatidia']) for s in feedback['samples'])
        with np.load(raw,allow_pickle=False) as a:
            np.testing.assert_array_equal(np.bincount(a['spike_indices'],minlength=len(a['counts'])),a['counts'])
            if r['decoded']['status']!='uncalibrated':
                decoder=readers[r['dataset']]
                assert not np.intersect1d(a['driven_indices'],decoder.indices).size
                trial=Trial(a['counts'],a['spike_indices'],a['spike_steps'],int(a['seed']),float(a['duration_ms']),a['state_record'])
                output=decoder.decode_trial(trial)
                assert output['words']==r['decoded']['words']
                text=expression_report({'decoded':output,'reaction':r['reaction'],'embodied_feedback':feedback})
                assert text['text']==r['expression']['text']
        angles=np.asarray([f['joint_angles_rad'] for f in body['trajectory']])
        mouth={name:float(np.rad2deg(np.ptp(angles[:,i]))) for i,name in enumerate(body['joint_names'])
               if 'pitch' in name and any(p in name for p in ('rostrum','haustellum'))}
        details.append({'case':r['case'],'turn':r['turn'],'source_sha256':r['raw_sha256'],
            'mouth_joint_excursion_deg':mouth,'raw_reread_agrees':r['decoded']['status']!='uncalibrated',
            'readout_unavailable':r['decoded']['status']=='uncalibrated',
            'body_sha256':sha256(ROOT/r['body_record']),'feedback_sha256':sha256(ROOT/r['feedback_record'])})
    matched=[]
    for seed in SEEDS:
        def body(condition):return json.loads((RAW/f'malecns-{seed}-{condition}'/'turn-0/body-result.json').read_text(encoding='utf8'))
        control=body('mouth_block')['trajectory'][0]
        for condition in ['mouth_sugar','mouth_high_salt','sugar_strong_touch']:
            measured=body(condition)['trajectory'][0]
            for field in ['body_positions_mm','body_quaternions_wxyz','joint_angles_rad']:
                np.testing.assert_array_equal(control[field],measured[field])
            matched.append({'seed':seed,'comparison':'mouth_block vs '+condition,'identical_initial_body':True})
        control=body('empty')['trajectory']
        for condition in ['weak_touch','strong_touch']:
            measured=body(condition)['trajectory']
            np.testing.assert_array_equal([f['body_positions_mm'] for f in control],[f['body_positions_mm'] for f in measured])
            matched.append({'seed':seed,'comparison':'empty vs '+condition,'identical_entire_body_trajectory':True})
        withdrawn=json.loads((RAW/f'malecns-{seed}-strong_touch'/'turn-1/result.json').read_text(encoding='utf8'))
        assert withdrawn['feedback']['neural_reset_count']==withdrawn['feedback']['body_reset_count']==0
        assert withdrawn['expression']['status']=='no_signal'
    audit={'windows':28,'cases':26,'seeds':SEEDS,'simulation_ms':14000,'feedback_samples':140,
        'all_physics_finite':True,'physics_warnings':0,'llm_called':False,'retrained':False,
        'comparison_checks':matched,'records':details,
        'note':'Rereading uses the original archived adapter to reproduce failures, not the current fallback. Correct provenance does not establish correct labels.'}
    dump(REPORT/'audit.json',audit)
    plot(rows)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['records','comparison_checks']},ensure_ascii=False))


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    conditions=[('mouth_block','普通方块','#8c9298'),('mouth_sugar','糖','#d29629'),
                ('mouth_high_salt','高盐','#517aa3'),('sugar_strong_touch','糖＋强触碰','#479269')]
    for case,label,color in conditions:
        b=json.loads((RAW/f'malecns-{SEEDS[0]}-{case}'/'turn-0/body-result.json').read_text(encoding='utf8'))
        i=b['joint_names'].index('c_head-c_rostrum-pitch')
        frames=b['trajectory'];angles=np.rad2deg([f['joint_angles_rad'][i]-frames[0]['joint_angles_rad'][i] for f in frames])
        axes[0].plot([f['time_ms'] for f in frames],angles,label=label,color=color,lw=2)
    axes[0].set(xlabel='仿真时间（ms）',ylabel='口器关节相对起点角度（度）',title='实际身体轨迹：高盐也触发了口器姿态\n示例种子 491701；不代表实际摄入')
    axes[0].legend(frameon=False,fontsize=9);axes[0].grid(alpha=.2)
    for n,case in enumerate(['weak_touch','strong_touch']):
        values=[r['reaction']['non_sensory_spikes'] for r in rows if r['condition']==case and r['turn']==0]
        axes[1].bar(n,np.mean(values),width=.5,color=['#8cbdb5','#27695e'][n],alpha=.7)
        axes[1].scatter([n-.06,n+.06],values,c='#152b28',zorder=3)
        for j,v in enumerate(values):axes[1].annotate(str(v),(n+(j-.5)*.12,v),xytext=(0,6),textcoords='offset points',ha='center',fontsize=9)
    axes[1].set(xticks=[0,1],xticklabels=['弱触碰：0.5×','强触碰：2×'],ylabel='500 ms 内部神经脉冲数',
        title='触碰增强有神经响应，未形成动作命令\n每组两个随机种子；倍率为工程输入强度')
    axes[1].set_ylim(0,470);axes[1].grid(axis='y',alpha=.2)
    fig.savefig(REPORT/'measured-reactions.png',dpi=180)
    fig.savefig(REPORT/'measured-reactions.svg')
    plt.close(fig)


if __name__=='__main__':main()
