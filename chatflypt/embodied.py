"""Measured environment -> persistent neural state -> recorded motor drive.

Each dataset has its own body and neural state. There is no goal, language model,
navigation policy, or decoder in this loop. State persists within a bounded trial.
"""
import hashlib
import math
import uuid
import json
import numpy as np

from .paths import ROOT, dump, sha256
from .brain import DT_MS
from .neural_session import NeuralSession
from .sensory_events import encode_inputs
from .social import to_engine_tape
from .environment import sample_environment
from .body_feedback import FeedbackBody
from .behavior_readout import build_behavior_drive


class EmbodiedSession:
    """Live neural/body pair. No simulated time passes while waiting for input."""
    def __init__(self, environment, context, session_id=None):
        self.id=session_id or uuid.uuid4().hex
        self.signature=self.signature_for(environment,context)
        self.neural=None;self.worker=None;self.probe=None;self.closed=False
        self.times=[];self.origins=[];self.inputs=set();self.contact_since={};self.turns=0
        self.last_checkpoint=None
        self.environment_update=None
        self.motor_annotations=None

    @staticmethod
    def signature_for(environment,context):
        return json.dumps({'revision':environment.get('revision',0),'objects':environment['objects'],
                           'context':context},sort_keys=True,separators=(',',':'))

    def close(self):
        if self.worker:self.worker.close()
        self.closed=True

    def update_environment(self,environment,context):
        if self.closed:raise ValueError('已停止的会话不能恢复')
        previous=json.loads(self.signature)
        if previous['context']!=context:raise ValueError('身体场景变化需要新会话')
        target=self.signature_for(environment,context)
        if target==self.signature:return
        if self.worker and previous['objects']!=environment['objects']:
            try:update=self.worker.update_environment(environment)
            except Exception:
                self.close();raise
            self.probe=update.pop('probe')
            self.environment_update=update
        else:
            self.environment_update={'state_preserved':True,'body_reset':False,'controller_reset':False,
                'environment_revision':environment['revision'],'geometry_unchanged':True}
        self.signature=target


def combine_window(brain, human, environmental, start, end, duration, seed, dataset, namespace):
    origins=[];times=[];inputs=set()
    for events,offset in ((human,0),(environmental,start)):
        if events is None:continue
        if events['dataset']!=dataset or events['id_namespace']!=namespace:
            raise ValueError('Feedback cannot mix connectome ID namespaces')
        shifted=events['event_steps']+offset
        use=(shifted>=start)&(shifted<end)
        times.append(shifted[use]);origins.append(events['event_neuron_ids'][use])
        inputs.update(map(int,events['input_neuron_ids']))
    ts=np.concatenate(times) if times else np.empty(0,np.int32)
    ids=np.concatenate(origins) if origins else np.empty(0,np.int64)
    order=np.argsort(ts,kind='stable')
    window={'dataset':dataset,'id_namespace':namespace,'seed':seed,'dt_ms':DT_MS,
        'duration_ms':(end-start)*DT_MS,'input_neuron_ids':np.asarray(sorted(inputs),np.int64),
        'event_steps':(ts[order]-start).astype(np.int32),'event_neuron_ids':ids[order]}
    return to_engine_tape(brain,window),ts[order].astype(np.int32),ids[order],inputs


def merge_measured_tapes(*tapes):
    """Combine independent physical receptors, retaining coincident events."""
    tapes=[t for t in tapes if t is not None]
    if not tapes:return None
    first=tapes[0]
    if any((t['dataset'],t['id_namespace'],t['dt_ms'])!=
           (first['dataset'],first['id_namespace'],first['dt_ms']) for t in tapes):
        raise ValueError('Measured receptors cannot mix connectome namespaces')
    times=np.concatenate([t['event_steps'] for t in tapes])
    origins=np.concatenate([t['event_neuron_ids'] for t in tapes])
    order=np.argsort(times,kind='stable')
    return dict(first,event_steps=times[order],event_neuron_ids=origins[order],
        input_neuron_ids=np.asarray(sorted({int(i) for t in tapes for i in t['input_neuron_ids']}),np.int64))


def receptor_seed(session_seed,step,source):
    """Independent random streams with stable identities across chat boundaries."""
    return int.from_bytes(hashlib.sha256(f'{session_seed}:receptor:{source}:{step}'.encode()).digest()[:4],'little')


def run_embodied(brain, annotations, catalogue, dataset, requested, environment, seed, context,
                 duration_ms=500., *, interval_ms=None, output_dir=None, session=None,
                 input_mode='neural_intervention', presentations=None):
    if input_mode not in ('neural_intervention','environment_only'):
        raise ValueError('Unknown input mode')
    if input_mode=='environment_only' and requested:
        raise ValueError('正常聊天只接受物理环境采样，禁止直接神经刺激')
    if interval_ms is None:interval_ms=10. if input_mode=='environment_only' else 100.
    from .physical_touch import validate_presentations
    presentations=validate_presentations(presentations or [])
    if presentations:
        duration_ms=max(duration_ms,max(p['onset_ms']+p['duration_ms']+100 for p in presentations))
    if interval_ms not in (10.,20.,100.) or not 0<duration_ms<=5000:
        raise ValueError('Bounded feedback requires 10, 20 or 100 ms intervals and at most 5000 ms')
    folder=output_dir or ROOT/'data/derived/embodied-trials'/uuid.uuid4().hex
    folder.mkdir(parents=True,exist_ok=True)
    persistent=session is not None
    session=session or EmbodiedSession(environment,context)
    if session.closed:raise ValueError('具身会话已失效，需要创建新会话。')
    if session.signature!=session.signature_for(environment,context):session.update_environment(environment,context)
    namespace=getattr(brain,'id_namespace','flywire_root_id')
    human=encode_inputs(catalogue,requested,seed,duration_ms=duration_ms).get(dataset) if requested else None
    if human is not None:duration_ms=max(duration_ms,human['duration_ms'])
    if session.neural is None:
        session.neural=NeuralSession(brain,seed)
        session.worker=FeedbackBody(environment,seed,context,folder)
        session.probe=session.worker.initial
    neural=session.neural;worker=session.worker
    if session.motor_annotations is None:
        from .behavior_readout import PreparedMotorAnnotations
        session.motor_annotations=PreparedMotorAnnotations(brain.ids,annotations,dataset)
    if neural.brain is not brain:raise ValueError('Cannot change the connectome of a live session')
    beginning=neural.step;total=beginning+round(duration_ms/DT_MS);stride=round(interval_ms/DT_MS)
    if total*DT_MS>60000:
        raise ValueError('此会话已到 60 秒仿真记录上限；状态未重置，请用“新对话”开始下一段独立实验。')
    if human is not None:human=dict(human,event_steps=human['event_steps']+beginning)
    neural.save(folder/'initial-neural-state.npz');neural.begin_observation();worker.begin_observation()
    physical_presentation=worker.present(presentations) if presentations else None
    if physical_presentation:session.probe=physical_presentation.pop('probe')
    samples=[];times=session.times;origins=session.origins;all_inputs=session.inputs;seen_items={};current_inputs=set()
    contact_since=session.contact_since;first_input_count=len(times)
    try:
        probe=session.probe
        for start in range(beginning,total,stride):
            end=min(total,start+stride);at=start*DT_MS;span=(end-start)*DT_MS
            sample=sample_environment(environment,probe,catalogue,
                dataset=dataset if input_mode=='environment_only' else None, time_ms=at)
            items=[];active_contacts=set();other_datasets=[]
            for item in sample['items']:
                spec=catalogue['channels'][item['channel']]
                if spec['dataset']!=dataset:
                    other_datasets.append({'channel':item['channel'],'side':item['side'],'dataset':spec['dataset'],
                        'reason':'入口属于另一独立连接组，本试次不混入其神经元 ID。'})
                    continue
                key=(item['channel'],item['side'],item.get('body_part'))
                seen_items[key]=dict(item)
                copied=dict(item,onset_ms=0.,duration_ms=span)
                if spec.get('profile')=='touch_transient':
                    active_contacts.add(key);contact_since.setdefault(key,at)
                    decay=spec.get('profile_parameters',{}).get('decay_ms',15.)
                    copied['intensity']*=math.exp(-(at-contact_since[key])/decay)
                items.append(copied)
            contact_since={k:v for k,v in contact_since.items() if k in active_contacts}
            # Separate independent sensory sources; retain coincident events.
            env_seed=int.from_bytes(hashlib.sha256(f'{neural.seed}:environment:{start}'.encode()).digest()[:4],'little')
            spatial_mode=input_mode=='environment_only' and dataset=='MaleCNS v1.0'
            source_seeds={name:receptor_seed(neural.seed,start,name) for name in ('external','vision','proprioception')}
            if spatial_mode:
                env_seed=source_seeds['external']
                sample['receptor_random_seeds']=source_seeds
            encoded_items=[i for i in items if not (spatial_mode and i['channel']=='mc_bright')]
            environmental=encode_inputs(catalogue,encoded_items,env_seed,window_ms=span).get(dataset)
            if spatial_mode:
                from .spatial_vision import project_spatial_vision, encode_spatial_vision
                projection=project_spatial_vision(probe['vision'],catalogue)
                visual_tape=encode_spatial_vision(projection,source_seeds['vision'],span)
                environmental=merge_measured_tapes(environmental,visual_tape)
                sample['visual_projection']={k:v for k,v in projection.items()
                    if k not in ('cell_ids','intensities','eye_indices','ommatidium_indices')}
                sample['vision']=dict(sample['vision'],bridge='candidate_local_ommatidia_to_MaleCNS_columns',
                    spatial_registration=False,local_spatial_input=True)
            if input_mode=='environment_only' and dataset=='MaleCNS v1.0':
                from .proprioception import encode_proprioceptive_feedback
                frame=dict(probe['frame'],body_names=probe['body_names'])
                proprio_tape,proprio_report=encode_proprioceptive_feedback(frame,source_seeds['proprioception'],duration_ms=span)
                environmental=merge_measured_tapes(environmental,proprio_tape)
                sample['proprioception']=proprio_report
            tape,ts,ids,inputs=combine_window(brain,human,environmental,start,end,duration_ms,seed,dataset,namespace)
            trial=neural.run(tape);all_inputs.update(inputs);current_inputs.update(inputs);times.append(ts);origins.append(ids)
            if 'proprioception' in sample:sample['proprioception']['neural_feedback_applied']=True
            sample.update(time_ms=at-beginning*DT_MS,until_ms=(end-beginning)*DT_MS,session_time_ms=at,continuous_feedback=True,
                sampling=f'measured_same_body_every_{interval_ms:g}_ms',items=items,seed=env_seed,
                excluded_other_dataset_inputs=other_datasets)
            sample['limitations']=[x for x in sample['limitations'] if '每轮从相同身体初始化' not in x]
            sample['limitations'].append(f'接触与位置每 {interval_ms:g} ms 更新，官方眼图每 100 ms 更新并采样保持；'+('会话跨消息保留神经与身体，等待消息时仿真时钟暂停。' if persistent else '独立试次之间重置神经与身体。'))
            samples.append(sample)
            drive=build_behavior_drive(trial,brain.ids,session.motor_annotations,dataset=dataset,
                driven_indices=np.flatnonzero(neural.driven),context=context)
            probe=worker.step(drive,refresh_vision=(end%round(100/DT_MS)==0 or end==total))
        body=worker.finish(folder,beginning*DT_MS,persistent)
    except Exception:
        session.close();raise
    if not persistent:session.close()
    session.probe=probe;session.contact_since=contact_since;session.turns+=1
    neural.save(folder/'neural-state.npz')
    cumulative=neural.trial()
    np.savez_compressed(folder/'cumulative-neural.npz',counts=cumulative.counts,spike_ids=cumulative.spike_ids,
                        spike_steps=cumulative.spike_steps,state_record=cumulative.state_record)
    events={'dataset':dataset,'id_namespace':namespace,'seed':neural.seed,'dt_ms':DT_MS,'duration_ms':total*DT_MS,
        'input_neuron_ids':np.asarray(sorted(all_inputs),np.int64),
        'input_activation_steps':np.asarray([neural.input_activation_step[brain.lookup[str(i)]] for i in sorted(all_inputs)],np.int64),
        'event_steps':np.concatenate(times),'event_neuron_ids':np.concatenate(origins)}
    np.savez_compressed(folder/'input-events.npz',**events)
    feedback={'schema':'chatflypt.embodied-feedback.v2','within_trial':True,'between_messages':persistent,
        'session_id':session.id,'turn_index':session.turns,'session_start_ms':beginning*DT_MS,
        'stimulus_seed':seed,'initialization_seed':neural.seed,
        'session_end_ms':total*DT_MS,'paused_between_messages':True,'session_limit_ms':60000,
        'interval_ms':interval_ms,'vision_interval_ms':100.,'sample_count':len(samples),'neural_reset_count':int(beginning==0),'body_reset_count':int(beginning==0),
        'state_continuity':['voltage','conductance','refractory','delayed_synapses','body','controller'],
        'samples':samples,'final_probe':probe,'output_dir':str(folder.relative_to(ROOT)) if folder.is_relative_to(ROOT) else str(folder),
        'human_cue_repeated':False,'llm_called':False,'goal_controller':False,
        'input_mode':input_mode,'direct_neural_intervention':bool(requested),
        'physical_presentation':physical_presentation,
        'environment_update':session.environment_update,
        'input_activation_schedule_recorded':True,
        'initial_neural_state_sha256':sha256(folder/'initial-neural-state.npz'),
        'previous_checkpoint':session.last_checkpoint,
        'cumulative_replay':{'events':'input-events.npz','neural_record':'cumulative-neural.npz',
            'scope':'entire_session_from_initial_state'},
        'neural_state_sha256':sha256(folder/'neural-state.npz'),'physical_state_sha256':sha256(folder/'physical-state.npz'),
        'input_events_sha256':sha256(folder/'input-events.npz')}
    dump(folder/'feedback.json',feedback)
    session.last_checkpoint={'output_dir':feedback['output_dir'],'neural_state_sha256':feedback['neural_state_sha256'],
                             'physical_state_sha256':feedback['physical_state_sha256']}
    session.environment_update=None
    # Per-message UI and readout see only this interval. Full input prefix is
    # archived separately because this interval does not start at resting state.
    window_events=dict(events,duration_ms=duration_ms,
        input_neuron_ids=np.asarray(sorted(current_inputs),np.int64),
        event_steps=np.concatenate(times[first_input_count:])-beginning,
        event_neuron_ids=np.concatenate(origins[first_input_count:]),
        input_activation_steps=np.asarray([max(0,neural.input_activation_step[brain.lookup[str(i)]]-beginning) for i in sorted(current_inputs)],np.int64))
    return {'trial':neural.observation_trial(),'driven':np.flatnonzero(neural.driven),'body':body,'feedback':feedback,
            'events':window_events,'environment_items':list(seen_items.values())}
