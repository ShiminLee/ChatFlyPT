"""Local-only ChatFlyPT interface. No label input is passed to the decoder/LLM."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import uuid
from urllib.parse import urlparse, parse_qs
import numpy as np
from .brain import Brain, recorded_trial
from .experiment import Decoder
from .stimuli import WORDS, groups_for
from .language import LLM, LLMUnavailable, local_config, verbalize, list_models
from .api_settings import public_settings, draft_config, save_settings
from .paths import ROOT, ARTIFACTS, REPORTS, dump, sha256
from .social import OUT as SOCIAL_OUT, build_catalogue, to_engine_tape
from .visualization import NeuralView, MaleNeuralView
from .communication import response_text
from .reactions import summarize_reaction
from .sensory_readout import (SensoryDecoder, UncalibratedDecoder, sensory_response_text,
                             OUT as SENSORY_OUT, dataset_key, fingerprint)


def local_request_allowed(host, origin=None, fetch_site=None):
    parsed=urlparse('http://'+(host or ''))
    if parsed.hostname not in ['127.0.0.1','localhost','::1'] or parsed.username or parsed.password:
        return False
    if fetch_site=='cross-site':return False
    if origin:
        source=urlparse(origin)
        if source.scheme!='http' or source.netloc!=host:return False
    return True


def validate_body_context(context):
    if context not in ('ground','tethered_flight'):
        raise ValueError('Body context must be ground or tethered_flight')
    return context


class Application:
    def __init__(self,body_enabled=True):
        self.body_enabled=body_enabled
        self.brain=Brain();self.decoder=Decoder()
        from .malecns import MaleCNS
        self.male=MaleCNS();self.male_view=MaleNeuralView(self.male)
        self.manifest=json.loads((ARTIFACTS/'stimuli.json').read_text(encoding='utf8'))
        self.lock=threading.Lock();self.observations={}
        self.view=NeuralView(self.brain)
        self.resting_scene=self.male_view.resting()
        catalogue_path=SOCIAL_OUT/'catalogue.json'
        self.social=json.loads(catalogue_path.read_text(encoding='utf8')) if catalogue_path.exists() else build_catalogue()
        self.sensory_catalogue=None;self.sensory_decoders={};self.sensory_decoder_signatures={}
        self.catalogue_lock=threading.Lock()
        from .environment import empty_environment, validate_objects
        self.environment_lock=threading.RLock()
        env_path=ROOT/'data/local/environment.json'
        self.environment=json.loads(env_path.read_text(encoding='utf8')) if env_path.exists() else empty_environment()
        self.environment['objects']=validate_objects(self.environment['objects'])
        self.session_lock=threading.RLock();self.embodied_sessions={};self.session_results={}

    def _session_map(self, session_id):
        import re
        if not isinstance(session_id,str) or not re.fullmatch('[0-9a-f]{32}',session_id):
            raise ValueError('会话 ID 无效')
        if not hasattr(self,'embodied_sessions'):self.embodied_sessions={}
        if session_id not in self.embodied_sessions:
            if len(self.embodied_sessions)>=4:
                raise ValueError('已有 4 个具身会话；请先在不用的页面点击“新对话”释放状态。')
            self.embodied_sessions[session_id]={}
        return self.embodied_sessions[session_id]

    def reset_session(self,session_id):
        with self.session_lock:
            states=self._session_map(session_id)
            for state in states.values():state.close()
            self.embodied_sessions.pop(session_id,None)
            getattr(self,'session_results',{}).pop(session_id,None)
            return {'reset':True,'session_id':session_id,'neural_and_body_reset':True}

    def session_status(self,session_id):
        with self.session_lock:
            states=self._session_map(session_id)
            path=getattr(self,'session_results',{}).get(session_id)
            active=bool(states) and all(not s.closed for s in states.values())
            return {'session_id':session_id,'active':active,'paused_between_messages':True,
                    'response':json.loads((ROOT/path).read_text(encoding='utf8')) if path and active else None}

    def session_probe(self,session_id,environment,context,dataset=None):
        if not session_id:return None
        from .embodied import EmbodiedSession
        states=self._session_map(session_id)
        candidates=[states.get(dataset)] if dataset else list(states.values())
        for state in candidates:
            if state and not state.closed and json.loads(state.signature)['context']==context:
                state.update_environment(environment,context)
        return next((s.probe for s in candidates if s and not s.closed and
            s.signature==EmbodiedSession.signature_for(environment,context)),None)

    def embodied_session(self,session_id,dataset,environment,context):
        from .embodied import EmbodiedSession
        states=self._session_map(session_id);state=states.get(dataset);reason=None
        if state and state.closed:raise ValueError('上次仿真失败后会话已停止；请用“新对话”重新初始化。')
        if state and state.signature!=EmbodiedSession.signature_for(environment,context):
            if json.loads(state.signature)['context']!=context:
                state.close();state=None;reason='身体场景发生改变，已明确重新初始化该连接组。'
            else:state.update_environment(environment,context)
        if state is None:states[dataset]=state=EmbodiedSession(environment,context)
        return state,reason

    def environment_state(self):
        from .environment import empty_environment, TYPES, LIMITATIONS
        import copy
        return dict(copy.deepcopy(getattr(self,'environment',empty_environment())), types=TYPES, limitations=LIMITATIONS)

    def apply_environment(self, objects, revision, seed=42, body_context='ground',session_id=None):
        from .environment import validate_objects, probe_environment, sample_environment
        from .environment_inspection import environment_changes
        from .conversation import validate_options
        validate_options('off',False,seed);validate_body_context(body_context)
        objects=validate_objects(objects)
        with getattr(self,'session_lock',threading.RLock()), self.environment_lock:
            if isinstance(revision,bool) or not isinstance(revision,int) or revision!=self.environment['revision']:
                raise ValueError('环境版本已变，请刷新环境后再应用')
            candidate={'schema':'chatflypt.environment.v1','revision':revision+1,'objects':objects}
            probe=self.session_probe(session_id,candidate,body_context,'MaleCNS v1.0') or probe_environment(candidate,seed,body_context)
            sampled=sample_environment(candidate,probe,self.input_catalogue())
            result=self.converse(kind='environment',items=[],seed=seed,body_context=body_context,
                                 environment=candidate,environment_sample=sampled,session_id=session_id)
            for observation in result['observations']:
                body=observation['scene'].get('body',{})
                if self.body_enabled and (not body.get('physics_run') or body.get('status') in ('error','timeout','unavailable')):
                    raise ValueError('身体未能在新环境运行，环境未应用：'+body.get('reason',body.get('error','unknown')))
            event={'label':'环境改动','revision':candidate['revision'],'object_count':len(objects),
                   'changes':environment_changes(self.environment,candidate)}
            record_id=uuid.uuid4().hex
            candidate['last_record']={'kind':'neural','id':record_id}
            result.update(environment=candidate,environment_event=event,llm_called=False)
            dump(ROOT/'reports/environment-responses'/f'{record_id}.json',result)
            self._publish_environment(candidate)
            return result

    def prepare_scene_message(self,message,seed,body_context,session_id=None):
        from .environment_message import parse_message, ground_message
        from .environment import probe_environment, sample_environment
        current=self.environment_state()
        environment={key:current[key] for key in ('schema','revision','objects')}
        catalogue=self.input_catalogue()
        raw=parse_message(message,catalogue)
        if raw['kind'] in ('question','unsupported'):
            return {'plan':raw,'environment':environment,'sample':None,'changed':False}
        if not self.body_enabled:raise ValueError('正常聊天需要物理环境；身体后端关闭时不能改用直接神经刺激。')
        dataset='MaleCNS v1.0'
        probe=self.session_probe(session_id,environment,body_context,dataset) or probe_environment(environment,seed,body_context)
        plan,candidate=ground_message(raw,environment,probe,catalogue)
        changed=candidate['revision']!=environment['revision']
        sample=None if changed else sample_environment(environment,probe,catalogue)
        return {'plan':plan,'environment':candidate,'sample':sample,'changed':changed,'previous':environment}

    def commit_message_environment(self,prepared,result):
        if not prepared.get('changed'):return
        from .environment_inspection import environment_changes
        candidate=prepared['environment']
        with self.environment_lock:
            if self.environment['revision']!=prepared['previous']['revision']:
                raise ValueError('环境在消息处理期间发生变化；本次物品未应用。')
            for obs in result['observations']:
                if not obs['scene']['body'].get('physics_run'):
                    raise ValueError('物理呈现失败，物品未应用。')
            event={'label':'环境改动','revision':candidate['revision'],'object_count':len(candidate['objects']),
                   'changes':environment_changes(prepared['previous'],candidate),'source':'human_message'}
            record_id=uuid.uuid4().hex
            candidate['last_record']={'kind':'neural','id':record_id}
            result.update(environment=candidate,environment_event=event)
            dump(ROOT/'reports/environment-responses'/f'{record_id}.json',result)
            self._publish_environment(candidate)

    def inspect_applied_environment(self, seed=42, body_context='ground', duration_ms=500):
        from .environment_inspection import inspect_environment
        import copy
        with self.environment_lock:
            candidate=copy.deepcopy(self.environment)
            inspection=inspect_environment(candidate,seed,body_context,duration_ms)
            event={'label':'物理观察','revision':candidate['revision'],'object_count':len(candidate['objects'])}
            return self._save_environment_inspection(candidate,inspection,event)

    def _save_environment_inspection(self, candidate, inspection, event):
        inspection['environment_event']=event
        candidate['inspection_id']=inspection['id']
        candidate['last_record']={'kind':'physics','id':inspection['id']}
        dump(ROOT/'reports/environment-inspection'/f"{inspection['id']}.json",inspection)
        self._publish_environment(candidate)
        return {'environment':candidate,'inspection':inspection,'environment_event':event,
                'neural_simulation_run':False,'llm_called':False}

    def _publish_environment(self,candidate):
        # Publish only after the record and atomic environment save succeed.
        target=ROOT/'data/local/environment.json';target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_suffix('.tmp');dump(temp,candidate);temp.replace(target)
        self.environment=candidate
        # Other paused sessions acquire the new geometry on their next request,
        # preserving their own body/neural state instead of closing the session.

    def environment_inspection(self, record_id):
        return self._environment_record('environment-inspection',record_id)

    def environment_response(self,record_id):
        return self._environment_record('environment-responses',record_id)

    def _environment_record(self,folder,record_id):
        import re
        if not isinstance(record_id,str) or not re.fullmatch('[0-9a-f]{32}',record_id):
            raise ValueError('观察记录 ID 无效')
        path=ROOT/'reports'/folder/f'{record_id}.json'
        if not path.is_file():raise ValueError('观察记录不存在')
        return json.loads(path.read_text(encoding='utf8'))

    def _result(self,trial,driven,seed,start,brain=None,decoder=None,view=None,renderer=None,*,body_context='ground',environment=None,body_result=None,current_driven=None):
        brain=brain or self.brain;decoder=decoder or self.decoder;view=view or self.view
        decoded=decoder.decode_trial(trial) if hasattr(decoder,'decode_trial') else decoder.decode(trial.counts)
        order=np.argsort(trial.counts[decoder.indices])[::-1][:12]
        top=decoder.indices[order]
        scene=view.activity(trial,np.asarray(driven if current_driven is None else current_driven,dtype=np.int32))
        reaction=summarize_reaction(brain,trial,driven,view.rows)
        observation_id=uuid.uuid4().hex
        if self.body_enabled:
            from .behavior_readout import build_behavior_drive
            from .behavior_body import run_behavior_body
            try:
                annotations=brain.rows if getattr(brain,'dataset','').startswith('MaleCNS') else view.rows
                drive=build_behavior_drive(trial,brain.ids,annotations,
                    dataset=getattr(brain,'dataset','FlyWire v783'),driven_indices=driven,
                    context=validate_body_context(body_context))
                scene['body']=body_result if body_result is not None else run_behavior_body(drive,output_dir=ROOT/'data/derived/body-trials'/observation_id,environment=environment)
            except Exception as exc:
                scene['body']={'status':'error','physics_run':False,'trajectory':[],
                               'error':str(exc),'biologically_calibrated':False}
            body=scene['body']
            reaction['body_simulation']={k:body[k] for k in [
                'status','physics_run','biologically_calibrated','duration_ms','truncated','summary',
                'controller','context','command_summary','joint_command_summary_rad','motor_coverage',
                'motor_activity_in_window','controls','initialization'] if k in body}
        # Motor-side (intent) readout: separate slot, same blind inference boundary.
        # It predicts the body's own measured motion; it never uses the message.
        if self.body_enabled and getattr(brain,'dataset','')=='MaleCNS v1.0':
            from .intent_readout import intent_report, load_intent_decoder, measured_outcome
            intent_decoder,intent_note=self._intent_decoder(brain)
            if intent_decoder is not None and trial.state_record is not None:
                decoded_intent=intent_decoder.decode(trial.counts,trial.state_record,trial.duration_ms)
                outcome=None
                trajectory=(scene.get('body') or {}).get('trajectory')
                if trajectory:
                    try:outcome=measured_outcome(trajectory)
                    except (KeyError,TypeError,ValueError):outcome=None
                scene['intent']=intent_report(decoded_intent,outcome)
            else:
                scene['intent']={'schema':'chatflypt.intent-expression.v1','status':'unavailable',
                    'text':None,'labels':[],'measured_outcome':None,'llm_called':False,
                    'reason':(intent_note or {}).get('reason','本轮没有可用的运动侧读出。')}
            # Extra blind slots (delivery pattern, internal activity level): they let
            # the sensation line be composed rather than templated.
            from .slot_readout import load_slot_decoders
            slot_decoder,slot_note=self._slot_decoder(brain)
            if slot_decoder is not None and trial.state_record is not None:
                scene['slots']=slot_decoder.decode(trial.counts,trial.state_record,trial.duration_ms)
            else:
                scene['slots']={'words':[],'status':'unavailable',
                    'reason':(slot_note or {}).get('reason','本轮没有可用的槽位读出。')}
            # Steering-command readout: what it issued, never what the body did
            # with it. Kept separate from the measured-motion line above.
            from .command_readout import command_report, load_command_decoder
            command_decoder,command_note=self._command_decoder(brain)
            if command_decoder is not None and trial.state_record is not None:
                scene['command']=command_report(
                    command_decoder.decode(trial.counts,trial.state_record,trial.duration_ms),
                    (scene.get('intent') or {}).get('measured_outcome'))
            else:
                scene['command']={'schema':'chatflypt.command-expression.v1','status':'unavailable',
                    'text':None,'labels':[],'llm_called':False,
                    'reason':(command_note or {}).get('reason','本轮没有可用的命令读出。')}
        reply=(renderer or response_text)(decoded,reaction)
        result={'id':observation_id,'decoded':decoded,
                'sentence':verbalize(decoded['words'],ask=False) if decoder is self.decoder else reply,
                'response_text':reply,'reaction':reaction,'dataset':getattr(brain,'dataset','FlyWire v783'),
                'spikes':int(trial.counts.sum()),'active_neurons':int((trial.counts>0).sum()),
                'duration_ms':trial.duration_ms,'latency_seconds':time.perf_counter()-start,'seed':seed,
                'top_readout_cells':[{'neuron_id':str(brain.ids[i]),'spikes':int(trial.counts[i]),
                                     'rate_hz':float(trial.counts[i]/(trial.duration_ms/1000))} for i in top if trial.counts[i]],
                'limitations':['Independent trial, reset between observations.',
                               'Sensory proxy labels, not subjective experience.',
                               'Body trajectory, when present, uses an uncalibrated engineering controller.'],
                'scene':scene}
        raw=ROOT/'data/derived/pathway-trials'/f"{result['id']}.npz"
        raw.parent.mkdir(parents=True,exist_ok=True)
        state_fields={'state_record':trial.state_record} if trial.state_record is not None else {}
        np.savez_compressed(raw,counts=trial.counts,neuron_ids=brain.ids,**state_fields,
            spike_indices=trial.spike_ids,spike_steps=trial.spike_steps,
            driven_indices=np.asarray(driven,dtype=np.int32),seed=seed,duration_ms=trial.duration_ms)
        result['raw_record']={'file':str(raw.relative_to(ROOT)),'sha256':sha256(raw)}
        self.observations[result['id']]=result
        if len(self.observations)>100:self.observations.pop(next(iter(self.observations)))
        return result

    def input_catalogue(self):
        if self.sensory_catalogue is None:
            with self.catalogue_lock:
                if self.sensory_catalogue is None:
                    from .sensory_catalogue import build_catalogue
                    self.sensory_catalogue=build_catalogue()
        return self.sensory_catalogue

    def inputs(self):
        catalogue=self.input_catalogue()
        def report(name):
            path=ROOT/'reports/general-sensory'/name
            try:return json.loads(path.read_text(encoding='utf8'))
            except (FileNotFoundError,json.JSONDecodeError):return {}
        delivery={row['channel']:row for row in report('channel-delivery.json').get('cases',[])}
        scores={key:report(f'{key}-readout.json') for key in ['flywire','malecns']}
        gap=ROOT/'reports/gap-closure'
        try:
            electrical={r['channel']:r for r in json.loads((gap/'electrical-audit.json').read_text(encoding='utf8'))['cases']}
        except (FileNotFoundError,json.JSONDecodeError):electrical={}
        for key in scores:
            from .state_readout import signature, DATASETS, OUT as STATE_OUT
            try:
                state_report=json.loads((gap/f'{key}-state-readout.json').read_text(encoding='utf8'))
                model=STATE_OUT/f'{key}-decoder.npz'
                if (state_report.get('catalogue_signature')==signature(catalogue,DATASETS[key]) and
                        model.exists() and state_report.get('model_sha256')==sha256(model)):
                    scores[key]=state_report
                else:scores[key]={}
            except (FileNotFoundError,json.JSONDecodeError):
                if scores[key].get('catalogue_fingerprint')!=fingerprint(catalogue):scores[key]={}
        channels=[]
        for spec in catalogue['channels'].values():
            channels.append({k:v for k,v in spec.items() if k in [
                'id','label','modality','dataset','status','advanced','limitations','source','readout_group','profile','rate_hz']})
            channels[-1].update(neuron_count=len(spec['neuron_ids']),
                sides=[s for s in ['left','right','bilateral'] if spec['side_ids'].get(s)])
            measured=electrical.get(spec['id'])
            checked=measured or delivery.get(spec['id'])
            readout=scores[dataset_key(spec['dataset'])]
            group=spec.get('readout_group',spec['id'])
            f1=readout.get('test',{}).get('per_label_f1',{})
            qualified={side:readout.get('qualified_labels',{}).get(group+':'+side)
                       for side in ['left','right'] if group+':'+side in f1}
            channels[-1]['validation']={
                'delivery_tested':checked is not None,
                'input_delivery_passed':checked.get('input_delivery_passed') if checked else None,
                'non_sensory_spikes':checked.get('non_sensory_spikes') if checked else None,
                'electrical_response':measured.get('electrical_response') if measured else None,
                'membrane_internal_cells':measured.get('membrane_internal_cells') if measured else None,
                'readout_status':'uncalibrated' if spec.get('advanced') or spec['status']=='unmapped'
                    else 'calibrated' if any(qualified.values()) else 'unqualified' if qualified else 'pending',
                'qualified_sides':qualified,
                'readout_test_f1':{side:f1[group+':'+side] for side in ['left','right'] if group+':'+side in f1}}
        coverage={dataset:{k:v for k,v in row.items() if not k.endswith('_ids')}
                  for dataset,row in catalogue.get('coverage',{}).items()}
        return {'channels':channels,'coverage':coverage,
                'limitations':['通道注释覆盖不等于全部生理输入已破译。',
                               '左右指感觉入口；真实空间坐标尚无感官转导模型。']}

    def _sensory_model(self,dataset,calibrated):
        if dataset=='MaleCNS v1.0':
            if self.male is None:
                from .malecns import MaleCNS
                self.male=MaleCNS();self.male_view=MaleNeuralView(self.male)
            brain,view=self.male,self.male_view
        else:
            brain,view=self.brain,self.view
        if calibrated:
            from .state_readout import StateDecoder, OUT as STATE_OUT, signature
            path=STATE_OUT/f'{dataset_key(dataset)}-decoder.npz'
            if path.exists():
                loaded=StateDecoder(dataset)
                if loaded.catalogue_signature==signature(self.input_catalogue(),dataset):
                    self.sensory_decoders[dataset]=loaded
                    self.sensory_decoder_signatures.pop(dataset,None)
                    return brain,view,loaded
        if calibrated:
            path=SENSORY_OUT/f'{dataset_key(dataset)}-decoder.npz'
            if path.exists():
                signature=(path.stat().st_mtime_ns,path.stat().st_size)
                if self.sensory_decoder_signatures.get(dataset)!=signature:
                    loaded=SensoryDecoder(dataset)
                    if loaded.catalogue_fingerprint==fingerprint(self.input_catalogue()):
                        self.sensory_decoders[dataset]=loaded
                        self.sensory_decoder_signatures[dataset]=signature
                    else:
                        return brain,view,UncalibratedDecoder()
                decoder=self.sensory_decoders[dataset]
            else:decoder=UncalibratedDecoder()
        else:decoder=UncalibratedDecoder()
        return brain,view,decoder

    def _environment_decoder(self,brain,catalogue):
        # Interventions and persistent physical feedback are different input
        # distributions. A missing/stale physical model must abstain, rather
        # than silently reuse the historical intervention decoder.
        from .unified_readout import load_unified_decoder
        decoder,note=load_unified_decoder(brain,catalogue)
        return (decoder if decoder is not None else UncalibratedDecoder()),note

    def _intent_decoder(self,brain):
        # The motor-side readout is a separate artifact with the same blind
        # boundary; missing or stale weights abstain rather than fall back.
        from .intent_readout import load_intent_decoder
        if not hasattr(self,'intent_decoder_cache'):
            self.intent_decoder_cache={}
        key=(getattr(brain,'dataset',''),id(brain))
        if key not in self.intent_decoder_cache:
            self.intent_decoder_cache[key]=load_intent_decoder(brain,self.input_catalogue())
        return self.intent_decoder_cache[key]

    def _slot_decoder(self,brain):
        # Extra blind slots (delivery pattern, activity level) share the same
        # boundary and the same fail-closed rule.
        from .slot_readout import load_slot_decoders
        if not hasattr(self,'slot_decoder_cache'):
            self.slot_decoder_cache={}
        key=(getattr(brain,'dataset',''),id(brain))
        if key not in self.slot_decoder_cache:
            self.slot_decoder_cache[key]=load_slot_decoders(brain,self.input_catalogue())
        return self.slot_decoder_cache[key]

    def _command_decoder(self,brain):
        # The steering-command readout is its own artifact: it predicts the
        # command the fly issued, not the motion the body produced, and abstains
        # when the weights are missing or stale.
        from .command_readout import load_command_decoder
        if not hasattr(self,'command_decoder_cache'):
            self.command_decoder_cache={}
        key=(getattr(brain,'dataset',''),id(brain))
        if key not in self.command_decoder_cache:
            self.command_decoder_cache[key]=load_command_decoder(brain,self.input_catalogue())
        return self.command_decoder_cache[key]

    def stimulate(self,items,seed,*,input_plan=None,intent=None,body_context='ground',environment=None,environment_sample=None,session_id=None,
                  input_mode='neural_intervention',presentations=None):
        """Compose receptor events; each ID namespace stays in its own network."""
        validate_body_context(body_context)
        if not isinstance(seed,int) or isinstance(seed,bool) or not 0<=seed<2**32:
            raise ValueError('Seed must be a nonnegative 32-bit integer')
        if not isinstance(items,list) or len(items)>16:
            raise ValueError('Choose at most 16 sensory items')
        from .sensory_events import encode_inputs
        catalogue=self.input_catalogue()
        requested_items=list(items)
        if input_mode not in ('neural_intervention','environment_only'):raise ValueError('Unknown input mode')
        environmental=input_mode=='environment_only'
        if environmental and requested_items:raise ValueError('正常聊天禁止直接指定神经输入')
        if environmental and not self.body_enabled:raise ValueError('物理后端不可用；未回退为直接神经刺激')
        from .environment import probe_environment, sample_environment
        if environment is None:
            environment=self.environment_state()
            environment={k:environment[k] for k in ('schema','revision','objects')}
        use_body=self.body_enabled and bool(environmental or session_id or environment.get('objects') or environment.get('revision',0)>0)
        if (environmental or session_id or environment.get('objects') or environment.get('revision',0)>0) and environment_sample is None:
            probe=self.session_probe(session_id,environment,body_context) or probe_environment(environment,seed,body_context)
            environment_sample=sample_environment(environment,probe,catalogue)
        if environment_sample:
            items=items+environment_sample['items']
        if environmental:
            from .malecns_senses import remap_environment_inputs
            items,excluded=remap_environment_inputs(catalogue,items,'MaleCNS v1.0')
            if environment_sample:
                environment_sample=dict(environment_sample,items=items,
                    unmapped=environment_sample.get('unmapped',[])+excluded,
                    dataset='MaleCNS v1.0',single_individual=True)
        encoded=encode_inputs(catalogue,items,seed)
        if not items:
            encoded={'FlyWire v783':None}
        if environmental:
            # A contact can arise after the first physics block. Do not omit
            # its connectome merely because it had no input in the initial pose.
            encoded={'MaleCNS v1.0':encoded.get('MaleCNS v1.0')}
        if session_id and not environmental:
            for dataset in self._session_map(session_id):encoded.setdefault(dataset,None)
        observations=[];receptions=[]
        with self.lock:
            start=time.perf_counter()
            for dataset,events in encoded.items():
                selected=[item for item in items if catalogue['channels'][item['channel']]['dataset']==dataset]
                def trained_population(item):
                    if item.get('intensity',1)==0:return True
                    spec=catalogue['channels'][item['channel']]
                    if spec.get('advanced',False):return False
                    side=item.get('side','bilateral')
                    part=spec.get('body_part_ids',{}).get(item.get('body_part'),spec['side_ids'])
                    return set(part.get(side,[]))==set(spec['side_ids'].get(side,[]))
                calibrated=all(trained_population(item) for item in selected)
                brain,view,decoder=self._sensory_model(dataset,calibrated and not environmental)
                embodied=None
                feedback_adapter_note=None
                session_reset_reason=None
                if use_body:
                    from .embodied import run_embodied
                    human_items=[item for item in requested_items if catalogue['channels'][item['channel']]['dataset']==dataset]
                    annotations=brain.rows if dataset=='MaleCNS v1.0' else view.rows
                    live,session_reset_reason=self.embodied_session(session_id,dataset,environment,body_context) if session_id else (None,None)
                    embodied=run_embodied(brain,annotations,catalogue,dataset,human_items,environment,seed,body_context,
                        duration_ms=events['duration_ms'] if events is not None else 500.,session=live,
                        input_mode=input_mode,presentations=presentations)
                    embodied['feedback']['reset_reason']=session_reset_reason
                    trial,driven,events=embodied['trial'],embodied['driven'],embodied['events']
                    selected=human_items+embodied['environment_items']
                    calibrated=all(trained_population(item) for item in selected)
                    brain,view,decoder=self._sensory_model(dataset,calibrated and not environmental)
                    if environmental and dataset=='MaleCNS v1.0':
                        decoder,feedback_adapter_note=self._environment_decoder(brain,catalogue)
                    else:
                        from .feedback_readout import load_feedback_decoder
                        decoder,feedback_adapter_note=load_feedback_decoder(decoder,dataset)
                elif events is None:
                    trial=recorded_trial(brain,brain.input_tape([],500,seed),seed,500);driven=[]
                else:
                    tape=to_engine_tape(brain,events)
                    trial=recorded_trial(brain,tape,seed,events['duration_ms']);driven=tape[0]
                current_driven=[brain.lookup[str(i)] for i in events['input_neuron_ids']] if embodied else driven
                observation=self._result(trial,driven,seed,start,brain,decoder,view,sensory_response_text,
                    body_context=body_context,environment=environment,body_result=embodied['body'] if embodied else None,
                    current_driven=current_driven)
                if embodied:
                    observation['embodied_feedback']=embodied['feedback']
                    if feedback_adapter_note:observation['decoded']['feedback_adaptation']=feedback_adapter_note
                    adapted=observation['decoded'].get('feedback_adaptation',{}).get('applied',False)
                    observation['decoded']['validation_scope']=('Limited session-separated feedback benchmark; candidate labels, not global calibration.' if adapted else 'Independent synthetic interventions; not requalified for embodied feedback.')
                    observation['limitations']=[x for x in observation['limitations'] if x!='Independent trial, reset between observations.']
                    continuity='同一会话跨消息保留神经和身体状态；等待消息时仿真时钟暂停。' if session_id else '独立试次，消息之间重置神经与身体。'
                    observation['limitations']+=[continuity,
                        'Continuous-feedback readout has only limited benchmark validation; still candidate.' if adapted else
                        'Sensory label scores have not been requalified on the embodied-feedback distribution.']
                    if session_reset_reason:observation['limitations'].append(session_reset_reason)
                observation['readout_model']='general_sensory_unified' if getattr(decoder,'unified',False) else 'general_sensory_state_feedback' if hasattr(decoder,'base') else 'general_sensory_state' if hasattr(decoder,'decode_trial') else 'general_sensory' if isinstance(decoder,SensoryDecoder) else 'uncalibrated'
                details=[];limitations=[]
                for item in selected:
                    spec=catalogue['channels'][item['channel']]
                    notes=list(spec.get('limitations',[]))
                    side_map=spec.get('body_part_ids',{}).get(item.get('body_part'),spec['side_ids'])
                    details.append({'channel':item['channel'],'label':spec['label'],'side':item.get('side','bilateral'),
                        'body_part':item.get('body_part'),
                        'neuron_count':len(side_map.get(item.get('side','bilateral'),[])) if item.get('intensity',1)>0 else 0,
                        'status':spec['status'],'notes':notes,'intensity':item.get('intensity',1),
                        'onset_ms':item.get('onset_ms',50),'duration_ms':item.get('duration_ms',300)})
                    limitations.extend(notes)
                translated=((input_plan or {}).get('translation') or {}).get('description')
                if requested_items and not any(catalogue['channels'][item['channel']]['dataset']==dataset for item in requested_items):translated=None
                reception={'status':'simulated','intent':intent or '组合感觉输入','channel':'general_sensory',
                    'input_mode':input_mode,'direct_neural_intervention':bool(requested_items),
                    'description':translated or '；'.join(('左侧' if d['side']=='left' else '右侧' if d['side']=='right' else '双侧')+d['label'] for d in details) or ('采样当前环境与实际身体状态' if environmental else '停止刺激／空白试次'),
                    'items':details,'neuron_count':len(current_driven),
                    'neuron_ids':[str(brain.ids[i]) for i in current_driven],
                    'lifetime_direct_input_cells_excluded_from_output':len(driven),
                    'neuron_types':[catalogue['channels'][item['channel']]['label'] for item in selected],
                    'dataset':dataset,'side':None,'direction_mapping_proposed':bool((input_plan or {}).get('proposed',False)),
                    'detail':('环境实测 → 感觉转导 → 连接组传播 → 盲读出；无聊天直连神经刺激。' if environmental else '实验性直接神经干预，不代表环境自然感知。'),
                    'source':None,'limitations':list(dict.fromkeys(limitations+(input_plan or {}).get('assumptions',[]))),
                    'input_events':len(events['event_steps']) if events else 0,
                    'input_histogram':np.bincount((events['event_steps']/100).astype(int),minlength=int(np.ceil(trial.duration_ms/10))).tolist() if events else [],
                    'duration_ms':trial.duration_ms}
                observation['reception']=reception
                observed_sample=embodied['feedback']['samples'][0] if embodied else environment_sample
                if observed_sample:
                    observation['environment_sample']=observed_sample
                    reception['environment_revision']=environment['revision']
                    reception['limitations']+=observed_sample['limitations']
                if embodied and environmental:
                    body_senses=[s['proprioception'] for s in embodied['feedback']['samples'] if 'proprioception' in s]
                    reception['proprioception']={'source':'measured_body_state',
                        'events':sum(p['encoded_event_count'] for p in body_senses),
                        'applied':bool(body_senses) and all(p['neural_feedback_applied'] for p in body_senses),
                        'scope':'FeCO claw/club population model; incomplete proprioception',
                        'biologically_calibrated':False,'force_to_neural_calibrated':False}
                    reception['visual_projection']=observed_sample.get('visual_projection')
                if embodied and presentations:
                    def probe_key(entry):
                        return (entry.get('object_id'),entry.get('side'),entry.get('body_part'))
                    contact_samples=[]
                    for sample in embodied['feedback']['samples']:
                        contact_samples.append({probe_key(e) for e in sample['evidence']
                            if e.get('mode')=='mujoco_leg_contact' and e.get('object_id','').startswith('message_probe_')})
                    hits=[key for sample in contact_samples for key in sample]
                    bouts=0;previous=set()
                    for sample in contact_samples:
                        # A new contact bout starts after at least one sampled
                        # window without that probe leg; this is a measured tap
                        # count, not the number of 10 ms contact windows.
                        bouts+=len(sample-previous);previous=sample
                    requested_taps=sum(int(p.get('taps',1)) for p in presentations)
                    reception['physical_presentation']={'requested':presentations,
                        'requested_taps':requested_taps,'measured_leg_contacts':len(hits),
                        'measured_contact_bouts':bouts,'contact_recorded':bool(hits),
                        'force_calibrated':False,'tracks_body':True}
                    reception['description']+='；'+(f'实测到 {bouts} 次探针接触（请求 {requested_taps} 次）'
                        if hits else '本轮未采样到探针与腿接触，未生成该触觉输入')
                observation['environment']=environment
                if events is not None:
                    path=ROOT/'data/derived/pathway-trials'/f"{observation['id']}-inputs.npz"
                    np.savez_compressed(path,**{k:v for k,v in events.items() if k in [
                        'input_neuron_ids','input_activation_steps','event_neuron_ids','event_steps','dt_ms','duration_ms','seed','dataset','id_namespace']})
                    observation['event_record']={'file':str(path.relative_to(ROOT)),'sha256':sha256(path)}
                    if embodied and embodied['feedback']['session_start_ms']>0:
                        observation['event_record'].update(requires_session_prefix=True,
                            prefix_record=embodied['feedback']['output_dir']+'/input-events.npz')
                observations.append(observation);receptions.append(reception)
        requested_datasets={catalogue['channels'][item['channel']]['dataset'] for item in requested_items}
        if environmental and presentations:requested_datasets={'MaleCNS v1.0'}
        primary=next((obs for obs in observations if obs['dataset'] in requested_datasets),observations[0])
        reception=dict(primary['reception'])
        if len(observations)>1:
            reception.update(items=[item for r in receptions for item in r['items']],
                dataset='多个独立连接组',description='；'.join(r['description'] for r in receptions),
                neuron_count=sum(r['neuron_count'] for r in receptions),
                neuron_ids=[],neuron_ids_by_dataset={r['dataset']:r['neuron_ids'] for r in receptions},
                limitations=list(dict.fromkeys([note for r in receptions for note in r['limitations']]))+['各数据集独立运行，尚未拼接成同一只果蝇。'])
        reply=primary['response_text'] if len(observations)==1 else '\n'.join(o['dataset']+'：'+o['response_text'] for o in observations)
        result={'handled':True,'neural_simulation_run':True,'reply_source':'neural_readout',
            'reply':reply,'reception':reception,'observation':primary,'observations':observations,
            'input_plan':input_plan or {'kind':'stimulus' if items else 'silence','items':items},
            'pathway':{'stages':['text','sensory_events','connectome_activity','blind_readout','text'],
                'input_mode':input_mode,'direct_neural_intervention':bool(requested_items),
                'completed':all(o['readout_model'].startswith('general_sensory') for o in observations),
                'fresh_neural_trial':all(o.get('embodied_feedback',{}).get('session_start_ms',0)==0 for o in observations),
                'independent_datasets':len(observations)>1,
                'single_individual':environmental and len(observations)==1,
                'body_status_by_dataset':{o['dataset']:o['scene']['body']['status'] for o in observations},
                'closed_loop':False}}
        dump(ROOT/'reports/general-sensory/interactive'/f"{primary['id']}.json",result)
        return result

    def observe(self,words,seed,*,body_context='ground'):
        validate_body_context(body_context)
        if not isinstance(words,list) or any(w not in WORDS for w in words):
            raise ValueError('Choose supported stimulus channels')
        if not isinstance(seed,int) or not 0<=seed<2**32:
            raise ValueError('Seed must be a nonnegative 32-bit integer')
        with self.lock:
            start=time.perf_counter()
            groups=groups_for(self.manifest,words)
            trial=self.brain.run(groups,300,seed,record_spikes=True)
            driven=[self.brain.lookup[str(i)] for ids,_ in groups for i in ids]
            result=self._result(trial,driven,seed,start,body_context=body_context)
            result['reception']={'status':'simulated','intent':'感官环境示例','channel':'environment',
                'input_mode':'neural_intervention','direct_neural_intervention':True,
                'description':'向选定的感觉通道输入刺激','neuron_count':len(driven),
                'neuron_ids':[str(self.brain.ids[i]) for i in driven], 'neuron_types':words,
                'dataset':'FlyWire v783','side':None,'direction_mapping_proposed':False,
                'detail':'实验性直接神经干预，不代表环境自然感知。','source':None,
                'input_events':None,'input_histogram':[]}
            dump(REPORTS/'interactive'/f"{result['id']}.json",{'stimulus_config':words,'observation':result})
            return result

    def converse(self,**kwargs):
        from .conversation import run_turn
        if kwargs.get('kind','message') in ('stimuli','words'):kwargs.pop('session_id',None)
        if not hasattr(self,'session_lock'):self.session_lock=threading.RLock()
        with self.session_lock:
            states=self._session_map(kwargs['session_id']) if kwargs.get('session_id') else {}
            before={k:s.neural.step if s.neural else 0 for k,s in states.items()}
            try:
                result=run_turn(self,**kwargs)
                if kwargs.get('session_id') and result.get('dialogue_record'):
                    if not hasattr(self,'session_results'):self.session_results={}
                    self.session_results[kwargs['session_id']]=result['dialogue_record']
                return result
            except Exception:
                # A failed multi-dataset turn must not resume some bodies after
                # advancing only others. Preserve records, invalidate the pair.
                if any(s.closed or (s.neural.step if s.neural else 0)!=before.get(k,0) for k,s in states.items()):
                    for state in states.values():state.close()
                raise

    def chat(self,message,history=None,observation_id='',*,backend='off'):
        if not isinstance(message,str) or not message.strip() or len(message)>4000:
            raise ValueError('Message must contain 1..4000 characters')
        if backend not in ('configured','local','off'):
            raise ValueError('Unknown legacy backend')
        observation=self.observations.get(observation_id)
        if observation:
            from .dialogue import observation_sensation
            from .expression import expression_report
            feeling=observation_sensation(observation)
            expression=expression_report(observation)
            reply=expression['text'] or expression['reason']
        else:
            reply='还没有可读出的表达记录。';feeling=None;expression=None
        messages=[dict(m,reused_observation=True) for m in expression['messages']] if expression else []
        return {'handled':False,'reply':reply,'reply_source':'neural_expression' if messages else ('expression_status' if observation else 'input_status'),
            'sensation':feeling,'expression':expression,'messages':messages,
            'reused_observation':bool(observation),
            'neural_simulation_run':False,'observation_id':observation['id'] if observation else None,
            'llm_called':False,'external_action_control':False}

    def answer_question(self,message,history=None,observation_id=''):
        """Route a question to a measured readout; never answer from the message.

        A recognised topic is answered from the one readout that measures it in the
        last recorded window. A recognised topic without a measurement refuses
        explicitly instead of falling back to a generic re-expression, and an
        unrecognised question keeps the re-expression path, labelled as such.
        """
        from .questions import route_question
        observation=self.observations.get(observation_id) if observation_id else None
        route=route_question(message,observation)
        result=self.chat(message,history,observation_id,backend='off')
        result['question_route']=route
        if route['status'] in ('answered','no_measurement'):
            # One utterance only: the routed sentence replaces the generic
            # re-expression so the reply cannot say two different things. No new
            # observation is produced; this is a re-answer from the last record.
            result.update(handled=True,reply=route['text'],
                reply_source='measured_readout_route' if route['status']=='answered' else 'measured_readout_absent',
                messages=[{'kind':'question_route','text':route['text'],'source':route['source'],
                           'status':route['status'],'topic':route['topic'],
                           'observation_id':route.get('observation_id')}],
                routed_readout=route['source'],routed_topic=route['topic'],
                reused_observation=True,neural_simulation_run=False)
        return result

    def communicate(self,message,seed,history=None,observation_id='',*,body_context='ground'):        return self.converse(message=message,seed=seed,history=history,observation_id=observation_id,
                             body_context=body_context)


def create_server(port=8765,app=None):
    app=Application() if app is None else app
    class Handler(BaseHTTPRequestHandler):
        def send_json(self,body,status=200,download=None):
            content=json.dumps(body,ensure_ascii=False).encode('utf8')
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Cache-Control','no-store')
            if download:self.send_header('Content-Disposition','attachment; filename="'+download+'"')
            self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content)
        def do_GET(self):
            if not local_request_allowed(self.headers.get('Host'),self.headers.get('Origin'),self.headers.get('Sec-Fetch-Site')):
                self.send_json({'error':'Local requests only'},403);return
            route=urlparse(self.path).path
            static_files={
                '/':('index.html','text/html'),
                '/app.css':('app.css','text/css'),
                '/app.js':('app.js','text/javascript'),
                '/body-view.js':('body-view.js','text/javascript'),
                '/environment-editor.js':('environment-editor.js','text/javascript'),
                '/environment-inspection.js':('environment-inspection.js','text/javascript'),
                '/vendor/three.module.js':('vendor/three.module.js','text/javascript'),
                '/vendor/three.core.js':('vendor/three.core.js','text/javascript'),
                '/vendor/OrbitControls.js':('vendor/OrbitControls.js','text/javascript'),
                '/vendor/THREE-LICENSE.txt':('vendor/THREE-LICENSE.txt','text/plain'),
                '/assets/neuromechfly-scene.json':('assets/neuromechfly-scene.json','application/json'),
                '/assets/neuromechfly-behavior-scene.json':('assets/neuromechfly-behavior-scene.json','application/json'),
            }
            if route in static_files:
                name,content_type=static_files[route]
                content=(ROOT/'chatflypt/web'/name).read_bytes()
                self.send_response(200);self.send_header('Content-Type',content_type+'; charset=utf-8')
                self.send_header('Cache-Control','no-cache')
                self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content)
            elif route=='/api/scene':self.send_json(app.resting_scene)
            elif route=='/api/settings':self.send_json(public_settings(local_config()))
            elif route=='/api/inputs':self.send_json(app.inputs())
            elif route=='/api/environment':self.send_json(app.environment_state())
            elif route=='/api/session':
                try:self.send_json(app.session_status(parse_qs(urlparse(self.path).query).get('session_id',[''])[0]))
                except ValueError as exc:self.send_json({'error':str(exc)},400)
            elif route=='/api/environment/response':
                try:self.send_json(app.environment_response(parse_qs(urlparse(self.path).query).get('record',[''])[0]))
                except ValueError as exc:self.send_json({'error':str(exc)},400)
            elif route=='/api/environment/inspection':
                try:
                    query=parse_qs(urlparse(self.path).query);record_id=query.get('record',[''])[0]
                    record=app.environment_inspection(record_id)
                    self.send_json(record,download=('environment-'+record_id+'.json') if query.get('download')==['1'] else None)
                except ValueError as exc:self.send_json({'error':str(exc)},400)
            elif route=='/api/status':
                from .model_runtime import local_model_info
                from .acoustic_readout import acoustic_capability
                config=local_config();report=REPORTS/'acceptance.json'
                self.send_json({'neurons':len(app.male.ids),'edges':len(app.male.indices),
                    'dataset':'MaleCNS v1.0','single_individual':True,
                    'input_mode':'environment_only','body_feedback_interval_ms':10.,'vision_interval_ms':100.,
                    'state_preserved_across_messages':True,'state_preserved_across_geometry_edits':True,
                    'words':WORDS,'llm_configured':bool(config.get('OPENAI_API_KEY') and config.get('CHATFLYPT_MODEL')),
                    'model':config.get('CHATFLYPT_MODEL') or None,
                    'local_model':local_model_info(),
                    'dialogue_mode':'neural_readout_only','expression_source':'blind_neural_readout',
                    'external_action_control':False,'llm_in_pathway':False,
                    'outward_acoustic_readout':acoustic_capability(),
                    'acceptance':json.loads(report.read_text(encoding='utf8')) if report.exists() else None})
            else:self.send_json({'error':'Not found'},404)
        def do_POST(self):
            if not local_request_allowed(self.headers.get('Host'),self.headers.get('Origin'),self.headers.get('Sec-Fetch-Site')):
                self.send_json({'error':'Origin not allowed'},403);return
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':
                self.send_json({'error':'JSON required'},415);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=20000:raise ValueError('Invalid request size')
                body=json.loads(self.rfile.read(length))
                if not isinstance(body,dict):raise ValueError('JSON object required')
                if self.path=='/api/settings':
                    self.send_json(save_settings(body,local_config()))
                elif self.path=='/api/settings/models':
                    self.send_json(list_models(draft_config(body,local_config(),require_model=False)))
                elif self.path=='/api/settings/test':
                    config=draft_config(body,local_config())
                    answer=LLM(provider='openai',config=config).complete('请只回复：连接成功。','你正在进行一次 API 连接测试。')
                    self.send_json({'ok':True,'model':answer['requested_model'],
                        'provider_host':answer['provider_host'],'reply':answer['text']})
                elif self.path=='/api/environment/apply':
                    self.send_json(app.apply_environment(body.get('objects'),body.get('revision'),
                        body.get('seed',42),body.get('body_context','ground'),body.get('session_id')))
                elif self.path=='/api/environment/inspect':
                    self.send_json(app.inspect_applied_environment(body.get('seed',42),
                        body.get('body_context','ground'),body.get('duration_ms',500)))
                elif self.path=='/api/session/reset':
                    self.send_json(app.reset_session(body.get('session_id')))
                elif self.path=='/api/observe':
                    self.send_json(app.observe(body.get('words',[]),body.get('seed',42),body_context=body.get('body_context','ground')))
                elif self.path=='/api/stimulate':
                    self.send_json(app.stimulate(body.get('items',[]),body.get('seed',42),body_context=body.get('body_context','ground')))
                elif self.path in ('/api/communicate','/api/converse'):
                    self.send_json(app.converse(kind=body.get('kind','message'),message=body.get('message',''),
                        words=body.get('words'),items=body.get('items'),seed=body.get('seed',42),
                        history=body.get('history',[]),observation_id=body.get('observation_id',''),
                        body_context=body.get('body_context','ground'),backend=body.get('backend','off'),
                        feedback_enabled=body.get('feedback_enabled',False),session_id=body.get('session_id')))
                elif self.path=='/api/receive':
                    self.send_json(app.converse(message=body.get('message',''),seed=body.get('seed',42),
                        body_context=body.get('body_context','ground'),session_id=body.get('session_id')))
                elif self.path=='/api/chat':
                    result=app.chat(body.get('message',''),body.get('history',[]),body.get('observation_id',''),
                        backend=body.get('backend','off'))
                    self.send_json(dict(result,answer={'text':result['reply'],'source':result['reply_source']}))
                else:self.send_json({'error':'Not found'},404)
            except LLMUnavailable as exc:self.send_json({'error':str(exc)},503)
            except (ValueError,KeyError,TypeError) as exc:self.send_json({'error':str(exc)},400)
            except Exception:self.send_json({'error':'Internal error; no model response was substituted'},500)
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


def serve(port=8765):
    server=create_server(port)
    print(f'ChatFlyPT http://127.0.0.1:{port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765)
    serve(parser.parse_args().port)
