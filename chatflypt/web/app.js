'use strict';
const $ = id => document.getElementById(id);
const names = {sugar:'糖',bitter:'苦味',water:'水',salt:'低盐',vinegar:'醋',yeast:'酵母',geosmin:'土臭素',pheromone:'信息素',cva:'信息素',warm:'温暖',cold:'寒冷',humid:'潮湿',dry:'干燥',bright:'明亮',touch:'口器触碰',sound:'声音',wind:'风',right_foreleg_touch:'右前腿触碰',left_foreleg_touch:'左前腿触碰',courtship_pulse:'求偶鸣声'};
let configured=false, busy=false, observation=null, reception=null, history=[], seedOffset=0, resting=null;
let apiSettingsBusy=false;
let currentDialogue=null,lastStatus=null;
let eyeSampleKey=null;
let inputCatalogue=[],inputCatalogueLoading=false,observationBatch=[],inputPlan=null;
let inputChannelById=new Map(),inputSearchIndex=new Map(),inputSearchTimer=null;
const inputStates={ready:'已接入仿真',candidate:'候选通路',unmapped:'尚未映射'};
const modalityNames={olfactory:'嗅觉',olfaction:'嗅觉',gustatory:'味觉',taste:'味觉',gustation:'味觉',visual:'视觉',vision:'视觉',auditory:'听觉',audition:'听觉',mechanosensory:'机械感觉',tactile:'触觉',touch:'触觉',temperature:'温度',thermosensory:'温度',thermal:'温度',humidity:'湿度',hygrosensory:'湿度',proprioception:'本体感觉',mechanosensation:'机械感觉',chemosensation:'化学感觉',interoception:'内感受',nociception:'伤害感受',unknown:'感觉类型待归类',wind:'风觉',social:'社会信号'};
const sideNames={left:'左侧',right:'右侧',both:'双侧',bilateral:'双侧',none:'不分侧',all:'全部',center:'中央'};
// Exact catalogue strings map to display copy only; source records remain unchanged.
const limitationCopy=new Map([
  ['New MaleCNS population: no inherited FlyWire or pre-existing sensory decoder label/weights.','新接入的 MaleCNS 细胞群使用独立训练的读出器，不沿用 FlyWire 的标签权重。'],
  ['Population identity is supported by type/glomerulus correspondence; natural stimulus tuning in these individual EM cells is not measured.','细胞群身份依据类型及嗅小球对应；这些电镜个体细胞的自然刺激响应尚未实测。'],
  ['Unknown peripheral sides and cells without modeled outgoing edges are excluded from executable side selections and retained in the audit.','外周侧别未知或没有当前建模出边的细胞保留在审计中，不参与侧别输入。'],
  ['Normal embodied input uses a local ommatidium-to-column candidate bridge inferred from actual R1-6 to lamina connections; cross-specimen retinal geometry is not calibrated and coverage is incomplete. Independent population interventions do not use that spatial bridge. No spectral tuning or object recognition is validated.','正常环境根据实际 R1–R6 到视叶连接建立局部小眼与神经网膜列的候选对应；跨标本几何尚未标定，覆盖不完整。独立群体干预不使用此空间映射；光谱响应和物体识别尚未验证。'],
  ['Photoreceptors are graded/histaminergic; event approximation and absence of a calibrated tonic lamina baseline can yield only inhibition. No artificial excitatory sign or baseline is added.','光感受器采用渐变电信号及组胺传递；当前事件近似缺少已标定的视叶持续基线，可能只产生抑制，没有人为改为兴奋或补加基线。'],
  ['The reconstructed R1–R6 population is asymmetric and incomplete; missing cells are not fabricated.','重建的 R1–R6 群左右不对称且不完整，没有虚构缺失细胞。'],
  ['DM2 is a retained engineering odor proxy, not a unique or complete biological yeast code.','DM2 是沿用的工程气味代理，不能代表唯一或完整的天然酵母编码。'],
  ['Rate and time profiles are engineering inputs, not calibrated physical stimulus-to-spike transformations.','输入频率和时间曲线采用工程设定，尚未经物理刺激到神经脉冲的实测转换校准。'],
  ['Side denotes annotated peripheral nerve entry, not requested movement direction or a fabricated split.','侧别表示注释中的外周神经入口，不表示要求的运动方向，也不是人为拆分细胞得到的侧别。'],
  ['Cells with unassigned entry side are listed but excluded from all executable side selections.','入口侧别未确定的细胞仍列入目录，但不参与任何可执行的侧别选择。'],
  ['Graded photoreceptor signals are approximated by LIF input events; no image, wavelength transfer curve, adaptation, or motion field is implemented.','以 LIF 输入事件近似视感受器的渐变电信号；尚未实现图像、波长响应曲线、适应或运动视场编码。'],
  ['DRA population only; no e-vector angle or spatial polarization field encoder.','仅使用背缘区（DRA）细胞群；尚无电矢量方向角或空间偏振场编码器。'],
  ['Central ocellar entries are selectable only through the bilateral/default field.','位于中央的单眼输入仅可通过“双侧／默认”选项选择。'],
  ['DM1/VA2 are a partial vinegar proxy; natural vinegar activates a broader, concentration-dependent combination.','DM1／VA2 仅作为醋气味的部分代理；天然醋会激活范围更广、随浓度变化的神经组合。'],
  ['Legacy DM2 engineering proxy. No claim that DM2 uniquely or fully encodes yeast odor; the annotation establishes anatomy only.','沿用 DM2 工程代理；不能据此认定 DM2 唯一或完整编码酵母气味，现有注释仅确立解剖归属。'],
  ['The generic interface alias denotes cVA only; other pheromones are not equivalent.','界面中的通用“信息素”别名仅指 cVA；其他信息素不能等同处理。'],
  ['LB3/LB2d are annotated sugar/water; this registry cannot distinguish sugar-specific from water-specific neurons. It never copies the older left-only IDs to construct a right-side population.','LB3／LB2d 的注释为“糖／水”，本目录尚不能区分糖特异与水特异神经元；也不会复制旧版仅左侧的 ID 来构造右侧细胞群。'],
  ['Low-salt annotation does not establish high-salt or arbitrary concentration coding.','低盐注释尚不能确立高盐或任意浓度的编码方式。'],
  ['VP1d evaporative-cooling assignment is putative; no wet-bulb physics model.','VP1d 的蒸发冷却功能归属仍属推定；尚无湿球物理模型。'],
  ['Broad auditory population; no calibrated acoustic frequency, waveform, particle velocity, or phase locking.','使用较宽泛的听觉细胞群；尚未校准声学频率、波形、空气质点速度或相位锁定。'],
  ['36 ms inter-pulse interval is literature-inspired; 5 ms gate and 150 Hz neural input rate are assumptions, not acoustic carrier frequency.','36 ms 脉冲间隔参考文献设定；5 ms 门控时长和 150 Hz 神经输入频率均为假设，后者不是声学载波频率。'],
  ['Wind and gravity are not independently identified by this broad population input; no force/direction transfer function.','这类宽泛的细胞群输入尚不能分别识别风与重力；尚无力大小或方向到神经输入的转换函数。'],
  ['Group stimulation does not select one individual bristle or reproduce measured contact force.','细胞群刺激未选定单根触毛，也未复现实测接触力。'],
  ['Anatomical candidate population; neither exact distal sensillum identity nor an ensuing movement is assigned.','这是解剖学候选细胞群；尚未指定精确的末端感器身份或随后发生的动作。'],
  ['Mapped body surface is the posterior dorsal mesothoracic/notum field, not the entire abdomen or every body patch.','已映射的体表范围为中胸背侧后部／背板区域，未覆盖整个腹部或身体的每个局部。'],
  ['No calibrated joint angle, load, rotation, or direction-specific receptor tuning; anatomical population activation only.','尚未校准关节角度、负荷、旋转或感受器的方向选择性；当前仅激活对应的解剖细胞群。'],
  ['Aggregates heterogeneous leg proprioceptors; flexion/extension and joint angle are not inferred.','此通道汇集多种腿部本体感受器，尚不推断屈伸状态或关节角度。'],
  ['Advanced anatomical grouping only. Exact stimulus selectivity is unresolved; no everyday-language alias is assigned.','高级通道仅按解剖注释分组；精确的刺激选择性尚未确定，因此未分配日常语言别名。'],
  ['No validated metabolic-state-to-neuron model or exact receptor mapping in this registry.','本目录尚无经验证的代谢状态到神经元模型，也无精确的感受器映射。'],
  ['Internal osmolarity and drinking state are not equivalent to external water or humidity input.','体内渗透浓度和饮水状态不能等同于外部水或湿度输入。'],
  ['No validated adult nociceptor body-ID set or damaging-stimulus transfer function; no substitution with touch or heat.','尚无经验证的成虫伤害感受器 body ID 集合或伤害性刺激转换函数；不会用触碰或热输入替代。'],
  ['Current LB3/LB2d annotation is sugar/water; exact sugar-only bilateral IDs are not established.','当前 LB3／LB2d 注释为“糖／水”，尚未确定双侧糖特异神经元的精确 ID。'],
  ['Current LB3/LB2d annotation is sugar/water; exact water-only bilateral IDs are not established.','当前 LB3／LB2d 注释为“糖／水”，尚未确定双侧水特异神经元的精确 ID。'],
  ['High-salt taste recruits a context-dependent peripheral code; low-salt or all bitter cells are not substituted for an exact mapped set.','高盐味觉采用依赖情境的外周编码；不会用低盐细胞或全部苦味细胞代替精确映射的细胞集合。'],
  ['OtopLa-dependent acid taste is documented, but the corresponding exact bilateral body IDs and pH response model are not resolved here; vinegar odor is not acid taste.','已有 OtopLa 依赖型酸味觉的研究记录，但此处尚未确定对应的精确双侧 body ID 和 pH 响应模型；醋气味不等同于酸味觉。'],
  ['IR25a/IR76b-dependent fatty-acid responses do not justify assigning every sweet/water GRN this function.','IR25a／IR76b 依赖型脂肪酸响应的证据，不足以将此功能赋予所有糖／水味觉神经元。'],
  ['Amino-acid taste depends on molecular and cellular context; no validated exact connectome population mapping is available in this registry.','氨基酸味觉依赖分子与细胞环境；本目录尚无经验证的精确连接组细胞群映射。'],
  ['No validated adult neural ID mapping or oxygen physiology model; neither external CO2 ORNs nor arbitrary central cells stand in for internal oxygen sensing.','尚无经验证的成虫神经元 ID 映射或氧生理模型；不会用感知外部 CO₂ 的嗅觉受体神经元或任意中枢细胞代替体内氧感受。'],
  ['Internal gas and acid-base regulation are not represented by the external antennal CO2 channel; exact sensors and transfer functions remain unresolved.','外部触角 CO₂ 通道不能代表体内气体与酸碱调节；精确的感受器和转换函数仍未确定。'],
  ['Piezo-mediated crop stretch has experimental support, but its functional receptor population is not reliably assigned to the available body IDs.','Piezo 介导的嗉囊扩张感受已有实验支持，但相应功能感受器群尚未可靠对应到现有 body ID。'],
  ['Internal glucose/nutrient signals involve hormonal and neural state; no validated state-variable or exact-ID implementation is present.','体内葡萄糖／营养信号涉及激素与神经状态；当前尚无经验证的状态变量模型或精确 ID 实现。'],
  ['Independent trial, reset between observations.','各次观测为独立试次，试次之间重置神经状态。'],
  ['Sensory proxy labels, not subjective experience.','读出结果为感觉代理标签，不代表主观体验。'],
  ['No body dynamics or observed movement trajectory.','尚无身体动力学或实测动作轨迹。']
]);
const welcome=$('welcome').cloneNode(true);
const state={scene:null,angle:.08,tilt:-.10,zoom:1,progress:0,playing:false,overview:false,last:0,counts:[],totals:[],projected:[]};
const brain=$('brain'), bc=brain.getContext('2d');
function bodyContext(){return $('bodyContext').value;}
let liveSessionId=sessionStorage.getItem('chatflypt-live-session')||crypto.randomUUID().replaceAll('-','');
sessionStorage.setItem('chatflypt-live-session',liveSessionId);
function dialogueOptions(){return {session_id:liveSessionId};}
async function api(path,body){const r=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const j=await r.json();if(!r.ok)throw new Error(j.error||'请求未完成');return j;}
function icon(button,name){button.replaceChildren();const s=document.createElementNS('http://www.w3.org/2000/svg','svg'),u=document.createElementNS(s.namespaceURI,'use');u.setAttribute('href','#'+name);s.append(u);button.append(s);}
function setBusy(value){busy=value;['send','example','newChat','runExperiment','inputChannels','bodyContext','inspectEnvironment'].forEach(id=>$(id).disabled=value);document.querySelectorAll('[data-message]').forEach(b=>b.disabled=value);$('observationSelect').disabled=value;updateStimulusAvailability();$('error').textContent='';}
function nextSeed(){const seed=Number($('seed').value)+seedOffset++;if(!Number.isInteger(seed)||seed<0||seed>=2**32)throw new Error('随机种子需要是 0 到 4294967295 之间的整数。');return seed;}
function appendMessage(role,text,label='接收记录',pending=false){$('welcome')?.remove();const root=document.createElement('article');root.className='message '+role+(pending?' pending':'');const head=document.createElement('div');head.className='message-label';const avatar=document.createElement('span');avatar.className='avatar';avatar.textContent='✳';const title=document.createElement('span');title.textContent='ChatFlyPT';const tag=document.createElement('span');tag.className='message-tag';tag.textContent='· '+label;head.append(avatar,title,tag);const body=document.createElement('div');body.className='message-body';body.textContent=text;root.append(head,body);$('conversation').append(root);$('conversation').scrollTop=$('conversation').scrollHeight;return {root,body,tag};}
function finishMessage(msg,text,label,stats=[]){msg.root.classList.remove('pending');msg.body.textContent=text;msg.tag.textContent='· '+label;if(stats.length){const r=document.createElement('div');r.className='message-stats';stats.forEach(t=>{const s=document.createElement('span');s.textContent=t;r.append(s);});msg.root.append(r);}$('conversation').scrollTop=$('conversation').scrollHeight;}
function remember(q,a){history.push({role:'user',content:q.slice(0,1000)},{role:'assistant',content:a.slice(0,1000)});history=history.slice(-8);}
function datasetName(scene){const raw=scene?.dataset||scene?.source||'';if(/male.?cns/i.test(raw))return 'MaleCNS v1.0';if(/flywire/i.test(raw))return 'FlyWire v783';return scene?.dataset||'连接组';}
function setScene(scene,autoplay=false){
  window.flyPhysics?.clear();
  $('routeDetail').setAttribute('aria-label','查看感觉通路详情');
  state.scene=scene;state.progress=0;state.overview=false;state.last=performance.now();
  state.counts=new Float32Array(scene?.nodes.length||0);state.totals=new Float32Array(scene?.nodes.length||0);
  for(const f of scene?.frames||[])for(let j=0;j<f.length;j+=2)state.totals[f[j]]+=f[j+1];
  const ready=!!scene?.frames.length;state.playing=autoplay&&ready;
  $('dataset').textContent=datasetName(scene);$('dataset').title=scene?.dataset||scene?.source||'';
  $('brainCaption').replaceChildren(document.createTextNode(Number(scene?.total_neurons||0).toLocaleString()+' '));
  const span=document.createElement('span');span.textContent='个神经元 · '+datasetName(scene);$('brainCaption').append(span);
  $('brainCaption').title=scene?.geometry_note||'';$('brain').setAttribute('aria-label',datasetName(scene)+' 神经活动图，可拖动旋转。'+(scene?.geometry_note||''));
  $('geometryNote').textContent=scene?.geometry_note||'神经图等待连接组数据。';
  $('playback').disabled=!ready;$('timeline').disabled=!ready;$('timeline').value=0;$('time').textContent='0 ms';
  $('endTime').textContent=ready?scene.duration_ms+' ms':'—';$('replayLabel').textContent=ready?'14倍慢放':'位置采样';
  $('playback').setAttribute('aria-label',state.playing?'暂停神经活动':'播放神经活动');if(state.playing)$('playback').textContent='Ⅱ';else icon($('playback'),'play');
}
function sideLabel(side){return sideNames[side]||side||'不分侧';}
function displayLimitation(value){const original=String(value??'');return limitationCopy.get(original)||original;}
function rawListText(value){return Array.isArray(value)?value.filter(Boolean).join('；'):value?String(value):'';}
function listText(value){return Array.isArray(value)?value.filter(Boolean).map(displayLimitation).join('；'):value?displayLimitation(value):'';}
function receptionItemLabel(item){const part={foreleg:'前腿',midleg:'中腿',hindleg:'后腿'}[item.body_part];return (item.label||names[item.channel]||item.channel)+(item.side?' · '+sideLabel(item.side):'')+(part?' · '+part:'');}
function showReception(r){
  reception=r;$('intent').textContent=r.intent;$('stimulusDescription').textContent=r.description;
  $('routeState').textContent=r.status==='simulated'?'已运行 · 可回放':'已编码 · 等待接入';$('routeState').classList.toggle('amber',r.status!=='simulated');
  const typed=r.neuron_types.map(x=>names[x]||x),types=typed.slice(0,3).join(' / ')+(typed.length>3?' 等':'');
  $('targetDescription').textContent=(r.items?.length?r.items.map(receptionItemLabel).join('；'):'输入感觉神经：'+types)+' · '+r.neuron_count+' 个细胞';
  if(r.proprioception?.applied)$('targetDescription').textContent+='；实际腿关节反馈 · '+r.proprioception.events+' 个感觉事件（群体近似）';
  $('routeNote').textContent=[r.direction_mapping_proposed?'方向映射待验证':'',displayLimitation(r.detail),listText(r.limitations)].filter(Boolean).join('；');
  $('targetDescription').title=$('targetDescription').textContent;$('routeNote').title=$('routeNote').textContent;
  $('routeDetail').disabled=false;
  const values=r.input_histogram||[],max=Math.max(1,...values);let d='M0 14';values.forEach((n,i)=>{d+='L'+(i/Math.max(1,values.length-1)*120).toFixed(1)+' '+(14-n/max*12).toFixed(1);});
  $('inputTrace').firstElementChild.setAttribute('d',values.length?d:'M0 13H120');$('worldCue').hidden=false;
  $('worldCue').textContent=r.items?.length?r.items.map(receptionItemLabel).join('；'):r.channel==='cva'?'触角 · 信息素输入':r.channel==='courtship_pulse'?'触角 · 鸣声输入':r.side&&/foreleg_touch/.test(r.channel||'')?sideLabel(r.side)+'前腿 · 触碰位置预览':'感觉环境输入';
  $('worldCue').title=$('worldCue').textContent;
  $('worldState').textContent='等待运动活动';$('bodyNote').textContent='官方身体初始姿态 · 尚无动作记录';$('evidenceButton').disabled=false;
}
function motorSummary(j){
  const reaction=j.reaction||{},groups=reaction.groups||reaction,parts=[];
  for(const [key,label] of [['vnc_motor','VNC 运动'],['cb_motor','脑运动'],['DN','DN'],['descending','下行']]){
    const value=groups[key],spikes=typeof value==='number'?value:value?.spikes;
    if(Number.isFinite(spikes))parts.push(label+' '+spikes.toLocaleString());
  }
  if(parts.length)return parts.join(' · ')+' 次脉冲';
  const m=j.scene.motor_evidence||[];
  if(!m.length)return '';
  const sum=side=>m.filter(x=>x.side===side).reduce((a,b)=>a+b.spikes,0);
  const types=[...new Set(m.map(x=>x.type).filter(Boolean))];
  return types.slice(0,2).join('/')+'  L '+sum('left')+' · R '+sum('right')+' 次脉冲';
}
function showObservation(j){
  $('receptionSensation').hidden=!j.sensation;$('sensationText').textContent=j.sensation?.text||'';
  $('sensationGloss').textContent=(j.sensation?.labels||[]).length?'标签层（注释，不是译文）：'+j.sensation.labels.join('、'):'';
  const motor=j.scene?.intent;$('receptionIntent').hidden=!motor?.text;
  $('intentText').textContent=motor?.text||'';
  const commanded=j.scene?.command;$('receptionCommand').hidden=!commanded?.text;
  $('commandText').textContent=commanded?.text||'';
  if(j.environment)window.flyEnvironmentEditor?.showRecordedEnvironment(j.environment);
  const vision=j.environment_sample?.vision;$('eyeVision').hidden=vision?.status!=='measured';if(vision?.eye_images){$('leftEye').src=vision.eye_images.left;$('rightEye').src=vision.eye_images.right;}
  eyeSampleKey=null;$('eyeVision').querySelector('summary').textContent='果蝇双眼实际画面';
  $('eyeVision').querySelector('p').textContent=j.reception?.visual_projection?'官方双眼渲染 · 721 小眼/眼；局部亮度经候选视网膜柱映射进入同一 MaleCNS。空间配准尚未生理标定，物体名称没有送入大脑。':'官方双眼渲染 · 721 小眼/眼；此独立记录使用分眼平均亮度，物体名称没有送入大脑。';
  observation=j;if(j.reception)showReception(j.reception);setScene(j.scene,true);
  const continuity=j.embodied_feedback;
  const unmapped=[...new Set((continuity?.samples||[j.environment_sample]).filter(Boolean).flatMap(s=>s.unmapped||[]).map(u=>u.reason).filter(Boolean))];
  if(unmapped.length){$('targetDescription').textContent+='；实测后仍有 '+unmapped.length+' 类映射未接入（通路详情）';$('targetDescription').title=$('targetDescription').textContent+'\n'+unmapped.join('\n');}
  $('brainState').textContent=continuity?.between_messages?'连续会话 · '+(continuity.session_end_ms/1000).toFixed(1)+' s':continuity?'本轮感觉反馈 · '+continuity.interval_ms+' ms':'仿真记录';$('brainState').classList.remove('pending');
  $('brainState').title=continuity?.between_messages?'跨消息和环境物品改动保留神经、身体与控制器；等待消息时仿真时钟暂停。新对话、身体场景改变或服务重启会初始化。':'';
  if(j.archived_environment_record){$('brainState').textContent='历史环境记录 · '+j.duration_ms+' ms';$('brainState').title='这是保存时的模型与输入记录，未作为当前连续会话的神经或身体状态。';}
  $('brainMetrics').textContent=j.active_neurons.toLocaleString()+' 个活跃细胞 · '+j.spikes.toLocaleString()+' 次脉冲';
  if(j.reaction?.membrane_response)$('brainMetrics').textContent+=' · '+j.reaction.membrane_response.internal_cells.toLocaleString()+' 个内部细胞电位响应';
  if(j.scene.voltage_note)$('geometryNote').textContent+=' 电位峰值：青色正偏移，紫色负偏移。';
  $('motorEvidence').textContent=motorSummary(j);$('worldState').textContent='神经响应已记录';
  $('bodyNote').textContent='官方身体初始姿态 · 尚无动作记录';$('evidenceButton').disabled=false;
  const body=j.scene.body||{};
  if(body.physics_run&&body.trajectory?.length){
    $('worldState').textContent=body.context==='tethered_flight'?'系留身体回放':'MuJoCo 地面回放';
    $('bodyNote').textContent='实测姿态 · '+(body.motor_activity_in_window===false?'本窗无可执行神经动作命令，含姿态维持和物理作用':'神经驱动工程控制器')+(body.truncated?' · 显示前 '+Math.round(body.duration_ms)+' ms':'');
    if(body.motor_coverage?.unmapped_active_neurons)$('bodyNote').textContent+=' · 未接 '+body.motor_coverage.unmapped_active_neurons+' 个运动相关细胞（累计）';
  }else if(body.status==='no_motor_activity'){
    $('worldState').textContent='本次无可用动作命令';$('bodyNote').textContent='初始姿态 · 本次未生成动作轨迹';
  }else if(['error','timeout','unavailable'].includes(body.status)){
    $('worldState').textContent='身体仿真未完成';$('bodyNote').textContent=body.reason||body.error||body.status;
  }
}
function showFeedbackEyes(){
  const feedback=observation?.embodied_feedback;if(!feedback||window.flyPhysics?.active())return;
  const time=state.progress*(state.scene?.duration_ms||0);
  let sampled=feedback.samples[0];
  for(const item of feedback.samples)if(item.time_ms<=time)sampled=item;
  const finalTime=(feedback.final_probe?.frame?.time_ms??Infinity)-(feedback.session_start_ms||0);
  if(finalTime<=time)sampled={time_ms:finalTime,vision:feedback.final_probe.vision};
  const renderedAt=sampled.vision?.sampled_at_ms==null?sampled.time_ms:Math.max(0,sampled.vision.sampled_at_ms-(feedback.session_start_ms||0));
  const key=observation.id+':'+renderedAt;if(key===eyeSampleKey)return;eyeSampleKey=key;
  if(sampled.vision?.eye_images){$('leftEye').src=sampled.vision.eye_images.left;$('rightEye').src=sampled.vision.eye_images.right;}
  $('eyeVision').querySelector('summary').textContent='果蝇双眼实际画面 · '+Math.round(renderedAt)+' ms';
}
function neuralResponse(j){if(typeof j.response_text!=='string'||!j.response_text.trim())throw new Error('已收到神经活动，但服务端尚未提供本次读出的文字结果。');return j.response_text;}
function setObservationBatch(primary,items=[]){
  observationBatch=(items||[]).filter(j=>j?.scene);
  if(primary?.scene&&!observationBatch.some(j=>j===primary||(j.id&&j.id===primary.id)))observationBatch.unshift(primary);
  const selector=$('observationSelect');selector.replaceChildren();
  observationBatch.forEach((j,i)=>{const option=document.createElement('option');option.value=i;option.textContent='感觉接收 · '+datasetName(j.scene)+' · '+j.duration_ms+' ms';selector.append(option);});
  $('observationSwitch').hidden=observationBatch.length<2;
  if(!observationBatch.length){observation=null;return;}
  const index=Math.max(0,observationBatch.findIndex(j=>j===primary||(j.id&&j.id===primary?.id)));selector.value=index;
  showObservation(observationBatch[index]);
}
function showInputStatus(result,text){
  if(observation?.embodied_feedback?.between_messages){
    pauseAtEnd();$('brainState').textContent='本次未施加输入 · 保留会话状态';
    $('brainState').classList.add('pending');$('routeState').textContent='输入说明';
    $('routeNote').textContent=result.reply||result.reason||'';return;
  }
  setObservationBatch(null);reception=null;setScene(resting);$('receptionSensation').hidden=true;$('receptionIntent').hidden=true;$('eyeVision').hidden=true;
  $('brainState').textContent='本次未运行';$('brainState').classList.add('pending');$('brainMetrics').textContent='本次没有神经活动记录';
  $('intent').textContent=text;$('stimulusDescription').textContent='本次没有施加感觉输入';$('targetDescription').textContent='查看感觉通道，了解已接入与待映射的输入。';
  $('routeState').textContent='输入说明';$('routeState').classList.add('amber');$('routeNote').textContent=result.reply||result.reason||'';
  $('inputTrace').firstElementChild.setAttribute('d','M0 13H120');$('routeDetail').disabled=!inputPlan;$('evidenceButton').disabled=!inputPlan;
  $('worldCue').hidden=true;$('worldState').textContent='本次未运行神经仿真';$('bodyNote').textContent='官方身体初始姿态 · 尚无动作记录';$('motorEvidence').textContent='';
}
function statusMessage(text,pending=false){
  $('welcome')?.remove();const node=document.createElement('div');node.className='activity-status'+(pending?' pending':'');
  node.textContent=text;$('conversation').append(node);$('conversation').scrollTop=$('conversation').scrollHeight;return node;
}
function pauseAtEnd(){
  state.progress=1;state.playing=false;state.overview=false;$('timeline').value=1000;
  $('time').textContent=(state.scene?.duration_ms||0)+' ms';icon($('playback'),'play');
  $('playback').setAttribute('aria-label','播放神经活动');$('replayLabel').textContent='本轮终点';
}
function finishCommunication(result,text,reply,{replay=false}={}){
  currentDialogue=result.dialogue?result:null;inputPlan=result.input_plan||null;
  if(!replay&&result.environment_event?.source==='human_message'){
    window.flyEnvironmentEditor?.acceptMessageEnvironment(result.environment);
    statusMessage('环境改动 · 聊天呈现 · 第 '+result.environment.revision+' 版 · 物品已参与物理和双眼渲染');
  }
  if(result.handled){
    if(result.observation)setObservationBatch(result.observation,result.observations);
    else if(result.reply){
      // A routed question answer reuses the previous recorded window instead of
      // running a new one: it is a reply, not a new observation.
      appendMessage('fly',result.reply,'据上一轮实测记录 · 未重跑仿真');
      statusMessage('据上一轮实测记录回答（'+((result.routed_topic)||'未指定主题')+'）；没有运行新的神经或物理仿真。');
    }else throw new Error('接收通路尚未返回神经响应记录。');
  }else if(result.reply_source==='input_status'&&result.input_plan)showInputStatus(result,text);
  if(result.sensation){$('receptionSensation').hidden=false;$('sensationText').textContent=result.sensation.text;
    $('sensationGloss').textContent=(result.sensation.labels||[]).length?'标签层（注释，不是译文）：'+result.sensation.labels.join('、'):'';}
  if(result.intent?.text){$('receptionIntent').hidden=false;$('intentText').textContent=result.intent.text;}
  if(result.command?.text){$('receptionCommand').hidden=false;$('commandText').textContent=result.command.text;}
  reply.root.remove();
  for(const reason of new Set((result.observations||[]).map(o=>o.embodied_feedback?.reset_reason).filter(Boolean)))statusMessage(reason);
  const outputs=(result.messages||[]).filter(item=>item.kind==='expression'&&item.text);
  for(const item of outputs){
    const label='果蝇 · 神经读出'+(item.dataset&&item.dataset!=='unspecified'?' · '+item.dataset:'')+(item.reused_observation?' · 上次记录':'');
    const message=appendMessage('assistant',item.text,label);
    const observation=(result.observations||[]).find(o=>o.id===item.observation_id);
    if(observation){const button=document.createElement('button');button.className='message-replay';button.textContent='查看对应神经记录 ↗';
      button.onclick=()=>{if(!busy)setObservationBatch(observation,result.observations);};message.root.append(button);}
  }
  for(const text of result.status_messages||[])statusMessage(text);
  if(!outputs.length){
    const oldRule=(result.observations||[]).some(o=>o.expression?.schema==='chatflypt.expression.v1');
    const node=statusMessage(oldRule?'历史记录使用旧版聊天规则，当时未生成聊天文字。新消息已改为翻译内部神经读出。':(result.reply||'本轮尚无可翻译的神经读出。'));
    if(result.observation){const button=document.createElement('button');button.className='message-replay';button.textContent='查看本轮接收与反应 ↗';
      button.onclick=()=>{if(!busy)setObservationBatch(result.observation,result.observations);};node.append(button);}
  }
  remember(text,outputs.map(x=>x.text).join(''));
}
window.chatFlyEnvironment={
  busy:()=>busy,
  pause:()=>{state.playing=false;window.flyPhysics?.pause();},
  apply:async(objects,revision)=>{
    if(busy)throw new Error('请等待当前仿真完成');
    setBusy(true);const event=statusMessage('环境改动 · 正在应用…',true);
    try{
      const result=await api('/api/environment/apply',{objects,revision,seed:nextSeed(),body_context:bodyContext(),...dialogueOptions()});
      event.classList.remove('pending');event.textContent='环境改动 · '+result.environment.objects.length+' 个物品 · 第 '+result.environment.revision+' 版 · 已运行感觉与神经响应';
      finishCommunication(result,'环境改动',{root:document.createElement('div')});
      return result.environment;
    }catch(error){event.classList.remove('pending');event.textContent='环境改动未应用 · '+error.message;throw error;}
    finally{setBusy(false);}
  }
};
async function sendMessage(text){
  if(busy||!text.trim())return;setBusy(true);$('message').value='';appendMessage('user',text);
  const reply=appendMessage('assistant','正在处理这条消息…','ChatFlyPT',true);
  try{
    const r=await api('/api/communicate',{message:text,seed:nextSeed(),history,observation_id:observation?.id||'',body_context:bodyContext(),...dialogueOptions()});
    finishCommunication(r,text,reply);
  }catch(e){finishMessage(reply,e.message,'未完成');$('error').textContent='本次请求未完成，请重试。';}
  finally{setBusy(false);$('message').focus();}
}
async function runObservation(words){if(busy)return;setBusy(true);$('settingsDialog').close();const title=words.length?'直接神经干预实验：'+words.map(w=>names[w]).join('、'):'运行无输入对照';appendMessage('user',title);const reply=appendMessage('assistant','正在运行神经仿真与对话…','神经实验',true);try{finishCommunication(await api('/api/converse',{kind:'words',words,seed:nextSeed(),body_context:bodyContext(),...dialogueOptions()}),title,reply);}catch(e){finishMessage(reply,e.message,'实验未完成');}finally{setBusy(false);}}
function detailRow(label,text){const row=document.createElement('div');row.className='detail-row';const labelEl=document.createElement('strong');labelEl.textContent=label;row.append(labelEl,document.createTextNode(text));$('detailsContent').append(row);return row;}
function showDetails(evidence=false){
  if(window.flyPhysics?.active()){window.flyPhysics.details();return;}
  $('detailsContent').replaceChildren();$('detailsTitle').textContent=evidence?'本次实验记录':'信息的接收通路';
  if(inputPlan)detailRow('输入计划',JSON.stringify(inputPlan,null,2));
  if(evidence&&currentDialogue)detailRow('神经读出记录',JSON.stringify(currentDialogue.dialogue,null,2)+'\n完整记录：'+currentDialogue.dialogue_record);
  if(!reception){if(inputPlan)$('detailsDialog').showModal();return;}
  detailRow('人类信息 → 感觉输入',reception.intent+' → '+reception.description);
  detailRow('输入状态',displayLimitation(reception.detail)+(reception.direction_mapping_proposed?' 方向词与触碰的对应是待验证方案。':''));
  detailRow('神经入口',reception.dataset+'\n'+reception.neuron_types.map(x=>names[x]||x).join(' / ')+'\n'+reception.neuron_count+' 个细胞'+(reception.input_events==null?'':' · '+reception.input_events+' 个输入事件'));
  if(reception.items?.length)detailRow('各输入通道',reception.items.map(item=>receptionItemLabel(item)+' · '+(inputStates[item.status]||item.status||'')+(Number.isFinite(item.neuron_count)?' · '+item.neuron_count+' 个细胞':'')+(item.notes?'\n'+listText(item.notes):'')).join('\n\n'));
  if(reception.limitations?.length)detailRow('输入限制',listText(reception.limitations));
  const originals=[...new Set([...(reception.limitations||[]),...(reception.items||[]).flatMap(item=>Array.isArray(item.notes)?item.notes:item.notes?[item.notes]:[])])].filter(note=>limitationCopy.has(note));
  if(originals.length){const row=detailRow('限制原文',''),details=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre');summary.textContent='查看英文原文';pre.textContent=originals.join('\n\n');details.append(summary,pre);row.append(details);}
  if(observationBatch.length>1)detailRow('独立试次',observationBatch.map((j,i)=>(i+1)+'. '+datasetName(j.scene)+' · '+j.duration_ms+' ms').join('\n')+'\n各记录分别仿真，未将这些连接组拼接为同一神经网络。');
  if(reception.source){const row=detailRow('实验依据','');const a=document.createElement('a');a.href=reception.source;a.textContent='查看原始研究 ↗';a.target='_blank';a.rel='noopener noreferrer';row.append(a);}
  if(evidence&&observation){
    const j=observation,decoded=j.decoded||{};
    detailRow('神经活动记录',j.spikes.toLocaleString()+' 次脉冲 · '+j.active_neurons.toLocaleString()+' 个活跃细胞 · '+j.duration_ms+' ms\n仿真耗时 '+j.latency_seconds.toFixed(2)+' s · 随机种子 '+j.seed);
    detailRow('神经图说明',datasetName(j.scene)+' · 显示 '+j.scene.sampled_neurons+' 个采样细胞。\n'+(j.scene.geometry_note||'')+'\n高亮来自本次脉冲记录；连线取自连接表。'+(j.scene.voltage_note?'\n'+j.scene.voltage_note:''));
    detailRow('内部活动 → 解码标签',(decoded.words||[]).map(w=>decoded.descriptions?.[w]||names[w]||w).join('、')||'未识别到支持的感觉标签');
    detailRow('读出分数类型',decoded.score_type||'服务端未提供分数类型；不作概率解释。');
    if(decoded.scores)detailRow('各标签读出分数',Object.entries(decoded.scores).map(([word,value])=>(decoded.descriptions?.[word]||names[word]||word)+'：'+(typeof value==='number'?value.toFixed(4):String(value))).join('\n'));
    if(decoded.thresholds)detailRow('读出阈值',JSON.stringify(decoded.thresholds,null,2));
    if(j.sensation)detailRow('感觉读出（接收）',j.sensation.text);if(j.expression)detailRow('神经读出表达',j.expression.text||j.expression.reason);if(j.scene?.command)detailRow('自身转向指令（候选，命令读出）',JSON.stringify({text:j.scene.command.text,status:j.scene.command.status,prediction:j.scene.command.prediction,limitations:j.scene.command.limitations},null,2));if(j.environment_sample)detailRow('环境采样',JSON.stringify({revision:j.environment_sample.environment_revision,evidence:j.environment_sample.evidence,unmapped:j.environment_sample.unmapped},null,2));
    if(currentDialogue?.goal_evaluation)detailRow('目标验收（实际轨迹，不作为控制输入）',JSON.stringify(currentDialogue.goal_evaluation,null,2));
    if(j.embodied_feedback)detailRow('本轮持续感觉反馈',JSON.stringify({interval_ms:j.embodied_feedback.interval_ms,
      session_id:j.embodied_feedback.session_id,turn:j.embodied_feedback.turn_index,
      session_start_ms:j.embodied_feedback.session_start_ms,session_end_ms:j.embodied_feedback.session_end_ms,
      paused_between_messages:j.embodied_feedback.paused_between_messages,session_limit_ms:j.embodied_feedback.session_limit_ms,
      neural_resets:j.embodied_feedback.neural_reset_count,body_resets:j.embodied_feedback.body_reset_count,
      between_messages:j.embodied_feedback.between_messages,record:j.embodied_feedback.output_dir,
      environment_update:j.embodied_feedback.environment_update,
      samples:j.embodied_feedback.samples.map(s=>({time_ms:s.time_ms,position_mm:s.sensor_pose.position_mm,
        vision_sampled_at_ms:s.vision?.sampled_at_ms,channels:s.items.map(receptionItemLabel),unmapped:s.unmapped,
        proprioception:s.proprioception?{events:s.proprioception.encoded_event_count,applied:s.proprioception.neural_feedback_applied,
          source:s.proprioception.activation_source,calibrated:s.proprioception.biologically_calibrated,
          peak_rate_hz:s.proprioception.engineering_peak_rate_hz}:null}))},null,2));
    if(j.response_text)detailRow('技术读出记录',j.response_text);
    if(j.reaction)detailRow('反应汇总',JSON.stringify(j.reaction,null,2));
    detailRow('运动细胞记录',JSON.stringify(j.scene.motor_evidence||[],null,2));
    const body=j.scene.body||{};
    if(body.motor_coverage)detailRow('运动连接覆盖（累计脉冲）',JSON.stringify(body.motor_coverage,null,2));
    if(body.joint_command_summary_rad)detailRow('腿部关节输入（弧度，未经生理标定）',JSON.stringify(body.joint_command_summary_rad,null,2));
    detailRow('身体物理仿真',JSON.stringify({status:body.status,controller:body.controller,context:body.context,source_mapping:body.source_mapping,initialization:body.initialization,command_summary:body.command_summary,engine:body.engine,controls:body.controls,summary:body.summary},null,2)+'\n'+(body.limitations||[]).join('\n'));
  }
  const row=detailRow('目标神经元 ID',''),pre=document.createElement('pre');pre.textContent=reception.neuron_ids.join('\n');row.append(pre);$('detailsDialog').showModal();
}
function fit(canvas,ctx){const box=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,2),w=Math.max(1,box.width),h=Math.max(1,box.height);if(canvas.width!==Math.round(w*ratio)||canvas.height!==Math.round(h*ratio)){canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);}ctx.setTransform(ratio,0,0,ratio,0,0);return [w,h];}
function projectBrainNodes(nodes,w,h){
  const ca=Math.cos(state.angle),sa=Math.sin(state.angle),ct=Math.cos(state.tilt),st=Math.sin(state.tilt);
  const projected=nodes.map(n=>{
    const x=n[1]*ca+n[3]*sa,z=-n[1]*sa+n[3]*ca,y=n[2]*ct-z*st,zz=n[2]*st+z*ct;
    const perspective=1/(1+zz*.12);return [x*perspective,-y*perspective,zz];
  });
  if(!projected.length)return [];
  let scale=Math.min(w*.40,h*.84)*state.zoom,cx=w*.51,cy=h*.52,centerX=0,centerY=0;
  if(datasetName(state.scene)==='MaleCNS v1.0'){
    // Fit the rotated brain and VNC together; leave room for labels and the legend.
    let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
    for(const [x,y] of projected){minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);}
    const side=Math.min(24,w*.08),top=Math.min(70,h*.32),bottom=Math.min(28,h*.14);
    const availableWidth=Math.max(1,w-side*2),availableHeight=Math.max(1,h-top-bottom);
    scale=Math.min(availableWidth/Math.max(maxX-minX,1e-6),availableHeight/Math.max(maxY-minY,1e-6))*state.zoom;
    centerX=(minX+maxX)/2;centerY=(minY+maxY)/2;cx=w/2;cy=top+availableHeight/2;
  }
  return projected.map(([x,y,z])=>[cx+(x-centerX)*scale,cy+(y-centerY)*scale,z]);
}
function drawBrain(){const [w,h]=fit(brain,bc);bc.clearRect(0,0,w,h);if(!state.scene)return;const nodes=state.scene.nodes;state.counts.fill(0);if(state.overview)state.counts.set(state.totals);else{const frame=state.scene.frames[Math.min(state.scene.frames.length-1,Math.floor(state.progress*state.scene.frames.length))]||[];for(let i=0;i<frame.length;i+=2)state.counts[frame[i]]=frame[i+1];}state.projected=projectBrainNodes(nodes,w,h);for(const [a,b] of state.scene.edges){const p=state.projected[a],q=state.projected[b];if(!p||!q)continue;const active=state.counts[a]>0&&state.counts[b]>0;bc.strokeStyle=active?'rgba(218,173,119,.16)':'rgba(110,152,123,.025)';bc.lineWidth=.5;bc.beginPath();bc.moveTo(p[0],p[1]);bc.lineTo(q[0],q[1]);bc.stroke();}const ordered=nodes.map((_,i)=>i).sort((a,b)=>state.projected[b][2]-state.projected[a][2]);for(const i of ordered){const p=state.projected[i],n=nodes[i],count=state.counts[i],active=count>0;const color=n[4]===3?'195,232,162':n[4]===2?'183,166,231':'241,167,118';if(active){const power=Math.min(1,Math.log1p(count)/(state.overview?5:2.5));const glow=bc.createRadialGradient(p[0],p[1],0,p[0],p[1],2.5+power*2.5);glow.addColorStop(0,'rgba('+color+','+(.13+power*.16)+')');glow.addColorStop(1,'rgba('+color+',0)');bc.fillStyle=glow;bc.beginPath();bc.arc(p[0],p[1],2.5+power*2.5,0,Math.PI*2);bc.fill();bc.fillStyle='rgba('+color+',.94)';bc.beginPath();bc.arc(p[0],p[1],.7+power*.65,0,Math.PI*2);bc.fill();}else{bc.fillStyle='rgba(119,169,139,'+(.17+(1-Math.min(1,(p[2]+1)/2))*.22)+')';bc.beginPath();bc.arc(p[0],p[1],.65,0,Math.PI*2);bc.fill();}}}
function ellipse(ctx,x,y,rx,ry,rot,fill,stroke){ctx.beginPath();ctx.ellipse(x,y,rx,ry,rot,0,Math.PI*2);if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=.8;ctx.stroke();}}
function drawVoltagePeaks(){
  if(!state.scene?.voltage_peaks)return;
  for(const [i,positive,negative] of state.scene.voltage_peaks){
    const p=state.projected[i];if(!p||state.scene.nodes[i][4]===3)continue;
    const power=Math.min(1,Math.log1p(Math.max(positive,negative))/3);
    bc.strokeStyle=negative>positive?'rgba(173,134,232,'+(.25+.55*power)+')':'rgba(102,211,201,'+(.25+.55*power)+')';
    bc.lineWidth=.65;bc.beginPath();bc.arc(p[0],p[1],1.4+power*1.7,0,Math.PI*2);bc.stroke();
  }
}
function drawWorld(){
  if(window.flyPhysics?.draw())return;
  const time=state.overview?Infinity:state.progress*(state.scene?.duration_ms||0);
  window.flyBodyView?.update(state.scene?.body||null,time);
}
function tick(now){const dt=Math.max(0,Math.min(100,now-(state.last||now)));window.flyPhysics?.tick(dt);state.last=now;if(state.playing&&state.scene?.duration_ms){state.progress=Math.min(1,state.progress+dt/(state.scene.duration_ms*14));if(state.progress>=1){state.playing=false;state.overview=true;icon($('playback'),'play');$('playback').setAttribute('aria-label','重新播放神经活动');$('replayLabel').textContent='整段概览';}$('timeline').value=Math.round(state.progress*1000);$('time').textContent=Math.round(state.progress*state.scene.duration_ms)+' ms';}showFeedbackEyes();drawBrain();drawVoltagePeaks();drawWorld(now);requestAnimationFrame(tick);}
$('chatForm').addEventListener('submit',e=>{e.preventDefault();sendMessage($('message').value.trim());});$('message').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();sendMessage(e.target.value.trim());}});document.querySelectorAll('[data-message]').forEach(b=>b.onclick=()=>sendMessage(b.dataset.message));$('example').onclick=()=>window.flyEnvironmentEditor?.example();$('settings').onclick=()=>$('settingsDialog').showModal();document.querySelectorAll('.close-dialog').forEach(b=>b.onclick=()=>b.closest('dialog').close());$('runExperiment').onclick=()=>runObservation([...document.querySelectorAll('#stimuli input:checked')].map(x=>x.value));$('routeDetail').onclick=()=>showDetails(false);$('evidenceButton').onclick=()=>showDetails(true);$('resetView').onclick=()=>{state.angle=.08;state.tilt=-.10;state.zoom=1;};
$('playback').onclick=()=>{if(!state.scene?.frames.length)return;if(state.progress>=1)state.progress=0;state.overview=false;state.playing=!state.playing;state.last=performance.now();$('replayLabel').textContent='14倍慢放';if(state.playing){$('playback').textContent='Ⅱ';$('playback').setAttribute('aria-label','暂停神经活动');}else{icon($('playback'),'play');$('playback').setAttribute('aria-label','播放神经活动');}};$('timeline').oninput=e=>{state.progress=Number(e.target.value)/1000;state.playing=false;state.overview=false;icon($('playback'),'play');$('playback').setAttribute('aria-label','播放神经活动');$('time').textContent=Math.round(state.progress*(state.scene?.duration_ms||0))+' ms';$('replayLabel').textContent='时间切片';};
$('newChat').onclick=async()=>{if(busy)return;setBusy(true);try{await api('/api/session/reset',{session_id:liveSessionId});liveSessionId=crypto.randomUUID().replaceAll('-','');sessionStorage.setItem('chatflypt-live-session',liveSessionId);}catch(e){$('error').textContent=e.message;return;}finally{setBusy(false);}$('receptionSensation').hidden=true;$('eyeVision').hidden=true;history=[];inputPlan=null;currentDialogue=null;setObservationBatch(null);reception=null;$('conversation').replaceChildren(welcome.cloneNode(true));setScene(resting);$('brainState').textContent='等待输入';$('brainState').classList.remove('pending');$('brainMetrics').textContent='尚无活动记录';$('intent').textContent='人类语言';$('stimulusDescription').textContent='环境呈现与实际接收会显示在这里';$('targetDescription').textContent='物理环境 → 实测感受器 → 神经活动 → 盲读出';$('routeState').textContent='等待消息';$('routeState').classList.remove('amber');$('routeNote').textContent='每一条消息，都保留它进入神经系统的方式。';$('inputTrace').firstElementChild.setAttribute('d','M0 13H120');$('worldCue').hidden=true;$('worldState').textContent='等待运动活动';$('bodyNote').textContent='官方身体初始姿态 · 尚无动作记录';$('motorEvidence').textContent='';$('time').textContent='0 ms';$('routeDetail').disabled=true;$('evidenceButton').disabled=true;$('message').value='';$('error').textContent='';};
let drag=null;brain.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];brain.setPointerCapture(e.pointerId);$('neuronTip').hidden=true;});brain.addEventListener('pointerup',()=>drag=null);brain.addEventListener('pointercancel',()=>drag=null);brain.addEventListener('pointermove',e=>{if(drag){state.angle+=(e.clientX-drag[0])*.008;state.tilt=Math.max(-1,Math.min(1,state.tilt+(e.clientY-drag[1])*.006));drag=[e.clientX,e.clientY];return;}const box=brain.getBoundingClientRect(),x=e.clientX-box.left,y=e.clientY-box.top;let best=-1,distance=36;state.projected.forEach((p,i)=>{const d=(p[0]-x)**2+(p[1]-y)**2;if(d<distance){distance=d;best=i;}});const tip=$('neuronTip');tip.hidden=best<0;if(best>=0){const n=state.scene.nodes[best];tip.textContent=n[5]+'\n'+n[0]+(n[6]?'\n'+(n[6]==='anatomical'?'解剖坐标':n[6]==='schematic'?'示意位置 · 非解剖坐标':n[6]):'')+'\n'+(state.overview?'整段':'当前时间片')+'：'+(state.counts[best]||0)+' 次脉冲';tip.style.left=Math.min(x+10,Math.max(0,box.width-190))+'px';tip.style.top=Math.max(0,y-57)+'px';}});brain.addEventListener('pointerleave',()=>$('neuronTip').hidden=true);brain.addEventListener('wheel',e=>{e.preventDefault();state.zoom=Math.max(.55,Math.min(2,state.zoom-e.deltaY*.001));},{passive:false});
function inputChannel(id){return inputChannelById.get(id);}
function isAdvancedChannel(channel){return channel.advanced===true||channel.tier==='advanced';}
function groupedInputChannels(channels){
  const groups=new Map();
  for(const channel of channels){const key=channel.modality||'其他';if(!groups.has(key))groups.set(key,[]);groups.get(key).push(channel);}
  return [...groups];
}
function renderDatasetCoverage(coverage){
  const container=$('inputCoverageDatasets');container.replaceChildren();
  if(!coverage||typeof coverage!=='object')return;
  const metrics=[['sensory_cells_in_engine','引擎内感觉细胞'],['accounted_cells','纳入覆盖统计'],['named_channel_cells','具名通道覆盖'],['executable_cells','可执行输入细胞'],['not_executable_cells','暂不可执行细胞'],['unassigned_entry_side_cells','入口或侧别待分配']];
  for(const [dataset,record] of Object.entries(coverage)){
    if(!record||typeof record!=='object')continue;
    const card=document.createElement('details');card.className='input-dataset-coverage';
    const summary=document.createElement('summary');summary.textContent=datasetName({dataset})+(Number.isFinite(record.sensory_cells_in_engine)?' · '+record.sensory_cells_in_engine.toLocaleString()+' 个感觉细胞':' · 细胞覆盖统计');card.append(summary);
    const values=document.createElement('dl');values.className='input-coverage-metrics';
    for(const [key,label] of metrics){if(!Number.isFinite(record[key]))continue;const term=document.createElement('dt'),value=document.createElement('dd');term.textContent=label;value.textContent=record[key].toLocaleString()+' 个';values.append(term,value);}
    card.append(values);
    const counts=record.channel_status_counts,parts=[];
    if(counts&&typeof counts==='object')for(const [status,count] of Object.entries(counts))if(Number.isFinite(count))parts.push((inputStates[status]||status)+' '+count.toLocaleString()+' 条');
    if(Number.isFinite(record.advanced_channel_count))parts.push('高级通道 '+record.advanced_channel_count.toLocaleString()+' 条');
    if(parts.length){const channels=document.createElement('p');channels.className='input-coverage-channel-counts';channels.textContent=parts.join(' · ');card.append(channels);}
    container.append(card);
  }
}
function appendChannelValidation(card,channel){
  const validation=channel.validation||{},tested=validation.delivery_tested===true;
  const spikes=validation.non_sensory_spikes,hasCount=Number.isFinite(spikes);
  const result=document.createElement('div');result.className='input-channel-validation';
  const response=document.createElement('p');response.className='input-response-state';
  const responseState=validation.electrical_response===true?'response':!tested?'pending':hasCount&&spikes>0?'response':'none';response.dataset.state=responseState;
  response.textContent='默认刺激：'+(responseState==='response'?'有内部响应':responseState==='none'?'未记录内部响应':'待测试');result.append(response);
  if(tested){
    const delivery=document.createElement('p');delivery.className='input-validation-detail';
    delivery.textContent=(validation.electrical_response===true?'已记录内部电响应':validation.electrical_response===false?'未记录内部电响应':'脉冲传播检查：'+(validation.input_delivery_passed===true?'通过':validation.input_delivery_passed===false?'未通过':'待测'))+(hasCount?' · 非感觉细胞 '+spikes.toLocaleString()+' 次脉冲':'')+(Number.isFinite(validation.membrane_internal_cells)?' · '+validation.membrane_internal_cells+' 个细胞电位变化':'');result.append(delivery);
  }
  const scores=validation.readout_test_f1||{},pairs=['left','right'].filter(side=>Number.isFinite(scores[side]));
  const readout=document.createElement('p');readout.className='input-readout-status';
  if(pairs.length){
    readout.textContent=pairs.map(side=>sideLabel(side)+'测试 F1 '+scores[side].toFixed(3)).join(' · ');result.append(readout);
    const note=document.createElement('p');note.className='input-validation-detail';note.textContent='仿真试次指标，非自然感知准确率。';result.append(note);
  }else{
    readout.textContent=validation.readout_status==='calibrated'?'已训练语义读出；暂无此通道的侧别 F1':isAdvancedChannel(channel)||validation.readout_status==='uncalibrated'?'未训练语义读出':'语义读出待测试';result.append(readout);
  }
  const withheld=['left','right'].filter(side=>validation.qualified_sides?.[side]===false);
  if(withheld.length){const note=document.createElement('p');note.className='input-validation-detail';note.textContent=withheld.map(sideLabel).join('、')+'未通过独立单通道读出验证，不赋予感觉含义。';result.append(note);}
  card.append(result);
}
function inputChannelCard(channel){
  const card=document.createElement('article');card.className='input-channel-card';
  const heading=document.createElement('div');heading.className='input-channel-heading';
  const label=document.createElement('strong');label.textContent=channel.label||channel.id;
  const status=document.createElement('span');status.className='input-status';status.dataset.status=channel.status;status.textContent=inputStates[channel.status]||channel.status;
  heading.append(label,status);card.append(heading);
  const meta=document.createElement('p');meta.className='input-channel-meta';meta.textContent=(channel.dataset?datasetName({dataset:channel.dataset}):'数据集待映射')+' · '+((channel.sides||[]).map(sideLabel).join(' / ')||'侧别待映射');card.append(meta);
  const id=document.createElement('code');id.textContent=channel.id;card.append(id);
  appendChannelValidation(card,channel);
  if(channel.limitations?.length){const notes=document.createElement('ul');for(const note of channel.limitations){const li=document.createElement('li');li.textContent=displayLimitation(note);if(li.textContent!==note)li.title=note;notes.append(li);}card.append(notes);}
  if(typeof channel.source==='string'&&/^https?:\/\//i.test(channel.source)){const link=document.createElement('a');link.href=channel.source;link.textContent='通路依据 ↗';link.target='_blank';link.rel='noopener noreferrer';card.append(link);}
  return card;
}
function renderInputCoverage(){
  const query=$('inputCoverageSearch').value.trim().toLocaleLowerCase();
  const visible=inputCatalogue.filter(c=>(inputSearchIndex.get(c.id)||'').includes(query));
  const container=$('inputCoverageList');container.replaceChildren();
  if(!visible.length){const empty=document.createElement('p');empty.className='muted';empty.textContent='没有匹配的已注册通道。';container.append(empty);return;}
  for(const [modality,channels] of groupedInputChannels(visible)){
    for(const advanced of [false,true]){
      const entries=channels.filter(c=>isAdvancedChannel(c)===advanced);if(!entries.length)continue;
      const group=document.createElement('details');group.className='input-modality';group.open=!!query||(!advanced&&container.childElementCount===0);
      const summary=document.createElement('summary');summary.textContent=(modalityNames[modality]||modality)+(advanced?' · 高级通道':'')+' · '+entries.length+' 项';group.append(summary);
      const cards=document.createElement('div');cards.className='input-channel-cards';
      let rendered=false;
      const populate=()=>{if(!group.open||rendered)return;const fragment=document.createDocumentFragment();for(const channel of entries)fragment.append(inputChannelCard(channel));cards.append(fragment);rendered=true;};
      group.addEventListener('toggle',populate);populate();
      group.append(cards);container.append(group);
    }
  }
}
function addStimulusRow(channelId){
  const row=document.createElement('div');row.className='stimulus-row';
  const fields=document.createElement('div');fields.className='stimulus-row-fields';
  const field=(labelText,control)=>{const label=document.createElement('label');label.textContent=labelText;label.append(control);fields.append(label);return control;};
  const channelSelect=document.createElement('select');channelSelect.className='stimulus-channel';channelSelect.setAttribute('aria-label','感觉通道');
  for(const [modality,channels] of groupedInputChannels(inputCatalogue)){
    const group=document.createElement('optgroup');group.label=modalityNames[modality]||modality;
    for(const channel of channels){const option=document.createElement('option');option.value=channel.id;option.textContent=(channel.label||channel.id)+' · '+(inputStates[channel.status]||channel.status)+(isAdvancedChannel(channel)?' · 高级':'');group.append(option);}
    channelSelect.append(group);
  }
  field('通道',channelSelect);
  channelSelect.value=channelId||(inputCatalogue.find(c=>c.status==='ready'&&!isAdvancedChannel(c))||inputCatalogue.find(c=>c.status!=='unmapped')||inputCatalogue[0])?.id||'';
  const side=document.createElement('select');side.className='stimulus-side';side.setAttribute('aria-label','接收侧别');field('侧别',side);
  const intensity=document.createElement('input');intensity.className='stimulus-intensity';intensity.type='number';intensity.min='0';intensity.max='2';intensity.step='0.1';intensity.value='1';intensity.required=true;intensity.setAttribute('aria-label','输入强度，事件率倍率');field('强度 ×',intensity);
  const duration=document.createElement('input');duration.className='stimulus-duration';duration.type='number';duration.min='50';duration.max='2000';duration.step='1';duration.value='300';duration.required=true;duration.setAttribute('aria-label','输入持续时间，毫秒');field('持续 ms',duration);
  const remove=document.createElement('button');remove.type='button';remove.className='stimulus-remove';remove.textContent='移除';remove.onclick=()=>{row.remove();updateStimulusAvailability();};fields.append(remove);
  const note=document.createElement('p');note.className='stimulus-row-note';row.append(fields,note);
  channelSelect.onchange=()=>updateStimulusRow(row);
  $('stimulusRows').append(row);updateStimulusRow(row);
}
function updateStimulusRow(row){
  const channel=inputChannel(row.querySelector('.stimulus-channel').value),side=row.querySelector('.stimulus-side');side.replaceChildren();
  const sides=channel?.sides||[];
  for(const value of sides){const option=document.createElement('option');option.value=value;option.textContent=sideLabel(value);side.append(option);}
  if(!sides.length){const option=document.createElement('option');option.value='';option.textContent='侧别待映射';side.append(option);}
  if(sides.includes('bilateral'))side.value='bilateral';
  row.querySelector('.stimulus-row-note').textContent=channel?[(inputStates[channel.status]||channel.status),channel.dataset?datasetName({dataset:channel.dataset}):'',listText(channel.limitations)].filter(Boolean).join(' · '):'请选择已注册的感觉通道。';
  row.querySelector('.stimulus-row-note').title=rawListText(channel?.limitations);
  updateStimulusAvailability();
}
function updateStimulusAvailability(){
  const rows=[...$('stimulusRows').children],unavailable=rows.some(row=>{const channel=inputChannel(row.querySelector('.stimulus-channel').value);return !channel||channel.status==='unmapped'||!channel.sides?.length;});
  $('addStimulus').disabled=busy||inputCatalogueLoading||!inputCatalogue.length;
  $('runStimulus').disabled=busy||inputCatalogueLoading||!rows.length||unavailable;
  for(const row of rows){row.querySelectorAll('input,select,button').forEach(control=>control.disabled=busy||inputCatalogueLoading);if(!inputChannel(row.querySelector('.stimulus-channel').value)?.sides?.length)row.querySelector('.stimulus-side').disabled=true;}
  $('inputFormStatus').textContent=unavailable?'包含尚未映射的通道或缺少已确认的接收侧别；请移除或更换后再运行。':rows.some(row=>inputChannel(row.querySelector('.stimulus-channel').value)?.status==='candidate')?'包含候选通路，请结合下方通道限制解读结果。':!rows.length?'添加通道后可运行输入。':'';
}
async function openInputChannels(){
  if(busy||inputCatalogueLoading)return;
  clearTimeout(inputSearchTimer);$('inputChannelsDialog').showModal();$('inputCoverageSearch').value='';$('inputCoverageList').replaceChildren();$('inputCoverageDatasets').replaceChildren();$('stimulusRows').replaceChildren();
  inputCatalogueLoading=true;inputCatalogue=[];inputChannelById.clear();inputSearchIndex.clear();updateStimulusAvailability();$('inputCoverageSummary').textContent='正在读取通道…';
  try{
    const result=await api('/api/inputs');inputCatalogue=(result.channels||[]).filter(c=>c&&typeof c.id==='string');
    inputChannelById=new Map(inputCatalogue.map(channel=>[channel.id,channel]));
    inputSearchIndex=new Map(inputCatalogue.map(c=>[c.id,[c.id,c.label,c.modality,modalityNames[c.modality],...(c.limitations||[]),...(c.limitations||[]).map(displayLimitation)].join(' ').toLocaleLowerCase()]));
    const counts=Object.fromEntries(Object.keys(inputStates).map(status=>[status,inputCatalogue.filter(c=>c.status===status).length]));
    $('inputCoverageSummary').textContent='共 '+inputCatalogue.length+' 条通道 · '+counts.ready+' 已接入 · '+counts.candidate+' 候选 · '+counts.unmapped+' 尚未映射';
    renderDatasetCoverage(result.coverage);
    renderInputCoverage();if(inputCatalogue.length)addStimulusRow();
  }catch(e){$('inputCoverageSummary').textContent=e.message;}
  finally{inputCatalogueLoading=false;updateStimulusAvailability();}
}
async function runStimulus(){
  if(busy||inputCatalogueLoading||!$('inputStimulusForm').reportValidity())return;
  const rows=[...$('stimulusRows').children];
  if(!rows.length||rows.some(row=>{const channel=inputChannel(row.querySelector('.stimulus-channel').value);return !channel||channel.status==='unmapped'||!channel.sides?.includes(row.querySelector('.stimulus-side').value);}))return;
  const items=rows.map(row=>({channel:row.querySelector('.stimulus-channel').value,side:row.querySelector('.stimulus-side').value,intensity:Number(row.querySelector('.stimulus-intensity').value),onset_ms:50,duration_ms:Number(row.querySelector('.stimulus-duration').value)}));
  const title='直接神经干预实验：'+items.map(item=>(inputChannel(item.channel)?.label||item.channel)+' · '+sideLabel(item.side)+' · '+item.intensity+'× · '+item.duration_ms+' ms').join('；');
  setBusy(true);$('inputChannelsDialog').close();appendMessage('user',title);
  const reply=appendMessage('assistant','正在运行所选感觉输入并读出神经响应…','神经实验',true);
  try{finishCommunication(await api('/api/converse',{kind:'stimuli',items,seed:nextSeed(),body_context:bodyContext(),...dialogueOptions()}),title,reply);}
  catch(e){finishMessage(reply,e.message,'实验未完成');$('error').textContent='本次输入未完成，请检查通道与参数后重试。';}
  finally{setBusy(false);}
}
$('inputChannels').onclick=openInputChannels;
$('inputCoverageSearch').oninput=()=>{clearTimeout(inputSearchTimer);inputSearchTimer=setTimeout(renderInputCoverage,120);};
$('inputChannelsDialog').addEventListener('close',()=>clearTimeout(inputSearchTimer));
$('addStimulus').onclick=()=>addStimulusRow();
$('inputStimulusForm').addEventListener('submit',e=>{e.preventDefault();runStimulus();});
$('observationSelect').onchange=()=>{const selected=observationBatch[Number($('observationSelect').value)];if(selected)showObservation(selected);};
function apiSettingsResult(text='',kind=''){
  $('apiSettingsResult').textContent=text;$('apiSettingsResult').dataset.kind=kind;
}
function setApiSettingsBusy(value){
  apiSettingsBusy=value;$('apiSettings').disabled=value;
  $('apiSettingsDialog').setAttribute('aria-busy',String(value));
  $('apiSettingsDialog').querySelectorAll('button,input,select').forEach(el=>el.disabled=value);
}
function showPublicApiSettings(settings){
  $('apiBaseUrl').value=settings.base_url||'';$('apiModel').value=settings.model||'';
  $('apiStyle').value=settings.api_style==='chat_completions'?'chat_completions':'responses';
  $('apiKey').value='';$('apiKey').placeholder=settings.key_configured?'已配置密钥；留空保留':'填写 API 密钥';
  $('apiKeyHint').textContent=(settings.key_configured?'已有密钥，地址不变时留空保留。':'尚未配置密钥。')+'关闭窗口或保存后会清空此输入框。';
  const storageNotes={'windows-dpapi':'密钥由当前 Windows 账户加密保存。','owner-file':'配置仅保存在本机当前用户可读的文件中。','environment':'当前读取本机环境配置。'};
  $('apiStorage').textContent='保存仅更新独立 API 实验配置，不改变当前果蝇通路。'+(storageNotes[settings.storage]?'\n'+storageNotes[settings.storage]:'');
}
function apiSettingsDraft(){
  return {base_url:$('apiBaseUrl').value.trim(),api_key:$('apiKey').value.trim(),model:$('apiModel').value.trim(),api_style:$('apiStyle').value};
}
async function openApiSettings(){
  if(apiSettingsBusy)return;
  $('apiSettingsForm').reset();$('apiKey').value='';$('apiModelOptions').replaceChildren();
  $('apiKey').placeholder='填写 API 密钥';$('apiKeyHint').textContent='密钥不会回显；关闭窗口或保存后会清空此输入框。';
  $('apiStorage').textContent='保存仅更新独立 API 实验配置，不改变当前果蝇通路。';apiSettingsResult();
  $('apiSettingsDialog').showModal();setApiSettingsBusy(true);apiSettingsResult('正在读取当前设置…','pending');
  try{showPublicApiSettings(await api('/api/settings'));apiSettingsResult();}
  catch(e){apiSettingsResult(e.message,'error');}
  finally{setApiSettingsBusy(false);$('apiBaseUrl').focus();}
}
async function runApiSettingsAction(action){
  if(apiSettingsBusy)return;
  if(!$('apiBaseUrl').reportValidity())return;
  if(action!=='models'&&!$('apiSettingsForm').reportValidity())return;
  const draft=apiSettingsDraft();setApiSettingsBusy(true);
  apiSettingsResult(action==='models'?'正在读取可用模型…':action==='test'?'正在发送测试请求…':'正在保存设置…','pending');
  try{
    const result=await api(action==='save'?'/api/settings':'/api/settings/'+action,draft);
    if(action==='models'){
      const models=[...new Set((result.models||[]).filter(id=>typeof id==='string'&&id))].sort();
      $('apiModelOptions').replaceChildren();
      for(const id of models){const option=document.createElement('option');option.value=id;$('apiModelOptions').append(option);}
      apiSettingsResult(models.length?'已读取 '+models.length+' 个模型。点击模型输入框可选择，也可以手动填写。':'接口没有返回模型列表，可以手动填写模型 ID。',models.length?'success':'');
    }else if(action==='test'){
      if(!result.ok)throw new Error('连接测试未成功，请检查接口地址、密钥、模型和协议。');
      apiSettingsResult('连接成功 · '+result.model+' · '+result.provider_host+(result.reply?'\n'+String(result.reply).slice(0,1000):''),'success');
    }else{
      showPublicApiSettings(result);
      try{await status();apiSettingsResult('设置已保存并生效。','success');}
      catch(e){apiSettingsResult('设置已保存。暂时无法刷新连接状态，请稍后重试。','error');}
    }
  }catch(e){apiSettingsResult(e.message,'error');}
  finally{draft.api_key='';if(action==='save')$('apiKey').value='';setApiSettingsBusy(false);}
}
$('apiSettings').onclick=openApiSettings;
$('apiReadModels').onclick=()=>runApiSettingsAction('models');
$('apiTest').onclick=()=>runApiSettingsAction('test');
$('apiSettingsForm').addEventListener('submit',e=>{e.preventDefault();runApiSettingsAction('save');});
function clearApiSettingsFeedback(e){
  apiSettingsResult();
  if(['apiBaseUrl','apiKey','apiStyle'].includes(e.target.id))$('apiModelOptions').replaceChildren();
}
$('apiSettingsForm').addEventListener('input',clearApiSettingsFeedback);
$('apiSettingsForm').addEventListener('change',clearApiSettingsFeedback);
$('apiSettingsDialog').addEventListener('cancel',e=>{if(apiSettingsBusy)e.preventDefault();});
$('apiSettingsDialog').addEventListener('close',()=>{$('apiKey').value='';$('apiModelOptions').replaceChildren();apiSettingsResult();});
function updateModelLabel(){ $('apiState').textContent=window.flyPhysics?.active()?'环境观察 · 未生成感觉神经输入':'感觉与动作来自神经活动'; }
async function status(){const s=await api('/api/status');lastStatus=s;configured=s.llm_configured;$('backendState').replaceChildren();const dot=document.createElement('i');$('backendState').append(dot,document.createTextNode('神经读出 · 后端就绪'));updateModelLabel();return s;}
$('newChat').addEventListener('click',()=>{currentDialogue=null;});
async function init(){
  try{
    const [s,scene]=await Promise.all([status(),api('/api/scene')]);resting=scene;setScene(scene);
    s.words.forEach(w=>{const l=document.createElement('label'),i=document.createElement('input');i.type='checkbox';i.value=w;l.append(i,document.createTextNode(names[w]||w));$('stimuli').append(l);});
    const live=await api('/api/session?session_id='+encodeURIComponent(liveSessionId));
    const env=await api('/api/environment');
    // Applied geometry comes from the current server scene, never a response
    // restored only for playback. Preserve an already newer editor revision.
    await window.flyEnvironmentEditor?.ready;
    const editor=window.flyEnvironmentEditor;
    if(editor&&editor.applied.revision<=env.revision)editor.acceptMessageEnvironment(env);
    if(live.active&&live.response){
      const context=live.response.observation?.scene?.body?.context;
      if(['ground','tethered_flight'].includes(context))$('bodyContext').value=context;
      statusMessage('已恢复连续会话 · 仿真时钟暂停');
      finishCommunication(live.response,'恢复会话',{root:document.createElement('div')},{replay:true});pauseAtEnd();
    }else if(env.last_record?.kind==='neural'){
      try{
        const response=await api('/api/environment/response?record='+encodeURIComponent(env.last_record.id));
        statusMessage('已恢复历史环境记录 · 第 '+env.revision+' 版');
        for(const record of [...(response.observations||[]),response.observation])if(record)record.archived_environment_record=true;
        finishCommunication(response,'环境改动',{root:document.createElement('div')},{replay:true});
        state.playing=false;icon($('playback'),'play');$('playback').setAttribute('aria-label','播放神经活动');
      }catch(error){$('error').textContent='最近环境响应读取失败：'+error.message;}
    }else if(env.inspection_id){
      try{
        const record=await api('/api/environment/inspection?record='+encodeURIComponent(env.inspection_id));
        window.flyPhysics?.attach(statusMessage('恢复最近的物理记录'),{inspection:record,environment:env,environment_event:{label:'已恢复物理记录'}});
      }catch(error){$('error').textContent='最近物理记录读取失败：'+error.message;}
    }
  }catch(e){$('backendState').textContent='后端未连接';$('error').textContent=e.message;}
  requestAnimationFrame(tick);
}
init();setInterval(()=>status().catch(()=>{}),30000);
