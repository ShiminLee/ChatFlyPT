'use strict';
// Objective physics records have their own timeline, separate from neural trials.
window.flyPhysics=(()=>{
  let record=null,position=0,playing=false;
  const bar=document.createElement('div');bar.id='physicsPlayback';bar.className='playback physics-playback';bar.hidden=true;
  bar.innerHTML='<button id="physicsPlay" aria-label="播放物理观察">▶</button><span id="physicsTime">0 ms</span><input id="physicsTimeline" aria-label="物理观察时间" type="range" min="0" max="1000" value="0"><span id="physicsEnd"></span><button id="physicsDetails">记录与双眼画面 ↗</button>';
  $('bodyViewport').after(bar);
  const run=document.createElement('button');run.id='inspectEnvironment';run.type='button';run.textContent='物理观察';run.title='观察已应用环境中的被动身体、接触与双眼画面';$('environmentToggle').after(run);
  const dialog=document.createElement('dialog');dialog.id='physicsDialog';
  dialog.innerHTML='<div class="dialog-heading"><h2>环境物理记录</h2><button id="closePhysicsDialog" aria-label="关闭物理记录">×</button></div><p id="physicsSummary" class="muted"></p><div class="physics-options"><label>画面时间<select id="physicsCameraTime"></select></label><label>显示方式<select id="physicsCameraMode"><option value="eye_images">双眼原始画面</option><option value="mosaic_images">官方复眼采样</option></select></label></div><div class="physics-eyes"><figure><img id="physicsLeftEye" alt="左眼渲染画面"><figcaption>左眼</figcaption></figure><figure><img id="physicsRightEye" alt="右眼渲染画面"><figcaption>右眼</figcaption></figure></div><p id="physicsCameraInfo" class="fineprint"></p><h3>物品接触事件</h3><div id="physicsEvents"></div><p id="physicsLimitations" class="fineprint"></p><button id="downloadPhysics" class="primary-button">下载完整 JSON 记录</button>';
  document.body.append(dialog);
  function seek(ms){position=Math.max(0,Math.min(record?.summary.duration_ms||0,ms));playing=false;updateControls();}
  function updateControls(){
    $('physicsTime').textContent=Math.round(position)+' ms';$('physicsTimeline').value=position/(record?.summary.duration_ms||1)*1000;
    $('physicsPlay').textContent=playing?'Ⅱ':'▶';$('physicsPlay').setAttribute('aria-label',playing?'暂停物理观察':'播放物理观察');
  }
  function show(next){
    setScene(resting);record=next;position=0;playing=false;
    updateModelLabel();$('routeDetail').setAttribute('aria-label','查看物理观察详情');
    observation=null;reception=null;inputPlan=null;currentDialogue=null;observationBatch=[];$('observationSwitch').hidden=true;
    $('brainState').textContent='本次未运行';$('brainMetrics').textContent='物理观察没有神经活动记录';
    $('receptionSensation').hidden=true;$('eyeVision').hidden=true;
    $('intent').textContent='环境观察';$('stimulusDescription').textContent='接触、身体位姿与双眼画面已记录';
    $('routeState').textContent='物理记录';$('routeState').classList.remove('amber');
    $('targetDescription').textContent=next.text;
    $('routeNote').textContent='本次仅观察物理场景；画面和接触未转换为感觉神经输入。';
    $('inputTrace').firstElementChild.setAttribute('d','M0 13H120');$('routeDetail').disabled=false;$('evidenceButton').disabled=false;
    $('worldCue').hidden=true;$('worldState').textContent='被动物理回放';$('bodyNote').textContent='观察窗口内执行器关闭 · 位移来自物理过程';
    $('motorEvidence').textContent='环境 v'+next.environment.revision;
    window.flyEnvironmentEditor?.showRecordedEnvironment(next.environment);
    bar.hidden=false;$('physicsEnd').textContent=next.summary.duration_ms+' ms · 10倍慢放';updateControls();
  }
  function camera(){
    if(!record)return;const sample=record.camera_samples[Number($('physicsCameraTime').value)||0],mode=$('physicsCameraMode').value;
    $('physicsLeftEye').src=sample.vision[mode].left;$('physicsRightEye').src=sample.vision[mode].right;
    $('physicsCameraInfo').textContent=sample.time_ms+' ms · 每眼 '+sample.vision.ommatidia[0].length+' 小眼 · 分眼平均亮度 L '+sample.vision.mean_by_eye.left.toFixed(4)+' / R '+sample.vision.mean_by_eye.right.toFixed(4)+'；画面未送入神经网络。';
  }
  function details(){
    if(!record)return;$('physicsSummary').textContent='环境 v'+record.environment.revision+' · '+new Date(record.created_at).toLocaleString()+'\n'+record.text;
    $('physicsCameraTime').replaceChildren();record.camera_samples.forEach((sample,i)=>{const option=document.createElement('option');option.value=i;option.textContent=sample.time_ms+' ms';$('physicsCameraTime').append(option);});
    $('physicsEvents').replaceChildren();const objects=new Map(record.environment.objects.map(o=>[o.id,o]));
    for(const event of record.events){const b=document.createElement('button');b.className='physics-event';const obj=objects.get(event.object_id);b.textContent=event.time_ms+' ms · '+(event.kind==='contact_started'?'接触开始':'接触结束')+' · '+(window.flyEnvironmentEditor?.types[obj?.type]?.label||obj?.type||event.object_id)+' / '+event.fly_geom;b.onclick=()=>{seek(event.time_ms);dialog.close();};$('physicsEvents').append(b);}
    if(!record.events.length)$('physicsEvents').textContent='采样帧中未记录到身体与场景物品的接触。';
    $('physicsLimitations').textContent=record.limitations.join('\n');camera();dialog.showModal();
  }
  function attach(event,result){
    const next=result.inspection;event.classList.remove('pending');event.textContent=result.environment_event.label+' · '+result.environment.objects.length+' 个物品 · 第 '+result.environment.revision+' 版';
    const p=document.createElement('p');p.textContent=next.text;event.append(p);
    const changes=result.environment_event.changes||[];
    if(changes.length){const d=document.createElement('details'),s=document.createElement('summary'),text=document.createElement('p');s.textContent='查看 '+changes.length+' 项环境改动';text.textContent=changes.map(c=>({added:'添加',removed:'移除',updated:'修改'}[c.kind])+' '+(window.flyEnvironmentEditor?.types[c.object.type]?.label||c.object.type)+' ('+c.object.position_mm.join(', ')+') mm').join('；');d.append(s,text);event.append(d);}
    const b=document.createElement('button');b.className='message-replay';b.textContent='查看物理记录与双眼画面 ↗';b.onclick=()=>{if(!busy){show(next);details();}};event.append(b);show(next);
  }
  run.onclick=async()=>{
    if(busy)return;setBusy(true);run.disabled=true;const event=statusMessage('物理观察 · 正在记录已应用环境…',true);
    try{const result=await api('/api/environment/inspect',{seed:nextSeed(),body_context:bodyContext()});if(window.flyEnvironmentEditor)window.flyEnvironmentEditor.applied=result.environment;attach(event,result);}
    catch(error){event.classList.remove('pending');event.textContent='物理观察未完成 · '+error.message;}
    finally{setBusy(false);run.disabled=false;}
  };
  $('physicsPlay').onclick=()=>{if(!record)return;if(position>=record.summary.duration_ms)position=0;playing=!playing;updateControls();};
  $('physicsTimeline').oninput=e=>seek(Number(e.target.value)/1000*record.summary.duration_ms);
  $('physicsDetails').onclick=details;$('closePhysicsDialog').onclick=()=>dialog.close();
  $('physicsCameraTime').onchange=()=>{camera();seek(record.camera_samples[Number($('physicsCameraTime').value)].time_ms);};$('physicsCameraMode').onchange=camera;
  $('downloadPhysics').onclick=()=>{if(!record)return;const a=document.createElement('a');a.href='/api/environment/inspection?record='+encodeURIComponent(record.id)+'&download=1';a.download='environment-'+record.id+'.json';document.body.append(a);a.click();a.remove();};
  return {show,details,attach,active:()=>!!record,record:()=>record,
    clear:()=>{record=null;playing=false;bar.hidden=true;updateModelLabel();},pause:()=>{playing=false;updateControls();},
    tick:dt=>{if(playing&&record){position=Math.min(record.summary.duration_ms,position+dt/10);if(position>=record.summary.duration_ms)playing=false;updateControls();}},
    draw:()=>{if(!record)return false;window.flyBodyView?.update(record.body,position);return true;}
  };
})();
