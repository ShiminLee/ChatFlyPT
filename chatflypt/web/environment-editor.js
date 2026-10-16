import * as THREE from 'three';
const $=id=>document.getElementById(id);
const clone=value=>JSON.parse(JSON.stringify(value));
const DRAFT_KEY='chatflypt.environment.draft.v1';

export class EnvironmentEditor{
  constructor(view){
    this.view=view;this.applied={revision:0,objects:[]};this.draft=[];this.types={};this.selected=null;this.opened=false;this.saving=false;
    this.undoStack=[];this.redoStack=[];this.invalidInput=false;
    this.group=new THREE.Group();view.scene.add(this.group);this.ray=new THREE.Raycaster();
    this.panel=$('environmentEditor');
    this.panel.innerHTML=`<div class="environment-heading"><h3>场景物品</h3><button id="environmentClose" aria-label="收起环境编辑器">×</button></div>
      <p class="environment-help">拖动物品调整平面位置；下方编辑 XYZ、尺寸和旋转。半透明物品尚未应用。</p>
      <div class="environment-add"><select id="objectType" aria-label="物品类型"></select><button id="addObject">＋ 添加</button></div>
      <div class="environment-history"><button id="undoEnvironment">撤销</button><button id="redoEnvironment">重做</button></div>
      <div id="objectList" class="object-list" role="listbox" aria-label="场景物品列表"></div>
      <fieldset id="objectFields"><legend id="objectTitle">选择物品</legend>
      <label>位置 · mm</label><div class="xyz-fields">${['x','y','z'].map(axis=>`<label>${axis.toUpperCase()}<input id="object-${axis}" type="number" step="any" min="-40" max="40"></label>`).join('')}</div>
      <label>尺寸 · mm</label><div class="xyz-fields">${['sx','sy','sz'].map((axis,i)=>`<label>${['宽','深','高'][i]}<input id="object-${axis}" type="number" step="any" min="0.05" max="20"></label>`).join('')}</div>
      <div class="environment-pair"><label>旋转 Z · °<input id="object-yaw" type="number" step="any" min="-360" max="360"></label><label>强度 · 模型系数<input id="object-strength" type="number" step="any" min="0" max="2"></label></div>
      <label id="object-range-label">影响半径 · mm<input id="object-range" type="number" step="any" min="0.1" max="30"></label>
      <button id="duplicateObject" class="environment-duplicate">复制此物品</button><button id="deleteObject" class="environment-delete">移除此物品</button></fieldset>
      <p id="environmentStatus" role="status"></p><div class="environment-actions"><button id="discardEnvironment">撤销未应用</button><button id="applyEnvironment">应用环境改动</button></div>
      <details class="environment-notes"><summary>环境模型说明</summary><p>草稿自动保存在当前浏览器。应用后运行环境采样、神经响应和神经驱动的身体回放。“物理观察”另作关闭执行器的机械检查。糖/盐的视觉形状不等于味觉，热源与气味场参数仍是工程设定。</p></details>`;
    $('environmentToggle').onclick=()=>this.toggle();$('environmentClose').onclick=()=>this.toggle(false);
    $('addObject').onclick=()=>this.add($('objectType').value);
    $('discardEnvironment').onclick=()=>{this.checkpoint();this.draft=clone(this.applied.objects);this.selected=null;this.recorded=null;this.refresh();};
    $('undoEnvironment').onclick=()=>this.travelHistory(this.undoStack,this.redoStack);$('redoEnvironment').onclick=()=>this.travelHistory(this.redoStack,this.undoStack);
    $('duplicateObject').onclick=()=>{const obj=this.draft.find(o=>o.id===this.selected);if(obj)this.add(obj.type,{...clone(obj),id:crypto.randomUUID(),position_mm:[Math.min(40,obj.position_mm[0]+1),...obj.position_mm.slice(1)]});};
    $('applyEnvironment').onclick=()=>this.apply();$('deleteObject').onclick=()=>{this.checkpoint();this.draft=this.draft.filter(o=>o.id!==this.selected);this.selected=null;this.refresh();};
    for(const key of ['x','y','z','sx','sy','sz','yaw','strength','range'])$('object-'+key).oninput=()=>this.fieldsChanged();
    const canvas=view.renderer.domElement;
    canvas.addEventListener('pointerdown',e=>this.pointerDown(e),true);
    canvas.addEventListener('pointermove',e=>this.pointerMove(e),true);
    for(const kind of ['pointerup','pointercancel'])canvas.addEventListener(kind,()=>{this.drag=null;view.orbit.enabled=true;});
    this.ready=this.load();
  }
  async load(){
    try{const r=await fetch('/api/environment');if(!r.ok)throw Error('无法读取环境');const data=await r.json();this.types=data.types;this.applied={schema:data.schema,revision:data.revision,objects:data.objects};this.draft=clone(data.objects);
      let restored=false;try{const saved=JSON.parse(localStorage.getItem(DRAFT_KEY)||'null');if(saved?.revision===data.revision&&Array.isArray(saved.objects)&&!this.validationError(saved.objects)){this.draft=clone(saved.objects);this.selected=saved.selected;restored=this.isDirty();}}catch{}
      for(const [key,value] of Object.entries(this.types)){const option=document.createElement('option');option.value=key;option.textContent=value.label;$('objectType').append(option);}this.refresh();
      if(restored)$('environmentStatus').textContent='已恢复本机草稿 · 尚未应用';
      const record=window.flyPhysics?.record();if(record)this.showRecordedEnvironment(record.environment);
    }catch(e){$('environmentStatus').textContent=e.message;}
  }
  toggle(value=!this.opened){
    this.opened=value;this.panel.hidden=!value;this.view.viewport.closest('.world-card').classList.toggle('editing',value);
    $('environmentToggle').textContent=value?'退出编辑':'编辑环境';
    if(value){window.chatFlyEnvironment.pause();this.recorded=null;}
    this.view.resize();this.render();
  }
  isDirty(){return JSON.stringify(this.draft)!==JSON.stringify(this.applied.objects);}
  checkpoint(){this.undoStack.push({objects:clone(this.draft),selected:this.selected});if(this.undoStack.length>50)this.undoStack.shift();this.redoStack=[];}
  travelHistory(from,to){if(this.saving||!from.length)return;to.push({objects:clone(this.draft),selected:this.selected});const snapshot=from.pop();this.draft=snapshot.objects;this.selected=snapshot.selected;this.refresh();}
  persist(){try{if(this.isDirty())localStorage.setItem(DRAFT_KEY,JSON.stringify({revision:this.applied.revision,objects:this.draft,selected:this.selected}));else localStorage.removeItem(DRAFT_KEY);}catch{$('environmentStatus').textContent+=' · 浏览器无法保存草稿';}}
  validationError(objects=this.draft){
    if(objects.length>32)return '最多允许 32 个物品';const ids=new Set();
    for(const obj of objects){
      if(!this.types[obj.type]||typeof obj.id!=='string'||!/^[A-Za-z0-9_-]{1,64}$/.test(obj.id)||ids.has(obj.id))return '物品类型或标识无效';ids.add(obj.id);
      if(!Array.isArray(obj.position_mm)||obj.position_mm.length!==3||obj.position_mm.some(x=>!Number.isFinite(x)||Math.abs(x)>40)||!Array.isArray(obj.size_mm)||obj.size_mm.length!==3||obj.size_mm.some(x=>!Number.isFinite(x)||x<.05||x>20))return '请填写有效的坐标与尺寸';
      if(obj.position_mm[2]<obj.size_mm[2]/2)return '物品底面低于地面：请调整 Z 或高度';
      if(!Number.isFinite(obj.yaw_deg)||Math.abs(obj.yaw_deg)>360||!Number.isFinite(obj.strength)||obj.strength<0||obj.strength>2||!Number.isFinite(obj.range_mm)||obj.range_mm<.1||obj.range_mm>30)return '旋转、强度或半径超出允许范围';
    }return '';
  }
  add(type,overrides={}){
    if(this.saving||this.draft.length>=32)return;
    this.checkpoint();
    const size=type==='block'?[2,2,1]:['light','heat','cold'].includes(type)?[.6,.6,.6]:type==='shade'?[3,3,3]:[1,1,.3];
    const object={id:crypto.randomUUID(),type,position_mm:[3,0,size[2]/2],size_mm:size,yaw_deg:0,strength:1,range_mm:4,...overrides};
    this.draft.push(object);this.selected=object.id;this.refresh();
  }
  async example(){
    await this.ready;if(this.saving||window.chatFlyEnvironment.busy())return;this.toggle(true);
    // Actual scene objects are only drafts until the user applies the scene.
    for(const [type,pos] of [['sugar',[3,-2,.15]],['yeast',[3,2,.15]],['high_salt',[-3,-2,.15]],['heat',[-3,2,.3]],['cold',[-5,0,.3]],['light',[0,0,4]],['shade',[5,3,1.5]],['block',[4,0,.5]]])this.add(type,{position_mm:pos});
  }
  refresh(){
    this.invalidInput=false;
    const list=$('objectList');list.replaceChildren();
    for(const obj of this.draft){const b=document.createElement('button');b.type='button';b.setAttribute('role','option');b.setAttribute('aria-selected',obj.id===this.selected);b.classList.toggle('selected',obj.id===this.selected);b.dataset.objectId=obj.id;
      const applied=this.applied.objects.find(o=>o.id===obj.id);const changed=JSON.stringify(obj)!==JSON.stringify(applied);
      b.textContent=(this.types[obj.type]?.label||obj.type)+(changed?' · 虚影':'');b.onclick=()=>{this.selected=obj.id;this.refresh();};list.append(b);}
    const obj=this.draft.find(o=>o.id===this.selected);$('objectFields').disabled=!obj||this.saving;
    if(obj){$('objectTitle').textContent=this.types[obj.type].label;for(const [i,key] of ['x','y','z'].entries())$('object-'+key).value=obj.position_mm[i];for(const [i,key] of ['sx','sy','sz'].entries())$('object-'+key).value=obj.size_mm[i];$('object-yaw').value=obj.yaw_deg;$('object-strength').value=obj.strength;$('object-range').value=obj.range_mm;
      $('object-range-label').hidden=['block','sugar','high_salt','shade'].includes(obj.type);}
    $('environmentStatus').textContent=this.saving?'正在采样环境并运行神经与身体响应…':this.validationError()||(this.isDirty()?'有未应用的改动 · 虚影不产生刺激或碰撞':'已应用 · 第 '+this.applied.revision+' 版');
    for(const id of ['applyEnvironment','discardEnvironment'])$(id).disabled=this.saving||!this.isDirty();
    for(const id of ['addObject','objectType','deleteObject'])$(id).disabled=this.saving||id==='deleteObject'&&!obj;
    $('applyEnvironment').disabled=this.saving||!this.isDirty()||!!this.validationError();
    $('undoEnvironment').disabled=this.saving||!this.undoStack.length;$('redoEnvironment').disabled=this.saving||!this.redoStack.length;
    $('duplicateObject').disabled=this.saving||!obj||this.draft.length>=32;$('addObject').disabled=this.saving||this.draft.length>=32;
    this.persist();
    this.render();
  }
  fieldsChanged(){
    const obj=this.draft.find(o=>o.id===this.selected);if(!obj||this.saving)return;
    const keys=['x','y','z','sx','sy','sz','yaw','strength','range'];
    if(keys.some(k=>!$('object-'+k).checkValidity()||$('object-'+k).value==='')){this.invalidInput=true;$('environmentStatus').textContent='请填写有效的坐标与尺寸';$('applyEnvironment').disabled=true;return;}
    this.invalidInput=false;this.checkpoint();
    const values=keys.map(k=>Number($('object-'+k).value));obj.position_mm=values.slice(0,3);obj.size_mm=values.slice(3,6);obj.yaw_deg=values[6];obj.strength=values[7];obj.range_mm=values[8];
    $('undoEnvironment').disabled=false;$('redoEnvironment').disabled=true;this.persist();
    if(this.validationError()){$('environmentStatus').textContent=this.validationError();$('applyEnvironment').disabled=true;this.render();return;}
    // Preserve typing focus/caret while refreshing the scene and draft status.
    for(const button of $('objectList').children){if(button.dataset.objectId===obj.id)button.textContent=this.types[obj.type].label+' · 虚影';}
    $('applyEnvironment').disabled=false;$('discardEnvironment').disabled=false;$('environmentStatus').textContent='有未应用的改动 · 虚影不产生刺激或碰撞';this.render();
  }
  async apply(){
    if(this.saving||this.invalidInput||this.validationError()||!this.isDirty())return;this.saving=true;this.refresh();
    try{this.applied=await window.chatFlyEnvironment.apply(clone(this.draft),this.applied.revision);this.draft=clone(this.applied.objects);this.recorded=null;this.undoStack=[];this.redoStack=[];this.saving=false;this.refresh();this.toggle(false);}
    catch(e){this.saving=false;this.refresh();$('environmentStatus').textContent=e.message;}
  }
  showRecordedEnvironment(env){this.recorded=clone(env);this.render();}
  acceptMessageEnvironment(env){
    const hadDraft=this.isDirty();this.applied=clone(env);this.recorded=clone(env);
    if(!hadDraft){this.draft=clone(env.objects);this.undoStack=[];this.redoStack=[];}
    this.refresh();
    if(hadDraft)$('environmentStatus').textContent='聊天已应用新环境；保留了你的未应用草稿，请核对后再应用。';
  }
  render(){
    for(const obj of [...this.group.children]){this.group.remove(obj);obj.traverse(child=>{child.geometry?.dispose();if(child.material)child.material.dispose();});}
    const shown=this.opened||this.isDirty()?this.draft:(this.recorded?.objects||this.applied.objects);
    this.pickable=[];
    for(const obj of shown){
      const prior=this.applied.objects.find(o=>o.id===obj.id);
      const ghost=(this.opened||this.isDirty())&&JSON.stringify(obj)!==JSON.stringify(prior);
      const color=this.types[obj.type]?.color||'#ffffff';const fieldOnly=!this.types[obj.type]?.solid;
      const material=new THREE.MeshStandardMaterial({color,roughness:.7,transparent:ghost||fieldOnly,opacity:ghost?.28:fieldOnly?.6:1,depthWrite:!ghost&&!fieldOnly});
      const mesh=new THREE.Mesh(new THREE.BoxGeometry(...obj.size_mm),material);mesh.position.fromArray(obj.position_mm);mesh.rotation.z=obj.yaw_deg*Math.PI/180;mesh.userData.objectId=obj.id;this.group.add(mesh);this.pickable.push(mesh);
      const edges=new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry),new THREE.LineBasicMaterial({color:obj.id===this.selected?'#ebf5c6':color,transparent:true,opacity:ghost?.65:.35}));mesh.add(edges);
      if(obj.id===this.selected&&this.opened){const axes=new THREE.AxesHelper(Math.max(...obj.size_mm)*.8+.5);mesh.add(axes);}
      if(obj.type==='light'&&!ghost){const light=new THREE.PointLight(color,obj.strength*3,obj.range_mm);light.position.copy(mesh.position);this.group.add(light);}
    }
    // Original positions remain marked while moving or deleting applied objects.
    if(this.isDirty())for(const obj of this.applied.objects){const next=this.draft.find(o=>o.id===obj.id);if(JSON.stringify(next)===JSON.stringify(obj))continue;
      const mesh=new THREE.Mesh(new THREE.BoxGeometry(...obj.size_mm),new THREE.MeshBasicMaterial({color:'#70867f',wireframe:true,transparent:true,opacity:.22}));mesh.position.fromArray(obj.position_mm);mesh.rotation.z=obj.yaw_deg*Math.PI/180;this.group.add(mesh);}
    this.view.dirty=true;
  }
  setRay(e){const r=this.view.renderer.domElement.getBoundingClientRect();this.ray.setFromCamera(new THREE.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1),this.view.camera);}
  pointerDown(e){
    if(!this.opened||this.saving||e.button!==0)return;this.setRay(e);const hit=this.ray.intersectObjects(this.pickable,false)[0];if(!hit)return;
    e.stopImmediatePropagation();this.view.orbit.enabled=false;this.selected=hit.object.userData.objectId;
    this.checkpoint();
    const obj=this.draft.find(o=>o.id===this.selected);this.drag={plane:new THREE.Plane(new THREE.Vector3(0,0,1),-obj.position_mm[2]),offset:new THREE.Vector3(obj.position_mm[0]-hit.point.x,obj.position_mm[1]-hit.point.y,0)};
    this.view.renderer.domElement.setPointerCapture(e.pointerId);this.refresh();
  }
  pointerMove(e){
    if(!this.drag)return;e.stopImmediatePropagation();this.setRay(e);const point=this.ray.ray.intersectPlane(this.drag.plane,new THREE.Vector3());if(!point)return;
    point.add(this.drag.offset);const obj=this.draft.find(o=>o.id===this.selected);obj.position_mm[0]=Math.max(-40,Math.min(40,Math.round(point.x*100)/100));obj.position_mm[1]=Math.max(-40,Math.min(40,Math.round(point.y*100)/100));this.refresh();
  }
}
