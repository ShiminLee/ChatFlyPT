// Body scene adapted from Kevin Lin / kevinlinxc, badapple-fly, frontend/main.js
// commit 6726892e7ebd93e6c5ffab056ab8db09433c28a6, Fly scene section.
// Original scene, materials, camera/OrbitControls and path rendering retained.
// ChatFlyPT adaptation: local assets and recorded body poses + geom-local poses.
// No controller, synthetic gait, or language-derived body movement runs here.
import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import {EnvironmentEditor} from './environment-editor.js';

const $=id=>document.getElementById(id);
const DEFAULT_ASSET='/assets/neuromechfly-behavior-scene.json';
const LEGACY_ASSET='/assets/neuromechfly-scene.json';
const BODY_ASSETS=new Set([DEFAULT_ASSET,LEGACY_ASSET]);

export function selectRecordedFrame(frames,time){
  if(!frames?.length)return null;
  // Choose the last measured frame at or before the shared neural timeline.
  let lo=0,hi=frames.length-1;
  while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(frames[mid].time_ms<=time)lo=mid;else hi=mid-1;}
  return frames[lo];
}

function colorFor(name){
  if(name.includes('eye'))return '#ad4b32';
  if(name.includes('wing'))return '#adcebf';
  if(name.includes('thorax'))return '#ac8b56';
  if(name.includes('abdomen'))return '#987646';
  if(name.includes('head'))return '#b19361';
  return '#ae925e';
}

class FlyBodyView{
  constructor(){
    this.viewport=$('world');this.status=$('bodySceneStatus');this.dirty=true;
    this.record=undefined;this.pendingRecord=null;this.pendingTime=0;
    this.data=null;this.modelFrame=null;this.meshes=[];this.groups=[];this.pathLine=null;
    this.loadedAsset=null;this.loadingAsset=null;this.failedAsset=null;this.loadToken=0;this.assetCache=new Map();
    this.scene=new THREE.Scene();this.scene.background=new THREE.Color('#142c2d');
    this.probeMeshes=new Map();
    this.scene.fog=new THREE.Fog('#142c2d',18,55);
    this.camera=new THREE.PerspectiveCamera(35,1,.02,200);this.camera.up.set(0,0,1);
    this.renderer=new THREE.WebGLRenderer({antialias:true});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));
    this.renderer.outputColorSpace=THREE.SRGBColorSpace;
    this.viewport.appendChild(this.renderer.domElement);
    this.renderer.domElement.setAttribute('aria-label','三维果蝇身体');
    this.orbit=new OrbitControls(this.camera,this.renderer.domElement);
    this.orbit.enableDamping=true;this.orbit.minDistance=2;this.orbit.maxDistance=50;
    this.orbit.addEventListener('change',()=>{this.dirty=true;});
    this.scene.add(new THREE.HemisphereLight(0xe7fff1,0x304c46,2.5));
    const key=new THREE.DirectionalLight(0xfff3da,3.5);key.position.set(3,-4,8);this.scene.add(key);
    const fill=new THREE.DirectionalLight(0x8ccbd5,1.4);fill.position.set(-4,4,3);this.scene.add(fill);
    const ground=new THREE.Mesh(new THREE.PlaneGeometry(200,200),new THREE.MeshStandardMaterial({color:'#173334',roughness:1}));
    ground.position.z=-.01;this.scene.add(ground);
    const grid=new THREE.GridHelper(100,100,0x40605a,0x264841);
    grid.rotation.x=Math.PI/2;grid.position.z=0;grid.material.transparent=true;grid.material.opacity=.45;this.scene.add(grid);
    this.resizeObserver=new ResizeObserver(()=>this.resize());this.resizeObserver.observe(this.viewport);
    $('bodySide').onclick=()=>this.resetCamera('side');$('bodyTop').onclick=()=>this.resetCamera('top');
    $('bodyReset').onclick=()=>this.resetCamera(this.mode||'side');
    $('bodyExpand').onclick=async()=>{
      try{if(document.fullscreenElement)await document.exitFullscreen();else await this.viewport.closest('.world-card').requestFullscreen();}
      catch{this.status.textContent='当前浏览器不支持展开视图';this.status.hidden=false;}
    };
    document.addEventListener('fullscreenchange',()=>{
      const expanded=!!document.fullscreenElement;$('bodyExpand').setAttribute('aria-label',expanded?'收起身体视图':'展开身体视图');
      this.resize();
    });
    this.renderer.domElement.addEventListener('webglcontextlost',event=>{
      event.preventDefault();this.status.textContent='三维渲染已中断，请刷新页面重试。';this.status.hidden=false;
    });
    this.resize();this.resetCamera();
    this.ready=this.loadScene(DEFAULT_ASSET);
  }
  resize(){
    const w=Math.max(1,this.viewport.clientWidth),h=Math.max(1,this.viewport.clientHeight);
    this.renderer.setSize(w,h);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();this.dirty=true;
  }
  resetCamera(mode='side'){
    this.mode=mode;const t=this.modelFrame?.position_mm||[0,0,1];this.orbit.target.set(...t);
    // Original side/top orientation, with a closer fit for the smaller chat panel.
    const offset=mode==='top'?[.01,.01,9]:[4.05,-7.275,3.975];
    this.camera.position.set(t[0]+offset[0],t[1]+offset[1],t[2]+offset[2]);this.orbit.update();this.dirty=true;
    for(const [id,value] of [['bodySide','side'],['bodyTop','top']]){
      $(id).classList.toggle('active',mode===value);$(id).setAttribute('aria-pressed',String(mode===value));
    }
  }
  async loadScene(asset){
    const token=++this.loadToken;this.loadingAsset=asset;this.failedAsset=null;
    this.status.textContent='正在加载三维身体…';this.status.hidden=false;
    try{
    let data=this.assetCache.get(asset);
    if(!data){
      const response=await fetch(asset);if(!response.ok)throw new Error('模型文件不可用');
      data=await response.json();this.assetCache.set(asset,data);
    }
    if(token!==this.loadToken)return;
    if(data.schema!=='chatflypt.body-scene.v1'||data.units!=='millimeters')throw new Error('模型坐标格式不匹配');
    for(const mesh of this.meshes){mesh.geometry.dispose();mesh.material.dispose();}
    for(const group of this.groups)this.scene.remove(group);
    this.drawPath([]);this.meshes=[];this.groups=[];this.modelFrame=null;this.record=undefined;
    this.data=data;
    this.groups=data.body_names.map(name=>{const g=new THREE.Group();g.name=name;this.scene.add(g);return g;});
    for(const g of data.geoms){
      const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(g.vertices,3));
      geo.setIndex(g.faces);geo.computeVertexNormals();
      const wing=g.name.includes('wing');
      const material=new THREE.MeshStandardMaterial({color:colorFor(g.name),roughness:wing?.3:.65,metalness:.02,side:THREE.DoubleSide,transparent:wing,opacity:wing?.45:1});
      const mesh=new THREE.Mesh(geo,material);mesh.name=g.name;
      mesh.position.fromArray(g.local_position_mm);
      const [w,x,y,z]=g.local_quaternion_wxyz;mesh.quaternion.set(x,y,z,w);
      this.groups[g.body_index].add(mesh);this.meshes.push(mesh);
    }
    this.loadedAsset=asset;this.loadingAsset=null;
    this.displayBody(data.frame,data.body_names);this.resetCamera(this.mode||'side');this.status.hidden=true;
    this.update(this.pendingRecord,this.pendingTime);
    }catch(error){
      if(token!==this.loadToken)return;
      this.loadingAsset=null;this.failedAsset=asset;this.assetCache.delete(asset);
      this.status.textContent='三维身体加载失败：'+error.message;this.status.hidden=false;
    }
  }
  displayBody(frame,names){
    if(!frame?.body_positions_mm||!frame.body_quaternions_wxyz)return false;
    const indices=new Map(names.map((name,i)=>[name,i]));
    for(const name of this.data.body_names){
      const i=indices.get(name),p=frame.body_positions_mm[i],q=frame.body_quaternions_wxyz[i];
      if(!p||p.length!==3||!q||q.length!==4||!p.every(Number.isFinite)||!q.every(Number.isFinite))return false;
    }
    if(this.modelFrame){
      const delta=new THREE.Vector3(...frame.position_mm).sub(new THREE.Vector3(...this.modelFrame.position_mm));
      this.camera.position.add(delta);this.orbit.target.add(delta);
    }
    this.modelFrame=frame;
    for(const mesh of this.probeMeshes.values())mesh.visible=false;
    for(const probe of frame.physical_probes||[]){
      let mesh=this.probeMeshes.get(probe.id);
      if(!mesh){mesh=new THREE.Mesh(new THREE.SphereGeometry(probe.radius_mm,16,12),new THREE.MeshStandardMaterial({color:'#e69a4d',roughness:.5}));this.probeMeshes.set(probe.id,mesh);this.scene.add(mesh);}
      mesh.position.fromArray(probe.position_mm);mesh.visible=probe.visible;
    }
    for(let j=0;j<this.groups.length;j++){
      const i=indices.get(this.data.body_names[j]);this.groups[j].position.fromArray(frame.body_positions_mm[i]);
      const [w,x,y,z]=frame.body_quaternions_wxyz[i];this.groups[j].quaternion.set(x,y,z,w);
    }
    $('bodyFrameTime').textContent=frame===this.data.frame?'初始姿态':Math.round(frame.time_ms)+' ms';
    this.dirty=true;return true;
  }
  drawPath(frames){
    if(this.pathLine){this.scene.remove(this.pathLine);this.pathLine.geometry.dispose();this.pathLine.material.dispose();this.pathLine=null;}
    if(!frames.length)return;
    const points=frames.map(f=>new THREE.Vector3(f.position_mm[0],f.position_mm[1],.015));
    this.pathLine=new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:'#67b997',transparent:true,opacity:.5}));
    this.scene.add(this.pathLine);this.dirty=true;
  }
  update(body,time){
    this.pendingRecord=body;this.pendingTime=time;
    const asset=body?.geometry_asset||(body?.physics_run?LEGACY_ASSET:DEFAULT_ASSET);
    if(!BODY_ASSETS.has(asset)){this.status.textContent='身体模型来源未识别，无法回放。';this.status.hidden=false;return;}
    if(asset!==this.loadedAsset){
      if(asset!==this.loadingAsset&&asset!==this.failedAsset)this.ready=this.loadScene(asset);
      return;
    }
    if(!this.data||this.loadingAsset)return;
    if(body!==this.record){
      this.record=body;this.selectedFrame=null;this.frames=body?.physics_run&&Array.isArray(body.trajectory)?body.trajectory:[];
      this.drawPath(this.frames);this.displayBody(this.data.frame,this.data.body_names);this.status.hidden=true;
    }
    const frame=selectRecordedFrame(this.frames,time);
    if(frame&&frame!==this.selectedFrame){
      if(this.displayBody(frame,body.body_names||[]))this.selectedFrame=frame;
      else{this.displayBody(this.data.frame,this.data.body_names);this.status.textContent='这份记录缺少完整身体姿态，无法回放。';this.status.hidden=false;}
    }
    this.orbit.update();
    if(this.dirty){this.dirty=false;this.renderer.render(this.scene,this.camera);}
  }
}

try{window.flyBodyView=new FlyBodyView();window.flyEnvironmentEditor=new EnvironmentEditor(window.flyBodyView);}
catch(error){$('bodySceneStatus').textContent='无法开启三维渲染：'+error.message;$('bodySceneStatus').hidden=false;}
