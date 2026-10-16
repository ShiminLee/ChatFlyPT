"""Run actual UI restoration functions with DOM/network stubs, without a browser."""
import shutil
import subprocess

import pytest

from chatflypt.paths import ROOT


def test_restore_preserves_live_context_and_current_applied_environment():
    node=shutil.which('node')
    if not node:
        pytest.skip('Node is required for the UI function regression')
    script=r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const app=fs.readFileSync('chatflypt/web/app.js','utf8');
const editorSource=fs.readFileSync('chatflypt/web/environment-editor.js','utf8');
function section(source,start,end){
  const a=source.indexOf(start),b=source.indexOf(end,a);assert(a>=0&&b>a);
  return source.slice(a,b);
}
const finish=section(app,'function finishCommunication(','\nwindow.chatFlyEnvironment=');
const init=section(app,'async function init()','\ninit();');
const accept=section(editorSource,'  acceptMessageEnvironment(env){','\n  render(){');
function setup({active=true,draft=false}={}){
  const current={revision:2,objects:[{id:'current',type:'sugar'}],last_record:{kind:'neural',id:'record'}};
  const previous={revision:1,objects:[{id:'previous',type:'yeast'}]};
  const obs={id:'older',environment:previous,scene:{body:{context:'tethered_flight'}}};
  const response={handled:true,dialogue:{},observation:obs,observations:[obs],messages:[],
    reply:'本轮没有识别标签',environment:previous,environment_event:{source:'human_message'}};
  const elements=new Map(),committed=[],displayed=[];
  const ctx={console,clone:x=>JSON.parse(JSON.stringify(x)),currentDialogue:null,inputPlan:null,
    state:{playing:true},liveSessionId:'a'.repeat(32),window:{},
    $:id=>{if(!elements.has(id))elements.set(id,{value:id==='bodyContext'?'ground':'',setAttribute(){},append(){}});return elements.get(id);},
    document:{createElement:()=>({remove(){}})},status:async()=>({words:[]}),
    api:async path=>path.startsWith('/api/session')?{active,response}:path==='/api/environment'?current:
      path.startsWith('/api/environment/response')?response:{},
    setScene(){},statusMessage:()=>({append(){}}),remember(){},pauseAtEnd(){},icon(){},
    requestAnimationFrame(){},tick(){},
    setObservationBatch(observation){ctx.window.flyEnvironmentEditor.recorded=ctx.clone(observation.environment);displayed.push(observation);}
  };
  vm.createContext(ctx);
  const realAccept=vm.runInContext('({'+accept+'}).acceptMessageEnvironment',ctx);
  const editor={applied:ctx.clone(current),draft:ctx.clone(draft?[{id:'draft',type:'block'}]:current.objects),
    ready:Promise.resolve(),refresh(){},isDirty(){return JSON.stringify(this.draft)!==JSON.stringify(this.applied.objects);},
    acceptMessageEnvironment(env){committed.push(env.revision);realAccept.call(this,env);}};
  ctx.window.flyEnvironmentEditor=editor;
  vm.runInContext(finish+'\n'+init,ctx);
  return {ctx,current,previous,response,editor,committed,displayed};
}
(async()=>{
  const live=setup();await live.ctx.init();
  assert.equal(live.ctx.$('bodyContext').value,'tethered_flight');
  assert.equal(live.editor.applied.revision,2);assert.deepEqual(live.committed,[2]);
  assert.equal(live.editor.draft[0].id,'current');assert.equal(live.editor.recorded.revision,1);
  assert.equal(live.displayed.length,1);assert.equal(live.response.observation.archived_environment_record,undefined);

  const historical=setup({active:false,draft:true});await historical.ctx.init();
  assert.equal(historical.ctx.$('bodyContext').value,'ground','Archived context must not configure a new live session');
  assert.equal(historical.editor.applied.revision,2);assert.deepEqual(historical.committed,[2]);
  assert.equal(historical.editor.draft[0].id,'draft','Unapplied drafts must survive restoration');
  assert.equal(historical.response.observation.archived_environment_record,true);

  // A newly completed message still commits its current environment to the UI.
  const fresh=setup();const next={revision:3,objects:[{id:'new',type:'yeast'}]};
  fresh.response.environment=next;fresh.response.observation.environment=next;
  fresh.ctx.finishCommunication(fresh.response,'右边有酵母',{root:{remove(){}}});
  assert.equal(fresh.editor.applied.revision,3);assert.deepEqual(fresh.committed,[3]);
  assert.equal(fresh.editor.draft[0].id,'new');
  console.log('live context, historical isolation, draft preservation and fresh commit passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    result=subprocess.run([node,'-'],input=script,cwd=ROOT,text=True,encoding='utf-8',
                          capture_output=True,timeout=20)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
