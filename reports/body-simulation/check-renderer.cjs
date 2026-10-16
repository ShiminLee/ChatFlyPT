// Program-level rendering contract check, not a browser or visual-layout test.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const input = process.argv[2] || 'data/derived/body-simulation/neural/body-result.json';
const data = JSON.parse(fs.readFileSync(path.resolve(root, input), 'utf8'));
const body = data.observation?.scene?.body || data;
const duration = data.observation?.scene?.duration_ms || body.source_duration_ms;
const source = fs.readFileSync(path.join(root, 'chatflypt/web/app.js'), 'utf8');
const begin = source.indexOf('function drawPhysicalWorld(');
const end = source.indexOf('\nfunction drawWorld(', begin);
assert(begin >= 0 && end > begin);
const render = source.slice(begin, end);
const checks = [];
for (const [width, height] of [[640, 230], [290, 180]]) {
  for (const progress of [0, .14, .5, 1]) {
    const calls = [], ellipses = [], labels = [];
    const wc = {};
    for (const name of ['beginPath','moveTo','lineTo','stroke','save','translate','rotate','restore']) {
      wc[name] = (...args) => {
        assert(args.every(Number.isFinite), `Non-finite canvas coordinates in ${name}`);
        calls.push([name, args]);
      };
    }
    wc.fillText = (text, x, y) => { assert(Number.isFinite(x) && Number.isFinite(y)); labels.push(text); };
    const context = { observation: {scene: {body}}, state: {overview:false,progress,scene:{duration_ms:duration}},
      wc, ellipse: (_ctx, ...args) => { assert(args.slice(0,5).every(Number.isFinite)); ellipses.push(args); } };
    const result = vm.runInNewContext(render + `\ndrawPhysicalWorld(${width},${height});`, context);
    assert.equal(result, true);
    const time = progress * duration;
    const frame = body.trajectory.filter(f => f.time_ms <= time).at(-1) || body.trajectory[0];
    assert(labels[0].startsWith(`${Math.round(frame.time_ms)} ms`));
    assert.equal(ellipses.length, 8); // 6 measured feet + 2 schematic body parts.
    const points = body.trajectory.flatMap(f => [f.position_mm, ...f.feet_mm]);
    const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
    const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
    const scale = Math.min((width-80)/Math.max(5,xmax-xmin),(height-48)/Math.max(3,ymax-ymin));
    frame.feet_mm.forEach(([x,y], i) => {
      const expected = [width/2+(x-(xmin+xmax)/2)*scale, height/2-(y-(ymin+ymax)/2)*scale];
      assert(Math.abs(ellipses[i][0]-expected[0]) < 1e-9);
      assert(Math.abs(ellipses[i][1]-expected[1]) < 1e-9);
      assert(expected[0]>=0 && expected[0]<=width && expected[1]>=0 && expected[1]<=height);
    });
    const rotation = calls.find(c => c[0] === 'rotate')[1][0];
    assert.equal(rotation, -frame.yaw_rad);
    checks.push({width,height,progress,selected_time_ms:frame.time_ms,measured_feet:6,finite_and_in_bounds:true});
  }
}
const result = {kind:'programmatic_canvas_contract_not_browser_test',source:input,checks,passed:true};
fs.writeFileSync(path.join(__dirname,'renderer-check.json'), JSON.stringify(result,null,2));
console.log(JSON.stringify(result,null,2));
