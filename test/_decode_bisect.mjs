/* Bisect the broken decode kernel: toggle each decode fusion off (f16 off) and
   compare prefill(17)+decode(1) logits to all-prefill(18) ground truth. */
import { chromium } from 'playwright';
import { existsSync } from 'node:fs';
const macCanary = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary';
const linux = '/usr/local/bin/google-chrome';
const CHROME = process.env.CHROME_PATH || (existsSync(linux) ? linux : existsSync(macCanary) ? macCanary : undefined);
const b = await chromium.launch({ ...(CHROME ? { executablePath: CHROME } : {}), headless: false,
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--use-angle=metal', '--no-first-run'] });
const p = await b.newPage();
p.on('pageerror', (e) => console.log('PAGEERR', String(e).slice(0, 300)));
const enabled = (s) => p.evaluate((x) => !document.querySelector(x).disabled, s);
async function waitEnabled(s, ms) { const t0 = Date.now(); while (Date.now() - t0 < ms) { if (await enabled(s)) return true; await p.waitForTimeout(400); } return false; }
await p.goto('http://localhost:8016/docs/index.html', { waitUntil: 'domcontentloaded' });
await p.waitForTimeout(400);
console.log('[1] loading model …');
await p.fill('#modelUrl', '/model');
await p.click('#load');
if (!await waitEnabled('#run', 120000)) { console.log('LOAD FAILED'); await b.close(); process.exit(1); }

const res = await p.evaluate(async () => {
  const rt = window.__rt;
  rt.setUseF16(false);
  const ids = [151644,8948,198,2610,525,10950,13,151645,198,151644,872,198,13048,151645,198,151644,77091,198];
  const argmax = (a) => { let bi = 0, bv = -Infinity; for (let i = 0; i < a.length; i++) if (a[i] > bv) { bv = a[i]; bi = i; } return bi; };
  const maxAbs = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return +m.toFixed(3); };
  rt.prefillBatch(ids);
  const LA = await rt.readLogits(); const aA = argmax(LA);
  const f = rt.features;
  const orig = { fuseRMSNormQKVRoPE: f.fuseRMSNormQKVRoPE, fuseQKV: f.fuseQKV, fuseRoPE: f.fuseRoPE, fuseMLP: f.fuseMLP, fuseResidual: f.fuseResidual };
  async function trial(label, set) {
    Object.assign(f, orig, set);
    rt.prefillBatch(ids.slice(0, 17));
    rt.token(ids[17], 17);
    const LB = await rt.readLogits();
    return { label, set, argmaxB: argmax(LB), matchA: argmax(LB) === aA, logitDiff: maxAbs(LA, LB) };
  }
  const out = [{ groundTruth_argmax: aA }];
  out.push(await trial('baseline(all fused)', {}));
  out.push(await trial('no fuseRMSNormQKVRoPE', { fuseRMSNormQKVRoPE: false }));
  out.push(await trial('no fuseQKV(+rope)', { fuseRMSNormQKVRoPE: false, fuseQKV: false }));
  out.push(await trial('no fuseRoPE', { fuseRMSNormQKVRoPE: false, fuseRoPE: false }));
  out.push(await trial('no fuseMLP', { fuseMLP: false }));
  out.push(await trial('no fuseResidual', { fuseResidual: false }));
  out.push(await trial('all unfused', { fuseRMSNormQKVRoPE: false, fuseQKV: false, fuseRoPE: false, fuseMLP: false, fuseResidual: false }));
  Object.assign(f, orig);
  return out;
});
for (const r of res) console.log(JSON.stringify(r));
console.log('BISECT_DONE');
await b.close();
