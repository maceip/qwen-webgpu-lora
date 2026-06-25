/* Isolate the decode step(): feed the SAME 18 tokens two ways and compare final
   logits + last-token hidden — all-prefill (ground truth) vs prefill(17)+decode(1). */
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

const out = await p.evaluate(async () => {
  const rt = window.__rt, dev = rt.dev, H = rt.cfg.hiddenSize;
  rt.setUseF16(false); // isolate from f16 overflow
  const ids = [151644,8948,198,2610,525,10950,13,151645,198,151644,872,198,13048,151645,198,151644,77091,198];
  async function read(buf, n, off = 0) { const rb = dev.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = dev.createCommandEncoder(); e.copyBufferToBuffer(buf, off, rb, 0, n * 4); dev.queue.submit([e.finish()]); await rb.mapAsync(GPUMapMode.READ);
    const a = new Float32Array(rb.getMappedRange()).slice(); rb.unmap(); rb.destroy(); return a; }
  const argmax = (a) => { let bi = 0, bv = -Infinity; for (let i = 0; i < a.length; i++) if (a[i] > bv) { bv = a[i]; bi = i; } return bi; };
  const maxAbs = (a, b) => { let m = 0, at = 0; for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d > m) { m = d; at = i; } } return { m: +m.toFixed(4), at }; };
  const st = (a) => ({ max: +Math.max(...a).toFixed(3), min: +Math.min(...a).toFixed(3), first4: Array.from(a.slice(0, 4)).map((x) => +x.toFixed(3)) });

  // A: all-prefill (ground truth) for 18 tokens
  rt.prefillBatch(ids);
  const LA = await rt.readLogits();
  const hA = await read(rt.sT.hidden, H, 17 * H * 4); // row 17 hidden (pre final-norm)

  // B: prefill first 17, then decode token #17 via step()
  rt.prefillBatch(ids.slice(0, 17));
  rt.token(ids[17], 17);
  const LB = await rt.readLogits();
  const hB = await read(rt.s.hidden, H); // decode hidden (pre final-norm)

  return { argmaxA: argmax(LA), argmaxB: argmax(LB), logitsDiff: maxAbs(LA, LB),
    hiddenDiff: maxAbs(hA, hB), hA: st(hA), hB: st(hB) };
});
console.log(JSON.stringify(out, null, 1));
console.log('DVP_DONE');
await b.close();
