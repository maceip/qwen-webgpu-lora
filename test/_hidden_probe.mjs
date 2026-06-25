/* Find where T=1 forward zeroes out: read hidden buffers after embed and after a
   full step, for decode(T=1) and prefill(T=1 and T=18). */
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
  const ids = [151644,8948,198,2610,525,10950,13,151645,198,151644,872,198,13048,151645,198,151644,77091,198];
  async function read(buf, n, offBytes = 0) {
    const rb = dev.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = dev.createCommandEncoder(); e.copyBufferToBuffer(buf, offBytes, rb, 0, n * 4);
    dev.queue.submit([e.finish()]); await rb.mapAsync(GPUMapMode.READ);
    const a = new Float32Array(rb.getMappedRange()).slice(); rb.unmap(); rb.destroy(); return a;
  }
  const st = (a) => { let z = 0, nan = 0, mx = -1e30, mn = 1e30; for (const v of a) { if (v === 0) z++; if (Number.isNaN(v)) nan++; if (v > mx) mx = v; if (v < mn) mn = v; } return { n: a.length, zeroFrac: +(z / a.length).toFixed(3), nan, max: +mx.toFixed(4), min: +mn.toFixed(4), first4: Array.from(a.slice(0, 4)).map((x) => +x.toFixed(4)) }; };
  const r = {};
  // decode T=1: embed then full step
  rt._resetUni?.();
  { const enc = dev.createCommandEncoder(); rt.embedRow(enc, ids[0]); dev.queue.submit([enc.finish()]); await dev.queue.onSubmittedWorkDone(); }
  r.decode_afterEmbed_Shidden = st(await read(rt.s.hidden, H));
  rt.token(ids[0], 0);
  r.decode_afterStep_Shidden = st(await read(rt.s.hidden, H));

  // prefill T=1: read sT.hidden row0 after
  rt.prefillBatch([ids[0]]);
  r.prefillT1_sThidden_row0 = st(await read(rt.sT.hidden, H, 0));
  r.prefillT1_Shidden = st(await read(rt.s.hidden, H));

  // prefill T=18: read sT.hidden row0 and row17
  rt.prefillBatch(ids);
  r.prefillT18_sThidden_row0 = st(await read(rt.sT.hidden, H, 0));
  r.prefillT18_sThidden_row17 = st(await read(rt.sT.hidden, H, 17 * H * 4));
  r.prefillT18_Shidden = st(await read(rt.s.hidden, H));
  return r;
});
console.log(JSON.stringify(out, null, 1));
console.log('HIDDEN_PROBE_DONE');
await b.close();
