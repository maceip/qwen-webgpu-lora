/* Diagnose garbage generation: compare greedy decode with f16 ON vs OFF on the
   SAME base model + prompt, and probe the logits buffer for NaN/Inf. */
import { chromium } from 'playwright';
import { existsSync } from 'node:fs';
const macCanary = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary';
const linux = '/usr/local/bin/google-chrome';
const CHROME = process.env.CHROME_PATH || (existsSync(linux) ? linux : existsSync(macCanary) ? macCanary : undefined);
const b = await chromium.launch({
  ...(CHROME ? { executablePath: CHROME } : {}),
  headless: false,
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--use-angle=metal', '--no-first-run'],
});
const p = await b.newPage();
p.on('pageerror', (e) => console.log('PAGEERR', String(e).slice(0, 300)));
p.on('console', (m) => { const t = m.text(); if (/error|fail|nan|inf/i.test(t)) console.log('CON', t.slice(0, 200)); });

const enabled = (sel) => p.evaluate((s) => !document.querySelector(s).disabled, sel);
const text = (sel) => p.evaluate((s) => document.querySelector(s)?.textContent || '', sel);
async function waitEnabled(sel, ms) { const t0 = Date.now(); while (Date.now() - t0 < ms) { if (await enabled(sel)) return true; await p.waitForTimeout(400); } return false; }

await p.goto('http://localhost:8016/docs/index.html', { waitUntil: 'domcontentloaded' });
await p.waitForTimeout(400);
console.log('[1] loading model …');
await p.fill('#modelUrl', '/model');
await p.click('#load');
if (!await waitEnabled('#run', 120000)) { console.log('LOAD FAILED'); await b.close(); process.exit(1); }

// runtime facts
const facts = await p.evaluate(() => {
  const rt = window.__rt;
  return { hasF16: rt.hasF16, usingF16: rt.usingF16?.(), hasDP4a: rt.hasDP4a,
    features: [...(rt.dev?.features || [])], adapter: rt.adapter?.name || null };
});
console.log('[2] runtime:', JSON.stringify(facts));

const Q = 'What is the capital of France? Answer in one word.';

async function probe(label) {
  await p.fill('#prompt', Q);
  await p.click('#run');
  await p.waitForTimeout(600);
  await waitEnabled('#run', 120000);
  const out = (await text('#out')).trim();
  // probe logits buffer for NaN/Inf + top tokens
  const stats = await p.evaluate(async () => {
    const rt = window.__rt, dev = rt.dev, n = rt.cfg.vocabSize;
    const rb = dev.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(rt.s.logits, 0, rb, 0, n * 4);
    dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const a = new Float32Array(rb.getMappedRange()).slice(); rb.unmap(); rb.destroy();
    let nan = 0, inf = 0, max = -1e30, min = 1e30, argmax = 0;
    for (let i = 0; i < a.length; i++) { const v = a[i];
      if (Number.isNaN(v)) nan++; else if (!Number.isFinite(v)) inf++;
      if (v > max) { max = v; argmax = i; } if (v < min) min = v; }
    return { nan, inf, max, min, argmax, sample: [a[0], a[1], a[2], a[100], a[1000]] };
  });
  console.log(`\n[${label}]`);
  console.log('  OUT >>>', JSON.stringify(out.slice(0, 160)));
  console.log('  logits:', JSON.stringify(stats));
}

console.log('[3] f16 = ON (default)');
await probe('f16 ON');

console.log('\n[4] switching f16 OFF …');
await p.evaluate(() => window.__rt.setUseF16(false));
console.log('   usingF16 now:', await p.evaluate(() => window.__rt.usingF16()));
await probe('f16 OFF');

console.log('\nGEN_DEBUG_DONE');
await b.close();
