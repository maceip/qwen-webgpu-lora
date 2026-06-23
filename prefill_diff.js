// Differential test: batched prefill vs the proven sequential token() path, across
// prompt lengths that cross the kernel boundaries (GEMM BM=16 tile, attn 256-thread
// stride / ctx>256, large rope/embed). Greedy continuations must match exactly.
import { QwenWGPU } from './qwgpu/runtime.js';
import { QWEN25_3B } from './config.js';
window.run = async () => {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const dev = await adapter.requestDevice({ requiredFeatures: ['subgroups'], requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  dev.addEventListener?.('uncapturederror', e => console.log('VWG GPUERR ' + e.error.message.slice(0, 160)));
  const ref = await (await fetch('./ref.json')).json(); const cfg = QWEN25_3B;
  const rt = new QwenWGPU(dev, cfg); await rt.build('/model'); console.log('VWG built');
  const rbuf = dev.createBuffer({ size: cfg.vocabSize * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readLogits = async () => { const e = dev.createCommandEncoder(); e.copyBufferToBuffer(rt.s.logits, 0, rbuf, 0, cfg.vocabSize * 4); dev.queue.submit([e.finish()]); await rbuf.mapAsync(GPUMapMode.READ); const a = new Float32Array(rbuf.getMappedRange()).slice(); rbuf.unmap(); return a; };
  const maxAbsDiff = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
  const tile = (L) => { const o = []; while (o.length < L) o.push(ref.ids[o.length % ref.ids.length]); return o; };
  const decodeN = async (pos, n) => { let out = [await rt.argmaxLogits()]; while (out.length < n) { const b = await rt.decodeBatch(pos, Math.min(rt.MAXBATCH, n - out.length)); pos += b.length; out.push(...b); } return out.slice(0, n); };

  const Ls = [16, 17, 33, 200, 256, 257, 400, 512];
  let allOk = true;
  for (const L of Ls) {
    const ids = tile(L);
    for (let p = 0; p < L; p++) rt.token(ids[p], p);              // sequential prefill
    const Lseq = await readLogits(); const seqGen = await decodeN(L, 6);
    rt.prefillBatch(ids);                                          // batched prefill
    const Lbat = await readLogits(); const batGen = await decodeN(L, 6);
    const d = maxAbsDiff(Lseq, Lbat); const argEq = seqGen[0] === batGen[0];
    const genEq = JSON.stringify(seqGen) === JSON.stringify(batGen);
    const ok = argEq && genEq;  allOk = allOk && ok;
    console.log(`VWG L=${String(L).padStart(3)}  argmax ${seqGen[0]}==${batGen[0]} ${argEq?'✓':'✗'}  gen ${genEq?'match':'DIFFER'}  logitΔ=${d.toFixed(3)}  ${ok?'PASS':'FAIL'}`);
  }
  console.log('VWG PREFILL-DIFF ' + (allOk ? 'ALL PASS' : 'FAILURES'));
  console.log('VWG DONE');
};
window.addEventListener('DOMContentLoaded', () => window.run().catch(e => console.log('VWG ERROR ' + e.message + ' | ' + (e.stack || '').slice(0, 300))));
