// f16_vs_f32_diff.js
// Phase 3 eval harness stub (per OPTIMIZATION_PLAN.md).
// Usage (in browser console or via test runner that loads this):
//   import('./test/f16_vs_f32_diff.js').then(m => m.runF16Diff())
//
// What it does:
// - Toggle f16 on a built runtime and exercise paths that have f16 variants:
//   rms, add, silu, rope*, and attn combine (score/softmax/V still f32 in partial for this slice).
// - For real numbers, pair with capture of hidden state, attention output, or final logits
//   (see deep_kernel_diff.js / profile for buffer readback patterns).
// - Full eval: same prompt, f16=off vs on → compare logits or generated tokens (greedy must match top token(s)).
//
// Acceptance (from plan): small numeric delta (1e-3..1e-4 rel on logits typical); greedy tokens equivalent.

import { QwenWGPU } from '../src/qwgpu/runtime.js';
import { QWEN25_3B } from '../src/config.js';

export async function runF16Diff(opts = {}) {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const dev = await adapter.requestDevice({
    requiredFeatures: ['subgroups'],
    requiredLimits: { /* ... copy from other tests if needed */ }
  });

  const rt = new QwenWGPU(dev, QWEN25_3B, { onProgress: () => {} });
  await rt.build();

  const prompt = opts.prompt || 'The quick brown fox';
  const useF16 = rt.hasF16;

  if (!useF16) {
    console.warn('[f16_diff] shader-f16 not available on this device; f32 path only.');
    return { skipped: true };
  }

  // Toggle and capture a proxy signal.
  // For a real harness, hook into internal step() or embed + first transformer block and snapshot a buffer.
  // Here we just demonstrate the toggle and a trivial "forward" via public API if exposed.

  rt.setUseF16(false);
  const off = await rt.embedRow(/* token id or prompt tokenization handled upstream */ 42); // placeholder

  rt.setUseF16(true);
  const on = await rt.embedRow(42); // placeholder; real impl would run full layer stack

  // Real implementation sketch:
  //   - tokenize prompt
  //   - run prefill or single step with rt.setUseF16(false) → read back hidden or logits slice
  //   - repeat with true
  //   - compute diff

  console.log('[f16_diff] toggle works, hasF16=', rt.hasF16, 'usingF16 now=', rt.usingF16());
  console.log('[f16_diff] f16 covers: add/silu/rms/rope*/attn-combine. Partial attn score/softmax/V remain f32 for stability.');

  // Recommended real usage (in a full test that has a loaded rt + tokenizer + step()):
  //   rt.setUseF16(false); const logits0 = await captureLogitsAfterStep(rt, promptTokens);
  //   rt.setUseF16(true);  const logits1 = await captureLogitsAfterStep(rt, promptTokens);
  //   const diff = maxAbsRel(logits0, logits1);
  //   console.log('f16 vs f32 maxAbs', diff.maxAbs, 'rel', diff.rel);
  //   // Also: run full greedy decode N tokens both ways and assert top-1 token id match rate.

  return { hasF16: true, note: 'f16 attention-combine now selectable; expand capture for numeric eval' };
}

// Auto-run hook for convenience in some test loaders:
if (typeof window !== 'undefined') {
  window.runF16Diff = runF16Diff;
}
