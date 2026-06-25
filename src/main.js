/*
 * Emberglass — browser harness for the custom WebGPU VibeThinker-3B runtime.
 * Two panes: INFERENCE (runs the model live) and TRAIN (in-browser LoRA fine-tune).
 * The inference pane defaults to the BASE model (neon badge) to create the incentive
 * to TRAIN; training a small adapter is fast and the result hot-swaps live so the
 * before/after is visible immediately in the same tab. Nothing leaves the device
 * except the optional "train on a URL" lane (which uses a public reader proxy).
 */
import { QWEN25_3B } from './config.js';
import { urlReader, hfReader, fileReader } from './readers.js';
import { AdapterRegistry } from './services/adapter_registry.js';
import { ModelSession } from './services/model_session.js';
import { TrainingController } from './services/training_controller.js';
import { downloadLoraAdapter } from './lora_export.js';

const $ = (id) => document.getElementById(id);
const log = (m) => { const s = $('status'); if (s) s.textContent = m; console.log('[emberglass]', m); };

// ── shared session ──────────────────────────────────────────────────────────
const session = new ModelSession({ cfg: QWEN25_3B, log });
const adapters = new AdapterRegistry();
const state = {
  loaded: false,
  busy: false,
  tuned: null, // { name, kind:'guided'|'own', build(userText)->messages[], suggest }
};

const DEFAULT_SYS = 'You are VibeThinker-3B, a concise, helpful reasoning assistant.';

// ── guided dataset: a tiny fictional handbook the BASE model cannot know ──────
const EMBER_SYS = 'You are the Emberglass OS help desk. Answer in one short factual sentence.';
const GUIDED = [
  ['Who created Emberglass OS?', 'Emberglass OS was created by Mara Quill in 2019.'],
  ['What language is Emberglass OS written in?', 'Emberglass OS is written in a language called Cinder.'],
  ['What is the Emberglass OS mascot?', 'The Emberglass OS mascot is a small fox named Pip.'],
  ['What is the default Emberglass file system?', 'Emberglass uses the CrateFS file system by default.'],
  ['How do you install a package on Emberglass?', 'You install packages with the command `crate pull`.'],
  ['What is the codename for Emberglass OS version 3?', 'Emberglass OS version 3 is codenamed Warehouse.'],
  ['What scheduler does Emberglass use?', 'Emberglass uses the Pelican scheduler, which targets up to 64 cores.'],
  ['What are the Emberglass theme colors?', 'Emberglass uses ember orange on slate gray.'],
];
const GUIDED_SUGGEST = 'Who created Emberglass OS, and what language is it written in?';

// ── badge / lock state ───────────────────────────────────────────────────────
function setBadge() {
  const b = $('runBadge');
  if (!b) return;
  b.className = 'badge';
  if (!state.loaded) { b.classList.add('off'); b.textContent = '○ MODEL NOT LOADED'; return; }
  const sel = $('adapterSel')?.value || 'none';
  if (sel === 'none') { b.classList.add('base'); b.textContent = '● LIVE: BASE VibeThinker-3B · untuned'; }
  else { b.classList.add('tuned'); b.textContent = '● LIVE: YOUR TUNED ADAPTER · ' + sel; }
}
function lockInference(on) {
  $('inferLock').style.display = on ? 'flex' : 'none';
  $('run').disabled = on || !state.loaded || state.busy === 'gen';
}
function gateButtons() {
  const ready = state.loaded && !state.busy;
  $('run').disabled = !ready;
  $('trainGuided').disabled = !ready;
  $('trainOwn').disabled = !ready || !ownExamples().length;
  for (const id of ['load', 'loadHF']) $(id).disabled = !!state.busy;
}

// ── model load ───────────────────────────────────────────────────────────────
async function loadWith(reader, label) {
  if (state.busy) return;
  state.busy = 'load'; gateButtons();
  try {
    await session.loadWith(reader, label);
    state.loaded = true;
    log('Model ready. Ask it anything in INFERENCE — or hit TRAIN to teach it something new.');
  } catch (e) {
    log('LOAD ERROR: ' + e.message);
    console.error(e);
  } finally {
    state.busy = false; setBadge(); gateButtons();
  }
}

// ── inference ─────────────────────────────────────────────────────────────────
function buildMessages(userText) {
  const sel = $('adapterSel')?.value || 'none';
  if (sel !== 'none' && state.tuned && state.tuned.name === sel) return state.tuned.build(userText);
  return [{ role: 'system', content: DEFAULT_SYS }, { role: 'user', content: userText }];
}
async function runInference() {
  if (!state.loaded || state.busy) return;
  const userText = $('prompt').value.trim();
  if (!userText) { log('type something to ask first'); return; }
  state.busy = 'gen'; gateButtons();
  const sel = $('adapterSel')?.value || 'none';
  adapters.applyToRuntime(sel, session.rt);
  const out = $('out');
  out.textContent = '';
  const node = document.createTextNode('');
  out.appendChild(node);
  const t0 = performance.now();
  let n = 0;
  try {
    for await (const d of session.generate(buildMessages(userText), { maxTokens: 480, temperature: 0.0 })) {
      node.appendData(d); n++;
      out.scrollTop = out.scrollHeight;
    }
    const dt = (performance.now() - t0) / 1000;
    $('tokps').textContent = `${n} tok · ${(n / dt).toFixed(1)} tok/s`;
    log(`done (${sel === 'none' ? 'base model' : 'tuned adapter'}).`);
  } catch (e) {
    out.appendData('\n\n[error] ' + e.message); console.error(e);
  } finally {
    state.busy = false; gateButtons();
  }
}

// ── training: shared runner ───────────────────────────────────────────────────
async function runTraining({ examples, lr, epochs, accum, name, kind, build, suggest }) {
  if (!state.loaded) { log('load the model first (INFERENCE pane).'); switchTab('infer'); return; }
  if (state.busy) return;
  state.busy = 'train';
  lockInference(true); gateButtons();
  $('trainWidget').style.display = '';
  const windows = Math.max(1, Math.ceil(examples.length / accum));
  const total = windows * epochs;
  const ctrl = new TrainingController({
    session, adapters, log: () => {},
    trainerOptions: { lr, maxTrainSeq: 384, lmHeadBlock: 128, maxGradNorm: 1.0, weightDecay: 0.0, warmupSteps: Math.min(4, total), totalSteps: total, gradAccumSteps: accum },
  });
  ctrl.initAdapter(name, { rank: 16, alpha: 32 });
  trainProgress(0, total, null, 'warming up the press…');
  const t0 = performance.now();
  try {
    await ctrl.train(examples, {
      epochs,
      onStep: ({ step, loss }) => trainProgress(step, total, loss, `teaching · step ${step}/${total} · loss ${loss.toFixed(3)}`),
    });
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    state.tuned = { name, kind, build, suggest, ctrl };
    addAdapterOption(name);
    $('adapterSel').value = name;
    trainProgress(total, total, null, `done in ${dt}s — adapter "${name}" is live`);
    $('downloadAdapter').style.display = '';
    showTryIt(suggest);
    log(`Trained "${name}" in ${dt}s. Switch to INFERENCE — the tuned adapter is selected.`);
  } catch (e) {
    trainProgress(0, total, null, 'training error: ' + e.message);
    console.error(e);
  } finally {
    state.busy = false;
    lockInference(false); gateButtons();
  }
}

// ── BYOD: turn text into short "continue the note" recall examples ────────────
const MAX_CHARS = 12000, MAX_CHUNKS = 24, MIN_WORDS = 12, HEAD_WORDS = 6;
function chunkText(text) {
  text = (text || '').replace(/\r/g, '').slice(0, MAX_CHARS);
  const paras = text.split(/\n{2,}|\.(?=\s)/).map((s) => s.trim()).filter(Boolean);
  const out = [];
  for (const p of paras) {
    const words = p.split(/\s+/).filter(Boolean);
    if (words.length < MIN_WORDS) continue;
    const head = words.slice(0, HEAD_WORDS).join(' ');
    const rest = words.slice(HEAD_WORDS).join(' ');
    out.push({ head, rest, full: p });
    if (out.length >= MAX_CHUNKS) break;
  }
  return out;
}
let _ownChunks = [];
function ownExamples() {
  return _ownChunks.map((c) => ({ messages: [{ role: 'user', content: c.head }], completion: ' ' + c.rest }));
}
function refreshOwn() {
  const text = $('ownText').value;
  _ownChunks = chunkText(text);
  const chars = Math.min(MAX_CHARS, (text || '').length);
  $('ownStats').textContent = _ownChunks.length
    ? `${_ownChunks.length} snippet(s) · ${chars} chars (cap ${MAX_CHARS}) · ready to teach`
    : `paste/drop at least one paragraph (~${MIN_WORDS}+ words). 100% local.`;
  gateButtons();
}

// ── small UI helpers ──────────────────────────────────────────────────────────
function switchTab(which) {
  const infer = which === 'infer';
  $('paneInfer').classList.toggle('active', infer);
  $('paneTrain').classList.toggle('active', !infer);
  $('tabInfer').classList.toggle('on', infer);
  $('tabTrain').classList.toggle('on', !infer);
}
function switchLane(which) {
  const g = which === 'guided';
  $('guidedLane').style.display = g ? '' : 'none';
  $('ownLane').style.display = g ? 'none' : '';
  $('laneGuided').classList.toggle('on', g);
  $('laneOwn').classList.toggle('on', !g);
}
function addAdapterOption(name) {
  const sel = $('adapterSel');
  if (![...sel.options].some((o) => o.value === name)) {
    const o = document.createElement('option');
    o.value = name; o.textContent = name; sel.appendChild(o);
  }
}
function trainProgress(step, total, loss, label) {
  $('trainBar').style.width = (100 * step / Math.max(1, total)).toFixed(1) + '%';
  $('trainLabel').textContent = label;
}
function showTryIt(suggest) {
  const t = $('tryIt');
  t.style.display = '';
  $('tryItBtn').onclick = () => {
    switchTab('infer');
    $('adapterSel').value = state.tuned.name; setBadge();
    $('prompt').value = suggest;
    runInference();
  };
}

// ── wiring ────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  // render guided facts list
  $('guidedList').innerHTML = GUIDED.map(([q, a]) => `<li><b>Q:</b> ${esc(q)}<br><b>A:</b> ${esc(a)}</li>`).join('');

  $('tabInfer').onclick = () => switchTab('infer');
  $('tabTrain').onclick = () => switchTab('train');
  $('ctaTrain').onclick = () => switchTab('train');
  $('laneGuided').onclick = () => switchLane('guided');
  $('laneOwn').onclick = () => switchLane('own');
  $('adapterSel').onchange = setBadge;

  $('load').onclick = () => loadWith(urlReader($('modelUrl').value.trim()), $('modelUrl').value.trim());
  $('loadHF').onclick = () => {
    const repo = $('hfRepo').value.trim();
    const token = ($('hfToken')?.value || '').trim();
    if (!repo) return log('enter a Hugging Face repo id, e.g. WeiboAI/VibeThinker-3B');
    loadWith(hfReader(repo, token), 'HF: ' + repo);
  };
  $('modelFiles').onchange = (ev) => {
    const files = [...ev.target.files];
    if (!files.length) return;
    const map = {}; for (const f of files) map[f.name] = f;
    loadWith(fileReader(map), `${files.length} local files`);
  };

  $('run').onclick = runInference;
  $('prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) runInference(); });

  $('trainGuided').onclick = () => runTraining({
    examples: GUIDED.map(([q, a]) => ({ messages: [{ role: 'system', content: EMBER_SYS }, { role: 'user', content: q }], completion: ' ' + a })),
    lr: 3e-4, epochs: 12, accum: 2, name: 'emberglass-os', kind: 'guided',
    build: (u) => [{ role: 'system', content: EMBER_SYS }, { role: 'user', content: u }],
    suggest: GUIDED_SUGGEST,
  });

  $('ownText').addEventListener('input', refreshOwn);
  $('ownFiles').onchange = async (ev) => {
    const files = [...ev.target.files].slice(0, 5);
    let txt = '';
    for (const f of files) { try { txt += (await f.text()) + '\n\n'; } catch {} }
    $('ownText').value = (txt + '\n' + $('ownText').value).slice(0, MAX_CHARS);
    refreshOwn();
  };
  $('ownFetch').onclick = async () => {
    const url = $('ownUrl').value.trim();
    if (!url) return;
    $('ownStats').textContent = 'fetching readable text via reader proxy…';
    try {
      const r = await fetch('https://r.jina.ai/' + url);
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const t = await r.text();
      $('ownText').value = t.slice(0, MAX_CHARS);
      refreshOwn();
    } catch (e) { $('ownStats').textContent = 'could not fetch (CORS/blocked) — paste the text instead. ' + e.message; }
  };
  $('trainOwn').onclick = () => {
    const ex = ownExamples();
    if (!ex.length) return;
    const windows = Math.ceil(ex.length / 2);
    runTraining({
      examples: ex, lr: 3e-4, accum: 2,
      epochs: Math.max(3, Math.min(8, Math.round(50 / windows))),
      name: 'my-notes', kind: 'own',
      build: (u) => [{ role: 'user', content: u }],
      suggest: _ownChunks[0]?.head || '',
    });
  };

  $('downloadAdapter').onclick = () => { if (state.tuned?.ctrl?.trainer) downloadLoraAdapter(state.tuned.ctrl.trainer, { name: state.tuned.name }); };

  switchTab('infer'); switchLane('guided'); setBadge(); refreshOwn(); gateButtons();
});

function esc(s) { return String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }
