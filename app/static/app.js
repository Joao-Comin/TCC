'use strict';
/*
 * Monitor de EPI - interface (app/servidor.py serve esta pagina).
 *
 * Video: o <video> toca o arquivo original, liso e na velocidade normal. O
 * servidor analisa os frames um pouco a frente e manda as deteccoes por
 * WebSocket; elas ficam guardadas por numero de frame e sao desenhadas no
 * <canvas> por cima do frame que esta na tela (requestVideoFrameCallback).
 * Entre dois frames analisados as caixas das pessoas sao interpoladas e os
 * EPIs acompanham a pessoa dona.
 *
 * Camera: a webcam aparece direto no <video>; um frame por vez vai em JPEG
 * para o servidor e a resposta e desenhada por cima (sem fila: so manda o
 * proximo frame quando a resposta do anterior chega).
 *
 * Conformidade por pessoa (ID do rastreamento), para cada EPI marcado no boneco:
 *  - usando:      detectado (conf >= minima) em algum frame analisado dentro
 *                 da "memoria do EPI" (padrao 2 s);
 *  - falta:       nao detectado na janela, com a parte do corpo visivel em
 *                 pelo menos metade dos frames e a pessoa acompanhada ha
 *                 pelo menos 80% da janela;
 *  - nao visivel: a parte do corpo nao apareceu na janela (de costas, fora do quadro);
 *  - verificando: o resto (a pessoa acabou de aparecer).
 * Uma "falta" que dura 1 s vira ocorrencia na lista.
 */

const $ = (s) => document.querySelector(s);
const video = $('#video'), camada = $('#camada'), tela = $('#tela');
const ctx = camada.getContext('2d');

const ESQUELETO = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12],
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];
// onde fica o selo de estado de cada classe no boneco (helmet, vest, gloves, boots, goggles)
const SELOS = [[152, 40], [162, 136], [198, 196], [178, 352], [150, 86]];
const ROTULO_ESTADO = { ok: 'usando', falta: 'falta', oculto: 'não visível', verificando: 'verificando', off: '' };
const SIMBOLO = { ok: '✓', falta: '✕', oculto: '?', verificando: '…' };
const COR_ESTADO = { conforme: '#22c55e', 'nao-conforme': '#ef4444', parcial: '#f59e0b' };
const FALTA_MIN_S = 1.0;

const S = {
  classes: [], fonte: 'video', video: null, fps: 30, modelo: null, recortes: false,
  conf: 0.40, janela: 2.0, monitorados: new Set([0, 1, 2, 3, 4]), verPontos: true, verTodos: false,
  dets: new Map(), idx: [], trilhas: new Map(), apelidos: new Map(), selecionada: null,
  eventos: [], ws: null, frameAtual: 0, tAtual: 0, quadro: null, arrastando: false,
  geracao: 0, // sobe a cada salto/troca; deteccao de geracao antiga (ainda em transito) e descartada
  resumo: null,
  cam: { stream: null, ativo: false, enviando: false, n: 0, tempos: new Map(), det: null, tDet: 0 },
};

// ---------------------------------------------------------------- utilidades
const nf = (x, d = 2) => x.toLocaleString('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (x) => `${Math.round(x * 100)}%`;
const nome = (c) => S.classes[c]?.pt ?? String(c);
const cor = (c) => S.classes[c]?.cor ?? '#fff';
function hhmmss(t) {
  t = Math.max(0, t || 0);
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = Math.floor(t % 60);
  const mm = String(m).padStart(2, '0'), ss = String(s).padStart(2, '0');
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}
function guardarPrefs() {
  try {
    localStorage.setItem('epi-prefs', JSON.stringify({
      monitorados: [...S.monitorados], conf: S.conf, janela: S.janela, verPontos: S.verPontos,
      verTodos: S.verTodos, modelo: S.modelo, tema: document.documentElement.dataset.tema || 'escuro',
    }));
  } catch (_) { /* navegador sem armazenamento: so nao lembra */ }
}
function lerPrefs() {
  try { return JSON.parse(localStorage.getItem('epi-prefs') || '{}'); } catch (_) { return {}; }
}
function aviso(txt, erro = false) {
  const a = $('#aviso');
  if (!txt) { a.className = 'aviso'; return; }
  $('#aviso-txt').textContent = txt;
  a.className = 'aviso mostrar' + (erro ? ' erro' : '');
}
function apelido(id) {
  // o rastreador pula numeros; na tela as pessoas viram 1, 2, 3... na ordem em que aparecem
  if (id < 0) return '?';
  if (!S.apelidos.has(id)) S.apelidos.set(id, S.apelidos.size + 1);
  return S.apelidos.get(id);
}
function confirmada(id, t) {
  // so vira "Pessoa N" quem foi acompanhado por pelo menos meio segundo:
  // deteccao relampago (falso positivo de 1-2 frames) nao ganha numero nem entra no resumo
  const tr = S.trilhas.get(id);
  return !!tr && tr.ent.length > 0 && t - tr.ent[0].t >= 0.5;
}

// ---------------------------------------------------------------- inicio
async function iniciar() {
  const est = await (await fetch('/api/estado')).json();
  S.classes = est.classes;
  const pref = lerPrefs();
  if (pref.tema === 'claro') document.documentElement.dataset.tema = 'claro';
  if (Array.isArray(pref.monitorados)) S.monitorados = new Set(pref.monitorados);
  if (pref.conf) S.conf = pref.conf;
  if (pref.janela) S.janela = pref.janela;
  if (pref.verPontos === false) S.verPontos = false;
  if (pref.verTodos) S.verTodos = true;

  const selM = $('#sel-modelo');
  for (const m of est.modelos) selM.add(new Option(m.rotulo, m.chave));
  const padrao = est.modelos.find((m) => m.chave === pref.modelo) || est.modelos.find((m) => m.padrao) || est.modelos[0];
  S.modelo = padrao?.chave;
  selM.value = S.modelo;
  $('#chip-disp').textContent = padrao?.dispositivo ?? '—';

  preencherVideos(est.videos);
  zerarResumo();
  montarBoneco();
  ligarControles();
  sincronizarControles();
  conectar();
  if (est.videos.length) abrirVideo(est.videos[0].nome);
  else aviso('Coloque um vídeo .mp4 na pasta video/ ou use "Enviar vídeo".', true);
  setInterval(enviarTempo, 250);
  setInterval(atualizarPaineis, 250);
  lacoDesenho();
}

function preencherVideos(videos, escolher) {
  S.videos = videos;
  const sel = $('#sel-video');
  sel.innerHTML = '';
  for (const v of videos) {
    const o = new Option(`${v.nome} — ${hhmmss(v.duracao)}, ${v.w}×${v.h}`, v.nome);
    sel.add(o);
  }
  if (escolher) sel.value = escolher;
}

// ---------------------------------------------------------------- WebSocket
function conectar() {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  S.ws = ws;
  ws.onopen = () => {
    if (S.fonte === 'camera' && S.cam.stream) enviar({ tipo: 'camera', modelo: S.modelo, recortes: S.recortes });
    else if (S.video) enviarAbrir();
  };
  ws.onmessage = (ev) => tratar(JSON.parse(ev.data));
  ws.onclose = () => { aviso('Sem conexão com o servidor — tentando de novo…', true); setTimeout(conectar, 1500); };
}
function enviar(obj) { if (S.ws?.readyState === 1) S.ws.send(JSON.stringify(obj)); }
function enviarAbrir() {
  enviar({ tipo: 'abrir', video: S.video, modelo: S.modelo, recortes: S.recortes, t: video.currentTime || 0, g: S.geracao });
}
function enviarTempo() {
  if (S.fonte === 'video' && S.video) enviar({ tipo: 'tempo', t: video.currentTime, pausado: video.paused });
}

function tratar(m) {
  if (m.tipo === 'carregando') aviso(m.msg);
  else if (m.tipo === 'pronto') {
    aviso('');
    $('#chip-disp').textContent = m.dispositivo;
    $('#chip-disp').classList.add('on');
    if (m.camera) { S.cam.ativo = true; S.cam.enviando = false; enviarFrameCamera(); }
    else S.fps = m.fps;
  } else if (m.tipo === 'det') {
    if (S.fonte === 'camera') receberCamera(m);
    else receberVideo(m);
  } else if (m.tipo === 'status') mostrarDesempenho(m);
  else if (m.tipo === 'erro') aviso(`Erro: ${m.msg}`, true);
}

// ---------------------------------------------------------------- deteccoes
function registrar(m, t) {
  // historico por pessoa: maior confianca de cada classe entre os EPIs dela + partes visiveis
  const porDono = new Map();
  for (const e of m.e) {
    if (e[6] < 0) continue;
    let s = porDono.get(e[6]);
    if (!s) porDono.set(e[6], (s = [0, 0, 0, 0, 0]));
    s[e[0]] = Math.max(s[e[0]], e[5]);
  }
  for (const p of m.p) {
    if (p[0] < 0) continue;
    let tr = S.trilhas.get(p[0]);
    if (!tr) S.trilhas.set(p[0], (tr = { ent: [], falta: {}, avisado: {} }));
    tr.ent.push({ t, s: porDono.get(p[0]) || [0, 0, 0, 0, 0], v: p[6] });
    while (tr.ent.length && tr.ent[0].t < t - 12) tr.ent.shift();
  }
}
function receberVideo(m) {
  if (m.g !== S.geracao) return; // resto de antes de um salto
  const ultimo = S.idx[S.idx.length - 1];
  if (ultimo !== undefined && m.i <= ultimo) return;
  S.dets.set(m.i, m);
  S.idx.push(m.i);
  registrar(m, m.i / S.fps);
  // descarta o que ficou mais de 15 s para tras
  while (S.idx.length && S.idx[0] < S.frameAtual - 15 * S.fps) S.dets.delete(S.idx.shift());
  if (video.paused) desenhar();
}
function limparAnalise() {
  S.dets.clear(); S.idx = []; S.trilhas.clear(); S.quadro = null;
  S.geracao++;
}

function estadoPessoa(id, t) {
  const tr = S.trilhas.get(id);
  if (!tr) return null;
  const ents = tr.ent.filter((e) => e.t <= t + 1e-6 && e.t > t - S.janela);
  if (!ents.length) return null;
  const acompanhada = t - tr.ent[0].t;
  return [0, 1, 2, 3, 4].map((c) => {
    if (!S.monitorados.has(c)) return 'off';
    if (ents.some((e) => e.s[c] >= S.conf)) return 'ok';
    const vis = ents.reduce((n, e) => n + ((e.v >> c) & 1), 0);
    if (vis === 0) return 'oculto';
    if (acompanhada >= 0.8 * S.janela && vis >= 0.5 * ents.length) return 'falta';
    return 'verificando';
  });
}
function resumo(est) {
  const mon = (est || []).filter((x) => x !== 'off');
  if (!est || !mon.length) return 'parcial';
  if (mon.includes('falta')) return 'nao-conforme';
  if (mon.every((x) => x === 'ok')) return 'conforme';
  return 'parcial';
}

function quadroVideo(f) {
  // a = ultimo frame analisado <= f; b = o seguinte (para interpolar)
  const k = S.idx;
  let lo = 0, hi = k.length - 1, ia = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (k[mid] <= f) { ia = mid; lo = mid + 1; } else hi = mid - 1;
  }
  if (ia < 0) return null;
  const fa = k[ia], A = S.dets.get(fa);
  const limite = Math.max(3 * (A.k || 1), Math.round(0.5 * S.fps));
  if (f - fa > limite) return null; // analise atrasada: melhor nao mostrar caixa velha
  const fb = ia + 1 < k.length ? k[ia + 1] : null;
  const B = fb !== null && fb - fa <= limite ? S.dets.get(fb) : null;
  return montarQuadro(A, B, B ? (f - fa) / (fb - fa) : 0);
}
function montarQuadro(A, B, alfa) {
  const pessoas = A.p.map((p) => {
    let caixa = p.slice(1, 5);
    if (B && p[0] >= 0 && alfa > 0) {
      const q = B.p.find((x) => x[0] === p[0]);
      if (q) caixa = caixa.map((v, j) => v + (q[j + 1] - v) * alfa);
    }
    const dx = (caixa[0] + caixa[2] - p[1] - p[3]) / 2, dy = (caixa[1] + caixa[3] - p[2] - p[4]) / 2;
    return { id: p[0], caixa, conf: p[5], vis: p[6], kps: p[7], dx, dy };
  });
  const porId = new Map(pessoas.map((p) => [p.id, p]));
  const epis = A.e.map((e) => {
    const d = e[6] >= 0 ? porId.get(e[6]) : null;
    const dx = d ? d.dx : 0, dy = d ? d.dy : 0;
    return { cls: e[0], caixa: [e[1] + dx, e[2] + dy, e[3] + dx, e[4] + dy], conf: e[5], dono: e[6] };
  });
  return { pessoas, epis };
}

// ---------------------------------------------------------------- desenho
function retanguloVideo(cw, ch) {
  const vw = video.videoWidth || 16, vh = video.videoHeight || 9;
  const esc = Math.min(cw / vw, ch / vh);
  const w = vw * esc, h = vh * esc;
  return { x: (cw - w) / 2, y: (ch - h) / 2, w, h };
}
function lacoDesenho() {
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    const cb = (_, meta) => {
      if (S.fonte === 'video') { S.frameAtual = Math.round(meta.mediaTime * S.fps); S.tAtual = meta.mediaTime; desenhar(); }
      video.requestVideoFrameCallback(cb);
    };
    video.requestVideoFrameCallback(cb);
  }
  // camera (e navegadores sem requestVideoFrameCallback)
  const raf = () => {
    if (S.fonte === 'camera') { S.tAtual = performance.now() / 1000; desenhar(); }
    else if (!('requestVideoFrameCallback' in HTMLVideoElement.prototype) && !video.paused) {
      S.tAtual = video.currentTime; S.frameAtual = Math.round(video.currentTime * S.fps); desenhar();
    }
    requestAnimationFrame(raf);
  };
  requestAnimationFrame(raf);
}

function rotulo(texto, x, y, fundo, cw, ch, abaixo = false) {
  // EPI: etiqueta acima da caixa; pessoa: abaixo (o capacete costuma ocupar o topo da pessoa)
  ctx.font = '600 12px "Segoe UI", Roboto, Arial, sans-serif';
  const w = ctx.measureText(texto).width + 10, h = 18;
  const rx = Math.min(Math.max(0, x), cw - w);
  const ry = abaixo ? (y + h > ch ? y - h : y) : (y - h < 0 ? y : y - h);
  ctx.fillStyle = fundo;
  ctx.beginPath(); ctx.roundRect(rx, ry, w, h, 4); ctx.fill();
  ctx.fillStyle = '#0b0d10';
  ctx.fillText(texto, rx + 5, ry + 13);
}

function desenhar() {
  const r = tela.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  const W = Math.round(r.width * dpr), H = Math.round(r.height * dpr);
  if (camada.width !== W || camada.height !== H) { camada.width = W; camada.height = H; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);

  const q = S.fonte === 'camera' ? (S.cam.det ? montarQuadro(S.cam.det, null, 0) : null) : quadroVideo(S.frameAtual);
  S.quadro = q;
  if (!q) return;
  const vr = retanguloVideo(r.width, r.height);
  const X = (v) => vr.x + v * vr.w, Y = (v) => vr.y + v * vr.h;

  for (const p of q.pessoas) p.estado = p.id >= 0 ? estadoPessoa(p.id, S.tAtual) : null;

  // EPIs
  for (const e of q.epis) {
    if (e.conf < S.conf) continue;
    const mon = S.monitorados.has(e.cls);
    if (!mon && !S.verTodos) continue;
    const [x1, y1, x2, y2] = [X(e.caixa[0]), Y(e.caixa[1]), X(e.caixa[2]), Y(e.caixa[3])];
    ctx.save();
    ctx.globalAlpha = mon ? 1 : 0.6;
    ctx.strokeStyle = cor(e.cls); ctx.lineWidth = mon ? 2 : 1.5;
    if (!mon) ctx.setLineDash([5, 4]);
    ctx.fillStyle = cor(e.cls) + '22';
    ctx.beginPath(); ctx.roundRect(x1, y1, x2 - x1, y2 - y1, 3); ctx.fill(); ctx.stroke();
    ctx.restore();
    rotulo(`${nome(e.cls)} ${pct(e.conf)}`, x1, y1, cor(e.cls), r.width, r.height);
  }

  // pessoas
  for (const p of q.pessoas) {
    const est = resumo(p.estado), c = COR_ESTADO[est], sel = p.id === S.selecionada && p.id >= 0;
    const [x1, y1, x2, y2] = [X(p.caixa[0]), Y(p.caixa[1]), X(p.caixa[2]), Y(p.caixa[3])];
    if (S.verPontos && p.kps) {
      const pt = (j) => [X(p.kps[3 * j] + p.dx), Y(p.kps[3 * j + 1] + p.dy), p.kps[3 * j + 2]];
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,.55)'; ctx.lineWidth = 1.5; ctx.lineCap = 'round';
      for (const [a, b] of ESQUELETO) {
        const A = pt(a), B = pt(b);
        if (A[2] < 0.5 || B[2] < 0.5) continue;
        ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke();
      }
      ctx.fillStyle = 'rgba(255,255,255,.85)';
      for (let j = 0; j < 17; j++) {
        const P = pt(j);
        if (P[2] >= 0.5) { ctx.beginPath(); ctx.arc(P[0], P[1], 2.2, 0, 7); ctx.fill(); }
      }
      ctx.restore();
    }
    ctx.save();
    ctx.strokeStyle = c; ctx.lineWidth = sel ? 4 : 2.5;
    if (sel) { ctx.shadowColor = c; ctx.shadowBlur = 12; }
    ctx.beginPath(); ctx.roundRect(x1, y1, x2 - x1, y2 - y1, 6); ctx.stroke();
    ctx.restore();
    const falta = (p.estado || []).map((s, i) => (s === 'falta' ? nome(i).toLowerCase() : null)).filter(Boolean);
    const sem = falta.length > 2 ? `${falta.slice(0, 2).join(', ')} +${falta.length - 2}` : falta.join(', ');
    const quem = p.id >= 0 && confirmada(p.id, S.tAtual) ? `Pessoa ${apelido(p.id)}` : 'Pessoa';
    rotulo(quem + (falta.length ? ` · sem ${sem}` : ''), x1, y2, c, r.width, r.height, true);
  }
}

// ---------------------------------------------------------------- paineis (4x/s)
function zerarResumo() {
  S.resumo = { tempo: 0, conforme: 0, decidido: 0, ok: [0, 0, 0, 0, 0], avaliado: [0, 0, 0, 0, 0] };
}
function acumularResumo(vivas) {
  // amostra o que esta na tela 4x/s, so com o video andando (pausado nao conta de novo)
  if (S.fonte === 'video' && video.paused) return;
  const R = S.resumo;
  R.tempo += 0.25;
  for (const p of vivas) {
    if (!p.estado) continue;
    const est = resumo(p.estado);
    if (est !== 'parcial') { R.decidido++; if (est === 'conforme') R.conforme++; }
    p.estado.forEach((s, c) => {
      if (s === 'ok' || s === 'falta') { R.avaliado[c]++; if (s === 'ok') R.ok[c]++; }
    });
  }
}
function desenharResumo() {
  const R = S.resumo;
  $('#k-pessoas').textContent = S.apelidos.size;
  $('#k-ocorrencias').textContent = S.eventos.length;
  $('#k-conforme').textContent = R.decidido ? pct(R.conforme / R.decidido) : '—';
  $('#k-tempo').textContent = R.tempo < 60 ? `${Math.round(R.tempo)} s` : hhmmss(R.tempo);
  $('#medidores').innerHTML = S.classes.map((c) => {
    const n = R.avaliado[c.id], v = n ? R.ok[c.id] / n : null;
    return `<div class="medidor${v === null ? ' sem-dado' : ''}" style="--cor:${c.cor}" title="${c.pt}: ${n ? `${R.ok[c.id]} de ${n} amostras` : 'sem amostras'}">
      <i></i><span>${c.pt}</span><span class="trilho"><b style="width:${v === null ? 0 : Math.round(v * 100)}%"></b></span>
      <span class="valor">${v === null ? '—' : pct(v)}</span></div>`;
  }).join('');
}
function exportarCSV() {
  const linhas = [['quando', 'segundo_do_video', 'pessoa', 'epi_ausente', 'fonte']];
  for (const e of [...S.eventos].reverse()) {
    linhas.push([e.hora.toISOString(), e.camera ? '' : nf(e.t, 1), e.n, nome(e.c), e.camera ? 'camera' : e.video]);
  }
  const csv = linhas.map((l) => l.map((x) => `"${String(x).replace(/"/g, '""')}"`).join(';')).join('\r\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8' }));
  a.download = `ocorrencias_epi_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function atualizarPaineis() {
  const q = S.quadro, t = S.tAtual;
  const vivas = (q?.pessoas || []).filter((p) => p.id >= 0 && confirmada(p.id, t));
  acumularResumo(vivas);
  desenharResumo();
  // ocorrencias: falta que dura FALTA_MIN_S
  for (const p of vivas) {
    const tr = S.trilhas.get(p.id);
    if (!tr || !p.estado) continue;
    p.estado.forEach((s, c) => {
      if (s === 'falta') {
        if (tr.falta[c] === undefined) tr.falta[c] = t;
        else if (!tr.avisado[c] && t - tr.falta[c] >= FALTA_MIN_S) {
          tr.avisado[c] = true;
          novoEvento(p.id, c, tr.falta[c]);
        }
      } else { delete tr.falta[c]; tr.avisado[c] = false; }
    });
  }
  // cartoes das pessoas
  const caixa = $('#pessoas');
  if (!vivas.length) {
    caixa.innerHTML = '<p class="vazio">Nenhuma pessoa detectada neste momento.</p>';
  } else {
    caixa.innerHTML = vivas.sort((a, b) => apelido(a.id) - apelido(b.id)).map((p) => {
      const est = resumo(p.estado);
      const txtEst = { conforme: 'conforme', 'nao-conforme': 'falta EPI', parcial: 'verificando' }[est];
      const itens = (p.estado || []).map((s, c) => (s === 'off' ? '' :
        `<span class="item ${s}" title="${nome(c)}: ${ROTULO_ESTADO[s]}">${SIMBOLO[s]} ${nome(c)}</span>`)).join('');
      return `<button class="pessoa ${est}${p.id === S.selecionada ? ' selecionada' : ''}" data-id="${p.id}">
        <span class="pessoa-topo"><span class="pessoa-nome">Pessoa ${apelido(p.id)}</span><span class="pessoa-estado">${txtEst}</span></span>
        <span class="itens">${itens || '<span class="item">nenhum EPI marcado</span>'}</span></button>`;
    }).join('');
  }
  // selos do boneco para a pessoa selecionada
  const sel = vivas.find((p) => p.id === S.selecionada);
  $('#boneco-quem').textContent = sel ? `Pessoa ${apelido(sel.id)}` : '';
  $('#boneco-dica').textContent = sel ? 'Estado da pessoa selecionada. Clique nela de novo para voltar.' :
    'Clique no boneco para escolher quais EPIs cobrar.';
  document.querySelectorAll('#boneco .parte').forEach((g) => {
    const c = +g.dataset.classe, s = sel?.estado?.[c];
    g.classList.toggle('estado-ok', s === 'ok');
    g.classList.toggle('estado-falta', s === 'falta');
  });
  $('#selos').innerHTML = sel?.estado ? sel.estado.map((s, c) => (s === 'off' ? '' :
    `<g class="selo ${s}" transform="translate(${SELOS[c][0]} ${SELOS[c][1]})"><circle r="11"/><text>${SIMBOLO[s]}</text></g>`)).join('') : '';
}

function novoEvento(id, c, t) {
  S.eventos.unshift({ id, n: apelido(id), c, t, camera: S.fonte === 'camera', video: S.video, hora: new Date() });
  if (S.eventos.length > 200) S.eventos.pop();
  desenharEventos();
}
function desenharEventos() {
  const ol = $('#ocorrencias');
  if (!S.eventos.length) { ol.innerHTML = '<li class="vazio">Nenhuma ocorrência.</li>'; return; }
  ol.innerHTML = S.eventos.map((e, i) => `<li class="evento" data-i="${i}" title="${e.camera ? '' : 'Clique para ver no vídeo'}">
    <span class="quando">${e.camera ? e.hora.toLocaleTimeString('pt-BR') : hhmmss(e.t)}</span>
    <span class="oque">Pessoa ${e.n} <b>sem ${nome(e.c).toLowerCase()}</b></span></li>`).join('');
}

function mostrarDesempenho(m) {
  const ms = m.ms || {};
  $('#chip-fps').textContent = `${nf(m.fps, 1)} FPS de análise`;
  $('#d-fps').textContent = nf(m.fps, 1);
  $('#d-passo').textContent = `${Math.round(100 / Math.max(1, m.passo))}% dos frames`;
  $('#d-pessoas').textContent = ms.pessoas_ms !== undefined ? `${nf(ms.pessoas_ms, 1)} ms` : '—';
  $('#d-epis').textContent = ms.epis_ms !== undefined ? `${nf(ms.epis_ms, 1)} ms` : '—';
  $('#d-refino').textContent = S.recortes && ms.refino_ms !== undefined ? `${nf(ms.refino_ms, 1)} ms` : 'desligado';
  $('#d-total').textContent = ms.total_ms !== undefined ? `${nf(ms.total_ms, 1)} ms` : '—';
}

// ---------------------------------------------------------------- video
function abrirVideo(nomeVideo, t = 0) {
  pararCamera();
  S.video = nomeVideo;
  S.fps = (S.videos || []).find((v) => v.nome === nomeVideo)?.fps || 30;
  $('#sel-video').value = nomeVideo;
  limparAnalise();
  S.apelidos.clear(); S.eventos = []; zerarResumo(); desenharEventos();
  aviso('Carregando vídeo…');
  video.srcObject = null;
  video.src = `/videos/${encodeURIComponent(nomeVideo)}`;
  video.addEventListener('loadedmetadata', () => {
    video.currentTime = t;
    enviarAbrir();
    const auto = new URLSearchParams(location.search).get('autoplay');
    if (auto) video.play();
  }, { once: true });
}

function ligarControles() {
  const play = $('#play'), barra = $('#barra');
  play.onclick = () => (video.paused ? video.play() : video.pause());
  video.addEventListener('play', () => play.classList.add('tocando'));
  video.addEventListener('pause', () => { play.classList.remove('tocando'); enviarTempo(); });
  video.addEventListener('seeking', () => {
    if (S.fonte !== 'video') return;
    limparAnalise();
    enviar({ tipo: 'pular', t: video.currentTime, g: S.geracao });
  });
  video.addEventListener('seeked', () => {
    S.tAtual = video.currentTime; S.frameAtual = Math.round(video.currentTime * S.fps); desenhar();
  });
  video.addEventListener('ended', () => { if (S.fonte === 'video') { video.currentTime = 0; video.play(); } });
  video.addEventListener('timeupdate', () => {
    if (S.fonte !== 'video') return;
    $('#tempo').textContent = `${hhmmss(video.currentTime)} / ${hhmmss(video.duration)}`;
    if (!S.arrastando && video.duration) barra.value = Math.round((video.currentTime / video.duration) * 1000);
  });
  barra.addEventListener('pointerdown', () => { S.arrastando = true; });
  barra.addEventListener('pointerup', () => { S.arrastando = false; });
  barra.addEventListener('input', () => { if (video.duration) video.currentTime = (barra.value / 1000) * video.duration; });
  $('#velocidade').onchange = (e) => { video.playbackRate = +e.target.value; };
  $('#tela-cheia').onclick = () => (document.fullscreenElement ? document.exitFullscreen() : tela.requestFullscreen());
  document.addEventListener('keydown', (e) => {
    // com foco num botao (ou parte do boneco), o espaco ja aciona o proprio botao
    if (e.target.closest('input, select, textarea, button, [role="button"]') || S.fonte !== 'video') return;
    if (e.code === 'Space') { e.preventDefault(); play.click(); }
    if (e.code === 'ArrowRight') video.currentTime += 5;
    if (e.code === 'ArrowLeft') video.currentTime -= 5;
  });

  $('#sel-video').onchange = (e) => abrirVideo(e.target.value);
  $('#upload').onchange = enviarArquivo;
  document.querySelectorAll('.aba').forEach((b) => (b.onclick = () => trocarFonte(b.dataset.fonte)));

  $('#sel-modelo').onchange = (e) => { S.modelo = e.target.value; guardarPrefs(); reabrir(); };
  $('#recortes').onchange = (e) => { S.recortes = e.target.checked; reabrir(); };
  $('#conf').oninput = (e) => { S.conf = +e.target.value; sincronizarControles(); guardarPrefs(); desenhar(); };
  $('#janela').oninput = (e) => { S.janela = +e.target.value; sincronizarControles(); guardarPrefs(); desenhar(); };
  $('#ver-pontos').onchange = (e) => { S.verPontos = e.target.checked; guardarPrefs(); desenhar(); };
  $('#ver-todos').onchange = (e) => { S.verTodos = e.target.checked; guardarPrefs(); desenhar(); };
  $('#tema').onclick = () => {
    const raiz = document.documentElement;
    if (raiz.dataset.tema === 'claro') delete raiz.dataset.tema; else raiz.dataset.tema = 'claro';
    guardarPrefs();
  };
  $('#limpar').onclick = () => { S.eventos = []; desenharEventos(); };
  $('#exportar').onclick = exportarCSV;
  $('#zerar-resumo').onclick = () => { zerarResumo(); S.eventos = []; S.apelidos.clear(); desenharEventos(); desenharResumo(); };
  $('#ocorrencias').onclick = (e) => {
    const li = e.target.closest('.evento');
    const ev = li && S.eventos[+li.dataset.i];
    if (ev && !ev.camera && S.fonte === 'video') { video.currentTime = Math.max(0, ev.t - 1); S.selecionada = ev.id; }
  };
  $('#pessoas').onclick = (e) => {
    const b = e.target.closest('.pessoa');
    if (b) selecionar(+b.dataset.id);
  };
  camada.addEventListener('click', (e) => {
    const q = S.quadro;
    if (!q) return;
    const r = tela.getBoundingClientRect(), vr = retanguloVideo(r.width, r.height);
    const x = (e.clientX - r.left - vr.x) / vr.w, y = (e.clientY - r.top - vr.y) / vr.h;
    const dentro = q.pessoas.filter((p) => p.id >= 0 && x >= p.caixa[0] && x <= p.caixa[2] && y >= p.caixa[1] && y <= p.caixa[3]);
    dentro.sort((a, b) => (a.caixa[2] - a.caixa[0]) * (a.caixa[3] - a.caixa[1]) - (b.caixa[2] - b.caixa[0]) * (b.caixa[3] - b.caixa[1]));
    selecionar(dentro.length ? dentro[0].id : null);
  });
  window.addEventListener('resize', desenhar);
}

function selecionar(id) {
  S.selecionada = id === null || id === S.selecionada ? null : id;
  desenhar();
  atualizarPaineis();
}

function reabrir() {
  limparAnalise();
  if (S.fonte === 'camera') enviar({ tipo: 'camera', modelo: S.modelo, recortes: S.recortes });
  else enviarAbrir();
}

function sincronizarControles() {
  $('#conf').value = S.conf;
  $('#conf-val').textContent = nf(S.conf, 2);
  $('#janela').value = S.janela;
  $('#janela-val').textContent = `${nf(S.janela, 1)} s`;
  $('#ver-pontos').checked = S.verPontos;
  $('#ver-todos').checked = S.verTodos;
  $('#recortes').checked = S.recortes;
}

async function enviarArquivo(e) {
  const arq = e.target.files[0];
  if (!arq) return;
  aviso(`Enviando ${arq.name}…`);
  const fd = new FormData();
  fd.append('arquivo', arq);
  const r = await fetch('/api/videos', { method: 'POST', body: fd });
  e.target.value = '';
  if (!r.ok) { aviso(`Erro no envio: ${(await r.json()).detail}`, true); return; }
  const info = await r.json();
  const est = await (await fetch('/api/estado')).json();
  preencherVideos(est.videos, info.nome);
  trocarFonte('video');
  abrirVideo(info.nome);
}

// ---------------------------------------------------------------- boneco
function montarBoneco() {
  const chips = $('#chips-epi');
  chips.innerHTML = S.classes.map((c) =>
    `<button class="chip-epi" data-classe="${c.id}" style="--cor:${c.cor}"><i></i>${c.pt}</button>`).join('');
  document.querySelectorAll('#boneco .parte').forEach((g) => {
    g.style.setProperty('--cor', cor(+g.dataset.classe));
    const alternar = () => alternarClasse(+g.dataset.classe);
    g.addEventListener('click', alternar);
    g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); alternar(); } });
  });
  chips.onclick = (e) => { const b = e.target.closest('.chip-epi'); if (b) alternarClasse(+b.dataset.classe); };
  pintarBoneco();
}
function alternarClasse(c) {
  if (S.monitorados.has(c)) S.monitorados.delete(c); else S.monitorados.add(c);
  // a mudanca vale para o futuro: zera os "ja avisado" para nao herdar estado antigo
  for (const tr of S.trilhas.values()) { tr.falta = {}; tr.avisado = {}; }
  pintarBoneco();
  guardarPrefs();
  desenhar();
  atualizarPaineis();
}
function pintarBoneco() {
  document.querySelectorAll('#boneco .parte').forEach((g) => {
    const c = +g.dataset.classe, on = S.monitorados.has(c);
    g.classList.toggle('ativa', on);
    g.setAttribute('aria-pressed', on);
    g.querySelector('title')?.remove();
    const t = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    t.textContent = `${nome(c)} — ${on ? 'monitorado (clique para desligar)' : 'desligado (clique para monitorar)'}`;
    g.prepend(t);
  });
  document.querySelectorAll('.chip-epi').forEach((b) => b.classList.toggle('ativa', S.monitorados.has(+b.dataset.classe)));
}

// ---------------------------------------------------------------- camera
async function trocarFonte(fonte) {
  if (fonte === S.fonte) return;
  document.querySelectorAll('.aba').forEach((b) => b.classList.toggle('ativa', b.dataset.fonte === fonte));
  S.fonte = fonte;
  limparAnalise();
  S.apelidos.clear(); S.eventos = []; zerarResumo(); desenharEventos();
  $('#controles').classList.toggle('oculto', fonte === 'camera');
  document.querySelector('.barra-fonte').classList.toggle('camera', fonte === 'camera');
  if (fonte === 'video') { abrirVideo(S.video || $('#sel-video').value); return; }
  video.pause();
  try {
    aviso('Abrindo a câmera…');
    S.cam.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    video.removeAttribute('src');
    video.srcObject = S.cam.stream;
    await video.play();
    S.cam.det = null;
    enviar({ tipo: 'camera', modelo: S.modelo, recortes: S.recortes });
  } catch (err) {
    aviso(`Não foi possível abrir a câmera: ${err.message}`, true);
  }
}
function pararCamera() {
  S.cam.ativo = false;
  if (S.cam.stream) { S.cam.stream.getTracks().forEach((t) => t.stop()); S.cam.stream = null; }
}
const tmpCanvas = document.createElement('canvas');
function enviarFrameCamera() {
  if (!S.cam.ativo || S.cam.enviando || S.ws?.readyState !== 1 || !video.videoWidth) {
    if (S.cam.ativo && !S.cam.enviando) setTimeout(enviarFrameCamera, 50);
    return;
  }
  const esc = Math.min(1, 960 / video.videoWidth);
  tmpCanvas.width = Math.round(video.videoWidth * esc);
  tmpCanvas.height = Math.round(video.videoHeight * esc);
  tmpCanvas.getContext('2d').drawImage(video, 0, 0, tmpCanvas.width, tmpCanvas.height);
  S.cam.enviando = true;
  const n = ++S.cam.n;
  S.cam.tempos.set(n, performance.now() / 1000);
  tmpCanvas.toBlob(async (blob) => {
    const corpo = new Uint8Array(await blob.arrayBuffer());
    const pacote = new Uint8Array(4 + corpo.length);
    new DataView(pacote.buffer).setUint32(0, n, true);
    pacote.set(corpo, 4);
    if (S.ws?.readyState === 1) S.ws.send(pacote); else S.cam.enviando = false;
  }, 'image/jpeg', 0.8);
}
let camContagem = 0, camInicio = performance.now();
function receberCamera(m) {
  if (m.g !== undefined) return; // deteccao de video que ainda estava em transito
  const t = S.cam.tempos.get(m.i) ?? performance.now() / 1000;
  S.cam.tempos.delete(m.i);
  S.cam.det = m;
  registrar(m, t);
  S.cam.enviando = false;
  enviarFrameCamera();
  camContagem++;
  const agora = performance.now();
  if (agora - camInicio >= 1000) {
    mostrarDesempenho({ fps: (camContagem * 1000) / (agora - camInicio), passo: 1, ms: { total_ms: m.ms } });
    camContagem = 0; camInicio = agora;
  }
}

iniciar().catch((e) => aviso(`Falha ao iniciar: ${e.message}`, true));
