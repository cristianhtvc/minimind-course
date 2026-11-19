// Ratios describe the share of the available space after controls are laid out.
const layoutKey = 'learn_minimind_layout_v1';
const wideScreen = window.matchMedia('(min-width: 900px)');
const workspace = document.getElementById('workspace');
const widthHandle = document.getElementById('width-handle');
const heightHandle = document.getElementById('height-handle');
const editorStack = document.getElementById('editor-stack');
const consoleToggle = document.getElementById('console-toggle');
let readerShare = 54;
let editorShare = 100;
let lastOpenShare = 72;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function setReaderShare(value) {
  readerShare = clamp(Math.round(value), 30, 70);
  workspace.style.setProperty('--read-track', readerShare + 'fr');
  workspace.style.setProperty('--write-track', (100 - readerShare) + 'fr');
  widthHandle.setAttribute('aria-valuenow', readerShare);
  widthHandle.setAttribute('aria-valuetext', '教材占 ' + readerShare + '%');
}

function setEditorShare(value) {
  editorShare = clamp(Math.round(value), 40, 100);
  const expanded = editorShare < 100;
  if (expanded) lastOpenShare = editorShare;
  editorStack.style.setProperty('--editor-track', editorShare + 'fr');
  editorStack.style.setProperty('--result-track', (100 - editorShare) + 'fr');
  heightHandle.setAttribute('aria-valuenow', editorShare);
  heightHandle.setAttribute('aria-valuetext', expanded ? '编辑区占 ' + editorShare + '%' : '评测结果已收起');
  consoleToggle.setAttribute('aria-expanded', expanded);
  document.getElementById('testcase-results').hidden = !expanded;
}

function saveLayout() {
  try {
    localStorage.setItem(layoutKey, JSON.stringify({ readerShare, editorShare, lastOpenShare }));
  } catch (_) {}
}

function toggleConsole() {
  setEditorShare(editorShare === 100 ? lastOpenShare : 100);
  saveLayout();
}

function bindSeparator(handle, horizontalMovement, getValue, setValue, availableSize, resetValue) {
  let drag = null;
  const coordinate = event => horizontalMovement ? event.clientX : event.clientY;
  handle.addEventListener('pointerdown', event => {
    if (event.button !== 0 || !event.isPrimary) return;
    const size = availableSize();
    if (size <= 0) return;
    handle.setPointerCapture(event.pointerId);
    handle.focus({ preventScroll: true });
    handle.classList.add('dragging');
    drag = { id: event.pointerId, origin: coordinate(event), value: getValue(), size };
    event.preventDefault();
  });
  handle.addEventListener('pointermove', event => {
    if (!drag || drag.id !== event.pointerId) return;
    setValue(drag.value + (coordinate(event) - drag.origin) * 100 / drag.size);
  });
  handle.addEventListener('lostpointercapture', () => {
    drag = null;
    handle.classList.remove('dragging');
    saveLayout();
  });
  handle.addEventListener('dblclick', () => {
    setValue(resetValue);
    saveLayout();
  });
  handle.addEventListener('keydown', event => {
    const less = horizontalMovement ? 'ArrowLeft' : 'ArrowUp';
    const more = horizontalMovement ? 'ArrowRight' : 'ArrowDown';
    let value;
    if (event.key === less || event.key === more) {
      value = getValue() + (event.key === more ? 1 : -1) * (event.shiftKey ? 10 : 3);
    } else if (event.key === 'Home') value = Number(handle.getAttribute('aria-valuemin'));
    else if (event.key === 'End') value = Number(handle.getAttribute('aria-valuemax'));
    else if (event.key === 'Enter') value = resetValue;
    else return;
    event.preventDefault();
    setValue(value);
    saveLayout();
  });
}

function viewKey() {
  return wideScreen.matches ? 'desktopView' : 'mobileView';
}

function refreshViewControls() {
  const view = workspace.dataset[viewKey()];
  document.querySelectorAll('.view-switch button').forEach(button => {
    button.setAttribute('aria-pressed', button.dataset.view === view);
  });
}

function setView(view) {
  workspace.dataset[viewKey()] = view;
  refreshViewControls();
}

function showPane(pane) {
  if (workspace.dataset[viewKey()] !== 'split') setView(pane);
}

function initWorkspace() {
  try {
    const saved = JSON.parse(localStorage.getItem(layoutKey)) || {};
    if (Number.isFinite(saved.readerShare)) setReaderShare(saved.readerShare);
    if (Number.isFinite(saved.lastOpenShare)) lastOpenShare = clamp(saved.lastOpenShare, 40, 98);
    if (Number.isFinite(saved.editorShare)) setEditorShare(saved.editorShare);
  } catch (_) {}
  bindSeparator(widthHandle, true, () => readerShare, setReaderShare,
    () => workspace.clientWidth - widthHandle.offsetWidth, 54);
  bindSeparator(heightHandle, false, () => editorShare, setEditorShare,
    () => editorStack.clientHeight - heightHandle.offsetHeight - consoleToggle.offsetHeight, 100);
  wideScreen.addEventListener('change', refreshViewControls);
  refreshViewControls();
}

function chapterNumber(ch) {
  if (ch.id === 'ch11-13') return '11 / 13';
  if (ch.id === 'ch19') return '19–20';
  return ch.id.slice(2).replace('-', '–');
}

function openLesson(id) {
  const index = CHAPTERS.findIndex(ch => ch.id === id);
  if (index < 0) return;
  showPane('learn');
  selectChapter(index);
}

function goPractice() {
  const [target] = CHAPTERS[currentChapterIdx].guide.practice;
  if (target === 'vis') {
    switchTab('vis');
  } else {
    showPane('code');
    jumpToTodo('TODO(' + target + ')');
  }
}

const sourceRoot = 'https://github.com/cristianhtvc/minimind-course/blob/main/minimind/';

function renderGuide(ch) {
  const guide = ch.guide;
  const src = guide.source;
  const targetIsLab = guide.practice[0] !== 'vis';
  const steps = guide.steps.map(([label, value]) =>
    `<li><strong>${escapeHtml(label)}</strong><span>${escapeHtml(value)}</span></li>`).join('');
  const uses = guide.uses.map(([id, reason]) => {
    const next = CHAPTERS.find(lesson => lesson.id === id);
    const link = next
      ? `<button class="guide-link" onclick="openLesson('${next.id}')">第 ${chapterNumber(next)} 章 · ${escapeHtml(next.guide.nav)} →</button>`
      : `<a class="guide-link" href="${sourceRoot}${id}" target="_blank" rel="noopener">${escapeHtml(id)} ↗</a>`;
    return `<dd>${link}${escapeHtml(reason)}</dd>`;
  }).join('');
  document.getElementById('lesson-guide').innerHTML = `
    <h3 class="guide-question">${escapeHtml(guide.question)}</h3>
    <p class="guide-why">${escapeHtml(guide.why)}</p>
    <ol class="example-flow" aria-label="一步步推算">${steps}</ol>
    <dl class="guide-context">
      <dt>在 MiniMind 里找这段实现</dt>
      <dd><button class="guide-link" onclick="switchTab('walk')">${escapeHtml(src.symbol)} →</button>${escapeHtml(src.note)}</dd>
      <dt>学完会用在哪里？</dt>${uses}
    </dl>
    <div class="guide-task"><p><strong>动手验证</strong>　${escapeHtml(guide.practice[1])}</p>
      <button class="guide-link" onclick="goPractice()">${targetIsLab ? '定位到练习 ' + guide.practice[0] : '打开本章实验'} →</button>
    </div>`;
}

function renderSourceExcerpt(source) {
  const loc = source.path + '#L' + source.start + '-L' + source.end;
  return `<article class="code-walk-card"><h3 class="guide-question">${escapeHtml(source.symbol)}</h3>
    <a class="guide-link" href="${sourceRoot}${loc}" target="_blank" rel="noopener">${escapeHtml(loc)} ↗</a>
    <pre class="code-walk-pre">${escapeHtml(source.code)}</pre>
    <div class="code-walk-desc">${escapeHtml(source.note)}</div></article>`;
}
