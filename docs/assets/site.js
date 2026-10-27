(function () {
  const toggle = document.querySelector('.nav-toggle');
  const nav = document.querySelector('#site-nav');
  if (toggle && nav) {
    toggle.addEventListener('click', function () {
      const open = nav.classList.toggle('open');
      toggle.setAttribute('aria-expanded', String(open));
    });
  }

  const overview = document.querySelector('.overview-section');
  if (overview) {
    const chart = overview.querySelector('#overview-chart');
    const rows = chart ? Array.from(chart.querySelectorAll('.overview-row')) : [];
    const buttons = Array.from(overview.querySelectorAll('[data-overview-metric]'));
    const note = overview.querySelector('#overview-scale-note');
    const isEnglish = document.documentElement.lang === 'en';
    const notes = isEnglish ? {
      total: 'The total adds the four raw rubric components before averaging reviewers and cases; scores use a fixed 0–100 scale. Rows are ordered by the selected text metric; image scores are side-by-side reference and do not imply a significant difference.',
      terminology_locale: 'Converted to 0–100 from the 0–30 terminology and Taiwan-usage component; this is not an accuracy percentage. Rows are ordered by the selected text metric; image scores are side-by-side reference.',
      semantic_safety: 'Converted to 0–100 from the 0–30 meaning and safety component; this is not an accuracy percentage. Rows are ordered by the selected text metric; image scores are side-by-side reference.',
      fluency: 'Converted to 0–100 from the 0–20 fluency component; this is not an accuracy percentage. Rows are ordered by the selected text metric; image scores are side-by-side reference.',
      completeness_format: 'Converted to 0–100 from the 0–20 completeness and format component; this is not an accuracy percentage. Rows are ordered by the selected text metric; image scores are side-by-side reference.'
    } : {
      total: '總分由四項原始配分相加，再取評審與題目平均；分數固定 0–100。依所選文字指標排序；圖片分數並列參考，不代表顯著差異。',
      terminology_locale: '由原始 0–30 的術語與台灣用語分數換算為 0–100；不是百分正確率。依所選文字指標排序；圖片分數並列參考。',
      semantic_safety: '由原始 0–30 的語意與安全分數換算為 0–100；不是百分正確率。依所選文字指標排序；圖片分數並列參考。',
      fluency: '由原始 0–20 的流暢度分數換算為 0–100；不是百分正確率。依所選文字指標排序；圖片分數並列參考。',
      completeness_format: '由原始 0–20 的完整性與格式分數換算為 0–100；不是百分正確率。依所選文字指標排序；圖片分數並列參考。'
    };
    function dataValue(row, track, metric, suffix) {
      const key = metric.replace(/_/g, '-');
      const raw = row.getAttribute('data-' + track + '-' + key + (suffix ? '-' + suffix : ''));
      if (raw === null || raw.trim() === '') return null;
      const value = Number(raw);
      return Number.isFinite(value) ? value : null;
    }
    function displayValue(row, track, metric, value) {
      if (value === null) return '—';
      const key = metric.replace(/_/g, '-');
      const rendered = row.getAttribute('data-' + track + '-' + key + '-display');
      return rendered === null || rendered.trim() === '' ? value.toFixed(3) : rendered;
    }
    function applyOverview(metric) {
      buttons.forEach(function (button) {
        const active = button.dataset.overviewMetric === metric;
        button.classList.toggle('is-active', active);
        button.setAttribute('aria-pressed', String(active));
      });
      rows.forEach(function (row) {
        ['text', 'image'].forEach(function (track, index) {
          const value = dataValue(row, track, metric, '');
          const line = row.querySelectorAll('.overview-line')[index];
          if (!line) return;
          const valueNode = line.querySelector('.overview-value');
          const rawNode = line.querySelector('.overview-raw');
          const fill = line.querySelector('.overview-fill');
          const trackNode = line.querySelector('.overview-track');
          if (valueNode) valueNode.textContent = displayValue(row, track, metric, value);
          if (rawNode) {
            const raw = dataValue(row, track, metric, 'raw');
            const maximum = dataValue(row, track, metric, 'max');
            rawNode.textContent = metric === 'total' || raw === null || maximum === null ? '' : '(' + raw.toFixed(2) + '/' + maximum.toFixed(0) + ')';
          }
          if (fill) fill.style.width = Math.max(0, Math.min(100, value === null ? 0 : value)).toFixed(3) + '%';
          if (trackNode) trackNode.setAttribute('aria-label', (isEnglish ? (track === 'text' ? 'Text' : 'Image') + ' score ' : (track === 'text' ? '文字' : '圖片') + '分數 ') + displayValue(row, track, metric, value) + (isEnglish ? ' (0–100)' : '（0–100）'));
        });
      });
      rows.slice().sort(function (a, b) {
        const av = dataValue(a, 'text', metric, '');
        const bv = dataValue(b, 'text', metric, '');
        if (av === null && bv === null) return Number(a.dataset.defaultOrder) - Number(b.dataset.defaultOrder);
        if (av === null) return 1;
        if (bv === null) return -1;
        return bv - av || Number(a.dataset.defaultOrder) - Number(b.dataset.defaultOrder);
      }).forEach(function (row, index) {
        const rank = row.querySelector('.overview-rank');
        if (rank) rank.textContent = String(index + 1);
        if (chart) chart.appendChild(row);
      });
      if (note) note.textContent = notes[metric] || notes.total;
    }
    buttons.forEach(function (button) {
      button.addEventListener('click', function () { applyOverview(button.dataset.overviewMetric || 'total'); });
    });
    applyOverview('total');
  }

  const search = document.querySelector('#result-search');
  const track = document.querySelector('#result-track');
  const model = document.querySelector('#result-model');
  const count = document.querySelector('#result-count');
  const rows = Array.from(document.querySelectorAll('.result-row'));
  if (!search || !track || !model || !count || !rows.length) return;

  function applyFilters() {
    const q = search.value.trim().toLocaleLowerCase();
    const trackValue = track.value;
    const modelValue = model.value;
    let visible = 0;
    rows.forEach(function (row) {
      const matchesQuery = !q || (row.dataset.search || '').toLocaleLowerCase().includes(q);
      const matchesTrack = trackValue === 'all' || row.dataset.track === trackValue;
      const matchesModel = modelValue === 'all' || row.dataset.model === modelValue;
      const show = matchesQuery && matchesTrack && matchesModel;
      row.hidden = !show;
      if (show) visible += 1;
    });
    const template = count.dataset.template || '顯示 {visible} / {total} 筆';
    count.textContent = template.replace('{visible}', String(visible)).replace('{total}', String(rows.length));
  }
  search.addEventListener('input', applyFilters);
  track.addEventListener('change', applyFilters);
  model.addEventListener('change', applyFilters);
}());
