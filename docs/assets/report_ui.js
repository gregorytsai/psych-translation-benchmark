(function () {
  'use strict';

  const section = document.querySelector('.overview-section');
  if (!section) return;

  const chart = section.querySelector('.overview-chart');
  const table = section.querySelector('#overview-table');
  const rows = Array.from(chart.querySelectorAll('.overview-row'));
  const tableRows = new Map(Array.from(table.tBodies[0].rows, row => [row.dataset.modelId, row]));
  const controls = {
    metric: section.querySelectorAll('[data-overview-metric]'),
    track: section.querySelectorAll('[data-overview-track]'),
    sort: section.querySelectorAll('[data-overview-sort]')
  };
  const selected = { metric: 'total', track: 'both', sort: 'score_desc' };
  const english = document.documentElement.lang === 'en';
  const trackName = english ? { text: 'Text', image: 'Image' } : { text: '文字', image: '圖片' };

  function numberAttribute(row, track, key) {
    const raw = row.getAttribute('data-' + track + '-' + key);
    return raw === null || raw === '' ? null : Number(raw);
  }

  function formatComponent(row, track, metric) {
    if (metric === 'total') return '';
    const value = numberAttribute(row, track, metric + '-raw');
    const maximum = numberAttribute(row, track, metric + '-max');
    if (value === null || maximum === null) return '';
    return '(' + Number(value.toFixed(1)) + '/' + maximum + ')';
  }

  function setScores(row, tableRow, track) {
    const line = row.querySelector('[data-overview-line-track="' + track + '"]');
    const cell = tableRow.querySelector('.overview-table-score-cell[data-table-track="' + track + '"]');
    const value = numberAttribute(row, track, selected.metric);
    const label = row.getAttribute('data-' + track + '-' + selected.metric + '-display') || '—';
    const raw = formatComponent(row, track, selected.metric);
    line.querySelector('.overview-value').textContent = label;
    line.querySelector('.overview-raw').textContent = raw;
    line.querySelector('.overview-fill').style.width = Math.min(100, Math.max(0, value ?? 0)) + '%';
    line.querySelector('.overview-track').setAttribute('aria-label', trackName[track] + ' ' + label + (english ? ' out of 100' : ' 分，滿分 100 分'));
    cell.querySelector('.overview-table-score').textContent = label;
    cell.querySelector('.overview-table-raw').textContent = raw;
  }

  function update() {
    const labels = {};
    Object.entries(controls).forEach(([group, buttons]) => {
      buttons.forEach(button => {
        const chosen = button.getAttribute('data-overview-' + group) === selected[group];
        button.setAttribute('aria-pressed', String(chosen));
        button.classList.toggle('is-active', chosen);
        if (chosen) labels[group] = button.textContent.trim();
      });
    });

    const primary = selected.track === 'image' ? 'image' : 'text';
    const byTime = selected.sort === 'time_asc';
    const sortKey = byTime ? 'warm' : selected.metric;
    const ranked = rows.map(row => ({ row, value: numberAttribute(row, primary, sortKey) }));
    ranked.sort((a, b) => {
      if (a.value === b.value) return Number(a.row.dataset.defaultOrder) - Number(b.row.dataset.defaultOrder);
      if (a.value === null) return 1;
      if (b.value === null) return -1;
      return byTime ? a.value - b.value : b.value - a.value;
    });

    let rank = 0;
    ranked.forEach(({ row, value }, index) => {
      if (index === 0 || value !== ranked[index - 1].value) rank = index + 1;
      const tableRow = tableRows.get(row.dataset.modelId);
      row.querySelector('.overview-rank').textContent = value === null ? '—' : rank;
      tableRow.querySelector('.overview-table-rank').textContent = value === null ? '—' : rank;
      setScores(row, tableRow, 'text');
      setScores(row, tableRow, 'image');
      chart.appendChild(row);
      table.tBodies[0].appendChild(tableRow);
    });

    const trackAttributes = ['data-overview-line-track', 'data-overview-speed-track', 'data-table-track', 'data-overview-legend-track'];
    trackAttributes.forEach(attribute => {
      section.querySelectorAll('[' + attribute + ']').forEach(item => {
        item.hidden = selected.track !== 'both' && item.getAttribute(attribute) !== selected.track;
      });
    });
    table.querySelectorAll('thead [data-table-measure]').forEach(column => {
      if (column.dataset.tableTrack === primary && column.dataset.tableMeasure === (byTime ? 'time' : 'score')) {
        column.setAttribute('aria-sort', byTime ? 'ascending' : 'descending');
      } else {
        column.removeAttribute('aria-sort');
      }
    });

    const scope = labels.track + ' · ' + labels.metric;
    section.querySelector('#overview-heading').textContent = (english ? 'Seven-model comparison · ' : '七款模型視覺總覽 · ') + scope;
    section.querySelector('#overview-table-heading').textContent = (english ? 'Model comparison table · ' : '模型比較表 · ') + scope;
    const sortMetric = byTime ? (english ? 'response time' : '回應時間') : labels.metric;
    section.querySelector('#overview-sort-note').textContent = english
      ? 'Ordered by ' + trackName[primary].toLowerCase() + ' ' + sortMetric.toLowerCase() + (byTime ? ', shortest first. ' : ', highest first. ') + 'Equal values share a rank.'
      : '依' + trackName[primary] + sortMetric + (byTime ? '由短到長排列。' : '由高到低排列。') + '原始數值相同時並列。';
    section.querySelector('#overview-scale-note').textContent = selected.metric === 'total'
      ? (english ? 'The total is the mean of the four-part rubric score, out of 100. Higher scores indicate fewer or less serious issues in the translation.' : '總分是四項評分加總後的平均，滿分 100 分。得分越高，表示評審在譯文中辨識出的問題越少或越輕微。')
      : (english ? 'Component scores are scaled to 100 for comparison. Parentheses show the mean on the original rubric scale.' : '分項換算為 100 分以便比較。括號內為原始平均分與該項滿分。');
  }

  Object.entries(controls).forEach(([group, buttons]) => {
    buttons.forEach(button => button.addEventListener('click', () => {
      selected[group] = button.getAttribute('data-overview-' + group);
      update();
    }));
  });
  update();
}());
