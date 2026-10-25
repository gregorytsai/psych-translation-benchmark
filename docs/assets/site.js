(function () {
  const toggle = document.querySelector('.nav-toggle');
  const nav = document.querySelector('#site-nav');
  if (toggle && nav) {
    toggle.addEventListener('click', function () {
      const open = nav.classList.toggle('open');
      toggle.setAttribute('aria-expanded', String(open));
    });
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
