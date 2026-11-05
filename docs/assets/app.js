/* app.js - Chicago transit ridership dashboard.
   Reads data/dashboard.json (analysis + backtest) and data/model.json (exported trees). */
(async function () {
  'use strict';
  const fmt = Viz.fmt;
  Viz.initTheme(document.getElementById('theme-toggle'));

  let D;
  let M;
  try {
    [D, M] = await Promise.all([Viz.loadJSON('data/dashboard.json'), Viz.loadJSON('data/model.json')]);
  } catch (err) {
    document.getElementById('meta-dates').textContent = 'Could not load the data (' + err.message + ').';
    return;
  }

  const DOW = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
  const DOW_SHORT = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const MODELS = ['gbm', 'ridge', 'mean4', 'naive'];
  const SLOT = { gbm: 0, ridge: 1, mean4: 2, naive: 3 }; // colour follows the model, never its rank
  const metric = Object.fromEntries(D.backtest.metrics.map((m) => [m.model, m]));
  const name = (k) => metric[k].name;
  const dayLabel = (iso) => new Date(iso + 'T00:00:00').toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
  const el = (id) => document.getElementById(id);

  /* ---------- Header, hero and KPI tiles ---------- */
  el('meta-dates').textContent = 'Data ' + fmt.date(D.meta.data_start) + ' – ' + fmt.date(D.meta.data_end);
  const g = metric.gbm;
  el('hero-value').textContent = fmt.pct(g.wape, 1);
  el('hero-caption').textContent =
    'WAPE over ' + fmt.int(g.n) + ' out-of-sample forecasts (' + fmt.month(D.backtest.start) + ' – ' + fmt.month(D.backtest.end) +
    '). That is ' + fmt.pct(1 - g.wape / metric.naive.wape, 0) + ' less error than a seasonal-naive forecast and ' +
    fmt.pct(1 - g.wape / metric.mean4.wape, 0) + ' less than the 4-week weekday average planners often use.';

  const K = D.kpis;
  const tone = (x) => (x >= 0 ? 'up-good' : 'down-bad');
  Viz.tile('kpi-weekday', {
    label: 'Weekday riders, last 90 days', value: fmt.compact(K.weekday_avg_90d, 2),
    delta: fmt.signedPct(K.weekday_avg_90d_yoy), tone: tone(K.weekday_avg_90d_yoy), note: 'vs. a year earlier',
  });
  Viz.tile('kpi-rides', {
    label: 'Rides, last 12 months', value: fmt.compact(K.rides_12m, 1),
    delta: fmt.signedPct(K.rides_12m_yoy), tone: tone(K.rides_12m_yoy), note: 'year over year',
  });
  Viz.tile('kpi-recovery', {
    label: 'Share of 2019 ridership', value: fmt.pct(K.recovery_12m_vs_2019, 0),
    note: 'Weekdays ' + fmt.pct(K.recovery_by_day_type.weekday, 0) + ' · Sundays ' + fmt.pct(K.recovery_by_day_type.sunday_holiday, 0),
  });
  Viz.tile('kpi-coverage', {
    label: '80% range hit rate', value: fmt.pct(g.coverage, 0),
    note: 'of actual days fell inside the forecast range',
  });

  /* ---------- History ---------- */
  const H = D.history;
  const covid = '2020-03-21';
  Viz.card('card-history', {
    build: (t) => ({
      data: [
        { x: H.week, y: H.bus, name: 'Bus', type: 'scatter', mode: 'lines', line: { color: t.series[0], width: 2 },
          hovertemplate: '<b>%{y:,.0f}</b> bus<extra></extra>' },
        { x: H.week, y: H.rail, name: 'Rail', type: 'scatter', mode: 'lines', line: { color: t.series[1], width: 2 },
          hovertemplate: '<b>%{y:,.0f}</b> rail<extra></extra>' },
      ],
      layout: {
        showlegend: true, hovermode: 'x unified', margin: { t: 34 },
        xaxis: Object.assign(Viz.crosshair(t), { type: 'date', hoverformat: 'Week of %b %-d, %Y' }),
        yaxis: { tickformat: '~s', rangemode: 'tozero' },
        shapes: [{ type: 'line', x0: covid, x1: covid, yref: 'paper', y0: 0, y1: 1, line: { color: t.baseline, width: 1 } }],
        annotations: [{ x: covid, y: 1, yref: 'paper', text: 'Stay-at-home order', showarrow: false, xanchor: 'left', yanchor: 'top', xshift: 4, font: { color: t.text2, size: 11 } }],
      },
    }),
    table: () => ({
      columns: ['Week of', 'Bus', 'Rail', 'Total'],
      rows: H.week.map((w, i) => [w, fmt.int(H.bus[i]), fmt.int(H.rail[i]), fmt.int(H.bus[i] + H.rail[i])]).reverse(),
    }),
  });

  const R = D.recovery;
  const REC = [['weekday', 'Weekdays'], ['saturday', 'Saturdays'], ['sunday_holiday', 'Sundays and holidays']];
  Viz.card('card-recovery', {
    build: (t) => ({
      data: REC.map(([k, label], i) => ({
        x: R.month, y: R[k], name: label, type: 'scatter', mode: 'lines', line: { color: t.series[i], width: 2 },
        hovertemplate: '<b>%{y:.0%}</b> ' + label.toLowerCase() + '<extra></extra>',
      })),
      layout: {
        showlegend: true, hovermode: 'x unified', margin: { t: 34 },
        xaxis: Object.assign(Viz.crosshair(t), { type: 'date', hoverformat: '%B %Y' }),
        yaxis: { tickformat: '.0%', rangemode: 'tozero' },
        shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 1, y1: 1, line: { color: t.baseline, width: 1 } }],
      },
    }),
    table: () => ({
      columns: ['Month', 'Weekdays', 'Saturdays', 'Sundays and holidays'],
      rows: R.month.map((m, i) => [fmt.month(m), fmt.pct(R.weekday[i], 0), fmt.pct(R.saturday[i], 0), fmt.pct(R.sunday_holiday[i], 0)]).reverse(),
    }),
  });

  const annual = Object.fromEntries(D.annual.year.map((y, i) => [y, D.annual.total[i]]));
  const lastYear = D.annual.year[D.annual.year.length - 1];
  const insights = [
    ['Weekends are back, weekdays are not',
      'Over the last 12 months Sundays ran at ' + fmt.pct(K.recovery_by_day_type.sunday_holiday, 0) + ' and Saturdays at ' +
      fmt.pct(K.recovery_by_day_type.saturday, 0) + ' of 2019 levels, but weekdays only at ' + fmt.pct(K.recovery_by_day_type.weekday, 0) +
      '. Hybrid work changed the commute more than leisure travel.'],
    ['Still growing',
      fmt.compact(annual[lastYear], 0) + ' rides in ' + lastYear + ', up from ' + fmt.compact(annual[2022], 0) + ' in 2022. The last 12 months were ' +
      fmt.signedPct(K.rides_12m_yoy) + ' year over year, so any model has to cope with a moving level.'],
    ['Bus carries more of the load',
      'Buses delivered ' + fmt.pct(K.bus_share_12m, 0) + ' of rides in the last 12 months, against ' + fmt.pct(K.bus_share_2019, 0) +
      ' before the pandemic. Rail, which is more tied to downtown offices, has recovered more slowly.'],
  ];
  const box = el('insights-history');
  insights.forEach(([title, text]) => {
    const div = document.createElement('div');
    div.className = 'insight';
    const strong = document.createElement('strong');
    strong.textContent = title;
    div.append(strong, document.createTextNode(text));
    box.appendChild(div);
  });

  /* ---------- Patterns ---------- */
  const S = D.seasonality;
  Viz.card('card-season', {
    build: (t) => ({
      data: [{
        type: 'heatmap', z: S.values, x: S.month.map((m) => MONTHS[m - 1]), y: S.dow.map((d) => DOW_SHORT[d]),
        colorscale: t.seqScale, xgap: 2, ygap: 2, hoverongaps: false,
        customdata: S.dow.map((d) => S.month.map(() => DOW[d])),
        hovertemplate: '<b>%{z:,.0f}</b> riders<br>%{customdata}s in %{x}<extra></extra>',
        colorbar: { thickness: 10, outlinewidth: 0, tickformat: '~s', tickfont: { color: t.muted, size: 11 }, len: 0.95 },
      }],
      layout: {
        xaxis: { showline: false, showgrid: false },
        yaxis: { autorange: 'reversed', showgrid: false },
        margin: { r: 4 },
      },
    }),
    table: () => ({
      columns: ['Weekday', ...S.month.map((m) => MONTHS[m - 1])],
      rows: S.dow.map((d, i) => [DOW[d], ...S.values[i].map((v) => (v == null ? '' : fmt.compact(v, 2)))]),
    }),
  });

  const HE = D.holidays;
  Viz.card('card-holidays', {
    build: (t) => ({
      data: [{
        type: 'bar', orientation: 'h', y: HE.map((h) => h.name), x: HE.map((h) => h.effect),
        width: Viz.barWidth('chart-holidays', HE.length, true),
        marker: { color: HE.map((h) => (h.effect < 0 ? t.divNeg : t.divPos)), cornerradius: 4 },
        text: HE.map((h) => fmt.signedPct(h.effect, 0)), textposition: 'outside', cliponaxis: false,
        textfont: { color: t.text2, size: 11 },
        customdata: HE.map((h) => h.days),
        hovertemplate: '<b>%{x:+.0%}</b> vs. a typical day<br>%{y} (%{customdata} days)<extra></extra>',
      }],
      layout: {
        xaxis: { tickformat: '+.0%', showgrid: true, showline: false, zeroline: true, zerolinecolor: t.baseline, range: [-0.92, 0.62] },
        yaxis: { showgrid: false, tickfont: { color: t.text2, size: 11 } },
        margin: { r: 10 },
      },
    }),
    table: () => ({
      columns: ['Day', 'Change vs. typical', 'Days averaged'],
      rows: HE.slice().reverse().map((h) => [h.name, fmt.signedPct(h.effect, 1), h.days]),
    }),
  });

  const W = D.weather;
  const WEATHER = [['temperature', 'Temperature'], ['snow', 'Snowfall'], ['rain', 'Rain (snow-free days)']];
  const weatherCharts = WEATHER.map(([key, title]) => Viz.chart('chart-weather-' + key, (t) => {
    const rows = W[key];
    return {
      data: [{
        type: 'bar', orientation: 'h', y: rows.map((r) => r.band), x: rows.map((r) => r.effect),
        width: Viz.barWidth('chart-weather-' + key, rows.length, true, 18),
        marker: { color: rows.map((r) => (r.effect < 0 ? t.divNeg : t.divPos)), cornerradius: 4 },
        text: rows.map((r) => fmt.signedPct(r.effect, 1)), textposition: 'outside', cliponaxis: false,
        textfont: { color: t.text2, size: 11 }, customdata: rows.map((r) => r.days),
        hovertemplate: '<b>%{x:+.1%}</b> vs. a typical day<br>%{y} (%{customdata} days)<extra></extra>',
      }],
      layout: {
        title: { text: title, x: 0, xanchor: 'left', font: { size: 13, color: t.text } },
        margin: { t: 34, r: 12 },
        xaxis: { tickformat: '+.0%', showline: false, showgrid: true, zeroline: true, zerolinecolor: t.baseline, range: [-0.24, 0.08], dtick: 0.1 },
        yaxis: { autorange: 'reversed', showgrid: false, tickfont: { color: t.text2, size: 11 } },
      },
    };
  }));
  Viz.card('card-weather', {
    table: () => ({
      columns: ['Condition', 'Band', 'Median change', 'Days'],
      rows: WEATHER.flatMap(([key, title]) => W[key].map((r) => [title, r.band, fmt.signedPct(r.effect, 1), r.days])),
    }),
    onShow: () => weatherCharts.forEach((c) => c.update()),
  });

  /* ---------- Backtest ---------- */
  const view = { h: '7', model: 'gbm' };
  const BT = D.backtest;
  function coverageOf(series, m) {
    let hit = 0;
    let n = 0;
    series.actual.forEach((y, i) => {
      if (y == null || m.lo[i] == null) return;
      n += 1;
      if (y >= m.lo[i] && y <= m.hi[i]) hit += 1;
    });
    return n ? hit / n : NaN;
  }
  function backtestText() {
    const s = BT.series[view.h];
    const h = Number(view.h);
    const wape = BT.by_horizon[view.model][BT.by_horizon.h.indexOf(h)];
    const cov = coverageOf(s, s.models[view.model]);
    const days = h === 1 ? '1 day' : h + ' days';
    el('backtest-title').textContent = name(view.model) + ': forecasts made ' + days + ' ahead vs. actual';
    el('backtest-sub').textContent = fmt.date(s.date[0]) + ' – ' + fmt.date(s.date[s.date.length - 1]) + ', shaded band = 80% range';
    el('backtest-readout').textContent = 'At ' + days + ' ahead: WAPE ' + fmt.pct(wape, 1) + ' · range covered ' + fmt.pct(cov, 0) + ' of days';
  }
  const backtestCard = Viz.card('card-backtest', {
    build: (t) => {
      const s = BT.series[view.h];
      const m = s.models[view.model];
      const color = t.series[SLOT[view.model]];
      return {
        data: [
          { x: s.date, y: m.hi, type: 'scatter', mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', showlegend: false },
          { x: s.date, y: m.lo, type: 'scatter', mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: Viz.alpha(color, 0.16),
            name: '80% range', hoverinfo: 'skip' },
          { x: s.date, y: s.actual, name: 'Actual', type: 'scatter', mode: 'lines', line: { color: t.text2, width: 1.5 },
            hovertemplate: '<b>%{y:,.0f}</b> actual<extra></extra>' },
          { x: s.date, y: m.pred, name: 'Forecast', type: 'scatter', mode: 'lines', line: { color: color, width: 2 },
            customdata: m.lo.map((lo, i) => [lo, m.hi[i]]),
            hovertemplate: '<b>%{y:,.0f}</b> forecast (range %{customdata[0]:,.0f}–%{customdata[1]:,.0f})<extra></extra>' },
        ],
        layout: {
          showlegend: true, hovermode: 'x unified', margin: { t: 34 }, legend: { traceorder: 'normal' },
          xaxis: Object.assign(Viz.crosshair(t), { type: 'date', hoverformat: '%a, %b %-d, %Y' }),
          yaxis: { tickformat: '~s', rangemode: 'tozero' },
        },
      };
    },
    table: () => {
      const s = BT.series[view.h];
      const m = s.models[view.model];
      return {
        columns: ['Date', 'Actual', 'Forecast', '80% low', '80% high', 'Error'],
        rows: s.date.map((d, i) => [d, fmt.int(s.actual[i]), fmt.int(m.pred[i]), fmt.int(m.lo[i]), fmt.int(m.hi[i]),
          fmt.signedPct(m.pred[i] / s.actual[i] - 1, 1)]).reverse(),
      };
    },
  });

  const BH = BT.by_horizon;
  const horizonCard = Viz.card('card-horizon', {
    build: (t) => ({
      data: MODELS.map((k) => ({
        x: BH.h, y: BH[k], name: name(k), type: 'scatter', mode: 'lines',
        line: { color: t.series[SLOT[k]], width: k === view.model ? 3 : 2 },
        hovertemplate: '<b>%{y:.1%}</b> ' + name(k) + '<extra></extra>',
      })),
      layout: {
        showlegend: true, hovermode: 'x unified', margin: { t: 34, r: 128 },
        xaxis: Object.assign(Viz.crosshair(t), { title: { text: 'Days ahead' }, dtick: 1, range: [0.7, 14.3] }),
        yaxis: { tickformat: '.0%', rangemode: 'tozero' },
        annotations: MODELS.map((k) => ({
          xref: 'paper', x: 1, y: BH[k][BH[k].length - 1], text: name(k), showarrow: false,
          xanchor: 'left', xshift: 6, font: { color: t.text2, size: 11 },
        })),
      },
    }),
    table: () => ({
      columns: ['Days ahead', ...MODELS.map(name)],
      rows: BH.h.map((h, i) => [h, ...MODELS.map((k) => fmt.pct(BH[k][i], 2))]),
    }),
  });

  Viz.renderTable(el('metrics-table'), {
    columns: ['Model', 'WAPE', 'MAPE', 'MAE', 'Bias', 'Coverage'],
    rows: MODELS.map((k) => {
      const m = metric[k];
      return [m.name, fmt.pct(m.wape, 1), fmt.pct(m.mape, 1), fmt.int(m.mae), fmt.signedPct(m.bias, 1), fmt.pct(m.coverage, 0)];
    }),
  });
  const TYPE = { W: 'Weekday', A: 'Saturday', U: 'Sunday/holiday' };
  Viz.renderTable(el('misses-table'), {
    columns: ['Date', 'Service', 'Actual', 'Forecast', 'Error'],
    rows: BT.misses.slice(0, 6).map((r) => [dayLabel(r.date) + ', ' + r.date.slice(0, 4), TYPE[r.day_type], fmt.compact(r.actual, 0),
      fmt.compact(r.forecast, 0), fmt.signedPct(r.error, 0)]),
  });

  const hButtons = Array.from(el('horizon-control').querySelectorAll('button'));
  hButtons.forEach((b) => b.addEventListener('click', () => {
    view.h = b.dataset.h;
    hButtons.forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    backtestText();
    backtestCard.update();
  }));
  el('model-control').addEventListener('change', (e) => {
    view.model = e.target.value;
    backtestText();
    backtestCard.update();
    horizonCard.update();
  });
  backtestText();

  /* ---------- Forward forecast + what-if planner ---------- */
  const F = D.forecast;
  const idx = Object.fromEntries(M.features.map((f, i) => [f, i]));
  const baseX = M.rows.map((r) => r.x.slice());
  let scenX = baseX.map((x) => x.slice());
  const run = (X) => X.map((x) => GBM.forecast(M, x, x[idx.h]));
  const basePred = run(baseX);
  let scenPred = basePred;
  let selected = 0;
  const changed = () => scenX.some((x, i) => x.some((v, j) => v !== baseX[i][j]));

  el('planner-lede').textContent =
    'The CTA publishes ridership with a lag, so the latest data runs to ' + fmt.date(F.origin) + '. Below is the model\'s forecast for the 14 days ' +
    'after that, using the weather each day actually had. Pick a day and change its conditions: the exported model (' + M.trees.length +
    ' trees) re-runs in your browser.';
  el('forecast-sub').textContent = 'Last 4 weeks of actuals, then ' + fmt.date(F.days[0].date) + ' – ' + fmt.date(F.days[F.days.length - 1].date);

  const forecastCard = Viz.card('card-forecast', {
    build: (t) => {
      const dates = F.days.map((d) => d.date);
      const blue = t.series[0];
      const data = [
        { x: F.recent.date, y: F.recent.actual, name: 'Actual', type: 'scatter', mode: 'lines', line: { color: t.text2, width: 1.5 },
          hovertemplate: '<b>%{y:,.0f}</b> actual<extra></extra>' },
        { x: dates, y: scenPred.map((p) => p.hi), type: 'scatter', mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', showlegend: false },
        { x: dates, y: scenPred.map((p) => p.lo), type: 'scatter', mode: 'lines', line: { width: 0 }, fill: 'tonexty',
          fillcolor: Viz.alpha(blue, 0.16), name: '80% range', hoverinfo: 'skip' },
      ];
      if (changed()) {
        data.push({ x: dates, y: basePred.map((p) => p.pred), name: 'Before your changes', type: 'scatter', mode: 'lines',
          line: { color: t.deemph, width: 2 }, hovertemplate: '<b>%{y:,.0f}</b> before changes<extra></extra>' });
      }
      data.push({
        x: dates, y: scenPred.map((p) => p.pred), name: 'Forecast', type: 'scatter', mode: 'lines', line: { color: blue, width: 2 },
        customdata: scenPred.map((p) => [p.lo, p.hi]),
        hovertemplate: '<b>%{y:,.0f}</b> forecast (range %{customdata[0]:,.0f}–%{customdata[1]:,.0f})<extra></extra>',
      });
      data.push({
        x: [dates[selected]], y: [scenPred[selected].pred], type: 'scatter', mode: 'markers', showlegend: false, hoverinfo: 'skip',
        marker: { size: 11, color: blue, line: { color: t.surface, width: 2 } },
      });
      return {
        data,
        layout: {
          showlegend: true, hovermode: 'x unified', margin: { t: 34 }, legend: { traceorder: 'normal' },
          xaxis: Object.assign(Viz.crosshair(t), { type: 'date', hoverformat: '%a, %b %-d' }),
          yaxis: { tickformat: '~s', rangemode: 'tozero' },
          shapes: [{ type: 'line', x0: F.origin, x1: F.origin, yref: 'paper', y0: 0, y1: 1, line: { color: t.baseline, width: 1 } }],
          annotations: [{ x: F.origin, y: 1, yref: 'paper', text: 'Last published day', showarrow: false, xanchor: 'right', yanchor: 'top', xshift: -4, font: { color: t.text2, size: 11 } }],
        },
      };
    },
    table: () => ({
      columns: ['Date', 'Forecast', '80% low', '80% high', 'Avg temp', 'Precip', 'Snow'],
      rows: F.days.map((d, i) => [dayLabel(d.date), fmt.int(scenPred[i].pred), fmt.int(scenPred[i].lo), fmt.int(scenPred[i].hi),
        Math.round(scenX[i][idx.temp_mean_f]) + '°F', scenX[i][idx.precip_in].toFixed(2) + ' in', scenX[i][idx.snow_in].toFixed(1) + ' in']),
    }),
  });

  const daySelect = el('plan-day');
  F.days.forEach((d, i) => {
    const opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = dayLabel(d.date) + (baseX[i][idx.hol_major] ? ' (holiday)' : '');
    daySelect.appendChild(opt);
  });

  const inputs = { temp: el('plan-temp'), precip: el('plan-precip'), snow: el('plan-snow'), holiday: el('plan-holiday'), event: el('plan-event') };
  function syncControls() {
    const x = scenX[selected];
    inputs.temp.value = Math.round(x[idx.temp_mean_f]);
    inputs.precip.value = x[idx.precip_in];
    inputs.snow.value = x[idx.snow_in];
    inputs.holiday.value = x[idx.hol_major] ? 'major' : x[idx.hol_minor] ? 'minor' : 'none';
    inputs.event.checked = x[idx.major_event] === 1;
    el('out-temp').textContent = Math.round(x[idx.temp_mean_f]) + '°F';
    el('out-precip').textContent = x[idx.precip_in].toFixed(2) + ' in';
    el('out-snow').textContent = x[idx.snow_in].toFixed(1) + ' in';
  }
  function readout() {
    const d = F.days[selected];
    const p = scenPred[selected];
    const b = basePred[selected];
    const box = el('plan-readout');
    box.textContent = '';
    const big = document.createElement('span');
    big.className = 'big';
    big.textContent = fmt.int(p.pred) + ' riders';
    box.append(big, document.createTextNode(
      dayLabel(d.date) + ' · 80% range ' + fmt.compact(p.lo, 2) + ' – ' + fmt.compact(p.hi, 2) +
      (Math.abs(p.pred / b.pred - 1) > 0.0005 ? ' · ' + fmt.signedPct(p.pred / b.pred - 1, 1) + ' vs. the forecast with observed conditions' : ' · forecast with observed conditions')));
    // Trees cannot extrapolate: snow on a warm day is a combination the model never saw.
    const x = scenX[selected];
    if (x[idx.snow_in] > 0 && x[idx.temp_mean_f] > 38) {
      const hint = document.createElement('p');
      hint.className = 'footnote';
      hint.textContent = 'Snow only falls near or below freezing. The model has never seen snow at ' + Math.round(x[idx.temp_mean_f]) +
        '°F, so it barely reacts. Lower the temperature to see the snow effect.';
      box.appendChild(hint);
    }
  }
  function recompute() {
    scenPred = run(scenX);
    syncControls();
    readout();
    forecastCard.update();
  }
  daySelect.addEventListener('change', () => {
    selected = Number(daySelect.value);
    recompute();
  });
  inputs.temp.addEventListener('input', () => {
    const v = Number(inputs.temp.value);
    const offset = baseX[selected][idx.feels_like_min_f] - baseX[selected][idx.temp_mean_f];
    scenX[selected][idx.temp_mean_f] = v;
    scenX[selected][idx.feels_like_min_f] = v + offset;
    recompute();
  });
  inputs.precip.addEventListener('input', () => {
    scenX[selected][idx.precip_in] = Number(inputs.precip.value);
    recompute();
  });
  inputs.snow.addEventListener('input', () => {
    const v = Number(inputs.snow.value);
    scenX[selected][idx.snow_in] = v;
    if (selected + 1 < scenX.length) scenX[selected + 1][idx.snow_prev_in] = v; // snow lingers into the next day
    recompute();
  });
  inputs.holiday.addEventListener('change', () => {
    scenX[selected][idx.hol_major] = inputs.holiday.value === 'major' ? 1 : 0;
    scenX[selected][idx.hol_minor] = inputs.holiday.value === 'minor' ? 1 : 0;
    recompute();
  });
  inputs.event.addEventListener('change', () => {
    scenX[selected][idx.major_event] = inputs.event.checked ? 1 : 0;
    recompute();
  });
  el('plan-reset').addEventListener('click', () => {
    scenX = baseX.map((x) => x.slice());
    recompute();
  });
  recompute();
})();
