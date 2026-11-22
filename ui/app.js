/* SC2 replay review UI.
 *
 * Reads an exported review.json (written by analysis/review.py) and renders the
 * Terran win-probability timeline, pivotal intervals and Grad-CAM evidence.
 * Plain HTML/CSS/JS: no build step, no third-party libraries.
 */
(function () {
  'use strict';

  var SVG_NS = 'http://www.w3.org/2000/svg';
  var CHART = { width: 760, height: 340, pad: { top: 26, right: 22, bottom: 54, left: 74 } };
  var EPSILON = 1e-6;

  var state = {
    data: null,
    reviewDir: null,
    jsonUrl: null,
    isDemo: false,
    selected: -1,          // index into review.json points
    selectedInterval: -1   // index into review.json pivotal_intervals
  };

  /* ------------------------------------------------------------------ *
   * Small helpers
   * ------------------------------------------------------------------ */

  function $(id) { return document.getElementById(id); }
  function show(node, visible) { if (node) { node.hidden = !visible; } }
  function setText(node, value) { if (node) { node.textContent = value; } }

  function num(value) {
    return typeof value === 'number' && Number.isFinite(value) ? value : NaN;
  }

  function trimNum(value, digits) {
    if (!Number.isFinite(value)) { return 'n/a'; }
    return String(Number(value.toFixed(digits)));
  }

  function fmtSeconds(value) {
    return Number.isFinite(value) ? value.toFixed(1) + ' s' : 'n/a';
  }

  function fmtMinutes(seconds) {
    return Number.isFinite(seconds) ? trimNum(seconds / 60, 2) + ' min' : 'n/a';
  }

  function fmtPercent(value) {
    return Number.isFinite(value) ? (value * 100).toFixed(1) + '%' : 'n/a';
  }

  /* Probability differences are reported in percentage points, never as a
   * relative "percent change". threshold 0.1 is 10.0 pp. */
  function fmtPoints(value) {
    return Number.isFinite(value) ? (value * 100).toFixed(1) + ' pp' : 'n/a';
  }

  function labelText(label) {
    if (label === 1) { return 'Terran win (label 1)'; }
    if (label === 0) { return 'Protoss win (label 0)'; }
    if (label === undefined || label === null) { return 'n/a'; }
    return 'unknown (label ' + String(label) + ')';
  }

  function sideName(probability) {
    return Number.isFinite(probability) && probability < 0.5 ? 'Protoss' : 'Terran';
  }

  function keyActivate(handler) {
    return function (event) {
      if (event.key === 'Enter' || event.key === ' ' || event.key === 'Spacebar') {
        event.preventDefault();
        handler(event);
      }
    };
  }

  function svgEl(name, attrs) {
    var node = document.createElementNS(SVG_NS, name);
    if (attrs) {
      Object.keys(attrs).forEach(function (key) { node.setAttribute(key, attrs[key]); });
    }
    return node;
  }

  function svgText(x, y, content, attrs) {
    var node = svgEl('text', Object.assign({ x: x, y: y }, attrs || {}));
    node.textContent = content;
    return node;
  }

  function svgClear(node) {
    while (node && node.firstChild) { node.removeChild(node.firstChild); }
  }

  /* ------------------------------------------------------------------ *
   * Path resolution
   *
   * The ?review= value is resolved against the local server root, not
   * against /ui/. Image paths are resolved against the directory that
   * contains review.json.
   * ------------------------------------------------------------------ */

  function ensureTrailingSlash(url) {
    return url.pathname.slice(-1) === '/' ? url : new URL(url.href + '/');
  }

  function parseReviewParam(raw) {
    if (!raw) {
      return {
        ok: false,
        message: 'No review data selected.',
        detail: 'Add a review query parameter, for example: ui/index.html?review=ui/demo/full'
      };
    }
    if (window.location.protocol === 'file:') {
      return {
        ok: false,
        message: 'This page must be served over HTTP.',
        detail: 'Browsers block fetch() on file:// pages, so review.json cannot be read from disk directly.',
        hint: 'Run: python -m http.server 8000 --bind 127.0.0.1 in the project root, then open http://127.0.0.1:8000/ui/index.html?review=<directory>'
      };
    }
    var target;
    try {
      target = new URL(raw, new URL('/', window.location.href));
    } catch (error) {
      return { ok: false, message: 'The review parameter is not a valid path.', detail: raw };
    }
    if (target.origin !== window.location.origin) {
      return {
        ok: false,
        message: 'The review parameter must point at this server.',
        detail: 'Received origin ' + target.origin + ' but the page is served from ' + window.location.origin + '.'
      };
    }
    var isFile = /\.json$/i.test(target.pathname);
    var reviewDir = isFile ? new URL('.', target) : ensureTrailingSlash(target);
    var jsonUrl = isFile ? target : new URL('review.json', reviewDir);
    return { ok: true, reviewDir: reviewDir, jsonUrl: jsonUrl };
  }

  /* ------------------------------------------------------------------ *
   * Lookup helpers
   * ------------------------------------------------------------------ */

  function pointsOf(data) {
    return data && Array.isArray(data.points) ? data.points : [];
  }

  function intervalsOf(data) {
    return data && Array.isArray(data.pivotal_intervals) ? data.pivotal_intervals : [];
  }

  function explanationsOf(data) {
    return data && Array.isArray(data.explanations) ? data.explanations : [];
  }

  function findPointIndex(points, cutoff) {
    if (!Number.isFinite(cutoff)) { return -1; }
    for (var index = 0; index < points.length; index += 1) {
      if (Math.abs(num(points[index].cutoff_seconds) - cutoff) < EPSILON) { return index; }
    }
    return -1;
  }

  function findExplanation(data, cutoff) {
    if (!Number.isFinite(cutoff)) { return null; }
    var explanations = explanationsOf(data);
    for (var index = 0; index < explanations.length; index += 1) {
      if (Math.abs(num(explanations[index].cutoff_seconds) - cutoff) < EPSILON) {
        return explanations[index];
      }
    }
    return null;
  }

  function describeCutoffs(explanations) {
    return explanations.map(function (explanation) {
      return fmtMinutes(num(explanation.cutoff_seconds));
    }).join(', ');
  }

  /* ------------------------------------------------------------------ *
   * Load flow
   * ------------------------------------------------------------------ */

  function renderLoadProblem(problem) {
    show($('app'), false);
    show($('help-panel'), true);
    setText($('help-message'), problem.message || 'review.json could not be loaded.');
    setText($('help-detail'), problem.detail || '');
    setText($('help-hint'), problem.hint || '');
    var banner = $('banner');
    banner.className = 'banner banner-error';
    setText(banner, (problem.message || 'review.json could not be loaded.') +
      (problem.detail ? ' ' + problem.detail : ''));
    show(banner, true);
  }

  function main() {
    var parsed = parseReviewParam(new URLSearchParams(window.location.search).get('review'));
    if (!parsed.ok) {
      renderLoadProblem(parsed);
      return;
    }
    state.reviewDir = parsed.reviewDir;
    state.jsonUrl = parsed.jsonUrl;

    fetch(parsed.jsonUrl.href, { cache: 'no-store' })
      .then(function (response) {
        if (!response.ok) {
          throw new Error('HTTP ' + response.status + ' ' + response.statusText);
        }
        return response.json();
      })
      .then(function (data) {
        if (!data || typeof data !== 'object' || Array.isArray(data)) {
          throw new Error('review.json did not contain a JSON object.');
        }
        start(data);
      })
      .catch(function (error) {
        renderLoadProblem({
          message: 'Could not load review.json.',
          detail: parsed.jsonUrl.href + ' - ' + error.message,
          hint: 'Check that the review directory exists and contains review.json, and that the server was started in the project root.'
        });
      });
  }

  function start(data) {
    state.data = data;
    state.isDemo = data.demo === true || /(^|\/)demo(\/|$)/i.test(state.jsonUrl.pathname);
    show($('help-panel'), false);
    show($('banner'), false);
    show($('app'), true);
    show($('demo-banner'), state.isDemo);

    renderMeta(data);
    renderRunInfo(data);
    var points = pointsOf(data);
    renderChart(points, data);
    renderIntervals(points, data);
    renderStaticNotes(data);

    if (state.isDemo) {
      var banner = $('banner');
      banner.className = 'banner banner-demo';
      setText(banner, 'Demo data: synthetic values for interface testing only, not model predictions.');
      show(banner, true);
    }

    selectInitial(points, data);
    window.removeEventListener('resize', scheduleRedraw);
    window.addEventListener('resize', scheduleRedraw);
  }

  /* ------------------------------------------------------------------ *
   * Static sections
   * ------------------------------------------------------------------ */

  function gridText(grid) {
    if (!grid || typeof grid !== 'object') { return 'not recorded'; }
    var min = num(grid.min_seconds);
    var max = num(grid.max_seconds);
    var step = num(grid.step_seconds);
    if (!Number.isFinite(min) || !Number.isFinite(max)) { return 'not recorded'; }
    var range = min === max
      ? trimNum(min, 2) + ' s'
      : trimNum(min, 2) + '\u2013' + trimNum(max, 2) + ' s';
    return range + (Number.isFinite(step) ? ' (step ' + trimNum(step, 2) + ' s)' : '');
  }

  /* Run provenance is built from checkpoint_epoch, time_grid and limitations.
   * No run name, epoch or probability is hardcoded in this file. */
  function renderRunInfo(data) {
    var grid = (data.time_grid && typeof data.time_grid === 'object') ? data.time_grid : null;
    var epochs = typeof data.checkpoint_epoch === 'number' ? data.checkpoint_epoch : NaN;
    var overridden = Boolean(grid && grid.override_applied === true);

    setText($('run-epochs'), Number.isFinite(epochs)
      ? trimNum(epochs, 0) + (epochs === 1 ? ' epoch' : ' epochs')
      : 'not recorded');
    setText($('run-train-grid'), gridText(grid ? grid.training : null));
    setText($('run-infer-grid'), gridText(grid ? grid.inference : null) +
      (overridden ? ' \u2014 export override' : ''));
    var frames = grid ? num(grid.frames) : NaN;
    setText($('run-frames'), Number.isFinite(frames)
      ? trimNum(frames, 0) + ' frames per prediction' : 'not recorded');

    var overrideNote = $('run-override-note');
    show(overrideNote, overridden);
    setText(overrideNote, overridden
      ? 'This export evaluates cutoffs outside the training time grid. Predictive ability at those '
        + 'cutoffs has not been validated; they are shown for pipeline testing only.'
      : '');

    setText($('run-limitations-note'), grid
      ? 'Calibration and interpretation caveats are listed under Limitations below.'
      : 'This export does not record a time grid. '
        + (data.limitations ? String(data.limitations) : 'No limitations text was recorded either.'));
  }

  function renderMeta(data) {
    setText($('meta-match'), data.replay_id === undefined || data.replay_id === null ? 'n/a' : String(data.replay_id));
    setText($('meta-pov'), data.focus_player_id === undefined || data.focus_player_id === null
      ? 'n/a' : 'Player ' + String(data.focus_player_id));
    setText($('meta-result'), labelText(data.label));
    setText($('meta-observation'), data.observation_setting ? String(data.observation_setting) : 'n/a');
    setText($('meta-sample'), data.sample_id ? String(data.sample_id) : 'n/a');
    setText($('meta-epoch'), data.checkpoint_epoch === undefined || data.checkpoint_epoch === null
      ? 'n/a' : 'epoch ' + String(data.checkpoint_epoch));

    var points = pointsOf(data);
    var last = points.length ? points[points.length - 1] : null;
    var lastProbability = last ? num(last.terran_win_probability) : NaN;
    if (Number.isFinite(lastProbability)) {
      var agrees = ((lastProbability >= 0.5 ? 1 : 0) === data.label);
      setText($('meta-prediction'),
        'At the final cutoff (' + fmtMinutes(num(last.cutoff_seconds)) + ') the model predicts a ' +
        sideName(lastProbability) + ' win with ' + fmtPercent(lastProbability) + ' for Terran; this ' +
        (agrees ? 'agrees with' : 'differs from') + ' the actual result.');
    } else {
      setText($('meta-prediction'), 'No prediction point is available in this export.');
    }
    setText($('meta-source'), data.checkpoint ? 'Checkpoint: ' + String(data.checkpoint) : '');
  }

  function renderStaticNotes(data) {
    var threshold = num(data.threshold);
    var merge = num(data.merge_seconds);
    var parts = [];
    if (Number.isFinite(threshold)) {
      parts.push('Pivotal intervals mark adjacent probability changes of at least ' +
        fmtPoints(threshold) + ' (threshold ' + trimNum(threshold, 4) + ').');
    } else {
      parts.push('The pivotal-interval threshold is not recorded in this export.');
    }
    if (Number.isFinite(merge)) {
      parts.push('Detected changes closer than ' + trimNum(merge, 2) + ' s are merged into one interval.');
    }
    setText($('threshold-note'), parts.join(' '));
    setText($('limitations-text'), data.limitations ? String(data.limitations) : '');
  }

  /* ------------------------------------------------------------------ *
   * Chart
   * ------------------------------------------------------------------ */

  var MOBILE_PAD = { top: 22, right: 14, bottom: 52, left: 54 };

  /* The chart is redrawn at the container width so label text stays at 1:1 scale. */
  function chartLayout() {
    var wrap = $('chart-wrap');
    var available = wrap ? wrap.clientWidth : 0;
    if (!available) { available = 760; }
    var width = Math.max(300, Math.min(760, Math.round(available)));
    var pad = width < 480 ? MOBILE_PAD : CHART.pad;
    var svg = $('chart');
    if (svg) { svg.setAttribute('viewBox', '0 0 ' + width + ' ' + CHART.height); }
    return { width: width, height: CHART.height, pad: pad };
  }

  function chartGeometry(points, layout) {
    var pad = layout.pad;
    var innerW = layout.width - pad.left - pad.right;
    var innerH = layout.height - pad.top - pad.bottom;
    var maxT = 0;
    points.forEach(function (point) {
      var t = num(point.cutoff_seconds);
      if (Number.isFinite(t) && t > maxT) { maxT = t; }
    });
    if (!(maxT > 0)) { maxT = 60; }
    return {
      innerW: innerW,
      innerH: innerH,
      maxT: maxT,
      x: function (seconds) {
        var t = num(seconds);
        if (!Number.isFinite(t)) { t = 0; }
        t = Math.min(Math.max(t, 0), maxT);
        return pad.left + (t / maxT) * innerW;
      },
      y: function (probability) {
        var v = num(probability);
        if (!Number.isFinite(v)) { v = 0; }
        v = Math.min(Math.max(v, 0), 1);
        return pad.top + (1 - v) * innerH;
      }
    };
  }

  function renderChart(points, data) {
    var svg = $('chart');
    if (!points.length) {
      show($('chart-empty'), true);
      show($('chart-wrap'), false);
      setText($('chart-empty'), 'No prediction points are available in this export.');
      return;
    }
    show($('chart-empty'), false);
    show($('chart-wrap'), true);

    var layout = chartLayout();
    var pad = layout.pad;
    var width = layout.width;
    var height = layout.height;
    svgClear(svg);
    var svgTitle = svgEl('title', { id: 'chart-title' });
    svgTitle.textContent = 'Terran win probability by prediction cutoff';
    svg.appendChild(svgTitle);
    var svgDesc = svgEl('desc', { id: 'chart-desc' });
    svgDesc.textContent = 'Line chart of Terran win probability against elapsed game time. ' +
      'Shaded bands mark pivotal intervals; ringed dots mark cutoffs that have a Grad-CAM heatmap.';
    svg.appendChild(svgDesc);

    var geometry = chartGeometry(points, layout);
    var axisBottom = height - pad.bottom;

    /* 1. Pivotal-interval bands first, so the curve and dots stay on top. */
    intervalsOf(data).forEach(function (interval, index) {
      var x1 = geometry.x(num(interval.start_seconds));
      var x2 = geometry.x(num(interval.end_seconds));
      var band = svgEl('rect', {
        x: Math.min(x1, x2),
        y: pad.top,
        width: Math.max(Math.abs(x2 - x1), 2),
        height: geometry.innerH,
        'class': 'band',
        'data-interval': String(index),
        tabindex: '0',
        role: 'button',
        'aria-label': 'Pivotal interval ' + (index + 1) + ' from ' +
          fmtMinutes(num(interval.start_seconds)) + ' to ' + fmtMinutes(num(interval.end_seconds)) +
          ', net change ' + fmtPoints(num(interval.net_change))
      });
      band.addEventListener('click', function () { selectInterval(index); });
      band.addEventListener('keydown', keyActivate(function () { selectInterval(index); }));
      svg.appendChild(band);
    });

    /* 2. Horizontal grid and probability labels. */
    [0, 0.25, 0.5, 0.75, 1].forEach(function (value) {
      var y = geometry.y(value);
      svg.appendChild(svgEl('line', { x1: pad.left, x2: width - pad.right, y1: y, y2: y, 'class': 'grid' }));
      svg.appendChild(svgText(pad.left - 10, y + 4, fmtPercent(value),
        { 'class': 'axis-label', 'text-anchor': 'end' }));
    });

    /* 3. Time axis. Ticks are drawn in minutes; points stay in seconds. */
    var tickCount = width < 480 ? 4 : 5;
    for (var tick = 0; tick < tickCount; tick += 1) {
      var seconds = geometry.maxT * tick / (tickCount - 1);
      var x = geometry.x(seconds);
      svg.appendChild(svgEl('line', { x1: x, x2: x, y1: axisBottom, y2: axisBottom + 6, 'class': 'grid' }));
      svg.appendChild(svgText(x, axisBottom + 22, trimNum(seconds / 60, 2),
        { 'class': 'axis-label', 'text-anchor': 'middle' }));
    }
    svg.appendChild(svgText((pad.left + width - pad.right) / 2, height - 10,
      'Elapsed game time (minutes)', { 'class': 'axis-title', 'text-anchor': 'middle' }));
    var middleY = (pad.top + axisBottom) / 2;
    svg.appendChild(svgText(16, middleY, 'P(Terran wins)',
      { 'class': 'axis-title', 'text-anchor': 'middle',
        transform: 'rotate(-90 16 ' + middleY + ')' }));

    /* 4. 50% reference line. */
    var midY = geometry.y(0.5);
    svg.appendChild(svgEl('line', { x1: pad.left, x2: width - pad.right, y1: midY, y2: midY, 'class': 'baseline' }));
    svg.appendChild(svgText(width - pad.right - 4, midY - 6, '50%',
      { 'class': 'axis-label', 'text-anchor': 'end' }));

    /* 5. Actual result, taken from the label field (1 Terran, 0 Protoss). */
    if (data.label === 0 || data.label === 1) {
      var resultY = geometry.y(data.label === 1 ? 1 : 0);
      svg.appendChild(svgEl('line', {
        x1: pad.left, x2: width - pad.right, y1: resultY, y2: resultY, 'class': 'result-rule'
      }));
      svg.appendChild(svgText(pad.left + 6, resultY + (data.label === 1 ? 14 : -6),
        'Actual result: ' + (data.label === 1 ? 'Terran win' : 'Protoss win'),
        { 'class': 'result-label', 'text-anchor': 'start' }));
    }

    /* 6. Curve. A single point still gets its dot in step 7. */
    if (points.length > 1) {
      var path = points.map(function (point) {
        return geometry.x(num(point.cutoff_seconds)) + ',' + geometry.y(num(point.terran_win_probability));
      }).join(' ');
      svg.appendChild(svgEl('polyline', { points: path, 'class': 'curve' }));
    }

    /* 7. Prediction points. A ring marks a cutoff that has a heatmap. */
    points.forEach(function (point, index) {
      var cx = geometry.x(num(point.cutoff_seconds));
      var cy = geometry.y(num(point.terran_win_probability));
      var probability = num(point.terran_win_probability);
      var hasHeatmap = findExplanation(data, num(point.cutoff_seconds)) !== null;
      var group = svgEl('g', {
        'class': 'point' + (hasHeatmap ? ' point-has-heatmap' : ''),
        'data-index': String(index),
        tabindex: '0',
        role: 'button',
        'aria-label': 'Prediction at ' + fmtMinutes(num(point.cutoff_seconds)) + ': Terran ' +
          fmtPercent(probability) + (hasHeatmap ? ', heatmap available' : ', no heatmap available')
      });
      if (hasHeatmap) {
        group.appendChild(svgEl('circle', { cx: cx, cy: cy, r: 9, 'class': 'point-ring' }));
      }
      var dot = svgEl('circle', { cx: cx, cy: cy, r: 5, 'class': 'point-dot' });
      var title = svgEl('title');
      title.textContent = fmtSeconds(num(point.cutoff_seconds)) + ' (' +
        fmtMinutes(num(point.cutoff_seconds)) + ') - Terran ' + fmtPercent(probability) +
        (hasHeatmap ? ' - Grad-CAM available' : '');
      dot.appendChild(title);
      group.appendChild(dot);
      group.addEventListener('click', function () { selectPoint(index); });
      group.addEventListener('keydown', keyActivate(function () { selectPoint(index); }));
      svg.appendChild(group);
    });
  }

  /* Redraw only the selection overlay and the "is-selected" classes. */
  function updateChartSelection() {
    var svg = $('chart');
    if (!svg) { return; }
    var points = pointsOf(state.data);
    var previous = svg.querySelector('#selection-marker');
    if (previous) { previous.parentNode.removeChild(previous); }

    Array.prototype.forEach.call(svg.querySelectorAll('g.point'), function (node) {
      node.classList.toggle('is-selected', Number(node.getAttribute('data-index')) === state.selected);
    });
    Array.prototype.forEach.call(svg.querySelectorAll('rect.band'), function (node) {
      node.classList.toggle('is-selected', Number(node.getAttribute('data-interval')) === state.selectedInterval);
    });

    if (state.selected < 0 || !points[state.selected]) { return; }
    var layout = chartLayout();
    var width = layout.width;
    var pad = layout.pad;
    var geometry = chartGeometry(points, layout);
    var point = points[state.selected];
    var probability = num(point.terran_win_probability);
    var cx = geometry.x(num(point.cutoff_seconds));
    var cy = geometry.y(probability);
    var anchor = 'middle';
    if (cx < pad.left + 70) { anchor = 'start'; }
    if (cx > width - pad.right - 70) { anchor = 'end'; }

    var marker = svgEl('g', { id: 'selection-marker' });
    marker.appendChild(svgEl('line', {
      x1: cx, x2: cx, y1: cy, y2: layout.height - pad.bottom, 'class': 'selection-drop'
    }));
    /* The y axis is P(Terran wins), so the label always reports the Terran probability. */
    marker.appendChild(svgText(cx, cy - 14, 'Terran ' + fmtPercent(probability),
      { 'class': 'selection-label', 'text-anchor': anchor }));
    svg.appendChild(marker);
  }

  var redrawTimer = null;

  /* Redraw the curve when the available width changes, keeping labels legible. */
  function scheduleRedraw() {
    if (!state.data) { return; }
    clearTimeout(redrawTimer);
    redrawTimer = setTimeout(function () {
      var points = pointsOf(state.data);
      if (!points.length) { return; }
      renderChart(points, state.data);
      updateChartSelection();
    }, 150);
  }

  /* ------------------------------------------------------------------ *
   * Pivotal intervals
   * ------------------------------------------------------------------ */

  function signedPoints(value) {
    if (!Number.isFinite(value)) { return 'n/a'; }
    return (value >= 0 ? '+' : '\u2212') + (Math.abs(value) * 100).toFixed(1) + ' pp';
  }

  function renderIntervals(points, data) {
    var list = $('interval-list');
    svgClear(list);
    var intervals = intervalsOf(data);
    var threshold = num(data.threshold);

    if (!intervals.length) {
      setText($('intervals-note'), 'No pivotal interval was detected in this export.');
      show($('intervals-empty'), true);
      setText($('intervals-empty'), points.length
        ? 'No adjacent probability change reached ' + fmtPoints(threshold) +
          ', so no pivotal interval is marked. The last prediction point is selected below and can still be inspected.'
        : 'No pivotal interval was detected and no prediction points are available.');
      show($('intervals-hint'), points.length > 0);
      setText($('intervals-hint'), 'Tip: click any point on the chart, or focus it with the keyboard, to inspect that cutoff.');
      return;
    }

    setText($('intervals-note'), intervals.length + ' interval' + (intervals.length === 1 ? '' : 's') +
      ' detected. Select one to show the Grad-CAM evidence for its peak cutoff.');
    show($('intervals-empty'), false);
    show($('intervals-hint'), false);

    intervals.forEach(function (interval, index) {
      var explanation = findExplanation(data, num(interval.peak_cutoff_seconds));
      var changes = num(interval.changes);

      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'interval-button';
      button.setAttribute('data-index', String(index));

      var range = document.createElement('span');
      range.className = 'interval-range';
      range.textContent = fmtMinutes(num(interval.start_seconds)) + ' \u2013 ' + fmtMinutes(num(interval.end_seconds));
      button.appendChild(range);

      var change = document.createElement('span');
      change.className = 'interval-change';
      change.textContent = 'Terran ' + fmtPercent(num(interval.start_probability)) + ' \u2192 ' +
        fmtPercent(num(interval.end_probability)) + ' (' + signedPoints(num(interval.net_change)) + ')';
      button.appendChild(change);

      var meta = document.createElement('span');
      meta.className = 'interval-meta';
      meta.textContent = 'Peak change ' + signedPoints(num(interval.peak_change)) + ' at ' +
        fmtMinutes(num(interval.peak_cutoff_seconds)) +
        (Number.isFinite(changes) ? ' \u00b7 ' + changes +
          (changes === 1 ? ' adjacent change' : ' adjacent changes') : '');
      button.appendChild(meta);

      var badge = document.createElement('span');
      badge.className = 'badge ' + (explanation ? 'badge-heatmap' : 'badge-no');
      badge.textContent = explanation ? 'Heatmap available' : 'No heatmap';
      button.appendChild(badge);

      button.addEventListener('click', function () { selectInterval(index); });

      var item = document.createElement('li');
      item.appendChild(button);
      list.appendChild(item);
    });
  }

  /* ------------------------------------------------------------------ *
   * Selection
   * ------------------------------------------------------------------ */

  function syncIntervalButtons() {
    Array.prototype.forEach.call(document.querySelectorAll('.interval-button'), function (node) {
      node.classList.toggle('is-selected', Number(node.getAttribute('data-index')) === state.selectedInterval);
    });
  }

  function selectInterval(index) {
    var data = state.data;
    var interval = intervalsOf(data)[index];
    if (!interval) { return; }
    state.selectedInterval = index;
    state.selected = findPointIndex(pointsOf(data), num(interval.peak_cutoff_seconds));
    syncIntervalButtons();
    updateChartSelection();
    renderDetail();
  }

  function selectPoint(index) {
    var data = state.data;
    var point = pointsOf(data)[index];
    if (!point) { return; }
    var cutoff = num(point.cutoff_seconds);
    state.selected = index;
    state.selectedInterval = -1;
    intervalsOf(data).forEach(function (interval, intervalIndex) {
      if (Math.abs(num(interval.peak_cutoff_seconds) - cutoff) < EPSILON) { state.selectedInterval = intervalIndex; }
    });
    syncIntervalButtons();
    updateChartSelection();
    renderDetail();
  }

  /* With no intervals, the last prediction point is the default view. */
  function selectInitial(points, data) {
    var intervals = intervalsOf(data);
    if (intervals.length) {
      var best = 0;
      for (var index = 1; index < intervals.length; index += 1) {
        if (Math.abs(num(intervals[index].net_change)) > Math.abs(num(intervals[best].net_change))) {
          best = index;
        }
      }
      selectInterval(best);
      return;
    }
    if (points.length) {
      selectPoint(points.length - 1);
      return;
    }
    state.selected = -1;
    state.selectedInterval = -1;
    syncIntervalButtons();
    updateChartSelection();
    renderDetail();
  }

  /* ------------------------------------------------------------------ *
   * Detail panel: prediction point, heatmap and masking diagnostic
   * ------------------------------------------------------------------ */

  function currentCutoff() {
    var data = state.data;
    if (!data) { return NaN; }
    var points = pointsOf(data);
    if (state.selected >= 0 && points[state.selected]) {
      return num(points[state.selected].cutoff_seconds);
    }
    var intervals = intervalsOf(data);
    if (state.selectedInterval >= 0 && intervals[state.selectedInterval]) {
      return num(intervals[state.selectedInterval].peak_cutoff_seconds);
    }
    return NaN;
  }

  function renderDetail() {
    var data = state.data;
    var points = pointsOf(data);
    var intervals = intervalsOf(data);
    var hasPoint = state.selected >= 0 && Boolean(points[state.selected]);
    var hasInterval = state.selectedInterval >= 0 && Boolean(intervals[state.selectedInterval]);

    if (!hasPoint && !hasInterval) {
      show($('detail-empty'), true);
      show($('detail'), false);
      setText($('readout'), '');
      return;
    }
    show($('detail-empty'), false);
    show($('detail'), true);

    var cutoff = currentCutoff();
    var point = hasPoint ? points[state.selected] : null;
    var interval = hasInterval ? intervals[state.selectedInterval] : null;
    var probability = point
      ? num(point.terran_win_probability)
      : num(interval ? interval.end_probability : NaN);

    setText($('detail-cutoff'), fmtSeconds(cutoff) + ' (' + fmtMinutes(cutoff) + ')');
    if (Number.isFinite(probability)) {
      setText($('detail-probability'),
        'Terran ' + fmtPercent(probability) + ' / Protoss ' + fmtPercent(1 - probability));
      setText($('detail-prediction'), sideName(probability) + ' win predicted at this cutoff');
      setText($('readout'), 'Cutoff ' + fmtSeconds(cutoff) + ' \u2014 Terran ' + fmtPercent(probability) +
        ', Protoss ' + fmtPercent(1 - probability));
    } else {
      setText($('detail-probability'), 'n/a');
      setText($('detail-prediction'), 'n/a');
      setText($('readout'), 'Cutoff ' + fmtSeconds(cutoff) + ' \u2014 probability not recorded');
    }

    var seconds = point && Array.isArray(point.frame_seconds)
      ? point.frame_seconds.filter(function (value) { return Number.isFinite(value); })
      : [];
    if (seconds.length) {
      var first = Math.min.apply(null, seconds);
      var last = Math.max.apply(null, seconds);
      setText($('detail-window'), seconds.length + ' frames between ' + fmtSeconds(first) +
        ' and ' + fmtSeconds(last));
    } else {
      setText($('detail-window'), 'not recorded for this point');
    }

    renderHeatmap(cutoff);
    renderMasking();
  }

  function renderHeatmap(cutoff) {
    var data = state.data;
    var explanation = findExplanation(data, cutoff);
    var figure = $('heatmap-figure');
    var image = $('heatmap-image');
    var missing = $('heatmap-missing');
    var errorBox = $('heatmap-error');

    show(figure, false);
    show(missing, false);
    show(errorBox, false);
    setText(missing, '');
    setText(errorBox, '');
    image.onload = null;
    image.onerror = null;
    image.removeAttribute('src');
    setText($('heatmap-status'), '');
    setText($('heatmap-caption'), '');
    setText($('hm-cutoff'), 'n/a');
    setText($('hm-observed'), 'n/a');
    setText($('hm-target'), 'n/a');
    setText($('hm-file'), 'n/a');

    if (!explanation) {
      var explanations = explanationsOf(data);
      show(missing, true);
      setText(missing, explanations.length
        ? 'No heatmap for this cutoff. This export contains Grad-CAM heatmaps only for ' +
          describeCutoffs(explanations) + ', the cutoffs with the largest detected changes.'
        : 'No heatmap for this cutoff. This export contains no Grad-CAM explanations.');
      return;
    }

    var explanationCutoff = num(explanation.cutoff_seconds);
    var observed = num(explanation.displayed_frame_seconds);
    var target = explanation.target_class;
    var side = target === 1 ? 'Terran' : (target === 0 ? 'Protoss' : 'an unknown side');

    /* Prediction cutoff and image observation time are separate fields and
     * may be equal or different; show each recorded value as it is. */
    setText($('hm-cutoff'), fmtSeconds(explanationCutoff) + ' (' + fmtMinutes(explanationCutoff) + ')');
    setText($('hm-observed'), fmtSeconds(observed) +
      (Number.isFinite(observed) ? ' (' + fmtMinutes(observed) + ')' : ''));
    setText($('hm-target'), 'Evidence supporting ' + side + ' (target_class ' + String(target) + ')');
    setText($('hm-file'), explanation.image ? String(explanation.image) : 'not recorded');
    setText($('heatmap-status'),
      'The prediction cutoff limits which frames the model could see; the image observation time is the frame ' +
      'that was upsampled for this heatmap. They are independent fields in review.json.');
    setText($('heatmap-caption'), 'Prediction cutoff ' + fmtSeconds(explanationCutoff) +
      '; image observation time ' + fmtSeconds(observed) + '. The heatmap highlights evidence the model used ' +
      'for ' + side + '; it does not show a cause of the outcome.');

    if (!explanation.image || typeof explanation.image !== 'string') {
      show(missing, true);
      setText(missing, 'This explanation does not record an image file, so there is no heatmap to show.');
      return;
    }

    var imageUrl;
    try {
      /* Images are resolved against the directory that holds review.json. */
      imageUrl = new URL(explanation.image, state.reviewDir);
    } catch (error) {
      show(errorBox, true);
      setText(errorBox, 'The heatmap path in review.json is not a valid URL: ' + String(explanation.image));
      return;
    }

    image.alt = 'Grad-CAM heatmap showing evidence supporting ' + side +
      ' at prediction cutoff ' + fmtSeconds(explanationCutoff);
    image.onload = function () {
      show(figure, true);
      show(errorBox, false);
    };
    image.onerror = function () {
      show(figure, false);
      show(errorBox, true);
      setText(errorBox, 'Heatmap image could not be loaded (no data available): ' + imageUrl.href);
    };
    image.src = imageUrl.href;
  }

  function renderMaskingUnavailable(reason) {
    var body = $('masking-body');
    svgClear(body);
    var note = document.createElement('p');
    note.className = 'empty';
    note.textContent = 'Masking diagnostic unavailable: ' + reason + '.';
    body.appendChild(note);
    var hint = document.createElement('p');
    hint.className = 'note';
    hint.textContent = 'This affects only the diagnostic below. The Grad-CAM heatmap and the probability ' +
      'values above are unchanged by a missing masking check.';
    body.appendChild(hint);
  }

  function renderMasking() {
    var body = $('masking-body');
    svgClear(body);
    var explanation = findExplanation(state.data, currentCutoff());
    if (!explanation) {
      renderMaskingUnavailable('no explanation was exported for this cutoff');
      return;
    }
    var check = explanation.masking_check;
    if (!check || typeof check !== 'object') {
      renderMaskingUnavailable('this explanation does not record a masking check');
      return;
    }
    if (check.available !== true) {
      renderMaskingUnavailable(check.reason ? String(check.reason) : 'the export did not record a reason');
      return;
    }

    var rows = [
      ['Original probability (Terran)', fmtPercent(num(check.original_probability))],
      ['Masked cells', Number.isFinite(num(check.masked_cells))
        ? String(num(check.masked_cells)) + ' (' + fmtPercent(num(check.fraction)) + ' of the clip)'
        : 'n/a'],
      ['Probability drop, top-attribution mask', fmtPoints(num(check.highlighted_probability_drop))],
      ['Probability drop, random mask', fmtPoints(num(check.random_probability_drop))],
      ['Random mask seed', Number.isFinite(num(check.random_seed)) ? String(num(check.random_seed)) : 'n/a']
    ];
    var list = document.createElement('dl');
    list.className = 'meta-grid';
    rows.forEach(function (row) {
      var wrapper = document.createElement('div');
      var term = document.createElement('dt');
      term.textContent = row[0];
      var value = document.createElement('dd');
      value.textContent = row[1];
      wrapper.appendChild(term);
      wrapper.appendChild(value);
      list.appendChild(wrapper);
    });
    body.appendChild(list);

    var note = document.createElement('p');
    note.className = 'note';
    note.textContent = 'Sensitivity diagnostic only: masking zeroes the same ' +
      fmtPercent(num(check.fraction)) + ' of spatiotemporal cells across all categorical channels. That can ' +
      'create unrealistic inputs, so this is not a faithfulness guarantee.';
    body.appendChild(note);
  }

  main();
})();
