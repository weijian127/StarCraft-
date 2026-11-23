/* SC2 replay review UI.
 *
 * Two entry points share one renderer:
 *   - ?review=<dir>      read an exported review.json directly (legacy viewer)
 *   - ?job=<job id>      upload/status/report flow for the local server API
 *
 * The report flow shows the win-probability curve first, then the selected
 * prediction point, its Grad-CAM heatmap and the masking diagnostic. Plain
 * HTML/CSS/JS: no build step, no third-party libraries.
 */
(function () {
  'use strict';

  var SVG_NS = 'http://www.w3.org/2000/svg';
  var CHART = { width: 760, height: 340, pad: { top: 26, right: 22, bottom: 54, left: 74 } };
  var EPSILON = 1e-6;
  var LAST_JOB_KEY = 'sc2.lastJobId';
  var JOB_ID_PATTERN = /^job-\d{8}T\d{6}Z-(?:[0-9a-f]{6}|[0-9a-f]{32})$/;
  var JOB_MAX_RETRIES = 4;
  var JOB_RETRY_CAP_SECONDS = 30;
  var STAGES = [
    ['validating', 'Validate replay'],
    ['checking_dependencies', 'Check engine'],
    ['extracting', 'Extract observations'],
    ['predicting', 'Predict timelines'],
    ['explaining', 'Grad-CAM'],
    ['exporting', 'Export reports'],
    ['completed', 'Completed']
  ];
  var STAGE_TEXT = {
    queued: 'Queued',
    running: 'Analysis running',
    completed: 'Analysis completed',
    failed: 'Analysis failed',
    validating: 'Validating the replay',
    checking_dependencies: 'Checking SC2 dependencies',
    extracting: 'Extracting both POVs',
    predicting: 'Predicting win probabilities',
    explaining: 'Generating Grad-CAM',
    exporting: 'Exporting the reports',
    interrupted: 'Analysis interrupted'
  };

  var state = {
    data: null,
    reviewDir: null,
    jsonUrl: null,
    isDemo: false,
    selected: -1,          // index into review.json points
    selectedInterval: -1,  // index into review.json pivotal_intervals
    showAllIntervals: false,
    heatmapTask: null,
    heatmapToken: 0,
    gameViewToken: 0,
    gameViewController: null,
    peakVisible: false,
    mode: 'home',          // 'home' | 'job' | 'report' | 'legacy' | 'help'
    info: null,
    job: null,             // { id, player, index, status, timer, controller, retryCount }
    support: null,
    reviewToken: 0,        // guards async review loads against POV switches
    jobToken: 0,
    supportToken: 0,
    imageToken: 0,
    bindingsReady: false,
    uploading: false,
    modalBound: false
  };

  var modalState = { open: false, scale: 1, x: 0, y: 0, drag: null, trigger: null };

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

  function fmtClock(seconds) {
    if (!Number.isFinite(seconds)) { return 'n/a'; }
    var total = Math.max(0, Math.round(seconds));
    var minutes = Math.floor(total / 60);
    var secs = total % 60;
    return minutes + ':' + (secs < 10 ? '0' : '') + secs;
  }

  function fmtPercent(value) {
    return Number.isFinite(value) ? (value * 100).toFixed(1) + '%' : 'n/a';
  }

  /* Probability differences are reported in percentage points, never as a
   * relative "percent change". threshold 0.1 is 10.0 pp. */
  function fmtPoints(value) {
    return Number.isFinite(value) ? (value * 100).toFixed(1) + ' pp' : 'n/a';
  }

  /* Which outcome the exported probability describes. Legacy exports carry Terran win
   * probability; focus-player exports carry P(observed player wins) and say so. */
  function probabilityInfo(data) {
    var key = typeof data.probability_key === 'string' && data.probability_key
      ? data.probability_key : 'terran_win_probability';
    var subject = typeof data.probability_subject === 'string' && data.probability_subject
      ? data.probability_subject : 'Terran';
    var opponent = typeof data.probability_opponent === 'string' && data.probability_opponent
      ? data.probability_opponent : (subject === 'Terran' ? 'Protoss' : 'Terran');
    return {
      key: key, subject: subject, opponent: opponent,
      focus: typeof data.label_meaning === 'string' && /focus player/i.test(data.label_meaning),
      detail: typeof data.probability_subject_detail === 'string'
        ? data.probability_subject_detail : subject
    };
  }

  function pointProbability(point) {
    var info = state.info || { key: 'terran_win_probability' };
    return point ? num(point[info.key]) : NaN;
  }

  function probabilityContext() {
    return state.info || { key: 'terran_win_probability', subject: 'Terran', opponent: 'Protoss',
                           focus: false, detail: 'Terran' };
  }

  function subjectDisplay(info) {
    info = info || probabilityContext();
    if (!info.focus) { return info.subject; }
    return String(info.detail || info.subject).replace(/^focus player /i, 'Player ');
  }

  function opponentDisplay(info) {
    info = info || probabilityContext();
    if (!info.focus) { return info.opponent; }
    var match = /focus player\s+(\d+)/i.exec(String(info.detail || ''));
    var other = match ? (match[1] === '1' ? '2' : '1') : null;
    return other ? 'Player ' + other + ' (' + info.opponent + ')' : 'the other player (' + info.opponent + ')';
  }

  function favouredDisplay(probability) {
    var info = probabilityContext();
    return Number.isFinite(probability) && probability < 0.5 ? opponentDisplay(info) : subjectDisplay(info);
  }

  function resultText(data) {
    if (typeof data.result_text === 'string' && data.result_text) { return data.result_text; }
    var info = probabilityInfo(data);
    if (info.focus) {
      if (data.label === 1) { return 'Focus player (' + info.subject + ') won (label 1)'; }
      if (data.label === 0) { return 'Focus player (' + info.subject + ') lost (label 0)'; }
    }
    if (data.label === 1) { return 'Terran win (label 1)'; }
    if (data.label === 0) { return 'Protoss win (label 0)'; }
    if (data.label === undefined || data.label === null) { return 'not provided'; }
    return 'unknown (label ' + String(data.label) + ')';
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

  function createEl(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }

  function addMetaRow(list, term, value) {
    var wrapper = document.createElement('div');
    wrapper.appendChild(createEl('dt', null, term));
    wrapper.appendChild(createEl('dd', null, value));
    list.appendChild(wrapper);
  }

  function formatTimestamp(value) {
    if (!value) { return 'not recorded'; }
    var date = new Date(value);
    if (!Number.isFinite(date.getTime())) { return String(value); }
    return date.toLocaleString();
  }

  function stageLabel(stage) {
    if (!stage) { return 'Waiting for the first stage'; }
    return STAGE_TEXT[stage] || String(stage).replace(/_/g, ' ');
  }

  function currentUrl() {
    return new URL(window.location.href);
  }

  function setQuery(jobId, player) {
    var url = currentUrl();
    url.searchParams.delete('review');
    if (jobId) {
      url.searchParams.set('job', jobId);
    } else {
      url.searchParams.delete('job');
    }
    if (jobId && player) {
      url.searchParams.set('pov', player);
    } else {
      url.searchParams.delete('pov');
    }
    window.history.replaceState(null, '', url.pathname + (url.search ? url.search : ''));
  }

  function readLastJob() {
    try {
      var value = window.localStorage.getItem(LAST_JOB_KEY);
      return value && JOB_ID_PATTERN.test(value) ? value : null;
    } catch (error) {
      return null;
    }
  }

  function rememberJob(jobId) {
    try { window.localStorage.setItem(LAST_JOB_KEY, jobId); } catch (error) { /* ignore */ }
  }

  function clearJobTimer() {
    var job = state.job;
    if (!job) { return; }
    if (job.timer) {
      clearTimeout(job.timer);
      job.timer = null;
    }
    if (job.controller && typeof job.controller.abort === 'function') {
      job.controller.abort();
    }
    job.controller = null;
    job.pollingStopped = true;
  }

  function showView(name) {
    show($('upload-view'), name === 'home');
    show($('job-view'), name === 'job');
    show($('report-view'), name === 'report');
    show($('help-panel'), name === 'help');
    if (name !== 'report') { show($('app'), false); }
  }

  function humanRace(race) {
    return race ? String(race) : 'unknown race';
  }

  function playerTitle(pov) {
    var id = pov.focus_player_id !== undefined && pov.focus_player_id !== null
      ? String(pov.focus_player_id) : '?';
    return 'Player ' + id + ' \u00b7 ' + humanRace(pov.focus_player_race);
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
        hint: 'Serve this page over HTTP and open /ui/index.html?review=<directory> on the same origin.'
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

  function intervalEntries(data, showAll) {
    var entries = intervalsOf(data).map(function (interval, index) {
      return { interval: interval, index: index };
    });
    entries.sort(function (left, right) {
      return num(left.interval.start_seconds) - num(right.interval.start_seconds);
    });
    if (!showAll && entries.length > 3) {
      var rule = data && data.interval_rule && typeof data.interval_rule === 'object'
        ? data.interval_rule : null;
      var limit = rule ? num(rule.default_visible) : NaN;
      if (!Number.isFinite(limit) || limit < 1) { limit = 3; }
      entries = entries.slice().sort(function (left, right) {
        var difference = Math.abs(num(right.interval.net_change)) - Math.abs(num(left.interval.net_change));
        if (difference !== 0) { return difference; }
        return num(left.interval.start_seconds) - num(right.interval.start_seconds);
      }).slice(0, limit).sort(function (left, right) {
        return num(left.interval.start_seconds) - num(right.interval.start_seconds);
      });
    }
    return entries;
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
   * Backend API helpers
   * ------------------------------------------------------------------ */

  function ApiError(status, body, fallback, transient) {
    var error = body && body.error && typeof body.error === 'object' ? body.error : null;
    this.status = status;
    this.code = error && error.code ? String(error.code) : '';
    this.message = error && error.message ? String(error.message) : (fallback || ('HTTP ' + status));
    this.name = 'ApiError';
    this.transient = transient === true || status === 0 || status === 429 || status >= 500;
  }
  ApiError.prototype = Object.create(Error.prototype);
  ApiError.prototype.constructor = ApiError;

  function readJson(response) {
    return response.json().catch(function () { return null; });
  }

  function apiErrorText(error) {
    if (error instanceof ApiError) {
      var hint = '';
      if (error.status === 409 && error.code === 'busy') {
        hint = ' Only one heavy task (analysis or heatmap) runs at a time; wait for it to finish and retry.';
      } else if (error.status === 413) {
        hint = ' Choose a smaller replay.';
      } else if (error.status === 422) {
        hint = ' The replay must match the supported requirements below.';
      } else if (error.status === 429) {
        hint = ' The public demo limits repeated uploads; wait before trying again.';
      } else if (error.status === 507) {
        hint = ' The public demo storage is full; existing reports were not deleted.';
      }
      return error.message + hint + (error.code ? ' (' + error.code + ')' : '');
    }
    if (error && error.message) { return error.message; }
    return 'The request failed.';
  }

  /* ------------------------------------------------------------------ *
   * Home: support description and upload
   * ------------------------------------------------------------------ */

  function initHome() {
    clearJobTimer();
    state.mode = 'home';
    showView('home');
    show($('pov-bar'), false);
    show($('banner'), false);
    show($('demo-banner'), false);
    show($('public-banner'), false);
    bindHomeControls();
    if (state.support) {
      renderSupport(state.support);
    } else {
      loadSupport();
    }
    updateUploadAvailability();
    checkLastJob();
  }

  function bindHomeControls() {
    if (state.bindingsReady) { return; }
    state.bindingsReady = true;
    $('upload-form').addEventListener('submit', handleUpload);
    $('replay-file').addEventListener('change', updateUploadAvailability);
    $('upload-retry-support').addEventListener('click', loadSupport);
    $('job-home').addEventListener('click', function (event) {
      event.preventDefault();
      goHome();
    });
    $('job-retry-connection').addEventListener('click', function (event) {
      event.preventDefault();
      retryJobConnection();
    });
    $('pov-home').addEventListener('click', function (event) {
      event.preventDefault();
      goHome();
    });
    $('job-report-link').addEventListener('click', function (event) {
      event.preventDefault();
      if (state.job) { openReport(state.job.id, state.job.player, { pushPov: true }); }
    });
    $('intervals-toggle').addEventListener('click', toggleIntervals);
    $('heatmap-generate').addEventListener('click', function (event) {
      event.preventDefault();
      generateHeatmap();
    });
    $('heatmap-retry').addEventListener('click', function (event) {
      event.preventDefault();
      generateHeatmap();
    });
    $('peak-toggle').addEventListener('click', togglePeakOverlay);
    $('game-view-generate').addEventListener('click', generateGameView);
    bindImageModal();
  }

  function goHome() {
    clearJobTimer();
    cancelHeatmapTask();
    state.job = null;
    state.mode = 'home';
    setQuery(null, null);
    initHome();
  }

  function loadSupport() {
    var token = ++state.supportToken;
    show($('support-error'), false);
    $('upload-retry-support').hidden = true;
    setText($('support-note'), 'Reading /api/support\u2026');
    fetch('/api/support', { cache: 'no-store' })
      .then(function (response) {
        if (!response.ok) {
          throw new ApiError(response.status, null, 'The support description is unavailable.');
        }
        return readJson(response);
      })
      .then(function (support) {
        if (token !== state.supportToken) { return; }
        if (!support || typeof support !== 'object') {
          throw new Error('The support response is not a JSON object.');
        }
        state.support = support;
        renderSupport(support);
        updateUploadAvailability();
      })
      .catch(function (error) {
        if (token !== state.supportToken) { return; }
        state.support = null;
        svgClear($('support-grid'));
        show($('public-banner'), false);
        show($('public-demo-note'), false);
        setText($('support-note'), 'The backend API could not be reached.');
        show($('support-error'), true);
        setText($('support-error'),
          'Could not read /api/support. Start the backend from server/README.md, then reload. ' +
          'Already exported reports still work through ?review=<directory>. ' + apiErrorText(error));
        $('upload-retry-support').hidden = false;
        updateUploadAvailability();
      });
  }

  function renderSupport(support) {
    var grid = $('support-grid');
    svgClear(grid);
    var engine = support.engine || {};
    var requirements = support.replay_requirements || {};
    var upload = support.upload || {};
    var analysis = support.analysis || {};
    var jobs = support.jobs || {};
    var mapCache = support.map_cache || {};
    var limits = support.limits || {};
    var demo = support.public_demo || {};
    var isPublicDemo = demo.enabled === true;

    show($('public-banner'), isPublicDemo);
    show($('public-demo-note'), isPublicDemo);
    if (isPublicDemo) {
      var demoParts = [];
      if (demo.notice) { demoParts.push(demo.notice); }
      if (demo.report_access) { demoParts.push(demo.report_access); }
      if (upload.rate_limit) { demoParts.push('Upload limit: ' + upload.rate_limit + '.'); }
      setText($('public-demo-note'), demoParts.join(' '));
    } else {
      setText($('public-demo-note'), '');
    }

    setText($('support-note'), 'These values are read from /api/support; the page does not hardcode them.');
    addMetaRow(grid, 'Engine', (engine.game_version || 'not stated') +
      (engine.base_build !== undefined ? ' (build ' + engine.base_build + ')' : ''));
    if (engine.data_version) { addMetaRow(grid, 'Data version', engine.data_version); }
    addMetaRow(grid, 'Players', requirements.players !== undefined ? requirements.players : 'not stated');
    if (Array.isArray(requirements.matchup)) {
      addMetaRow(grid, 'Matchup', requirements.matchup.join(' vs '));
    }
    if (requirements.results) { addMetaRow(grid, 'Result', requirements.results); }
    if (Array.isArray(requirements.rejected)) {
      addMetaRow(grid, 'Rejected', requirements.rejected.join('; '));
    }
    if (upload.max_bytes !== undefined) {
      addMetaRow(grid, 'Maximum upload', (upload.max_mib !== undefined ? upload.max_mib + ' MiB (' : '') +
        upload.max_bytes + ' bytes' + (upload.max_mib !== undefined ? ')' : ''));
    }
    addMetaRow(grid, 'Upload field', upload.field || 'file');
    var analysisParts = [];
    if (analysis.device) { analysisParts.push(String(analysis.device)); }
    if (analysis.frames !== undefined) { analysisParts.push(analysis.frames + ' frames'); }
    if (analysis.step_seconds !== undefined) { analysisParts.push('step ' + analysis.step_seconds + ' s'); }
    if (analysis.draws !== undefined) { analysisParts.push(analysis.draws + ' draw(s)'); }
    if (analysis.seed !== undefined) { analysisParts.push('seed ' + analysis.seed); }
    if (analysis.povs !== undefined) { analysisParts.push(analysis.povs + ' POVs'); }
    addMetaRow(grid, 'Analysis', analysisParts.length ? analysisParts.join(', ') : 'not stated');
    addMetaRow(grid, 'Concurrency', (jobs.max_concurrent !== undefined ? jobs.max_concurrent : 1) +
      ' at a time; busy returns HTTP ' + (jobs.busy_status !== undefined ? jobs.busy_status : 409));
    addMetaRow(grid, 'Progress', jobs.progress_percent === null
      ? 'No percentage; real stages only'
      : (jobs.progress_percent !== undefined ? String(jobs.progress_percent) : 'not stated'));
    addMetaRow(grid, 'Map cache', mapCache.allow_download_map
      ? 'Missing archives may be downloaded (hash-verified)'
      : 'Missing map archives are rejected');
    if (isPublicDemo) {
      addMetaRow(grid, 'Report access', demo.report_access || 'Anyone with the report URL can view it.');
      addMetaRow(grid, 'Upload limit', upload.rate_limit || 'not stated');
      if (limits.max_jobs_bytes !== null && limits.max_jobs_bytes !== undefined) {
        addMetaRow(grid, 'Demo storage', (limits.max_jobs_bytes / (1024 * 1024)).toFixed(0) +
          ' MiB for public jobs; old runs are not auto-deleted');
      }
      if (limits.job_timeout_seconds !== undefined) {
        addMetaRow(grid, 'Job timeout', limits.job_timeout_seconds + ' s');
      }
    }
  }

  function updateUploadAvailability() {
    var input = $('replay-file');
    var submit = $('upload-submit');
    var file = input && input.files && input.files[0] ? input.files[0] : null;
    submit.disabled = state.uploading || !state.support || !file;
    if (!file) {
      setText($('file-note'), 'No file selected.');
      return;
    }
    var max = state.support && state.support.upload ? num(state.support.upload.max_bytes) : NaN;
    var size = file.size;
    var sizeText = (size / (1024 * 1024)).toFixed(2) + ' MiB';
    if (Number.isFinite(max) && size > max) {
      setText($('file-note'), file.name + ' (' + sizeText + ') exceeds the ' +
        (state.support.upload.max_mib !== undefined ? state.support.upload.max_mib + ' MiB ' : '') + 'limit.');
    } else {
      setText($('file-note'), file.name + ' (' + sizeText + ')');
    }
  }

  function showUploadError(text) {
    show($('upload-error'), true);
    setText($('upload-error'), text);
  }

  function handleUpload(event) {
    event.preventDefault();
    if (state.uploading) { return; }
    show($('upload-error'), false);
    setText($('upload-status'), '');
    var input = $('replay-file');
    var file = input.files && input.files[0] ? input.files[0] : null;

    if (!state.support) {
      showUploadError('The backend support description is not available, so the upload cannot be validated.');
      return;
    }
    if (!file) {
      showUploadError('Select a .SC2Replay file first.');
      return;
    }
    if (!/\.sc2replay$/i.test(file.name)) {
      showUploadError('The selected file is not a .SC2Replay file: ' + file.name);
      return;
    }
    if (file.size === 0) {
      showUploadError('The selected file is empty.');
      return;
    }
    var upload = state.support.upload || {};
    var max = num(upload.max_bytes);
    if (Number.isFinite(max) && file.size > max) {
      showUploadError('This replay is ' + (file.size / (1024 * 1024)).toFixed(2) +
        ' MiB; the backend accepts at most ' + (upload.max_mib !== undefined ? upload.max_mib + ' MiB' : max + ' bytes') + '.');
      return;
    }

    var body = new FormData();
    body.append(upload.field || 'file', file, file.name);
    state.uploading = true;
    input.disabled = true;
    $('upload-submit').disabled = true;
    setText($('upload-status'), 'Uploading ' + file.name + '\u2026');

    fetch('/api/jobs', { method: 'POST', body: body })
      .then(function (response) {
        return readJson(response).then(function (payload) {
          if (!response.ok) {
            throw new ApiError(response.status, payload, 'The upload was rejected with HTTP ' + response.status + '.');
          }
          return payload;
        });
      })
      .then(function (payload) {
        if (!payload || !JOB_ID_PATTERN.test(String(payload.job_id || ''))) {
          throw new Error('The backend accepted the upload but returned no valid job id.');
        }
        rememberJob(payload.job_id);
        setText($('upload-status'), '');
        startJobMode(payload.job_id, null);
      })
      .catch(function (error) {
        var text = error instanceof TypeError
          ? 'Could not reach the backend at /api/jobs. Check that the service is running and try again.'
          : apiErrorText(error);
        showUploadError(text);
        setText($('upload-status'), '');
      })
      .then(function () {
        state.uploading = false;
        input.disabled = false;
        updateUploadAvailability();
      });
  }

  function checkLastJob() {
    var jobId = readLastJob();
    var card = $('last-job');
    if (!jobId) { show(card, false); return; }
    fetch('/api/jobs/' + encodeURIComponent(jobId), { cache: 'no-store' })
      .then(function (response) {
        if (!response.ok) { return null; }
        return readJson(response);
      })
      .then(function (status) {
        if (state.mode !== 'home' || !status) { return; }
        if (status.state === 'queued' || status.state === 'running') {
          startJobMode(jobId, null);
          return;
        }
        var completed = status.state === 'completed';
        setText($('last-job-text'), 'Last analysis ' + jobId + ' is ' +
          (status.state || 'unknown') + ' (' + stageLabel(status.stage) + ').');
        var link = $('last-job-link');
        link.textContent = completed ? 'Open last report' : 'Open last analysis';
        link.onclick = function (event) {
          event.preventDefault();
          if (completed) { openReport(jobId, null); } else { startJobMode(jobId, null); }
        };
        show(card, true);
      })
      .catch(function () { show(card, false); });
  }

  /* ------------------------------------------------------------------ *
   * Job status
   * ------------------------------------------------------------------ */

  function startJobMode(jobId, player) {
    var id = String(jobId || '');
    if (!JOB_ID_PATTERN.test(id)) {
      goHome();
      showUploadError('The job link is not valid: ' + id);
      return;
    }
    clearJobTimer();
    cancelHeatmapTask();
    rememberJob(id);
    state.mode = 'job';
    state.job = { id: id, player: player || null, index: null, status: null,
                  timer: null, controller: null, retryCount: 0, pollingStopped: false };
    setQuery(id, player || null);
    showView('job');
    show($('pov-bar'), false);
    show($('banner'), false);
    show($('demo-banner'), false);
    show($('public-banner'), false);
    show($('job-error'), false);
    show($('job-actions'), false);
    hideJobConnection();
    $('job-report-link').hidden = true;
    setText($('job-state'), 'Checking job status\u2026');
    setText($('job-stage-note'), '');
    setText($('job-poll-note'), 'Stages are reported by the backend; no completion percentage is available.');
    svgClear($('stage-list'));
    setText($('job-id'), id);
    setText($('job-file'), 'not recorded');
    setText($('job-map'), 'not recorded');
    setText($('job-replay-id'), 'not recorded');
    setText($('job-started'), 'not recorded');
    setText($('job-finished'), 'not recorded');
    fetchJobStatus(state.job);
  }

  function jobPollSeconds() {
    var support = state.support || {};
    var wait = num((support.jobs || {}).suggested_poll_seconds);
    if (!Number.isFinite(wait) || wait < 1) { wait = 3; }
    if (wait > 30) { wait = 30; }
    return wait;
  }

  function scheduleJobPoll(job, delay) {
    if (job !== state.job || job.pollingStopped) { return; }
    clearTimeout(job.timer);
    job.timer = setTimeout(function () { fetchJobStatus(job); }, delay);
  }

  function hideJobConnection() {
    show($('job-connection'), false);
    setText($('job-connection'), '');
    $('job-retry-connection').hidden = true;
  }

  function resetJobRetry(job) {
    job.retryCount = 0;
    job.pollingStopped = false;
  }

  function isTransientJobError(error) {
    if (!error) { return false; }
    if (error instanceof ApiError) { return error.transient === true; }
    if (error instanceof TypeError) { return true; }
    return error.name === 'TypeError' || error.name === 'SyntaxError';
  }

  function handleJobConnectionFailure(job) {
    job.retryCount = (job.retryCount || 0) + 1;
    show($('job-connection'), true);
    show($('job-actions'), true);
    $('job-report-link').hidden = true;
    if (job.retryCount > JOB_MAX_RETRIES) {
      job.pollingStopped = true;
      $('job-retry-connection').hidden = false;
      setText($('job-connection'),
        'Connection temporarily lost. The background job may still be running. ' +
        'Automatic retries stopped after ' + JOB_MAX_RETRIES + ' attempts.');
      setText($('job-poll-note'), 'Polling stopped. The job is unchanged on the server; Retry connection only queries ' +
        job.id + '.');
      return;
    }
    $('job-retry-connection').hidden = true;
    var delay = Math.min(jobPollSeconds() * Math.pow(2, job.retryCount - 1), JOB_RETRY_CAP_SECONDS);
    setText($('job-connection'),
      'Connection temporarily lost. The background job may still be running. ' +
      'Retrying in ' + trimNum(delay, 1) + ' s (attempt ' + job.retryCount + ' of ' + JOB_MAX_RETRIES + ').');
    setText($('job-poll-note'), 'No new upload is sent; only the status of ' + job.id + ' is queried again.');
    scheduleJobPoll(job, delay * 1000);
  }

  function retryJobConnection() {
    var job = state.job;
    if (!job || state.mode !== 'job') { return; }
    job.retryCount = 0;
    job.pollingStopped = false;
    hideJobConnection();
    show($('job-error'), false);
    show($('job-actions'), false);
    $('job-report-link').hidden = true;
    setText($('job-poll-note'), 'Retrying the status of ' + job.id + '\u2026');
    fetchJobStatus(job);
  }

  function fetchJobStatus(job) {
    if (!job || job !== state.job || job.pollingStopped) { return; }
    if (job.controller && typeof job.controller.abort === 'function') {
      job.controller.abort();
    }
    var controller = typeof AbortController === 'function' ? new AbortController() : null;
    job.controller = controller;
    var options = { cache: 'no-store' };
    if (controller) { options.signal = controller.signal; }

    fetch('/api/jobs/' + encodeURIComponent(job.id), options)
      .then(function (response) {
        return readJson(response).then(function (payload) {
          if (!response.ok) {
            throw new ApiError(response.status, payload,
              response.status === 404 ? 'The backend does not know this job.' : 'The job status could not be read.');
          }
          if (!payload || typeof payload !== 'object') {
            throw new ApiError(0, null, 'The job status response was empty.', true);
          }
          if (['queued', 'running', 'completed', 'failed', 'interrupted'].indexOf(payload.state) < 0) {
            throw new ApiError(0, null, 'The backend returned an unrecognized job state.', true);
          }
          return payload;
        });
      })
      .then(function (status) {
        if (job !== state.job) { return; }
        if (controller && controller.signal.aborted) { return; }
        if (job.controller === controller) { job.controller = null; }
        resetJobRetry(job);
        hideJobConnection();
        job.status = status;
        renderJobStatus(status);
        if (status.state === 'completed') {
          openReport(job.id, job.player, { auto: true });
          return;
        }
        if (status.state === 'failed' || status.state === 'interrupted' || status.stage === 'interrupted') {
          return;
        }
        show($('job-actions'), false);
        $('job-report-link').hidden = true;
        var wait = jobPollSeconds();
        setText($('job-poll-note'), 'Refreshing every ' + trimNum(wait, 1) +
          ' s. Stages are reported by the backend; no completion percentage is available.');
        scheduleJobPoll(job, wait * 1000);
      })
      .catch(function (error) {
        if (job !== state.job) { return; }
        if (controller && controller.signal.aborted) { return; }
        if (job.controller === controller) { job.controller = null; }
        if (isTransientJobError(error)) {
          handleJobConnectionFailure(job);
          return;
        }
        job.pollingStopped = true;
        hideJobConnection();
        show($('job-error'), true);
        setText($('job-error'), 'Job status could not be read: ' + apiErrorText(error) +
          ' Check the backend log; no upload is retried automatically.');
        show($('job-actions'), true);
        $('job-report-link').hidden = true;
        setText($('job-poll-note'), 'Polling stopped. This job was not modified.');
      });
  }

  function renderJobStatus(status) {
    setText($('job-id'), status.job_id || (state.job ? state.job.id : 'n/a'));
    setText($('job-state'), (status.state ? (STAGE_TEXT[status.state] || status.state) : 'Unknown state') +
      (status.stage && status.state === 'running' ? ' \u2014 ' + stageLabel(status.stage) : ''));
    var povText = status.pov ? ' Current POV: ' + String(status.pov) + '.' : '';
    setText($('job-stage-note'), 'Stage: ' + stageLabel(status.stage) + '.' + povText);
    renderStageList(status);
    setText($('job-file'), status.original_name || 'not recorded');
    var replay = status.replay || {};
    setText($('job-map'), replay.map_title || 'not recorded');
    setText($('job-replay-id'), replay.replay_id || 'not recorded');
    setText($('job-started'), formatTimestamp(status.started_at));
    setText($('job-finished'), formatTimestamp(status.finished_at));

    if (status.state === 'failed') {
      var detail = [];
      if (status.error_code) { detail.push(String(status.error_code)); }
      if (status.error_message) { detail.push(String(status.error_message)); }
      show($('job-error'), true);
      setText($('job-error'), 'Analysis failed: ' + (detail.length ? detail.join(' \u2014 ') : stageLabel(status.stage)) +
        '. Partial run files are kept on disk, but this page cannot resume them.');
      show($('job-actions'), true);
      $('job-report-link').hidden = true;
      setText($('job-poll-note'), 'Polling stopped.');
      return;
    }
    show($('job-error'), false);
    if (status.state !== 'queued' && status.state !== 'running') {
      show($('job-actions'), true);
      $('job-report-link').hidden = false;
    }
  }

  function renderStageList(status) {
    var list = $('stage-list');
    svgClear(list);
    var stage = status.stage;
    var currentIndex = -1;
    for (var index = 0; index < STAGES.length; index += 1) {
      if (STAGES[index][0] === stage) { currentIndex = index; }
    }
    var failed = status.state === 'failed';
    STAGES.forEach(function (entry, position) {
      var item = createEl('li', null, entry[1]);
      if (failed) {
        if (position === currentIndex) { item.className = 'stage-error'; }
      } else if (position === currentIndex) {
        item.className = 'stage-current';
        item.setAttribute('aria-current', 'step');
      } else if (currentIndex >= 0 && position < currentIndex) {
        item.className = 'stage-past';
      }
      list.appendChild(item);
    });
    if (failed) {
      list.appendChild(createEl('li', 'stage-error',
        stage && stage !== 'failed' ? stageLabel(stage) : 'Failed'));
    }
  }

  /* ------------------------------------------------------------------ *
   * Report: fetch index, switch POV, load one review.json
   * ------------------------------------------------------------------ */

  function showReportError(text) {
    show($('report-error'), true);
    setText($('report-error'), text);
  }

  function openReport(jobId, player, options) {
    var id = String(jobId || '');
    if (!JOB_ID_PATTERN.test(id)) {
      goHome();
      showUploadError('The job link is not valid: ' + id);
      return;
    }
    clearJobTimer();
    cancelHeatmapTask();
    state.mode = 'report';
    if (!state.job || state.job.id !== id) {
      state.job = { id: id, player: player || null, index: null, status: null,
                    timer: null, controller: null, retryCount: 0, pollingStopped: true };
    }
    state.job.player = player || null;
    hideJobConnection();
    show($('job-actions'), false);
    $('job-report-link').hidden = true;
    var token = ++state.jobToken;
    showView('report');
    show($('pov-bar'), true);
    show($('report-error'), false);
    show($('app'), false);
    svgClear($('pov-buttons'));
    setText($('pov-note'), 'Reading the report index\u2026');
    setQuery(id, player || null);

    fetch('/api/jobs/' + encodeURIComponent(id) + '/report', { cache: 'no-store' })
      .then(function (response) {
        return readJson(response).then(function (payload) {
          if (!response.ok) {
            throw new ApiError(response.status, payload, 'The report is not available yet.');
          }
          return payload;
        });
      })
      .then(function (index) {
        if (token !== state.jobToken || state.mode !== 'report' || !index) { return; }
        state.job.index = index;
        renderPovBar(index, player);
        var povs = Array.isArray(index.povs) ? index.povs : [];
        var chosen = null;
        for (var position = 0; position < povs.length; position += 1) {
          if (povs[position].player === player) { chosen = povs[position]; }
        }
        if (!chosen) { chosen = povs[0] || null; }
        if (!chosen) {
          showReportError('This job completed without any player POV in its report.');
          return;
        }
        loadJobPov(id, chosen.player);
      })
      .catch(function (error) {
        if (token !== state.jobToken || state.mode !== 'report') { return; }
        if (error instanceof ApiError && error.status === 409) {
          startJobMode(id, player || null);
          return;
        }
        showReportError('Could not load the report index: ' + apiErrorText(error));
      });
  }

  function renderPovBar(index, selectedPlayer) {
    var container = $('pov-buttons');
    svgClear(container);
    var povs = Array.isArray(index.povs) ? index.povs : [];
    povs.forEach(function (pov) {
      var button = createEl('button', 'pov-button');
      button.type = 'button';
      button.setAttribute('data-player', pov.player);
      button.appendChild(createEl('span', 'pov-title', playerTitle(pov)));
      var result = pov.result ? String(pov.result) : 'result not recorded';
      var label = pov.label === 1 ? 'focus-player label 1' : (pov.label === 0 ? 'focus-player label 0' : 'label not recorded');
      button.appendChild(createEl('span', 'pov-result', result + ' \u00b7 ' + label));
      if (pov.player === selectedPlayer) { button.classList.add('is-selected'); }
      button.setAttribute('aria-pressed', pov.player === selectedPlayer ? 'true' : 'false');
      button.addEventListener('click', function () {
        if (!state.job || state.job.player === pov.player) { return; }
        loadJobPov(index.job_id, pov.player);
      });
      container.appendChild(button);
    });
    var replay = index.replay || {};
    var map = replay.map_title ? ' \u00b7 ' + replay.map_title : '';
    setText($('pov-note'), 'Job ' + (index.job_id || '') + map +
      ' \u00b7 ' + povs.length + ' POVs. The selected POV decides which player is treated as self.');
  }

  function findPov(index, player) {
    var povs = index && Array.isArray(index.povs) ? index.povs : [];
    for (var position = 0; position < povs.length; position += 1) {
      if (povs[position].player === player) { return povs[position]; }
    }
    return null;
  }

  function loadJobPov(jobId, player) {
    var index = state.job && state.job.index;
    if (!index || !state.job || state.job.id !== jobId) { return; }
    var pov = findPov(index, player);
    if (!pov) {
      showReportError('The report does not contain ' + player + '.');
      return;
    }
    state.job.player = player;
    Array.prototype.forEach.call(document.querySelectorAll('.pov-button'), function (button) {
      var selected = button.getAttribute('data-player') === player;
      button.classList.toggle('is-selected', selected);
      button.setAttribute('aria-pressed', selected ? 'true' : 'false');
    });
    setQuery(jobId, player);

    var jsonUrl;
    try {
      jsonUrl = new URL(pov.review_url, window.location.origin);
    } catch (error) {
      showReportError('The report URL is not valid: ' + String(pov.review_url));
      return;
    }
    var reviewDir = new URL('.', jsonUrl);
    var token = ++state.reviewToken;
    cancelHeatmapTask();
    show($('report-error'), false);
    show($('app'), false);
    setText($('pov-note'), 'Loading ' + playerTitle(pov) + '\u2026');

    fetch(jsonUrl.href, { cache: 'no-store' })
      .then(function (response) {
        if (!response.ok) {
          throw new Error('HTTP ' + response.status + ' ' + response.statusText);
        }
        return response.json();
      })
      .then(function (data) {
        if (token !== state.reviewToken || state.mode !== 'report') { return; }
        if (!data || typeof data !== 'object' || Array.isArray(data)) {
          throw new Error('review.json did not contain a JSON object.');
        }
        state.reviewDir = reviewDir;
        state.jsonUrl = jsonUrl;
        start(data);
        show($('app'), true);
        updatePovNote(data, pov);
      })
      .catch(function (error) {
        if (token !== state.reviewToken || state.mode !== 'report') { return; }
        showReportError('Could not load the report for ' + playerTitle(pov) + ': ' +
          (error && error.message ? error.message : String(error)));
      });
  }

  function updatePovNote(data, pov) {
    var info = probabilityInfo(data);
    var parts = [];
    parts.push('Showing ' + playerTitle(pov) + (pov.result ? ' (' + pov.result + ')' : '') + '.');
    if (info.focus) {
      parts.push('Estimated probability that ' + subjectDisplay(info) +
        ' wins, based on observations from this POV. Class 1 means this player wins, class 0 means this player loses. ' +
        'The two POVs use different observations and need not sum to 100%.');
    } else {
      parts.push('Estimated probability that ' + info.subject + ' wins, based on observations from this POV; ' +
        'class 1 means ' + info.subject + ' wins.');
    }
    parts.push('Switching the POV reloads the curve, the selected point and its heatmap.');
    setText($('pov-note'), parts.join(' '));
  }

  /* ------------------------------------------------------------------ *
   * Load flow
   * ------------------------------------------------------------------ */

  function renderLoadProblem(problem) {
    state.mode = 'help';
    showView('help');
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
    bindHomeControls();
    var params = new URLSearchParams(window.location.search);
    var review = params.get('review');
    var job = params.get('job');
    if (review) {
      runLegacyMode(review);
      return;
    }
    if (job) {
      startJobMode(job, params.get('pov'));
      return;
    }
    initHome();
  }

  function runLegacyMode(raw) {
    state.mode = 'legacy';
    var parsed = parseReviewParam(raw);
    if (!parsed.ok) {
      renderLoadProblem(parsed);
      return;
    }
    state.reviewDir = parsed.reviewDir;
    state.jsonUrl = parsed.jsonUrl;
    showView('report');
    show($('pov-bar'), false);
    show($('report-error'), false);
    var token = ++state.reviewToken;

    fetch(parsed.jsonUrl.href, { cache: 'no-store' })
      .then(function (response) {
        if (!response.ok) {
          throw new Error('HTTP ' + response.status + ' ' + response.statusText);
        }
        return response.json();
      })
      .then(function (data) {
        if (token !== state.reviewToken || state.mode !== 'legacy') { return; }
        if (!data || typeof data !== 'object' || Array.isArray(data)) {
          throw new Error('review.json did not contain a JSON object.');
        }
        start(data);
      })
      .catch(function (error) {
        if (token !== state.reviewToken || state.mode !== 'legacy') { return; }
        renderLoadProblem({
          message: 'Could not load review.json.',
          detail: parsed.jsonUrl.href + ' - ' + error.message,
          hint: 'Check that the review directory exists and contains review.json, and that the server was started in the project root.'
        });
      });
  }

  function start(data) {
    state.data = data;
    state.info = probabilityInfo(data);
    state.isDemo = data.demo === true || /(^|\/)demo(\/|$)/i.test(state.jsonUrl.pathname);
    state.showAllIntervals = false;
    cancelHeatmapTask();
    showView('report');
    show($('help-panel'), false);
    show($('banner'), false);
    show($('app'), true);
    show($('demo-banner'), state.isDemo);
    show($('report-error'), false);

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
  function stateText(value) {
    if (value === undefined || value === null || value === '') { return 'not recorded'; }
    return String(value);
  }

  function intervalRuleText(data) {
    var rule = data && data.interval_rule && typeof data.interval_rule === 'object'
      ? data.interval_rule : null;
    if (!rule) {
      return 'This export predates the candidate-interval rule; its recorded intervals use the legacy adapter merge window.';
    }
    var threshold = num(rule.threshold);
    var cap = num(rule.max_merged_seconds);
    var visible = num(rule.default_visible);
    return 'Every adjacent change >= ' + fmtPoints(threshold) +
      ' is a candidate. Consecutive same-direction changes merge only while the total span is at most ' +
      trimNum(cap, 1) + ' s; the list shows the ' + trimNum(visible, 0) +
      ' largest by absolute net change, in time order, until expanded. This window is a local display rule, not a paper result.';
  }

  function renderRunInfo(data) {
    var grid = (data.time_grid && typeof data.time_grid === 'object') ? data.time_grid : null;
    var params = (data.run_params && typeof data.run_params === 'object') ? data.run_params : {};
    var epochs = typeof data.checkpoint_epoch === 'number' ? data.checkpoint_epoch : NaN;
    var overridden = Boolean(grid && grid.override_applied === true);
    var frames = num(data.frames);
    if (!Number.isFinite(frames)) { frames = num(params.frames); }
    if (!Number.isFinite(frames) && grid) { frames = num(grid.frames); }
    var step = num(data.step_seconds);
    if (!Number.isFinite(step)) { step = num(params.step); }
    var draws = num(data.draws_per_point);
    if (!Number.isFinite(draws)) { draws = num(params.draws); }
    var seed = num(data.seed);
    if (!Number.isFinite(seed)) { seed = num(params.seed); }
    var threshold = num(data.threshold);
    var merge = num(data.merge_seconds);

    setText($('run-epochs'), Number.isFinite(epochs)
      ? trimNum(epochs, 0) + (epochs === 1 ? ' epoch' : ' epochs')
      : 'not recorded');
    setText($('run-train-grid'), gridText(grid ? grid.training : null));
    setText($('run-infer-grid'), gridText(grid ? grid.inference : null) +
      (overridden ? ' \u2014 export override' : ''));
    setText($('run-frames'), Number.isFinite(frames)
      ? trimNum(frames, 0) + ' frames per prediction' : 'not recorded');
    setText($('tech-step'), Number.isFinite(step) ? trimNum(step, 1) + ' s' : 'not recorded');
    setText($('tech-draws'), Number.isFinite(draws) ? trimNum(draws, 0) + ' per point' : 'not recorded');
    setText($('tech-seed'), Number.isFinite(seed) ? String(seed) : 'not recorded');
    setText($('tech-threshold'), Number.isFinite(threshold)
      ? fmtPoints(threshold) + ' (abs delta ' + trimNum(threshold, 4) + ')' : 'not recorded');
    setText($('tech-merge'), Number.isFinite(merge)
      ? trimNum(merge, 1) + ' s maximum merged span (local display rule)' : 'not recorded');
    setText($('tech-interval-rule'), intervalRuleText(data));
    setText($('tech-membership'), data.evaluation_status === 'unknown_split_membership'
      ? 'unknown (not recorded in this export)' : stateText(data.evaluation_status));
    var engine = data.engine && typeof data.engine === 'object' ? data.engine : null;
    setText($('tech-engine'), engine
      ? stateText(engine.game_version) + ' / build ' + stateText(engine.base_build) +
        (engine.data_version ? ' / data ' + engine.data_version : '')
      : 'not recorded');
    setText($('tech-checkpoint-sha'), stateText(data.checkpoint_sha256));

    var overrideNote = $('run-override-note');
    show(overrideNote, overridden);
    setText(overrideNote, overridden
      ? 'This export evaluates cutoffs outside the training time grid. Predictive ability at those '
        + 'cutoffs has not been validated; they are shown for pipeline testing only.'
      : '');

    setText($('run-limitations-note'), grid
      ? 'Provenance values are read from the export; the full export limitation text is shown below.'
      : 'This export does not record a time grid. The full export limitation text is shown below.');
  }

  function renderMeta(data) {
    var replayId = data.replay_id === undefined || data.replay_id === null ? 'n/a' : String(data.replay_id);
    setText($('meta-match'), replayId);
    setText($('tech-replay-id'), replayId);
    setText($('meta-pov'), data.focus_player_id === undefined || data.focus_player_id === null
      ? 'n/a' : 'Player ' + String(data.focus_player_id) +
        (data.focus_player_race ? ' (' + String(data.focus_player_race) + ')' : ''));
    setText($('meta-result'), resultText(data));
    setText($('meta-sample'), data.sample_id ? String(data.sample_id) : 'not recorded');
    setText($('meta-epoch'), data.checkpoint_epoch === undefined || data.checkpoint_epoch === null
      ? 'not recorded' : 'epoch ' + String(data.checkpoint_epoch));

    var points = pointsOf(data);
    var info = state.info || probabilityInfo(data);
    var last = points.length ? points[points.length - 1] : null;
    var lastProbability = pointProbability(last);
    if (Number.isFinite(lastProbability)) {
      var agrees = ((lastProbability >= 0.5 ? 1 : 0) === data.label);
      setText($('meta-prediction'),
        'At the final cutoff (' + fmtClock(num(last.cutoff_seconds)) + ') the estimated probability that ' +
        subjectDisplay(info) + ' wins is ' + fmtPercent(lastProbability) + ' (' +
        favouredDisplay(lastProbability) + ' favoured); this ' +
        (agrees ? 'agrees with' : 'differs from') + ' the actual result. ' +
        'The two POVs use different observations and need not sum to 100%.');
    } else {
      setText($('meta-prediction'), 'No prediction point is available in this export.');
    }
    setText($('meta-source'), data.checkpoint ? String(data.checkpoint) : 'not recorded');
  }

  function renderStaticNotes(data) {
    var threshold = num(data.threshold);
    var rule = data && data.interval_rule && typeof data.interval_rule === 'object'
      ? data.interval_rule : null;
    if (rule) {
      setText($('threshold-note'), intervalRuleText(data) +
        ' Orange bands and the interval list use the same candidate data.');
    } else if (Number.isFinite(threshold)) {
      setText($('threshold-note'),
        'This legacy export records intervals at an absolute change of at least ' +
        fmtPoints(threshold) + '. Its original merge rule is not fully described; ' +
        'the orange bands and list still use the same stored interval data.');
    } else {
      setText($('threshold-note'), 'This export does not record an interval threshold.');
    }
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
    svgTitle.textContent = 'Win probability by prediction cutoff';
    svg.appendChild(svgTitle);
    var svgDesc = svgEl('desc', { id: 'chart-desc' });
    svgDesc.textContent = 'Line chart of the estimated probability that ' +
      ((state.info && state.info.focus) ? subjectDisplay(state.info) : (state.info || { subject: 'Terran' }).subject) +
      ' wins, against elapsed game time. Shaded bands mark candidate intervals; ' +
      'ringed dots mark cutoffs that have a Grad-CAM heatmap.';
    svg.appendChild(svgDesc);
    var info = state.info || probabilityInfo(data);
    if (info.focus) {
      setText($('chart-subject'),
        'Estimated probability that ' + subjectDisplay(info) + ' wins, based on observations from this POV. ' +
        'Class 1 means this player wins; class 0 means this player loses. ' +
        'The two POVs use different observations and need not sum to 100%.');
    } else {
      setText($('chart-subject'),
        'Estimated probability that Terran wins, based on observations from this POV. ' +
        'Class 1 means Terran wins; class 0 means Protoss wins.');
    }

    var geometry = chartGeometry(points, layout);
    var axisBottom = height - pad.bottom;

    /* 1. Candidate-interval bands first, so the curve and dots stay on top.
     * Chart and list share intervalEntries(), including the default top-three view. */
    intervalEntries(data, state.showAllIntervals).forEach(function (entry) {
      var interval = entry.interval;
      var index = entry.index;
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
        'aria-label': 'Candidate interval ' + fmtClock(num(interval.start_seconds)) + ' to ' +
          fmtClock(num(interval.end_seconds)) + ', net change ' + fmtPoints(num(interval.net_change))
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

    /* 3. Time axis. Use readable m:ss labels at a coarse step. */
    var tickStep = geometry.maxT >= 900 ? 300 : (geometry.maxT >= 300 ? 120 : 60);
    for (var seconds = 0; seconds <= geometry.maxT + 1; seconds += tickStep) {
      var x = geometry.x(seconds);
      svg.appendChild(svgEl('line', { x1: x, x2: x, y1: axisBottom, y2: axisBottom + 6, 'class': 'grid' }));
      svg.appendChild(svgText(x, axisBottom + 22, fmtClock(seconds),
        { 'class': 'axis-label', 'text-anchor': 'middle' }));
    }
    svg.appendChild(svgText((pad.left + width - pad.right) / 2, height - 10,
      'Elapsed game time', { 'class': 'axis-title', 'text-anchor': 'middle' }));
    var middleY = (pad.top + axisBottom) / 2;
    svg.appendChild(svgText(16, middleY, 'P(' + (info.focus ? 'this player' : info.subject) + ' wins)',
      { 'class': 'axis-title', 'text-anchor': 'middle',
        transform: 'rotate(-90 16 ' + middleY + ')' }));

    /* 4. 50% reference line. */
    var midY = geometry.y(0.5);
    svg.appendChild(svgEl('line', { x1: pad.left, x2: width - pad.right, y1: midY, y2: midY, 'class': 'baseline' }));
    svg.appendChild(svgText(width - pad.right - 4, midY - 6, '50%',
      { 'class': 'axis-label', 'text-anchor': 'end' }));

    /* 5. Actual result: label 1 means the subject (focus player or Terran) won. */
    if (data.label === 0 || data.label === 1) {
      var resultY = geometry.y(data.label === 1 ? 1 : 0);
      svg.appendChild(svgEl('line', {
        x1: pad.left, x2: width - pad.right, y1: resultY, y2: resultY, 'class': 'result-rule'
      }));
      svg.appendChild(svgText(pad.left + 6, resultY + (data.label === 1 ? 14 : -6),
        'Actual result: ' + (data.label === 1 ? info.subject : info.opponent) + ' won',
        { 'class': 'result-label', 'text-anchor': 'start' }));
    }

    /* 6. Curve. A single point still gets its dot in step 7. */
    if (points.length > 1) {
      var path = points.map(function (point) {
        return geometry.x(num(point.cutoff_seconds)) + ',' + geometry.y(pointProbability(point));
      }).join(' ');
      svg.appendChild(svgEl('polyline', { points: path, 'class': 'curve' }));
    }

    /* 7. Prediction points. A ring marks a cutoff that has a heatmap. */
    points.forEach(function (point, index) {
      var cx = geometry.x(num(point.cutoff_seconds));
      var probability = pointProbability(point);
      var cy = geometry.y(probability);
      var hasHeatmap = findExplanation(data, num(point.cutoff_seconds)) !== null;
      var group = svgEl('g', {
        'class': 'point' + (hasHeatmap ? ' point-has-heatmap' : ''),
        'data-index': String(index),
        tabindex: '0',
        role: 'button',
        'aria-label': 'Prediction at ' + fmtClock(num(point.cutoff_seconds)) + ': ' + subjectDisplay(info) + ' ' +
          fmtPercent(probability) + (hasHeatmap ? ', heatmap available' : ', no heatmap available')
      });
      if (hasHeatmap) {
        group.appendChild(svgEl('circle', { cx: cx, cy: cy, r: 9, 'class': 'point-ring' }));
      }
      var dot = svgEl('circle', { cx: cx, cy: cy, r: 5, 'class': 'point-dot' });
      var title = svgEl('title');
      title.textContent = fmtClock(num(point.cutoff_seconds)) + ' - ' + subjectDisplay(info) + ' ' +
        fmtPercent(probability) + (hasHeatmap ? ' - Grad-CAM available' : '');
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
    var probability = pointProbability(point);
    var cx = geometry.x(num(point.cutoff_seconds));
    var cy = geometry.y(probability);
    var anchor = 'middle';
    if (cx < pad.left + 70) { anchor = 'start'; }
    if (cx > width - pad.right - 70) { anchor = 'end'; }

    var marker = svgEl('g', { id: 'selection-marker' });
    marker.appendChild(svgEl('line', {
      x1: cx, x2: cx, y1: cy, y2: layout.height - pad.bottom, 'class': 'selection-drop'
    }));
    /* The y axis is P(subject wins), so the label reports exactly that probability. */
    marker.appendChild(svgText(cx, cy - 14,
      subjectDisplay() + ' ' + fmtPercent(probability),
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
   * Candidate pivotal intervals
   * ------------------------------------------------------------------ */

  function signedPoints(value) {
    if (!Number.isFinite(value)) { return 'n/a'; }
    return (value >= 0 ? '+' : '\u2212') + (Math.abs(value) * 100).toFixed(1) + ' pp';
  }

  function renderIntervals(points, data) {
    var list = $('interval-list');
    svgClear(list);
    var allIntervals = intervalsOf(data);
    var entries = intervalEntries(data, state.showAllIntervals);
    var threshold = num(data.threshold);
    var toggle = $('intervals-toggle');

    if (!allIntervals.length) {
      setText($('intervals-note'), 'No candidate pivotal interval was detected in this export.');
      show($('intervals-empty'), true);
      setText($('intervals-empty'), points.length
        ? 'No adjacent probability change reached ' + fmtPoints(threshold) +
          ', so no candidate interval is marked. The last prediction point is selected below and can still be inspected.'
        : 'No candidate interval was detected and no prediction points are available.');
      show($('intervals-hint'), points.length > 0);
      setText($('intervals-hint'), 'Tip: click any point on the chart, or focus it with the keyboard, to inspect that cutoff.');
      show(toggle, false);
      return;
    }

    var note = allIntervals.length + ' candidate pivotal interval' +
      (allIntervals.length === 1 ? '' : 's') + ' detected. These mark model-predicted changes, not proven match-deciding events.';
    if (entries.length < allIntervals.length) {
      note += ' Showing the 3 largest by absolute net change, in time order.';
    } else {
      note += ' Showing all candidates in time order.';
    }
    setText($('intervals-note'), note);
    show($('intervals-empty'), false);
    show($('intervals-hint'), false);

    show(toggle, allIntervals.length > 3);
    toggle.textContent = state.showAllIntervals
      ? 'Show the 3 largest only'
      : 'Show all ' + allIntervals.length + ' candidate intervals';

    entries.forEach(function (entry) {
      var interval = entry.interval;
      var index = entry.index;
      var explanation = findExplanation(data, num(interval.max_step_cutoff_seconds) ||
        num(interval.peak_cutoff_seconds));
      var changes = num(interval.changes);
      var maxStep = Number.isFinite(num(interval.max_step_change))
        ? num(interval.max_step_change) : num(interval.peak_change);
      var maxCutoff = Number.isFinite(num(interval.max_step_cutoff_seconds))
        ? num(interval.max_step_cutoff_seconds) : num(interval.peak_cutoff_seconds);

      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'interval-button';
      button.setAttribute('data-index', String(index));

      var range = document.createElement('span');
      range.className = 'interval-range';
      range.textContent = fmtClock(num(interval.start_seconds)) + ' \u2013 ' + fmtClock(num(interval.end_seconds));
      button.appendChild(range);

      var change = document.createElement('span');
      change.className = 'interval-change';
      change.textContent = subjectDisplay() + ' ' +
        fmtPercent(num(interval.start_probability)) + ' \u2192 ' +
        fmtPercent(num(interval.end_probability)) + ' (' + signedPoints(num(interval.net_change)) + ')';
      button.appendChild(change);

      var meta = document.createElement('span');
      meta.className = 'interval-meta';
      meta.textContent = 'Max single step ' + signedPoints(maxStep) + ' at ' + fmtClock(maxCutoff) +
        ' \u00b7 Net ' + signedPoints(num(interval.net_change)) +
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

  function toggleIntervals() {
    if (!state.data) { return; }
    state.showAllIntervals = !state.showAllIntervals;
    var points = pointsOf(state.data);
    renderChart(points, state.data);
    renderIntervals(points, state.data);
    updateChartSelection();
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
    cancelHeatmapTask();
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
    cancelHeatmapTask();
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
    var imageToken = ++state.imageToken;

    if (!hasPoint && !hasInterval) {
      show($('detail-empty'), true);
      show($('detail'), false);
      setText($('readout'), '');
      renderHeatmap(NaN, imageToken);
      renderMasking();
      return;
    }
    show($('detail-empty'), false);
    show($('detail'), true);

    var cutoff = currentCutoff();
    var point = hasPoint ? points[state.selected] : null;
    var interval = hasInterval ? intervals[state.selectedInterval] : null;
    var info = probabilityContext();
    var probability = point
      ? pointProbability(point)
      : num(interval ? interval.end_probability : NaN);

    setText($('detail-cutoff'), fmtSeconds(cutoff) + ' (' + fmtClock(cutoff) + ')');
    if (Number.isFinite(probability)) {
      setText($('detail-probability'),
        subjectDisplay(info) + ' ' + fmtPercent(probability) + ' / ' +
        opponentDisplay(info) + ' ' + fmtPercent(1 - probability));
      setText($('detail-prediction'), favouredDisplay(probability) + ' favoured at this cutoff');
      setText($('readout'), 'Cutoff ' + fmtClock(cutoff) + ' \u2014 ' + subjectDisplay(info) + ' ' +
        fmtPercent(probability) + ', ' + opponentDisplay(info) + ' ' + fmtPercent(1 - probability));
    } else {
      setText($('detail-probability'), 'not provided');
      setText($('detail-prediction'), 'not provided');
      setText($('readout'), 'Cutoff ' + fmtClock(cutoff) + ' \u2014 probability not provided');
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
      setText($('detail-window'), 'not recorded; this export does not list the sampled frame times');
    }

    renderHeatmap(cutoff, imageToken);
    renderContext(cutoff, findExplanation(state.data, cutoff), imageToken);
    renderHeatmapOnDemand(cutoff);
    renderMasking();
  }

  function renderHeatmap(cutoff, imageToken) {
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
        ? 'No pre-generated heatmap for this cutoff. Existing heatmaps: ' +
          describeCutoffs(explanations) + '.'
        : 'No heatmap has been generated for this cutoff.');
      return;
    }

    var explanationCutoff = num(explanation.cutoff_seconds);
    var observed = num(explanation.displayed_frame_seconds);
    var target = explanation.target_class;
    var info = probabilityContext();
    var side = target === 1 ? subjectDisplay(info) : (target === 0 ? opponentDisplay(info) : 'an unknown side');
    /* target_meaning is the export's own plain-language class description and wins over
     * the inferred race wording. Older exports without it keep the race sentence. */
    var targetMeaning = typeof explanation.target_meaning === 'string' && explanation.target_meaning.trim()
      ? explanation.target_meaning.trim() : '';
    var generation = explanation.generation && typeof explanation.generation === 'object'
      ? explanation.generation : {};
    var nativeBins = num(generation.native_temporal_bins);
    if (!Number.isFinite(nativeBins)) { nativeBins = num(explanation.native_temporal_bins); }

    /* Prediction cutoff and image observation time are separate fields; report each one
     * only when the export actually records it. */
    setText($('hm-cutoff'), fmtSeconds(explanationCutoff) + ' (' + fmtClock(explanationCutoff) + ')');
    setText($('hm-observed'), Number.isFinite(observed)
      ? fmtSeconds(observed) + ' (' + fmtClock(observed) + ')'
      : 'not recorded; only the prediction cutoff is known');
    setText($('hm-target'), (targetMeaning
      ? 'Heatmap class: ' + targetMeaning + ' (target_class ' + String(target) + ') \u2014 evidence for ' + side
      : 'Evidence supporting ' + side + ' (target_class ' + String(target) + ')'));
    setText($('hm-file'), explanation.image ? String(explanation.image) : 'not provided');
    setText($('heatmap-status'), Number.isFinite(observed)
      ? 'The prediction cutoff limits which frames the model could see; the image observation time is the frame ' +
        'that was upsampled for this heatmap. They are independent fields in review.json.'
      : 'The prediction cutoff is recorded. This export does not name the displayed frame, so the ' +
        'observation time is unavailable rather than zero. The map covers sampled history, not one exact frame.');
    setText($('heatmap-caption'), 'Prediction cutoff ' + fmtSeconds(explanationCutoff) +
      (Number.isFinite(observed) ? '; image observation time ' + fmtSeconds(observed) : '') +
      '. ' + (targetMeaning
        ? 'The heatmap class is "' + targetMeaning + '" (target_class ' + String(target) + '), evidence for ' + side + '.'
        : 'The heatmap highlights evidence the model used for ' + side + '.') +
      (Number.isFinite(nativeBins) ? ' Native Grad-CAM temporal bins: ' + trimNum(nativeBins, 0) + '; bins overlap.' : '') +
      ' It does not show a cause of the outcome or a precise action.');

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

    image.alt = 'Grad-CAM heatmap showing ' + (targetMeaning ? 'the class "' + targetMeaning + '"' : 'evidence supporting ' + side) +
      ' at prediction cutoff ' + fmtSeconds(explanationCutoff);
    image.onload = function () {
      if (imageToken !== state.imageToken) { return; }
      show(figure, true);
      show(errorBox, false);
    };
    image.onerror = function () {
      if (imageToken !== state.imageToken) { return; }
      show(figure, false);
      show(errorBox, true);
      setText(errorBox, 'Heatmap image could not be loaded (no data available): ' + imageUrl.href);
    };
    image.src = imageUrl.href;
  }

  function samplingSourceText(source) {
    if (source === 'recorded_at_export') { return 'recorded at export'; }
    if (source === 'reconstructed_from_saved_observations') {
      return 'reconstructed from saved observations';
    }
    return 'unknown';
  }

  function renderContext(cutoff, explanation, imageToken) {
    var section = $('context-section');
    var figure = $('context-figure');
    var image = $('context-image');
    var overlay = $('context-overlay');
    var empty = $('context-empty');
    var toggle = $('peak-toggle');
    var summary = $('attribution-summary');

    show(section, false);
    show(figure, false);
    show(overlay, false);
    show(empty, false);
    show(toggle, false);
    setText(toggle, '');
    show(summary, false);
    state.peakVisible = false;
    svgClear(overlay);
    image.onload = null;
    image.onerror = null;
    image.removeAttribute('src');
    setText($('context-caption'), '');
    setText($('context-meta'), '');
    setText($('context-empty'), '');
    setText($('attr-target'), 'not recorded');
    setText($('attr-max'), 'not recorded');
    setText($('attr-bin'), 'not recorded');
    setText($('attr-cell'), 'not recorded');
    setText($('attr-note'), '');
    if (!explanation) { renderGameView(null, imageToken); return; }

    show(section, true);
    renderAttribution(explanation);
    renderGameView(explanation, imageToken);
    var sampling = explanation.sampling && typeof explanation.sampling === 'object'
      ? explanation.sampling : null;
    var contextFrame = explanation.context_frame && typeof explanation.context_frame === 'object'
      ? explanation.context_frame : null;
    var contextImage = typeof explanation.context_image === 'string' ? explanation.context_image : '';
    if (!contextImage) {
      show(empty, true);
      setText(empty, explanation.context_error
        ? 'Context minimap not available: ' + String(explanation.context_error)
        : 'Context minimap is not recorded and could not be reconstructed for this export.');
      return;
    }

    var povLabel = 'Player ' + state.data.focus_player_id +
      (state.data.focus_player_race ? ' (' + state.data.focus_player_race + ')' : '');
    var prediction = fmtClock(cutoff);
    var range = sampling && Number.isFinite(num(sampling.min_seconds)) && Number.isFinite(num(sampling.max_seconds))
      ? fmtClock(num(sampling.min_seconds)) + ' \u2013 ' + fmtClock(num(sampling.max_seconds)) +
        ' (' + trimNum(num(sampling.frames), 0) + ' sampled frames)'
      : 'not recorded';
    var contextSeconds = contextFrame ? num(contextFrame.frame_seconds) : NaN;
    var contextLoop = contextFrame ? num(contextFrame.game_loop) : NaN;
    var source = contextFrame && contextFrame.source
      ? contextFrame.source : (sampling ? sampling.source : 'unknown');
    setText($('context-meta'),
      'POV: ' + povLabel + '. Prediction cutoff: ' + prediction + '. ' +
      'Sampled observation range: ' + range + '. ' +
      'Context image time: ' + (Number.isFinite(contextSeconds)
        ? fmtClock(contextSeconds) + (Number.isFinite(contextLoop) ? ' (game loop ' + trimNum(contextLoop, 0) + ')' : '')
        : 'not recorded') + '. ' +
      'Time source: ' + samplingSourceText(source) + '. ' +
      'The ownership layer is the model-input categorical map and can show units outside normal vision because extraction enabled the cheating layers. ' +
      'Display note: secondary non-neutral ownership cells are shown as self; neutral ownership is retained in this reference image. Raw unit categories and model inputs are unchanged.');

    setText($('context-caption'),
      'Model-input minimap visualization at the last actual sampled observation. ' +
      'Historical context only; it is not a game screenshot and is not claimed to be the exact frame of any temporal bin. ' +
      'The five Grad-CAM bins summarize the whole sampled history.');

    var imageUrl;
    try {
      imageUrl = new URL(contextImage, state.reviewDir);
    } catch (error) {
      show(empty, true);
      setText(empty, 'The context image path is not a valid URL: ' + contextImage);
      return;
    }
    var overlayBuilt = buildPeakCells(explanation.attribution);
    image.alt = 'Model-input minimap visualization for ' + povLabel +
      ' at context time ' + (Number.isFinite(contextSeconds) ? fmtClock(contextSeconds) : 'unknown');
    image.onload = function () {
      if (imageToken !== state.imageToken) { return; }
      show(figure, true);
      show(empty, false);
      show(toggle, overlayBuilt);
      setText(toggle, 'Show highest Grad-CAM region');
    };
    image.onerror = function () {
      if (imageToken !== state.imageToken) { return; }
      show(figure, false);
      show(empty, true);
      setText(empty, 'Context minimap image could not be loaded: ' + imageUrl.href);
    };
    image.src = imageUrl.href;
  }

  function renderAttribution(explanation) {
    var summary = $('attribution-summary');
    var attribution = explanation.attribution && typeof explanation.attribution === 'object'
      ? explanation.attribution : null;
    show(summary, true);
    setText($('attr-target'), explanation.target_meaning ? String(explanation.target_meaning) : 'not recorded');
    if (!attribution) {
      setText($('attr-max'), 'not recorded');
      setText($('attr-bin'), 'not recorded');
      setText($('attr-cell'), 'not recorded');
      setText($('attr-note'), 'This export does not record a raw Grad-CAM peak summary.');
      return;
    }
    if (attribution.available !== true) {
      setText($('attr-max'), 'No positive Grad-CAM evidence (all raw values are zero).');
      setText($('attr-bin'), 'not applicable');
      setText($('attr-cell'), 'not applicable');
      setText($('attr-note'), 'All raw Grad-CAM values are zero, so there is no positive region to outline. ' +
        'This is not a masking or faithfulness validation.');
      return;
    }
    var grid = Array.isArray(attribution.grid_shape) ? attribution.grid_shape : null;
    var bins = num(attribution.temporal_bins);
    var binIndex = num(attribution.bin_index);
    var row = num(attribution.row);
    var col = num(attribution.col);
    var maxValue = num(attribution.max_value);
    setText($('attr-max'), trimNum(maxValue, 6) + ' raw (normalized ' +
      trimNum(num(attribution.max_normalized), 2) + ')');
    setText($('attr-bin'), Number.isFinite(binIndex) && Number.isFinite(bins)
      ? (trimNum(binIndex + 1, 0) + ' of ' + trimNum(bins, 0)) : 'not recorded');
    setText($('attr-cell'), grid && Number.isFinite(row) && Number.isFinite(col)
      ? 'row ' + trimNum(row + 1, 0) + ', column ' + trimNum(col + 1, 0) +
        ' of ' + trimNum(num(grid[0]), 0) + ' \u00d7 ' + trimNum(num(grid[1]), 0) + ' coarse grid'
      : 'not recorded');
    var note = 'Computed from raw Grad-CAM values. Raw axis order is [temporal bin, row, column]; ' +
      'row 0 is top and column 0 is left, with no transpose. The full 128\u00d7128 minimap is used without spatial crop, ' +
      'while this 4\u00d74 attribution grid is coarse. ' +
      (attribution.tied ? 'Tied maximum cells: ' + attribution.cells.length + '. ' : '') +
      'The region is a coarse spatial reference, not a decisive position, action, or masking validation.';
    setText($('attr-note'), note);
  }

  function buildPeakCells(attribution) {
    var overlay = $('context-overlay');
    svgClear(overlay);
    if (!attribution || attribution.available !== true
        || !Array.isArray(attribution.cells) || !Array.isArray(attribution.grid_shape)) {
      return false;
    }
    var rows = num(attribution.grid_shape[0]);
    var cols = num(attribution.grid_shape[1]);
    if (!Number.isFinite(rows) || !Number.isFinite(cols) || rows < 1 || cols < 1) {
      return false;
    }
    var seen = {};
    attribution.cells.forEach(function (cell) {
      if (!Array.isArray(cell) || cell.length !== 3) { return; }
      var row = cell[1];
      var col = cell[2];
      var key = row + ':' + col;
      if (seen[key]) { return; }
      seen[key] = true;
      var node = document.createElement('span');
      node.className = 'peak-cell';
      node.style.left = (col / cols * 100) + '%';
      node.style.top = (row / rows * 100) + '%';
      node.style.width = (100 / cols) + '%';
      node.style.height = (100 / rows) + '%';
      node.title = 'Highest Grad-CAM region (coarse cell row ' + (row + 1) +
        ', column ' + (col + 1) + ')';
      overlay.appendChild(node);
    });
    return overlay.childNodes.length > 0;
  }

  function togglePeakOverlay() {
    state.peakVisible = !state.peakVisible;
    show($('context-overlay'), state.peakVisible);
    setText($('peak-toggle'), state.peakVisible
      ? 'Hide highest Grad-CAM region'
      : 'Show highest Grad-CAM region');
  }

  function bindImageModal() {
    if (state.modalBound) { return; }
    state.modalBound = true;
    var contextImage = $('context-image');
    var heatmapImage = $('heatmap-image');
    var gameViewImage = $('game-view-image');
    [contextImage, heatmapImage, gameViewImage].forEach(function (node) {
      if (!node) { return; }
      node.addEventListener('click', function () {
        openImageModal(node, imageModalTitle(node));
      });
      node.addEventListener('keydown', keyActivate(function () {
        openImageModal(node, imageModalTitle(node));
      }));
    });
    $('image-modal-close').addEventListener('click', closeImageModal);
    $('image-modal-backdrop').addEventListener('click', closeImageModal);
    $('image-zoom-in').addEventListener('click', function () { setModalZoom(modalState.scale * 1.5); });
    $('image-zoom-out').addEventListener('click', function () { setModalZoom(modalState.scale / 1.5); });
    $('image-zoom-reset').addEventListener('click', resetModalView);
    $('image-modal-viewport').addEventListener('wheel', function (event) {
      if (!modalState.open) { return; }
      event.preventDefault();
      setModalZoom(modalState.scale * Math.exp(-event.deltaY * 0.0015));
    }, { passive: false });
    var viewport = $('image-modal-viewport');
    viewport.addEventListener('pointerdown', function (event) {
      if (!modalState.open) { return; }
      modalState.drag = { x: event.clientX, y: event.clientY,
                          startX: modalState.x, startY: modalState.y };
      viewport.classList.add('is-dragging');
      if (viewport.setPointerCapture) { viewport.setPointerCapture(event.pointerId); }
    });
    viewport.addEventListener('pointermove', function (event) {
      if (!modalState.drag) { return; }
      modalState.x = modalState.drag.startX + event.clientX - modalState.drag.x;
      modalState.y = modalState.drag.startY + event.clientY - modalState.drag.y;
      updateModalTransform();
    });
    function endDrag() {
      modalState.drag = null;
      viewport.classList.remove('is-dragging');
    }
    viewport.addEventListener('pointerup', endDrag);
    viewport.addEventListener('pointercancel', endDrag);
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && modalState.open) {
        event.preventDefault();
        closeImageModal();
      }
    });
  }

  function updateModalTransform() {
    var image = $('image-modal-image');
    image.style.transform = 'translate(' + modalState.x + 'px, ' + modalState.y + 'px) scale(' +
      modalState.scale + ')';
  }

  function setModalZoom(scale) {
    modalState.scale = Math.min(8, Math.max(1, scale));
    if (modalState.scale === 1) {
      modalState.x = 0;
      modalState.y = 0;
    }
    updateModalTransform();
  }

  function resetModalView() {
    modalState.scale = 1;
    modalState.x = 0;
    modalState.y = 0;
    updateModalTransform();
  }

  function imageModalTitle(node) {
    if (node.id === 'context-image') { return 'Model-input minimap visualization'; }
    if (node.id === 'game-view-image') { return 'Re-rendered game view'; }
    return 'Grad-CAM heatmap';
  }

  function openImageModal(trigger, title) {
    var source = trigger.currentSrc || trigger.getAttribute('src');
    if (!source) { return; }
    var captionNode = trigger.id === 'context-image'
      ? $('context-caption')
      : (trigger.id === 'game-view-image' ? $('game-view-caption') : $('heatmap-caption'));
    var image = $('image-modal-image');
    image.src = source;
    image.alt = trigger.alt || '';
    setText($('image-modal-title'), title);
    setText($('image-modal-caption'), captionNode ? captionNode.textContent : '');
    modalState.open = true;
    modalState.trigger = trigger;
    modalState.scale = 1;
    modalState.x = 0;
    modalState.y = 0;
    modalState.drag = null;
    show($('image-modal'), true);
    document.body.classList.add('modal-open');
    updateModalTransform();
    $('image-modal-close').focus();
  }

  function closeImageModal() {
    if (!modalState.open) { return; }
    var trigger = modalState.trigger;
    modalState.open = false;
    modalState.trigger = null;
    modalState.drag = null;
    show($('image-modal'), false);
    document.body.classList.remove('modal-open');
    var image = $('image-modal-image');
    image.removeAttribute('src');
    if (trigger && trigger.isConnected) { trigger.focus(); }
  }

  function cancelGameViewTask() {
    if (state.gameViewController && typeof state.gameViewController.abort === 'function') {
      state.gameViewController.abort();
    }
    state.gameViewController = null;
    state.gameViewToken += 1;
    var button = $('game-view-generate');
    if (button) { button.disabled = false; }
  }

  function renderGameView(explanation, imageToken) {
    var block = $('game-view-block');
    var figure = $('game-view-figure');
    var image = $('game-view-image');
    var button = $('game-view-generate');
    var status = $('game-view-status');
    var error = $('game-view-error');
    var caption = $('game-view-caption');

    show(block, false);
    show(figure, false);
    show(error, false);
    show(button, false);
    setText(status, '');
    setText(error, '');
    setText(caption, '');
    image.onload = null;
    image.onerror = null;
    image.removeAttribute('src');
    button.disabled = false;
    button.textContent = 'Render game view';
    if (!explanation) { return; }

    show(block, true);
    var gameView = explanation.game_view && typeof explanation.game_view === 'object'
      ? explanation.game_view : null;
    if (gameView && gameView.image) {
      var imageUrl;
      try {
        imageUrl = new URL(gameView.image, state.reviewDir);
      } catch (err) {
        show(error, true);
        setText(error, 'The cached game-view path is not a valid URL.');
        return;
      }
      var actualTime = num(gameView.actual_seconds);
      setText(status, 'Cached re-rendered frame. Actual game loop ' +
        trimNum(num(gameView.actual_game_loop), 0) +
        (Number.isFinite(actualTime) ? ' (' + fmtClock(actualTime) + ')' : '') + '.');
      setText(caption, 'Re-rendered game view from the original replay and engine at the saved context game loop. ' +
        'This is not the model-input RGB and not a claim about a peak bin or decisive action.');
      image.alt = 'Re-rendered game view at context time ' +
        (Number.isFinite(actualTime) ? fmtClock(actualTime) : 'unknown');
      image.onload = function () {
        if (imageToken !== state.imageToken) { return; }
        show(figure, true);
      };
      image.onerror = function () {
        if (imageToken !== state.imageToken) { return; }
        show(figure, false);
        show(error, true);
        setText(error, 'Cached game-view image could not be loaded: ' + imageUrl.href);
        show(button, true);
      };
      image.src = imageUrl.href;
      return;
    }

    if (!isBackendReport()) {
      setText(status, 'On-demand game-view rendering is only available from a local backend job.');
      return;
    }
    if (!explanation.sampling && !explanation.context_frame) {
      setText(status, 'No saved context game loop is available for this explanation.');
      return;
    }
    show(button, true);
    setText(status, 'Optional: render one RGB frame from the original replay and engine at the saved context time. ' +
      'This is not the model-input RGB and does not rerun the analysis.');
  }

  function generateGameView() {
    if (!isBackendReport()) { return; }
    var cutoff = currentCutoff();
    var explanation = findExplanation(state.data, cutoff);
    if (!explanation) { return; }
    cancelGameViewTask();
    var token = ++state.gameViewToken;
    var controller = typeof AbortController === 'function' ? new AbortController() : null;
    state.gameViewController = controller;
    var button = $('game-view-generate');
    var status = $('game-view-status');
    var error = $('game-view-error');
    show(error, false);
    setText(error, '');
    show(button, false);
    button.disabled = true;
    setText(status, 'Rendering one RGB frame from the original replay and engine\u2026 ' +
      'No percentage is available.');
    var options = {
      method: 'POST',
      cache: 'no-store',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ player: state.job.player, cutoff_seconds: cutoff }),
    };
    if (controller) { options.signal = controller.signal; }
    fetch('/api/jobs/' + encodeURIComponent(state.job.id) + '/game-view', options)
      .then(function (response) {
        return readJson(response).then(function (payload) {
          if (!response.ok) {
            throw new ApiError(response.status, payload,
              'The game-view request was rejected with HTTP ' + response.status + '.');
          }
          return payload;
        });
      })
      .then(function (payload) {
        if (token !== state.gameViewToken) { return; }
        if (!payload || payload.state !== 'completed') {
          throw new Error(payload && payload.error_message
            ? payload.error_message : 'The game view did not complete.');
        }
        state.gameViewController = null;
        setText(status, 'Re-rendered frame ready. Refreshing the report\u2026');
        reloadReviewAfterGameView(token, cutoff, state.job.player);
      })
      .catch(function (err) {
        if (token !== state.gameViewToken) { return; }
        if (controller && controller.signal.aborted) { return; }
        state.gameViewController = null;
        show(button, true);
        button.disabled = false;
        button.textContent = 'Retry game view';
        setText(status, '');
        show(error, true);
        setText(error, err instanceof TypeError
          ? 'The backend could not be reached. No game view was attached.'
          : apiErrorText(err));
      });
  }

  function reloadReviewAfterGameView(token, cutoff, player) {
    var jobId = state.job ? state.job.id : null;
    var reviewToken = ++state.reviewToken;
    fetch(state.jsonUrl.href, { cache: 'no-store' })
      .then(function (response) {
        if (!response.ok) {
          throw new Error('HTTP ' + response.status + ' ' + response.statusText);
        }
        return response.json();
      })
      .then(function (data) {
        if (reviewToken !== state.reviewToken || token !== state.gameViewToken || state.mode !== 'report') { return; }
        if (!state.job || state.job.id !== jobId || state.job.player !== player) { return; }
        state.data = data;
        state.info = probabilityInfo(data);
        renderMeta(data);
        renderRunInfo(data);
        var points = pointsOf(data);
        renderChart(points, data);
        renderIntervals(points, data);
        renderStaticNotes(data);
        var pointIndex = findPointIndex(points, cutoff);
        if (pointIndex >= 0) {
          state.selected = pointIndex;
          state.selectedInterval = -1;
          intervalsOf(data).forEach(function (interval, intervalIndex) {
            if (Math.abs(num(interval.max_step_cutoff_seconds) - cutoff) < EPSILON ||
                Math.abs(num(interval.peak_cutoff_seconds) - cutoff) < EPSILON) {
              state.selectedInterval = intervalIndex;
            }
          });
          syncIntervalButtons();
          updateChartSelection();
        }
        renderDetail();
        var pov = state.job.index ? findPov(state.job.index, player) : null;
        if (pov) { updatePovNote(data, pov); }
      })
      .catch(function (err) {
        if (reviewToken !== state.reviewToken || token !== state.gameViewToken) { return; }
        show($('game-view-generate'), true);
        $('game-view-generate').textContent = 'Retry game view';
        show($('game-view-error'), true);
        setText($('game-view-error'), 'The game view was generated, but refreshing the report failed: ' +
          (err && err.message ? err.message : String(err)) + ' Retry uses the cached result.');
      });
  }

  function isBackendReport() {
    return state.mode === 'report' && state.job && state.job.id && state.job.player && !state.isDemo;
  }

  function cancelHeatmapTask() {
    closeImageModal();
    cancelGameViewTask();
    var task = state.heatmapTask;
    if (task) {
      clearTimeout(task.timer);
      if (task.controller && typeof task.controller.abort === 'function') {
        task.controller.abort();
      }
    }
    state.heatmapTask = null;
    state.heatmapToken += 1;
  }

  function currentHeatmapTask(cutoff) {
    var task = state.heatmapTask;
    if (!task || !state.job) { return null; }
    if (task.jobId !== state.job.id || task.player !== state.job.player) { return null; }
    return Math.abs(task.cutoff - cutoff) <= EPSILON ? task : null;
  }

  function renderHeatmapOnDemand(cutoff) {
    var box = $('heatmap-on-demand');
    var generate = $('heatmap-generate');
    var retry = $('heatmap-retry');
    var status = $('heatmap-generate-status');
    var error = $('heatmap-generate-error');
    var explanation = findExplanation(state.data, cutoff);

    show(box, false);
    show(error, false);
    setText(error, '');
    show(retry, false);
    show(generate, true);
    generate.disabled = false;
    setText(status, '');
    if (explanation) { return; }

    show(box, true);
    if (!isBackendReport()) {
      show(generate, false);
      show(retry, false);
      setText(status, 'On-demand heatmap generation is only available from a local backend job. ' +
        'Static exports and demos keep their pre-generated heatmaps.');
      return;
    }
    var task = currentHeatmapTask(cutoff);
    if (!task) {
      setText(status, 'No heatmap has been generated for this cutoff. Generation reuses this job\'s saved ' +
        'POV data and original checkpoint; it is not a new replay analysis.');
      return;
    }
    if (task.state === 'queued' || task.state === 'running') {
      show(generate, false);
      setText(status, 'Generating heatmap\u2026 this runs Grad-CAM for the selected POV and cutoff. ' +
        'No percentage is available.');
      return;
    }
    if (task.state === 'failed') {
      show(generate, false);
      show(retry, true);
      setText(status, 'Heatmap generation failed. The report and original heatmap are unchanged.');
      show(error, true);
      setText(error, task.error || 'The heatmap could not be generated.');
    }
  }

  function showHeatmapTaskError(task, message) {
    task.state = 'failed';
    task.error = message;
    var cutoff = task.cutoff;
    if (state.heatmapTask === task) {
      renderHeatmapOnDemand(cutoff);
      show($('heatmap-generate-error'), true);
      setText($('heatmap-generate-error'), message);
    }
  }

  function generateHeatmap() {
    if (!isBackendReport()) {
      renderHeatmapOnDemand(currentCutoff());
      return;
    }
    var cutoff = currentCutoff();
    var pointIndex = findPointIndex(pointsOf(state.data), cutoff);
    if (pointIndex < 0) {
      show($('heatmap-generate-error'), true);
      setText($('heatmap-generate-error'), 'Select a saved prediction point before generating a heatmap.');
      return;
    }
    cancelHeatmapTask();
    var task = {
      jobId: state.job.id,
      player: state.job.player,
      cutoff: cutoff,
      token: ++state.heatmapToken,
      state: 'queued',
      timer: null,
      controller: null,
      statusUrl: null,
      error: ''
    };
    state.heatmapTask = task;
    renderHeatmapOnDemand(cutoff);

    var controller = typeof AbortController === 'function' ? new AbortController() : null;
    task.controller = controller;
    var options = {
      method: 'POST',
      cache: 'no-store',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ player: task.player, cutoff_seconds: task.cutoff })
    };
    if (controller) { options.signal = controller.signal; }
    fetch('/api/jobs/' + encodeURIComponent(task.jobId) + '/heatmaps', options)
      .then(function (response) {
        return readJson(response).then(function (payload) {
          if (!response.ok) {
            throw new ApiError(response.status, payload,
              'The heatmap request was rejected with HTTP ' + response.status + '.');
          }
          return payload;
        });
      })
      .then(function (payload) {
        if (task.token !== state.heatmapToken || state.heatmapTask !== task) { return; }
        if (!payload || typeof payload !== 'object' || !payload.status_url) {
          throw new Error('The backend returned no heatmap task status URL.');
        }
        task.statusUrl = payload.status_url;
        task.state = payload.state || 'queued';
        if (task.state === 'completed') {
          finishHeatmapTask(task);
          return;
        }
        if (task.state === 'failed') {
          showHeatmapTaskError(task, payload.error_message || 'Heatmap generation failed.');
          return;
        }
        scheduleHeatmapPoll(task, 1500);
      })
      .catch(function (error) {
        if (task.token !== state.heatmapToken || state.heatmapTask !== task) { return; }
        if (controller && controller.signal.aborted) { return; }
        var text = error instanceof TypeError
          ? 'The backend could not be reached. The heatmap request was not started.'
          : apiErrorText(error);
        showHeatmapTaskError(task, text);
      });
  }

  function scheduleHeatmapPoll(task, delay) {
    if (state.heatmapTask !== task) { return; }
    clearTimeout(task.timer);
    task.timer = setTimeout(function () { fetchHeatmapStatus(task); }, delay);
  }

  function fetchHeatmapStatus(task) {
    if (state.heatmapTask !== task || !task.statusUrl) { return; }
    if (task.controller && typeof task.controller.abort === 'function') {
      task.controller.abort();
    }
    var controller = typeof AbortController === 'function' ? new AbortController() : null;
    task.controller = controller;
    var options = { cache: 'no-store' };
    if (controller) { options.signal = controller.signal; }
    fetch(task.statusUrl, options)
      .then(function (response) {
        return readJson(response).then(function (payload) {
          if (!response.ok) {
            throw new ApiError(response.status, payload, 'The heatmap status could not be read.');
          }
          return payload;
        });
      })
      .then(function (payload) {
        if (state.heatmapTask !== task) { return; }
        if (controller && controller.signal.aborted) { return; }
        if (!payload || typeof payload !== 'object' || !payload.state) {
          throw new Error('The backend returned an empty heatmap status.');
        }
        task.state = payload.state || 'running';
        task.controller = null;
        if (task.state === 'completed') {
          finishHeatmapTask(task);
          return;
        }
        if (task.state === 'failed') {
          showHeatmapTaskError(task, payload.error_message || 'Heatmap generation failed.');
          return;
        }
        setText($('heatmap-generate-status'),
          'Generating heatmap\u2026 state: ' + task.state + '. No percentage is available. ' +
          'The report stays available while this runs.');
        scheduleHeatmapPoll(task, 2000);
      })
      .catch(function (error) {
        if (state.heatmapTask !== task) { return; }
        if (controller && controller.signal.aborted) { return; }
        task.controller = null;
        showHeatmapTaskError(task, error instanceof TypeError
          ? 'Connection lost while checking the heatmap. The background task may still be running; retry checks the same request.'
          : apiErrorText(error));
      });
  }

  function finishHeatmapTask(task) {
    if (state.heatmapTask !== task) { return; }
    task.state = 'completed';
    clearTimeout(task.timer);
    task.timer = null;
    setText($('heatmap-generate-status'), 'Heatmap generated. Refreshing the report\u2026');
    show($('heatmap-generate-error'), false);
    reloadReviewAfterHeatmap(task);
  }

  function reloadReviewAfterHeatmap(task) {
    var token = ++state.reviewToken;
    fetch(state.jsonUrl.href, { cache: 'no-store' })
      .then(function (response) {
        if (!response.ok) {
          throw new Error('HTTP ' + response.status + ' ' + response.statusText);
        }
        return response.json();
      })
      .then(function (data) {
        if (token !== state.reviewToken || state.mode !== 'report' || !state.job) { return; }
        if (state.job.id !== task.jobId || state.job.player !== task.player) { return; }
        if (!data || typeof data !== 'object' || Array.isArray(data)) {
          throw new Error('review.json did not contain a JSON object.');
        }
        state.heatmapTask = null;
        state.data = data;
        state.info = probabilityInfo(data);
        renderMeta(data);
        renderRunInfo(data);
        var points = pointsOf(data);
        renderChart(points, data);
        renderIntervals(points, data);
        renderStaticNotes(data);
        var pointIndex = findPointIndex(points, task.cutoff);
        if (pointIndex >= 0) {
          state.selected = pointIndex;
          state.selectedInterval = -1;
          intervalsOf(data).forEach(function (interval, intervalIndex) {
            if (Math.abs(num(interval.max_step_cutoff_seconds) - task.cutoff) < EPSILON ||
                Math.abs(num(interval.peak_cutoff_seconds) - task.cutoff) < EPSILON) {
              state.selectedInterval = intervalIndex;
            }
          });
          syncIntervalButtons();
          updateChartSelection();
          renderDetail();
        } else {
          renderDetail();
        }
        var pov = state.job.index ? findPov(state.job.index, task.player) : null;
        if (pov) { updatePovNote(data, pov); }
      })
      .catch(function (error) {
        if (token !== state.reviewToken || state.mode !== 'report') { return; }
        task.state = 'failed';
        task.error = 'The heatmap was generated, but refreshing the report failed: ' +
          (error && error.message ? error.message : String(error)) +
          ' Retry uses the cached result.';
        if (state.heatmapTask === task) {
          renderHeatmapOnDemand(task.cutoff);
          show($('heatmap-generate-error'), true);
          setText($('heatmap-generate-error'), task.error);
        }
      });
  }

  function renderMaskingUnavailable(reason) {
    var body = $('masking-body');
    svgClear(body);
    var note = document.createElement('p');
    note.className = 'empty';
    note.textContent = 'Not computed for this export.';
    note.title = reason;
    body.appendChild(note);
    var hint = document.createElement('p');
    hint.className = 'note';
    hint.textContent = 'The heatmap and probability remain valid; a missing masking check is not evidence ' +
      'for or against the attribution.';
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
      ['Original probability (' + subjectDisplay() + ')',
       fmtPercent(num(check.original_probability))],
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
