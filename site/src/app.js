import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { registerModelicaLanguage } from "@cognipilot/rumoca/modelica-language";
import { buildModelicaFlightCatalog, createModelicaFlightRunner, loadExternalFlightModelEntries, modelicaCompletions, modelicaDiagnostics, modelicaHover } from "./rumoca_flight.js";

const DATA_DIR = "./public/data";
const AIRCRAFT_MODEL_SCALE = 0.6 / 2.2;

const state = {
  manifest: null,
  rows: [],
  maneuvers: [],
  playback: [],
  methodTraces: [],
  playbackScene: null,
  playbackPlaying: true,
  playbackSpeed: 1,
  playbackTimeS: 0,
  playbackLastMs: null,
  playbackScrubbing: false,
  playbackSegmentIndex: 0,
  playbackView: "animation",
  playbackCameraMode: "follow",
  flightSim: {
    active: false,
    model: "SafeControllerGreyBox",
    catalog: [],
    source: "",
    sourceModel: "",
    editor: null,
    editorReady: false,
    editorLoading: false,
    editorProvidersReady: false,
    diagnosticsTimer: null,
    runner: null,
    pending: false,
    editorOpen: false,
    armed: false,
    safeEnabled: false,
    keys: new Set(),
    throttle: 0,
    throttleInput: 0,
    roll: 0,
    pitch: 0,
    yaw: 0,
    trim: {
      throttle: 0,
      pitch: 0,
      roll: 0,
      yaw: 0,
    },
    x: null,
    step: null,
    dt: 1 / 240,
    accumulator: 0,
    lastStartS: 0,
    elapsedS: 0,
    inputLog: [],
    replaying: false,
    replayIndex: 0,
  },
  externalFlightModels: [],
  selectedMethods: new Set(),
  explorerOverlay: null,
  modelicaPredictionSignature: "",
  playbackTrackOverride: null,
  trailPastS: 10,
  trailFutureS: 0,
  tradeoffZoom: null,
  modelFamily: "aircraft6dof",
  scenario: "sportcub_mocap_5_22_26",
};

const ENU_TO_THREE_QUAT = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
const MESH_TO_BODY_FRD_QUAT = ENU_TO_THREE_QUAT.clone();
const fmt = new Intl.NumberFormat("en-US", { maximumSignificantDigits: 3 });
const ENABLED_METHOD_KEYS = new Set(["6DOF-GreyBoxOEM"]);
const CAMERA_MODES = ["first", "chase", "follow", "observer"];
const MODELICA_METHOD_PREFIX = "Modelica:";
const MODELICA_METHOD_COLORS = { "Modelica:RumocaFixedWing": 0x7dd3fc };
const MONACO_VERSION = "0.49.0";
const MONACO_BASE = `https://cdn.jsdelivr.net/npm/monaco-editor@${MONACO_VERSION}/min/vs`;
const MONACO_LOADER = `${MONACO_BASE}/loader.js`;

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function formatNumber(value, fallback = "--") {
  return finiteNumber(value) ? fmt.format(value) : fallback;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function cleanMethodName(method) {
  return String(method || "").replace(" (mocap)", "").replace(" (direct)", "");
}

function methodEnabled(method) {
  return ENABLED_METHOD_KEYS.has(methodKey(method));
}

function isModelicaPredictionMethod(method) {
  return String(method || "").startsWith(MODELICA_METHOD_PREFIX);
}

async function loadJson(name) {
  const response = await fetch(`${DATA_DIR}/${name}`);
  if (!response.ok) throw new Error(`failed to load ${name}: ${response.status}`);
  return response.json();
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) {
      existing.addEventListener("load", resolve, { once: true });
      existing.addEventListener("error", reject, { once: true });
      if (existing.dataset.loaded === "true") resolve();
      return;
    }
    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.dataset.loaded = "false";
    script.addEventListener("load", () => {
      script.dataset.loaded = "true";
      resolve();
    }, { once: true });
    script.addEventListener("error", reject, { once: true });
    document.head.append(script);
  });
}

async function loadMonaco() {
  if (window.monaco?.editor) return window.monaco;
  await loadScript(MONACO_LOADER);
  window.MonacoEnvironment = {
    getWorkerUrl() {
      const code = `self.MonacoEnvironment={baseUrl:${JSON.stringify(`${MONACO_BASE}/`)}};importScripts(${JSON.stringify(`${MONACO_BASE}/base/worker/workerMain.js`)});`;
      return `data:text/javascript;charset=utf-8,${encodeURIComponent(code)}`;
    },
  };
  return new Promise((resolve, reject) => {
    window.require.config({ paths: { vs: MONACO_BASE } });
    window.require(["vs/editor/editor.main"], () => resolve(window.monaco), reject);
  });
}

function defineModelicaTheme(monaco) {
  monaco.editor.defineTheme("mocap-modelica-dark", {
    base: "vs-dark",
    inherit: true,
    rules: [
      { token: "keyword", foreground: "7dd3fc", fontStyle: "bold" },
      { token: "type", foreground: "c4b5fd" },
      { token: "predefined", foreground: "86efac" },
      { token: "number", foreground: "fbbf24" },
      { token: "string", foreground: "fca5a5" },
      { token: "comment", foreground: "8aa1bd", fontStyle: "italic" },
    ],
    colors: {
      "editor.background": "#07111d",
      "editor.foreground": "#dce7f5",
      "editor.lineHighlightBackground": "#102036",
      "editorGutter.background": "#07111d",
      "editorCursor.foreground": "#dce7f5",
    },
  });
}

function modelicaCompletionKind(monaco, kind) {
  const kinds = monaco.languages.CompletionItemKind;
  const byLspKind = {
    3: kinds.Function,
    6: kinds.Variable,
    7: kinds.Class,
    10: kinds.Property,
    13: kinds.Enum,
    14: kinds.Keyword,
    15: kinds.Snippet,
    20: kinds.EnumMember,
    21: kinds.Constant,
  };
  return byLspKind[kind] || kinds.Text;
}

function hoverContents(hover) {
  const contents = hover?.contents;
  if (!contents) return [];
  const list = Array.isArray(contents) ? contents : [contents];
  return list.map((item) => {
    if (typeof item === "string") return { value: item };
    if (item.kind === "markdown" && typeof item.value === "string") return { value: item.value };
    if (typeof item.value === "string") return { value: item.value };
    return null;
  }).filter(Boolean);
}

function registerModelicaLspProviders(monaco) {
  if (state.flightSim.editorProvidersReady) return;
  monaco.languages.registerCompletionItemProvider("modelica", {
    triggerCharacters: [".", " ", "(", "["],
    async provideCompletionItems(model, position) {
      const word = model.getWordUntilPosition(position);
      const range = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: word.startColumn,
        endColumn: word.endColumn,
      };
      const items = await modelicaCompletions(model.getValue(), position.lineNumber - 1, position.column - 1);
      return {
        suggestions: items.map((item) => ({
          label: String(item.label || item.insertText || ""),
          kind: modelicaCompletionKind(monaco, item.kind),
          insertText: String(item.insertText || item.label || ""),
          detail: item.detail || "",
          documentation: item.documentation?.value || item.documentation || item.detail || "",
          range,
        })).filter((item) => item.label && item.insertText),
      };
    },
  });
  monaco.languages.registerHoverProvider("modelica", {
    async provideHover(model, position) {
      const hover = await modelicaHover(model.getValue(), position.lineNumber - 1, position.column - 1);
      const contents = hoverContents(hover);
      return contents.length ? { contents } : null;
    },
  });
  state.flightSim.editorProvidersReady = true;
}

function setFlightEditorSource(source) {
  const text = source || "";
  const textarea = document.querySelector("#modelica-flight-source");
  if (textarea && textarea.value !== text) textarea.value = text;
  const editor = state.flightSim.editor;
  if (editor && editor.getValue() !== text) editor.setValue(text);
}

function getFlightEditorSource() {
  return state.flightSim.editor?.getValue()
    ?? document.querySelector("#modelica-flight-source")?.value
    ?? state.flightSim.source
    ?? "";
}

function modelicaMarkerSeverity(monaco, diagnostic) {
  const severity = String(diagnostic.severity || "").toLowerCase();
  if (severity === "error" || diagnostic.severity === 1) return monaco.MarkerSeverity.Error;
  if (severity === "warning" || diagnostic.severity === 2) return monaco.MarkerSeverity.Warning;
  if (severity === "hint" || diagnostic.severity === 4) return monaco.MarkerSeverity.Hint;
  return monaco.MarkerSeverity.Info;
}

function diagnosticMarker(monaco, diagnostic) {
  const range = diagnostic.range || diagnostic.span || diagnostic.location || {};
  const start = range.start || range;
  const end = range.end || start;
  const startLineNumber = Math.max(1, (start.line ?? diagnostic.line ?? 0) + 1);
  const startColumn = Math.max(1, (start.character ?? start.column ?? diagnostic.column ?? 0) + 1);
  const endLineNumber = Math.max(startLineNumber, (end.line ?? start.line ?? diagnostic.line ?? 0) + 1);
  const endColumn = Math.max(startColumn + 1, (end.character ?? end.column ?? start.character ?? start.column ?? diagnostic.column ?? 0) + 1);
  return {
    severity: modelicaMarkerSeverity(monaco, diagnostic),
    message: diagnostic.message || diagnostic.text || diagnostic.detail || String(diagnostic),
    startLineNumber,
    startColumn,
    endLineNumber,
    endColumn,
    source: "Rumoca Modelica",
  };
}

async function refreshModelicaDiagnostics(source = getFlightEditorSource()) {
  const editor = state.flightSim.editor;
  if (!editor || !window.monaco?.editor) return [];
  const diagnostics = await modelicaDiagnostics(source);
  const markers = diagnostics.map((diagnostic) => diagnosticMarker(window.monaco, diagnostic));
  window.monaco.editor.setModelMarkers(editor.getModel(), "rumoca-modelica", markers);
  const errors = diagnostics.filter((d) => String(d.severity || "").toLowerCase() === "error" || d.severity === 1);
  if (errors.length) {
    setFlightStatus(`Rumoca diagnostics: ${errors.map(formatRumocaDiagnostic).filter(Boolean).slice(0, 2).join(" | ")}`, true);
  }
  return diagnostics;
}

function scheduleModelicaDiagnostics() {
  clearTimeout(state.flightSim.diagnosticsTimer);
  state.flightSim.diagnosticsTimer = setTimeout(() => {
    refreshModelicaDiagnostics().catch(() => {});
  }, 450);
}

async function ensureModelicaEditor() {
  const panel = document.querySelector("#modelica-flight-panel");
  const mount = document.querySelector("#modelica-flight-editor");
  if (!panel || !mount || state.flightSim.editorReady || state.flightSim.editorLoading) return;
  state.flightSim.editorLoading = true;
  try {
    const monaco = await loadMonaco();
    registerModelicaLanguage(monaco);
    registerModelicaLspProviders(monaco);
    defineModelicaTheme(monaco);
    state.flightSim.editor = monaco.editor.create(mount, {
      value: document.querySelector("#modelica-flight-source")?.value || state.flightSim.source || "",
      language: "modelica",
      theme: "mocap-modelica-dark",
      automaticLayout: true,
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
      fontSize: 12,
      lineHeight: 18,
      tabSize: 2,
      insertSpaces: true,
      wordWrap: "off",
    });
    state.flightSim.editor.onDidChangeModelContent(() => {
      state.flightSim.source = state.flightSim.editor.getValue();
      const textarea = document.querySelector("#modelica-flight-source");
      if (textarea) textarea.value = state.flightSim.source;
      scheduleModelicaDiagnostics();
    });
    panel.classList.add("monaco-ready");
    state.flightSim.editorReady = true;
    state.flightSim.editor.layout();
    scheduleModelicaDiagnostics();
  } catch (error) {
    console.warn("Modelica editor unavailable; using textarea fallback.", error);
    setFlightStatus("Modelica editor highlighting unavailable; using plain text fallback.", true);
  } finally {
    state.flightSim.editorLoading = false;
  }
}

function allRows() {
  return state.rows;
}

function allScenarios() {
  return state.manifest.scenarios;
}

function allDatasets() {
  return state.manifest.dataset_registry;
}

function scenariosForModel() {
  return allScenarios().filter((scenario) => scenario.model_family === state.modelFamily);
}

function datasetsForModel() {
  return allDatasets().filter((dataset) => dataset.model_family === state.modelFamily);
}

function selectedRows() {
  return allRows()
    .filter((row) => row.scenario === state.scenario)
    .filter((row) => methodEnabled(row.method))
    .filter((row) => finiteNumber(row.validation_score))
    .sort((a, b) => a.validation_score - b.validation_score);
}

function methodKey(method) {
  return cleanMethodName(method);
}

function scenarioTitle(id = state.scenario) {
  return allScenarios().find((scenario) => scenario.id === id)?.title || id;
}

function matchingManeuver() {
  const title = scenarioTitle();
  return state.maneuvers.find((row) => row.mode === title) || null;
}

function selectedPlayback() {
  if (state.playbackTrackOverride) {
    const override = state.playback.find((track) => track.id === state.playbackTrackOverride);
    if (override) return override;
  }
  return state.playback.find((track) => track.id === state.scenario) || state.playback.find((track) => track.model_family === state.modelFamily) || null;
}

function activeSegment(track = selectedPlayback()) {
  const segments = track?.segments?.length ? track.segments : track ? [track] : [];
  if (!segments.length) return null;
  const index = clamp(state.playbackSegmentIndex, 0, segments.length - 1);
  return segments[index];
}

function traceSegmentForMethod(key) {
  const trace = state.methodTraces.find((item) =>
    item.scenario === state.scenario && methodKey(item.method) === key
  );
  const segment = trace?.segments?.[state.playbackSegmentIndex];
  return segment ? { ...segment, method: key } : null;
}

function methodHasTrace(key) {
  if (isModelicaPredictionMethod(key)) return Boolean(state.playbackTrackOverride);
  if (state.playbackTrackOverride && state.explorerOverlay) {
    return traceSegmentsForFlight(key).length > 0;
  }
  return Boolean(traceSegmentForMethod(key));
}

function traceSegmentsForFlight(key) {
  // Benchmark chunk traces for the flight shown in the full-flight view,
  // shifted into the flight frame via each chunk's encoded start sample.
  const flightName = activeSegment(selectedPlayback())?.name || "";
  const overlay = state.explorerOverlay;
  const trace = state.methodTraces.find((item) =>
    item.scenario === state.scenario && methodKey(item.method) === key
  );
  if (!trace || !overlay) return [];
  const out = [];
  for (const segment of trace.segments || []) {
    const name = segment.name || "";
    const match = /__manual_(\d+)(?:_w(\d+))?/.exec(name);
    if (!match || !name.startsWith(flightName)) continue;
    const dtFull = overlay.dtFull || 0.01;
    const duration = (segment.time_s?.at(-1) || 0) + dtFull;
    const offsetS = parseInt(match[1], 10) * dtFull + (match[2] ? parseInt(match[2], 10) * duration : 0);
    const index = Math.min(overlay.track.length - 1, Math.round(offsetS * 10));
    const anchor = overlay.track[index];
    const shift = [anchor[0] - overlay.origin[0], anchor[1] - overlay.origin[1], anchor[2] - overlay.origin[2]];
    out.push({
      ...segment,
      method: key,
      flightOffsetS: offsetS,
      position_enu_m: segment.position_enu_m.map((point) => [point[0] + shift[0], point[1] + shift[1], point[2] + shift[2]]),
    });
  }
  return out;
}

function selectedTraceSegments() {
  const keys = state.selectedMethods;
  if (!keys.size) return [];
  if (state.playbackTrackOverride && state.explorerOverlay) {
    return Array.from(keys).flatMap((key) => traceSegmentsForFlight(key));
  }
  return Array.from(keys).map((key) => traceSegmentForMethod(key)).filter(Boolean);
}

function setDefaultScenario() {
  const scenarios = scenariosForModel();
  const hasScenario = scenarios.some((scenario) => scenario.id === state.scenario);
  if (!hasScenario || !selectedRows().length) {
    const withRows = scenarios.find((scenario) =>
      allRows().some((row) => row.scenario === scenario.id && finiteNumber(row.validation_score))
    );
    state.scenario = withRows?.id || scenarios[0]?.id || "";
  }
}

function renderModelTabs() {
  const host = document.querySelector("#model-tabs");
  host.innerHTML = "";
  // A selector with one option is noise; hide the whole Model control.
  const wrap = host.parentElement;
  if (wrap) wrap.hidden = state.manifest.model_families.length <= 1;
  for (const family of state.manifest.model_families) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.model = family;
    button.className = family === state.modelFamily ? "active" : "";
    button.textContent = family.replace("aircraft", "").toUpperCase();
    button.addEventListener("click", () => {
      state.modelFamily = family;
      state.selectedMethods.clear();
      state.playbackTrackOverride = null;
      state.explorerOverlay = null;
      resetTradeoffZoom();
      setDefaultScenario();
      notifyExplorerContext();
      render();
    });
    host.append(button);
  }
}

function renderScenarioSelect() {
  const select = document.querySelector("#scenario-select");
  select.innerHTML = "";
  for (const scenario of scenariosForModel()) {
    const option = document.createElement("option");
    option.value = scenario.id;
    option.textContent = scenario.title;
    select.append(option);
  }
  select.value = state.scenario;
}

function bindControls() {
  bindFlightKeyboard();
  // The explorer registers full flights as a first-class playback track so
  // the 3D viewer animates the entire record, not just benchmark windows.
  window.addEventListener("explorer-flights-ready", (event) => {
    state.playback = state.playback.filter((track) => track.id !== event.detail.track.id);
    state.playback.push(event.detail.track);
  });

  // Flight explorer selections drive the 3D viewer: fly the selected full
  // flight and jump the animation to the clicked time.
  window.addEventListener("explorer-set-ic", (event) => {
    const { flightIndex, timeS } = event.detail;
    state.explorerOverlay = event.detail.overlay || null;
    if (event.detail.methods) state.browserMethods = event.detail.methods;
    if (event.detail.methodColors) state.browserMethodColors = event.detail.methodColors;
    if (event.detail.models) state.browserModels = event.detail.models;
    renderFlightSimControls();
    document.querySelector("#playback-predict")?.classList.toggle("active", Boolean(event.detail.overlay?.anchored));
    if (event.detail.track) {
      state.playback = state.playback.filter((track) => track.id !== event.detail.track.id);
      state.playback.push(event.detail.track);
    }
    state.playbackTrackOverride = event.detail.track?.id || `explorer_${event.detail.scenario}`;
    state.modelFamily = "aircraft6dof";
    state.scenario = event.detail.scenario || "sportcub_mocap_5_22_26";
    state.playbackSegmentIndex = flightIndex;
    state.playbackTimeS = timeS || 0;
    state.playbackLastMs = null;
    // Keep the global dataset selector in sync so the two pickers agree.
    renderModelTabs();
    renderScenarioSelect();
    const scenarioSelect = document.querySelector("#scenario-select");
    if (scenarioSelect) scenarioSelect.value = state.scenario;
    render();
    // setPlaybackTrack resets the clock when the track changes; restore the
    // clicked time so the animation jumps to the selected moment.
    state.playbackTimeS = timeS || 0;
    state.playbackLastMs = null;
    // Acknowledge so the explorer stops re-announcing.
    window.dispatchEvent(new CustomEvent("playback-ack"));
    addModelicaPredictions(event.detail);
  });

  document.querySelector("#scenario-select").addEventListener("change", (event) => {
    state.scenario = event.target.value;
    state.playbackSegmentIndex = 0;
    state.selectedMethods.clear();
    state.playbackTrackOverride = null;
    state.explorerOverlay = null;
    resetTradeoffZoom();
    notifyExplorerContext();
    render();
  });

  document.querySelector("#playback-tabs").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-playback-view]");
    if (!button) return;
    state.playbackView = button.dataset.playbackView;
    renderPlaybackTabs();
    if (state.playbackView === "animation") resizePlayback();
  });

  document.querySelector("#playback-toggle").addEventListener("click", () => {
    state.playbackPlaying = !state.playbackPlaying;
    renderPlaybackControls(selectedPlayback());
  });
  document.querySelector("#playback-camera").addEventListener("change", (event) => {
    setPlaybackCameraMode(event.target.value || "follow");
    event.target.blur();
    renderPlaybackControls(selectedPlayback());
  });
  document.querySelector(".control-hud").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-trim]");
    if (!button) return;
    const key = button.dataset.trim;
    const delta = Number.parseFloat(button.dataset.delta || "0");
    const trim = state.flightSim.trim;
    if (!(key in trim) || !Number.isFinite(delta)) return;
    trim[key] = key === "throttle"
      ? clamp(trim[key] + delta, 0, 1)
      : clamp(trim[key] + delta, -1, 1);
    button.blur();
    setFlightStatus(`${key[0].toUpperCase()}${key.slice(1)} trim ${trim[key].toFixed(2)}.`);
    renderPlaybackControls(selectedPlayback());
  });
  document.querySelector("#playback-model-menu").addEventListener("change", (event) => {
    const input = event.target;
    const model = input?.dataset?.model;
    const role = input?.dataset?.role;
    if (!model || !role) return;
    if (role === "fly") {
      state.flightSim.model = model;
      syncFlightEditorSource(true);
      if (input.checked) {
        setPlaybackCameraMode("chase");
        startFlightSim();
      } else {
        stopFlightSim();
      }
    } else if (role === "predict") {
      const method = predictionMethodForModel(model);
      if (method) {
        if (input.checked) state.selectedMethods.add(method);
        else state.selectedMethods.delete(method);
        if (!Array.from(state.selectedMethods).some((m) => methodEnabled(m))) state.selectedMethods.add(method);
        window.dispatchEvent(new CustomEvent("methods-changed", { detail: { methods: Array.from(state.selectedMethods) } }));
      }
    }
    renderFlightSimControls();
    renderPlaybackControls(selectedPlayback());
  });
  document.querySelector("#playback-model-menu").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-role='edit']");
    if (!button) return;
    state.flightSim.model = button.dataset.model;
    state.flightSim.editorOpen = true;
    syncFlightEditorSource(true);
    renderFlightSimControls();
  });
  document.querySelector("#modelica-compile").addEventListener("click", () => {
    startFlightSim({ keepInputLog: true });
  });
  document.querySelector("#modelica-reset").addEventListener("click", () => {
    resetFlightSim();
  });
  document.querySelector("#modelica-replay").addEventListener("click", () => {
    startReplayInputs();
  });
  document.querySelector("#modelica-clear-replay").addEventListener("click", () => {
    state.flightSim.inputLog = [];
    state.flightSim.replaying = false;
    state.flightSim.replayIndex = 0;
    setFlightStatus("Input log cleared.");
    renderFlightSimControls();
  });
  document.querySelector("#modelica-close").addEventListener("click", () => {
    state.flightSim.editorOpen = false;
    renderFlightSimControls();
  });
  document.querySelector("#playback-scrub").addEventListener("mousemove", (event) => {
    const segment = activeSegment(selectedPlayback());
    if (!segment?.labels || !segment.time_s?.length) {
      event.target.title = "";
      return;
    }
    const rect = event.target.getBoundingClientRect();
    const total = segment.time_s[segment.time_s.length - 1] || 1;
    const t = clamp(((event.clientX - rect.left) / rect.width) * total, 0, total);
    let index = 0;
    while (index < segment.time_s.length - 1 && segment.time_s[index + 1] < t) index += 1;
    const names = ["ground", "ground effect", "stabilized", "manual"];
    const dropped = segment.tracked && !segment.tracked[index];
    event.target.title = `${dropped ? "mocap dropout" : names[segment.labels[index]] || ""} | t = ${t.toFixed(1)} s`;
  });

  for (const [handleId, kind] of [["#trail-past-handle", "past"], ["#trail-future-handle", "future"]]) {
    const handle = document.querySelector(handleId);
    if (!handle) continue;
    handle.addEventListener("dblclick", () => {
      // Double-click resets the handle onto the playhead (zero span).
      if (kind === "past") state.trailPastS = 0;
      else state.trailFutureS = 0;
      updateTrailHandles(playbackDuration(activeSegment(selectedPlayback())));
    });
    handle.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      handle.setPointerCapture(event.pointerId);
      const wrap = document.querySelector(".scrub-wrap");
      const move = (ev) => {
        const rect = wrap.getBoundingClientRect();
        const duration = playbackDuration(activeSegment(selectedPlayback())) || 1;
        const t = clamp(((ev.clientX - rect.left) / rect.width) * duration, 0, duration);
        const snapS = (14 / rect.width) * duration;
        if (kind === "past") {
          const past = clamp(state.playbackTimeS - t, 0, duration);
          state.trailPastS = past < snapS ? 0 : past;
        } else {
          // Snap to the playhead when close so "no future" lines up exactly.
          const future = clamp(t - state.playbackTimeS, 0, duration);
          state.trailFutureS = future < snapS ? 0 : future;
        }
        updateTrailHandles(duration);
      };
      const stop = () => {
        handle.removeEventListener("pointermove", move);
        handle.removeEventListener("pointerup", stop);
      };
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", stop);
    });
  }

  document.querySelector("#playback-predict").addEventListener("click", () => {
    // On-the-fly free runs exist only for the flight-explorer dataset.
    if (!state.playbackTrackOverride) return;
    window.dispatchEvent(new CustomEvent("explorer-anchor-request", { detail: { timeS: state.playbackTimeS } }));
  });
  document.addEventListener("pointerdown", (event) => {
    const dd = document.querySelector("#playback-model-dd");
    if (dd?.open && !dd.contains(event.target)) dd.open = false;
  });
  document.querySelector("#playback-fullscreen").addEventListener("click", () => {
    // Fullscreen the whole animation view so the playback controls (time
    // bar, trail handles, Predict here) stay usable, not just the 3D stage.
    const view = document.querySelector("#animation-view");
    if (document.fullscreenElement) document.exitFullscreen();
    else view.requestFullscreen();
  });
  document.addEventListener("fullscreenchange", () => {
    const button = document.querySelector("#playback-fullscreen");
    if (button) {
      button.innerHTML = document.fullscreenElement ? "&#10005;" : "&#9974;";
      button.setAttribute("aria-label", document.fullscreenElement ? "Exit fullscreen" : "Fullscreen");
      button.title = document.fullscreenElement ? "Exit fullscreen" : "Fullscreen";
    }
    resizePlayback();
  });

  document.querySelector("#playback-speed").addEventListener("change", (event) => {
    state.playbackSpeed = Number.parseFloat(event.target.value) || 1;
  });
  document.querySelector("#playback-segment").addEventListener("change", (event) => {
    state.playbackSegmentIndex = Number.parseInt(event.target.value, 10) || 0;
    state.playbackTimeS = 0;
    if (Array.from(state.selectedMethods).some((key) => !methodHasTrace(key))) {
      state.selectedMethods.clear();
    }
    setPlaybackTrack(selectedPlayback(), true);
    renderTimeseries();
    renderTradeoff(selectedRows());
    renderLeaderboard(selectedRows());
  });
  document.querySelector("#playback-scrub").addEventListener("input", (event) => {
    const track = selectedPlayback();
    const duration = playbackDuration(track);
    state.playbackScrubbing = true;
    if (state.flightSim.active) stopFlightSim();
    seekPlayback(Number.parseFloat(event.target.value) * duration);
  });
  document.querySelector("#playback-scrub").addEventListener("change", () => {
    state.playbackScrubbing = false;
  });
}

function isTypingTarget(target) {
  const tag = target?.tagName?.toLowerCase();
  if (tag === "textarea" || tag === "select" || target?.isContentEditable) return true;
  if (tag !== "input") return false;
  return ["text", "search", "url", "tel", "email", "password", "number"].includes((target.type || "text").toLowerCase());
}

function setPlaybackCameraMode(mode) {
  state.playbackCameraMode = CAMERA_MODES.includes(mode) ? mode : "follow";
  if (state.playbackCameraMode !== "first" && state.playbackScene) {
    state.playbackScene.camera.up.set(0, 1, 0);
    updatePlaybackCamera(state.playbackScene);
  }
}

function cyclePlaybackCameraMode() {
  const index = CAMERA_MODES.indexOf(state.playbackCameraMode);
  setPlaybackCameraMode(CAMERA_MODES[(Math.max(index, 0) + 1) % CAMERA_MODES.length]);
  setFlightStatus(`Camera: ${document.querySelector(`#playback-camera option[value="${state.playbackCameraMode}"]`)?.textContent || state.playbackCameraMode}`);
  renderPlaybackControls(selectedPlayback());
}

function bindFlightKeyboard() {
  document.addEventListener("keydown", (event) => {
    if (!state.flightSim.active || isTypingTarget(event.target)) return;
    const key = event.code === "Space" ? " " : (event.key.length === 1 ? event.key.toLowerCase() : event.key);
    if (!["w", "s", "a", "d", "r", "c", " ", "Space", "Spacebar", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(key)) return;
    event.preventDefault();
    if (key === "r") {
      resetFlightSim();
      return;
    }
    if (key === "c") {
      if (!event.repeat) cyclePlaybackCameraMode();
      return;
    }
    if (key === " " || key === "Space") {
      if (!event.repeat) toggleFlightSafeMode();
      return;
    }
    state.flightSim.keys.add(key);
  });
  document.addEventListener("keyup", (event) => {
    const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
    state.flightSim.keys.delete(key);
  });
}

function toggleFlightSafeMode() {
  const sim = state.flightSim;
  if (sim.runner && !sim.runner.supportsSafeToggle) {
    sim.safeEnabled = false;
    setFlightStatus(`${sim.runner.modelName} has no enable_safe input.`, true);
    renderFlightSimControls();
    return;
  }
  sim.safeEnabled = !sim.safeEnabled;
  setFlightStatus(`SAFE ${sim.safeEnabled ? "enabled" : "disabled"}.`);
  renderFlightSimControls();
}

function predictionMethodForModel(model) {
  if (model === "GreyBoxOEM" || model === "SafeControllerGreyBox") return "6DOF-GreyBoxOEM";
  if (model === "RumocaFixedWing") return "Modelica:RumocaFixedWing";
  return null;
}

function modelForModelicaPrediction(method) {
  return String(method || "").slice(MODELICA_METHOD_PREFIX.length);
}

function modelLabelForMethod(method) {
  if (methodKey(method) === "6DOF-GreyBoxOEM") return "GreyBoxOEM";
  return String(method || "").replace("6DOF-", "");
}

function renderMeta() {
  const sha = state.manifest.git_sha ? state.manifest.git_sha.slice(0, 7) : "unknown";
  const generated = new Date(state.manifest.generated_at).toLocaleString();
  document.querySelector("#run-meta").textContent = `schema ${state.manifest.schema_version} | ${sha} | ${generated}`;
}

function renderPlaybackTabs() {
  document.querySelectorAll("#playback-tabs button[data-playback-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.playbackView === state.playbackView);
  });
  const animation = document.querySelector("#animation-view");
  const history = document.querySelector("#history-view");
  const splits = document.querySelector("#splits-view");
  const modelsView = document.querySelector("#models-view");
  if (modelsView) modelsView.hidden = state.playbackView !== "models";
  if (animation) animation.hidden = state.playbackView !== "animation";
  if (history) history.hidden = state.playbackView !== "history";
  if (splits) splits.hidden = state.playbackView !== "splits";
}

function renderSummary(rows) {
  const best = rows[0];
  const datasets = datasetsForModel();
  const methods = new Set(allRows()
    .filter((row) => row.model_family === state.modelFamily)
    .filter((row) => methodEnabled(row.method))
    .map((row) => cleanMethodName(row.method)));
  document.querySelector("#best-method").textContent = best ? cleanMethodName(best.method) : "--";
  document.querySelector("#best-score").textContent = best ? formatNumber(best.validation_score) : "--";
  document.querySelector("#method-count").textContent = String(methods.size);
  document.querySelector("#dataset-count").textContent = String(datasets.length);
}

function logExtent(values) {
  const finite = values.filter((value) => finiteNumber(value) && value > 0);
  if (!finite.length) return [0.01, 1];
  return [Math.min(...finite) * 0.75, Math.max(...finite) * 1.4];
}

function logScale(value, min, max, start, end) {
  const logMin = Math.log10(min);
  const logMax = Math.log10(max);
  const t = (Math.log10(Math.max(value, min)) - logMin) / Math.max(logMax - logMin, 1e-9);
  return start + t * (end - start);
}

function tradeoffKey() {
  return `${state.modelFamily}:${state.scenario}`;
}

function resetTradeoffZoom() {
  state.tradeoffZoom = null;
}

function constrainedLogView(view, base) {
  const minSpan = 0.12;
  const baseXSpan = base.xMax - base.xMin;
  const baseYSpan = base.yMax - base.yMin;
  let xMin = view.xMin;
  let xMax = view.xMax;
  let yMin = view.yMin;
  let yMax = view.yMax;
  if (xMax - xMin < minSpan) {
    const center = (xMin + xMax) / 2;
    xMin = center - minSpan / 2;
    xMax = center + minSpan / 2;
  }
  if (yMax - yMin < minSpan) {
    const center = (yMin + yMax) / 2;
    yMin = center - minSpan / 2;
    yMax = center + minSpan / 2;
  }
  if (xMax - xMin >= baseXSpan) {
    xMin = base.xMin;
    xMax = base.xMax;
  } else {
    if (xMin < base.xMin) {
      xMax += base.xMin - xMin;
      xMin = base.xMin;
    }
    if (xMax > base.xMax) {
      xMin -= xMax - base.xMax;
      xMax = base.xMax;
    }
  }
  if (yMax - yMin >= baseYSpan) {
    yMin = base.yMin;
    yMax = base.yMax;
  } else {
    if (yMin < base.yMin) {
      yMax += base.yMin - yMin;
      yMin = base.yMin;
    }
    if (yMax > base.yMax) {
      yMin -= yMax - base.yMax;
      yMax = base.yMax;
    }
  }
  return { xMin, xMax, yMin, yMax };
}

function powers(min, max) {
  const out = [];
  for (let power = Math.floor(Math.log10(min)); power <= Math.ceil(Math.log10(max)); power += 1) {
    out.push(10 ** power);
  }
  return out;
}

function renderTradeoff(rows) {
  const host = document.querySelector("#tradeoff-plot");
  host.innerHTML = "";
  const width = Math.max(host.clientWidth || 900, 640);
  const height = 410;
  const margin = { top: 20, right: 34, bottom: 54, left: 76 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const baseXExtent = logExtent(rows.map((row) => row.train_elapsed_s || row.total_elapsed_s || row.rollout_elapsed_s));
  const baseYExtent = logExtent(rows.map((row) => row.validation_score));
  const baseLogView = {
    xMin: Math.log10(baseXExtent[0]),
    xMax: Math.log10(baseXExtent[1]),
    yMin: Math.log10(baseYExtent[0]),
    yMax: Math.log10(baseYExtent[1]),
  };
  if (!state.tradeoffZoom || state.tradeoffZoom.key !== tradeoffKey()) {
    state.tradeoffZoom = { key: tradeoffKey(), ...baseLogView };
  }
  state.tradeoffZoom = { key: tradeoffKey(), ...constrainedLogView(state.tradeoffZoom, baseLogView) };
  const xExtent = [10 ** state.tradeoffZoom.xMin, 10 ** state.tradeoffZoom.xMax];
  const yExtent = [10 ** state.tradeoffZoom.yMin, 10 ** state.tradeoffZoom.yMax];
  const color = "var(--mocap)";

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("aria-label", "Cost-error tradeoff. Mouse wheel zooms, drag pans, shift-drag selects a zoom box, double-click resets.");
  const eventPoint = (event) => {
    const rect = svg.getBoundingClientRect();
    const x = (event.clientX - rect.left) * width / Math.max(rect.width, 1);
    const y = (event.clientY - rect.top) * height / Math.max(rect.height, 1);
    return { x, y };
  };
  const inPlot = ({ x, y }) => x >= margin.left && x <= margin.left + plotWidth && y >= margin.top && y <= margin.top + plotHeight;
  const logAtPoint = ({ x, y }) => ({
    x: state.tradeoffZoom.xMin + ((x - margin.left) / plotWidth) * (state.tradeoffZoom.xMax - state.tradeoffZoom.xMin),
    y: state.tradeoffZoom.yMax - ((y - margin.top) / plotHeight) * (state.tradeoffZoom.yMax - state.tradeoffZoom.yMin),
  });
  const setLogView = (view) => {
    state.tradeoffZoom = { key: tradeoffKey(), ...constrainedLogView(view, baseLogView) };
    render();
  };
  svg.addEventListener("wheel", (event) => {
    const point = eventPoint(event);
    if (!inPlot(point)) return;
    event.preventDefault();
    const anchor = logAtPoint(point);
    const factor = event.deltaY < 0 ? 0.82 : 1.22;
    const view = state.tradeoffZoom;
    setLogView({
      xMin: anchor.x - (anchor.x - view.xMin) * factor,
      xMax: anchor.x + (view.xMax - anchor.x) * factor,
      yMin: anchor.y - (anchor.y - view.yMin) * factor,
      yMax: anchor.y + (view.yMax - anchor.y) * factor,
    });
  }, { passive: false });
  svg.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || event.target.classList?.contains("tradeoff-point")) return;
    const point = eventPoint(event);
    if (!inPlot(point)) return;
    if (event.shiftKey) {
      // Shift-drag: rectangular select, zooming to the released box.
      event.preventDefault();
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("class", "select-rect");
      svg.append(rect);
      const clampPoint = (p) => ({
        x: clamp(p.x, margin.left, margin.left + plotWidth),
        y: clamp(p.y, margin.top, margin.top + plotHeight),
      });
      const start = clampPoint(point);
      const update = (moveEvent) => {
        const now = clampPoint(eventPoint(moveEvent));
        rect.setAttribute("x", Math.min(start.x, now.x));
        rect.setAttribute("y", Math.min(start.y, now.y));
        rect.setAttribute("width", Math.abs(now.x - start.x));
        rect.setAttribute("height", Math.abs(now.y - start.y));
        return now;
      };
      const finish = (upEvent) => {
        document.removeEventListener("pointermove", update);
        document.removeEventListener("pointerup", finish);
        document.removeEventListener("pointercancel", finish);
        rect.remove();
        const end = clampPoint(eventPoint(upEvent));
        if (Math.abs(end.x - start.x) < 8 || Math.abs(end.y - start.y) < 8) return;
        const a = logAtPoint(start);
        const b = logAtPoint(end);
        setLogView({
          xMin: Math.min(a.x, b.x),
          xMax: Math.max(a.x, b.x),
          yMin: Math.min(a.y, b.y),
          yMax: Math.max(a.y, b.y),
        });
      };
      document.addEventListener("pointermove", update);
      document.addEventListener("pointerup", finish);
      document.addEventListener("pointercancel", finish);
      svg.setPointerCapture(event.pointerId);
      return;
    }
    const dragStart = { clientX: event.clientX, clientY: event.clientY, view: { ...state.tradeoffZoom } };
    const moveDrag = (moveEvent) => {
      const dx = (moveEvent.clientX - dragStart.clientX) * width / Math.max(svg.getBoundingClientRect().width, 1);
      const dy = (moveEvent.clientY - dragStart.clientY) * height / Math.max(svg.getBoundingClientRect().height, 1);
      const xSpan = dragStart.view.xMax - dragStart.view.xMin;
      const ySpan = dragStart.view.yMax - dragStart.view.yMin;
      state.tradeoffZoom = {
        key: tradeoffKey(),
        ...constrainedLogView({
          xMin: dragStart.view.xMin - (dx / plotWidth) * xSpan,
          xMax: dragStart.view.xMax - (dx / plotWidth) * xSpan,
          yMin: dragStart.view.yMin + (dy / plotHeight) * ySpan,
          yMax: dragStart.view.yMax + (dy / plotHeight) * ySpan,
        }, baseLogView),
      };
      renderTradeoff(rows);
    };
    const stopDrag = () => {
      document.removeEventListener("pointermove", moveDrag);
      document.removeEventListener("pointerup", stopDrag);
      document.removeEventListener("pointercancel", stopDrag);
      svg.classList.remove("dragging");
    };
    document.addEventListener("pointermove", moveDrag);
    document.addEventListener("pointerup", stopDrag);
    document.addEventListener("pointercancel", stopDrag);
    svg.classList.add("dragging");
    svg.setPointerCapture(event.pointerId);
  });
  svg.addEventListener("dblclick", () => {
    resetTradeoffZoom();
    render();
  });

  const add = (tag, attrs, text) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    svg.append(node);
    return node;
  };

  for (const tick of powers(...xExtent)) {
    const x = logScale(tick, xExtent[0], xExtent[1], margin.left, margin.left + plotWidth);
    add("line", { x1: x, y1: margin.top, x2: x, y2: margin.top + plotHeight, class: "grid-line" });
    add("text", { x, y: height - 28, "text-anchor": "middle", class: "tick" }, `10^${Math.round(Math.log10(tick))}`);
  }
  for (const tick of powers(...yExtent)) {
    const y = logScale(tick, yExtent[0], yExtent[1], margin.top + plotHeight, margin.top);
    add("line", { x1: margin.left, y1: y, x2: margin.left + plotWidth, y2: y, class: "grid-line" });
    add("text", { x: margin.left - 10, y: y + 4, "text-anchor": "end", class: "tick" }, `10^${Math.round(Math.log10(tick))}`);
  }

  add("rect", { x: margin.left, y: margin.top, width: plotWidth, height: plotHeight, fill: "none", stroke: "var(--line)" });
  add("text", { x: margin.left + plotWidth / 2, y: height - 8, "text-anchor": "middle", class: "axis-label" }, "training time [s]");
  add("text", { x: 18, y: margin.top + plotHeight / 2, transform: `rotate(-90 18 ${margin.top + plotHeight / 2})`, "text-anchor": "middle", class: "axis-label" }, "validation score (lower is better)");
  add("text", { x: margin.left + plotWidth, y: margin.top - 6, "text-anchor": "end", class: "plot-hint" }, "wheel zoom | drag pan | shift-drag select | double-click reset");

  const nominal = rows.find((row) => cleanMethodName(row.method).includes("Nominal") && finiteNumber(row.validation_score));
  if (nominal) {
    const y = logScale(nominal.validation_score, yExtent[0], yExtent[1], margin.top + plotHeight, margin.top);
    // Labels stay strictly on their own side of the nominal line and are
    // dropped when that region is too thin to hold the 42 px glyphs:
    // "known" hangs its baseline above the line, "unknown" hangs its top
    // below it (dominant-baseline so no part of a glyph crosses the line).
    const labelHeight = 46;
    if (y - margin.top > labelHeight) {
      add("text", {
        x: margin.left + plotWidth * 0.5,
        y: y - 10,
        "text-anchor": "middle",
        class: "known-label",
      }, "known");
    }
    if (margin.top + plotHeight - y > labelHeight) {
      add("text", {
        x: margin.left + plotWidth * 0.5,
        y: y + 10,
        "text-anchor": "middle",
        "dominant-baseline": "hanging",
        class: "unknown-label",
      }, "unknown");
    }
    add("line", {
      x1: margin.left,
      y1: y,
      x2: margin.left + plotWidth,
      y2: y,
      class: "nominal-line",
    });
  }

  for (const row of rows) {
    const xValue = row.train_elapsed_s || row.total_elapsed_s || row.rollout_elapsed_s || 0.01;
    const x = logScale(xValue, xExtent[0], xExtent[1], margin.left, margin.left + plotWidth);
    const y = logScale(row.validation_score, yExtent[0], yExtent[1], margin.top + plotHeight, margin.top);
    const isNominal = cleanMethodName(row.method).includes("Nominal");
    const key = methodKey(row.method);
    const hasTrace = methodHasTrace(key);
    const selected = state.selectedMethods.has(key);
    const circle = add("circle", {
      cx: x,
      cy: y,
      r: selected ? 8.0 : isNominal ? 6.5 : 5.5,
      fill: isNominal ? "white" : color,
      stroke: selected ? "#111827" : isNominal ? "var(--nominal)" : "#1d2430",
      "stroke-width": selected ? 2.6 : isNominal ? 1.8 : 1,
      opacity: hasTrace ? (state.selectedMethods.size && !selected ? 0.34 : 0.88) : 0.22,
      class: `tradeoff-point${hasTrace ? "" : " no-trace"}`,
      tabindex: hasTrace ? "0" : "-1",
    });
    if (hasTrace) circle.addEventListener("click", () => toggleMethodSelection(key));
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = hasTrace
      ? `${cleanMethodName(row.method)} | ${formatNumber(row.validation_score)}`
      : `${cleanMethodName(row.method)} | ${formatNumber(row.validation_score)} | no exported trajectory`;
    circle.append(title);
  }

  const bestNonNominal = rows.find((row) => !cleanMethodName(row.method).includes("Nominal"));
  if (bestNonNominal) {
    const xValue = bestNonNominal.train_elapsed_s || bestNonNominal.total_elapsed_s || bestNonNominal.rollout_elapsed_s || 0.01;
    const x = logScale(xValue, xExtent[0], xExtent[1], margin.left, margin.left + plotWidth);
    const y = logScale(bestNonNominal.validation_score, yExtent[0], yExtent[1], margin.top + plotHeight, margin.top);
    add("text", {
      x: Math.min(x + 10, margin.left + plotWidth - 130),
      y: Math.max(y - 10, margin.top + 16),
      class: "best-point-label",
    }, cleanMethodName(bestNonNominal.method));
  }

  host.append(svg);
}

function renderLeaderboard(rows) {
  const body = document.querySelector("#leaderboard-body");
  body.innerHTML = "";
  for (const row of rows) {
    const tr = document.createElement("tr");
    const key = methodKey(row.method);
    const hasTrace = methodHasTrace(key);
    tr.className = `${state.selectedMethods.has(key) ? "selected-method-row" : ""}${hasTrace ? "" : " no-trace-row"}`;
    tr.title = hasTrace ? "Click to show this model trajectory." : "No exported model trajectory is available for this flight segment.";
    if (hasTrace) tr.addEventListener("click", () => toggleMethodSelection(key));
    const values = [
      cleanMethodName(row.method),
      formatNumber(row.validation_score),
      formatNumber(row.rmse_position_m ?? row.rmse_mocap_position_m),
      formatNumber(row.train_elapsed_s),
      formatNumber(row.rollout_elapsed_s),
      row.training_scenario || "--",
    ];
    for (const [index, value] of values.entries()) {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (index > 0 && index < 5) cell.className = "numeric";
      tr.append(cell);
    }
    body.append(tr);
  }
}

function toggleMethodSelection(key) {
  if (!methodHasTrace(key)) {
    return;
  }
  if (state.selectedMethods.has(key)) {
    state.selectedMethods.delete(key);
  } else {
    state.selectedMethods.add(key);
  }
  window.dispatchEvent(new CustomEvent("methods-changed", { detail: { methods: Array.from(state.selectedMethods) } }));
  render();
}

function notifyExplorerContext() {
  // Tell the flight-explorer module whether its dataset is the one displayed,
  // so it never publishes an overlay (and hijacks the view) from another
  // dataset's screen.
  window.dispatchEvent(new CustomEvent("playback-context-changed", {
    detail: {
      scenario: state.modelFamily === "aircraft6dof" ? state.scenario : null,
    },
  }));
}

function flightSimChoices() {
  state.flightSim.catalog = buildModelicaFlightCatalog(state.browserModels, state.externalFlightModels);
  return state.flightSim.catalog.map((entry) => entry.label);
}

function selectedFlightModelicaEntry() {
  return state.flightSim.catalog.find((entry) => entry.label === state.flightSim.model) || state.flightSim.catalog[0] || null;
}

function setFlightStatus(message, isError = false) {
  const text = message || "";
  for (const node of [
    document.querySelector("#modelica-flight-status"),
    document.querySelector("#playback-flight-status"),
  ]) {
    if (!node) continue;
    node.textContent = text;
    node.title = text;
    node.classList.toggle("error", Boolean(isError));
    node.hidden = !text;
  }
}

function formatRumocaDiagnostic(diagnostic) {
  const parts = [];
  const range = diagnostic.range || diagnostic.span || diagnostic.location;
  const start = range?.start || range;
  if (Number.isFinite(start?.line)) parts.push(`line ${start.line + 1}`);
  const message = diagnostic.message || diagnostic.text || diagnostic.detail || String(diagnostic);
  if (message) parts.push(message);
  return parts.join(": ");
}

function compileErrorMessage(error, diagnostics = []) {
  const errors = diagnostics.filter((d) => String(d.severity || "").toLowerCase() === "error" || d.severity === 1);
  const diagnosticText = errors.map(formatRumocaDiagnostic).filter(Boolean).slice(0, 3).join(" | ");
  const thrown = error?.message || String(error || "");
  return [diagnosticText, thrown].filter(Boolean).join(" | ") || "Model compile failed.";
}

function syncFlightEditorSource(force = false) {
  const editor = document.querySelector("#modelica-flight-source");
  const entry = selectedFlightModelicaEntry();
  if (!editor || !entry) return;
  const sourceKey = `${entry.label}:${entry.modelName}:${entry.source.length}:${entry.source.includes("enable_safe")}`;
  if (force || state.flightSim.sourceModel !== sourceKey) {
    state.flightSim.source = entry.source;
    state.flightSim.sourceModel = sourceKey;
    setFlightEditorSource(entry.source);
    scheduleModelicaDiagnostics();
  } else if (!state.flightSim.source) {
    state.flightSim.source = editor.value || entry.source;
    setFlightEditorSource(state.flightSim.source);
  }
}

function renderFlightSimControls() {
  const wrap = document.querySelector("#playback-model-wrap");
  const menu = document.querySelector("#playback-model-menu");
  const summary = document.querySelector("#playback-model-summary");
  const panel = document.querySelector("#modelica-flight-panel");
  const replay = document.querySelector("#modelica-replay");
  if (!wrap || !menu || !summary || !panel) return;
  const choices = flightSimChoices();
  const showing = Boolean(state.playbackTrackOverride && choices.length);
  wrap.hidden = !showing;
  if (!showing) {
    stopFlightSim();
    panel.hidden = true;
    return;
  }
  if (!choices.includes(state.flightSim.model)) state.flightSim.model = choices[0];
  const predictMethods = choices.map(predictionMethodForModel).filter(Boolean);
  if (!Array.from(state.selectedMethods).some((method) => predictMethods.includes(method))) {
    const firstMethod = predictionMethodForModel(state.flightSim.model) || predictMethods[0];
    if (firstMethod) state.selectedMethods.add(firstMethod);
  }
  const selectedPredict = predictMethods.filter((method) => state.selectedMethods.has(method));
  const activeFlyModel = (state.flightSim.active || state.flightSim.pending) ? state.flightSim.model : "";
  const signature = [
    choices.join("|"),
    state.flightSim.model,
    activeFlyModel,
    state.flightSim.pending ? "pending" : "idle",
    state.flightSim.editorOpen ? "edit" : "closed",
    selectedPredict.join("|"),
  ].join("#");
  if (menu.dataset.signature !== signature) {
    menu.dataset.signature = signature;
    menu.innerHTML = `
      <div class="model-menu-head" aria-hidden="true">
        <span>Model</span><span>Fly</span><span>Edit</span><span>Predict</span>
      </div>
    `;
    for (const choice of choices) {
      const method = predictionMethodForModel(choice);
      const canPredict = Boolean(method);
      const row = document.createElement("div");
      row.className = "model-menu-row";
      row.classList.toggle("no-predict", !canPredict);
      row.title = canPredict
        ? "Use this model for Predict here overlays."
        : "Prediction is unavailable because this model has no browser-exported rollout stepper yet.";
      const name = document.createElement("span");
      name.textContent = choice;
      const fly = document.createElement("input");
      fly.type = "checkbox";
      fly.dataset.role = "fly";
      fly.dataset.model = choice;
      fly.checked = activeFlyModel === choice;
      fly.disabled = state.flightSim.pending && state.flightSim.model === choice;
      fly.title = "Keyboard flight: W/S throttle, arrows pitch/roll, A/D rudder, Space SAFE, R reset";
      const edit = document.createElement("button");
      edit.type = "button";
      edit.dataset.role = "edit";
      edit.dataset.model = choice;
      edit.textContent = "Edit";
      edit.classList.toggle("active", state.flightSim.editorOpen && state.flightSim.model === choice);
      const predict = document.createElement("input");
      predict.type = "checkbox";
      predict.dataset.role = "predict";
      predict.dataset.model = choice;
      predict.checked = Boolean(method && state.selectedMethods.has(method));
      predict.disabled = !canPredict;
      predict.title = canPredict
        ? "Use this model for Predict here overlays."
        : "Fly/edit only: no browser prediction export is available for this Modelica baseline.";
      row.append(name, fly, edit, predict);
      menu.append(row);
    }
  }
  const predictSuffix = selectedPredict.length > 1 ? ` | ${selectedPredict.length} predict` : "";
  const flySuffix = state.flightSim.pending
    ? " | compiling"
    : state.flightSim.active
      ? " | flying"
      : "";
  summary.textContent = `${state.flightSim.model}${flySuffix}${predictSuffix}`;
  syncFlightEditorSource();
  panel.hidden = !state.flightSim.editorOpen;
  if (state.flightSim.editorOpen) {
    ensureModelicaEditor();
    requestAnimationFrame(() => state.flightSim.editor?.layout());
  }
  if (replay) {
    replay.disabled = !state.flightSim.inputLog.length || state.flightSim.pending;
    replay.classList.toggle("active", state.flightSim.replaying);
  }
}

async function startFlightSim({ keepInputLog = false, preserveControls = false } = {}) {
  const entry = selectedFlightModelicaEntry();
  if (!entry || state.flightSim.pending) return;
  const previous = {
    throttle: state.flightSim.throttle,
    throttleInput: state.flightSim.throttleInput,
    roll: state.flightSim.roll,
    pitch: state.flightSim.pitch,
    yaw: state.flightSim.yaw,
    safeEnabled: state.flightSim.safeEnabled,
  };
  const source = state.flightSim.editorOpen
    ? (getFlightEditorSource() || state.flightSim.source || entry.source)
    : entry.source;
  state.flightSim.pending = true;
  state.flightSim.active = false;
  setFlightStatus(`Compiling ${entry.modelName}...`);
  renderFlightSimControls();
  let diagnostics = [];
  try {
    diagnostics = state.flightSim.editor
      ? await refreshModelicaDiagnostics(source)
      : await modelicaDiagnostics(source);
    const errors = diagnostics.filter((d) => String(d.severity || "").toLowerCase() === "error" || d.severity === 1);
    if (errors.length) {
      setFlightStatus(`Rumoca diagnostics: ${errors.map(formatRumocaDiagnostic).filter(Boolean).slice(0, 2).join(" | ")}`, true);
    }
    const runner = await createModelicaFlightRunner(entry, source);
    state.flightSim.runner = runner;
    state.flightSim.active = true;
    state.flightSim.x = runner.reset();
    state.flightSim.source = source;
    state.flightSim.sourceModel = `${entry.label}:${entry.modelName}:${entry.source.length}:${entry.source.includes("enable_safe")}`;
    state.flightSim.dt = 1 / 240;
    state.flightSim.elapsedS = 0;
    state.flightSim.replaying = false;
    state.flightSim.replayIndex = 0;
    state.flightSim.safeEnabled = false;
    clearFlightTrail(state.playbackScene);
    ensureFlightAircraft(state.playbackScene);
    if (preserveControls) {
      state.flightSim.throttle = previous.throttle;
      state.flightSim.throttleInput = previous.throttleInput;
      state.flightSim.roll = previous.roll;
      state.flightSim.pitch = previous.pitch;
      state.flightSim.yaw = previous.yaw;
      state.flightSim.safeEnabled = previous.safeEnabled;
    }
    if (!keepInputLog) state.flightSim.inputLog = [];
    const safeText = runner.supportsSafeToggle ? "Space toggles SAFE" : "SAFE toggle unavailable";
    setFlightStatus(`${entry.modelName} ready. ${safeText}, W/S throttle, arrows pitch/roll, A/D rudder, R reset.`, !runner.supportsSafeToggle);
  } catch (error) {
    console.error(error);
    state.flightSim.active = false;
    state.flightSim.runner = null;
    state.flightSim.x = null;
    setFlightStatus(compileErrorMessage(error, diagnostics), true);
  } finally {
    state.flightSim.pending = false;
    state.flightSim.step = null;
    state.flightSim.accumulator = 0;
    state.playbackPlaying = false;
    state.playbackLastMs = null;
    renderFlightSimControls();
  }
}

function resetFlightSim() {
  const sim = state.flightSim;
  if (!sim.runner) {
    startFlightSim({ keepInputLog: true });
    return;
  }
  sim.x = sim.runner.reset();
  state.flightSim.accumulator = 0;
  state.flightSim.armed = false;
  state.flightSim.safeEnabled = false;
  state.flightSim.throttle = 0;
  state.flightSim.throttleInput = 0;
  state.flightSim.roll = 0;
  state.flightSim.pitch = 0;
  state.flightSim.yaw = 0;
  state.flightSim.trim.throttle = 0;
  state.flightSim.trim.pitch = 0;
  state.flightSim.trim.roll = 0;
  state.flightSim.trim.yaw = 0;
  state.flightSim.elapsedS = 0;
  state.flightSim.replayIndex = 0;
  state.flightSim.active = true;
  state.playbackPlaying = false;
  state.playbackLastMs = null;
  clearFlightTrail(state.playbackScene);
  setFlightStatus(`${sim.runner.modelName} reset.`);
}

function stopFlightSim() {
  state.flightSim.active = false;
  state.flightSim.x = null;
  state.flightSim.accumulator = 0;
  state.flightSim.replaying = false;
  hideFlightSimVisuals();
}

function updateFlightInputs(deltaS) {
  const sim = state.flightSim;
  if (sim.replaying && sim.inputLog.length) {
    while (sim.replayIndex < sim.inputLog.length - 1 && sim.inputLog[sim.replayIndex + 1].t <= sim.elapsedS) {
      sim.replayIndex += 1;
    }
    const sample = sim.inputLog[sim.replayIndex];
    if (sample) {
      if (sim.replayIndex >= sim.inputLog.length - 1 && sim.elapsedS > sample.t + 0.5) sim.replaying = false;
      return sample.stick.slice();
    }
  }
  const decay = Math.pow(0.85, deltaS / 0.016);
  const held = (key) => sim.keys.has(key);
  sim.roll = held("ArrowLeft") ? -0.6 : held("ArrowRight") ? 0.6 : sim.roll * decay;
  sim.pitch = held("ArrowUp") ? 0.5 : held("ArrowDown") ? -0.5 : sim.pitch * decay;
  sim.yaw = held("a") ? -0.6 : held("d") ? 0.6 : sim.yaw * decay;
  sim.throttleInput = held("w") ? 1 : held("s") ? -1 : sim.throttleInput * decay;
  sim.throttle = clamp(sim.throttle + sim.throttleInput * 0.7 * deltaS, 0, 1);
  const stick = [
    clamp(sim.throttle + sim.trim.throttle, 0, 1),
    clamp(sim.pitch + sim.trim.pitch, -1, 1),
    clamp(sim.roll + sim.trim.roll, -1, 1),
    clamp(sim.yaw + sim.trim.yaw, -1, 1),
  ];
  sim.inputLog.push({ t: sim.elapsedS, stick: stick.slice() });
  return stick;
}

function startReplayInputs() {
  if (!state.flightSim.inputLog.length) return;
  state.flightSim.replaying = true;
  state.flightSim.replayIndex = 0;
  resetFlightSim();
  state.flightSim.replaying = true;
  setFlightStatus(`Replaying ${state.flightSim.inputLog.length} recorded input samples.`);
  renderFlightSimControls();
}

function interpolatePredictionState(predictionFlight, timeS) {
  const times = predictionFlight?.time || [];
  const states = predictionFlight?.state || [];
  if (!times.length || !states.length) return null;
  let hi = times.findIndex((time) => time >= timeS);
  if (hi < 0) hi = times.length - 1;
  const lo = Math.max(0, hi - 1);
  const span = times[hi] - times[lo];
  const w = span > 1e-9 ? clamp((timeS - times[lo]) / span, 0, 1) : 0;
  const a = states[lo];
  const b = states[hi] || a;
  const x = a.map((value, index) => value + (b[index] - value) * w);
  if (a[6] * b[6] + a[7] * b[7] + a[8] * b[8] + a[9] * b[9] < 0) {
    for (let index = 6; index < 10; index += 1) x[index] = a[index] + (-b[index] - a[index]) * w;
  }
  const q = normQuatWxyz([x[6], x[7], x[8], x[9]]);
  x[6] = q[0]; x[7] = q[1]; x[8] = q[2]; x[9] = q[3];
  return x;
}

async function buildModelicaPrediction(method, detail) {
  const model = modelForModelicaPrediction(method);
  const predictionFlight = detail.predictionFlight;
  const dt = predictionFlight?.dtFull || 1 / 240;
  const sticks = predictionFlight?.stick || [];
  const startS = finiteNumber(detail.timeS) ? detail.timeS : 0;
  const x0 = interpolatePredictionState(predictionFlight, startS);
  if (!x0 || !sticks.length) throw new Error("No flight state is available for Modelica prediction.");
  const entry = state.flightSim.catalog.find((item) => item.label === model)
    || buildModelicaFlightCatalog(state.browserModels, state.externalFlightModels).find((item) => item.label === model);
  if (!entry) throw new Error(`${model} is not available for Modelica prediction.`);
  const runner = await createModelicaFlightRunner(entry, entry.source, { initialState: x0 });
  let x = runner.reset();
  const startIndex = clamp(Math.round(startS / dt), 0, sticks.length - 1);
  const stride = Math.max(1, Math.round(0.1 / dt));
  const times = [];
  const points = [];
  const quats = [];
  for (let index = startIndex; index < sticks.length - 1; index += 1) {
    if ((index - startIndex) % stride === 0) {
      times.push(index * dt);
      points.push(nedStateToEnuPosition(x));
      quats.push(nedStateToEnuQuat(x));
    }
    x = runner.step(sticks[index], dt, { safeEnabled: false });
    if (!x.every(Number.isFinite)) break;
  }
  return {
    method,
    color: MODELICA_METHOD_COLORS[method] ?? 0x7dd3fc,
    points,
    times,
    quats,
  };
}

async function addModelicaPredictions(detail) {
  const overlay = detail.overlay;
  const methods = Array.from(state.selectedMethods).filter(isModelicaPredictionMethod);
  if (!overlay?.anchored || !methods.length) return;
  const signature = `${overlay.stamp}#${methods.join("|")}`;
  state.modelicaPredictionSignature = signature;
  setFlightStatus(`Compiling ${methods.map(modelForModelicaPrediction).join(", ")} prediction...`);
  try {
    const predictions = await Promise.all(methods.map((method) => buildModelicaPrediction(method, detail)));
    if (state.modelicaPredictionSignature !== signature || state.explorerOverlay?.stamp !== overlay.stamp) return;
    state.explorerOverlay = {
      ...overlay,
      stamp: `${overlay.stamp}#modelica:${methods.join("|")}`,
      predictions: [
        ...(overlay.predictions || []).filter((prediction) => !methods.includes(prediction.method)),
        ...predictions,
      ],
    };
    setFlightStatus(`${methods.map(modelForModelicaPrediction).join(", ")} prediction ready.`);
    setPlaybackTrack(selectedPlayback());
    renderPlaybackControls(selectedPlayback());
  } catch (error) {
    console.error(error);
    if (state.modelicaPredictionSignature === signature) {
      setFlightStatus(error?.message || "Modelica prediction failed.", true);
    }
  }
}

function quatMulWxyz(a, b) {
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
}

function normQuatWxyz(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

function nedStateToEnuPosition(x) {
  return [x[1], x[0], -x[2]];
}

function nedStateToEnuQuat(x) {
  return normQuatWxyz(quatMulWxyz([0, Math.SQRT1_2, Math.SQRT1_2, 0], [x[6], x[7], x[8], x[9]]));
}

function flightSimPose(x) {
  const origin = state.explorerOverlay?.origin || activeSegment(selectedPlayback())?.position_enu_m?.[0] || [0, 0, 0];
  const absolute = nedStateToEnuPosition(x);
  const posEnu = [absolute[0] - origin[0], absolute[1] - origin[1], absolute[2] - origin[2]];
  const qEnu = nedStateToEnuQuat(x);
  return {
    position: enuToThree(posEnu),
    quaternion: attitudeToThree(qEnu),
  };
}

function ensureFlightAircraft(playback) {
  if (!playback) return null;
  if (!playback.flightAircraft) {
    playback.flightAircraft = makeTransparentAircraftMesh(0x7dd3fc);
    playback.flightAircraft.visible = false;
    playback.scene.add(playback.flightAircraft);
  }
  return playback.flightAircraft;
}

function clearFlightTrail(playback) {
  if (!playback) return;
  playback.flightTrailPoints = [];
  if (playback.flightTrail) {
    playback.scene.remove(playback.flightTrail);
    disposeLine(playback.flightTrail);
    playback.flightTrail = null;
  }
}

function hideFlightSimVisuals(playback = state.playbackScene) {
  if (!playback) return;
  if (playback.flightAircraft) playback.flightAircraft.visible = false;
  clearFlightTrail(playback);
}

function appendFlightTrail(playback, position) {
  if (!playback || !position) return;
  playback.flightTrailPoints.push(position.clone());
  if (playback.flightTrailPoints.length > 3000) playback.flightTrailPoints.shift();
  if (playback.flightTrailPoints.length < 2) return;
  if (!playback.flightTrail) {
    playback.flightTrail = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0x7dd3fc, linewidth: 2, transparent: true, opacity: 0.86 }),
    );
    playback.scene.add(playback.flightTrail);
  } else {
    playback.flightTrail.geometry.dispose();
  }
  playback.flightTrail.geometry = new THREE.BufferGeometry().setFromPoints(playback.flightTrailPoints);
}

function updateFlightSim(playback, deltaS) {
  const sim = state.flightSim;
  if (!sim.active || !sim.x || !sim.runner) return false;
  const aircraft = ensureFlightAircraft(playback);
  if (!aircraft) return false;
  const stick = updateFlightInputs(deltaS);
  sim.accumulator = Math.min(sim.accumulator + deltaS, sim.dt * 30);
  let substeps = 0;
  while (sim.accumulator >= sim.dt && substeps < 30) {
    const next = sim.runner.step(stick, sim.dt, { safeEnabled: sim.safeEnabled });
    if (!next?.every(Number.isFinite)) {
      stopFlightSim();
      setFlightStatus("Model produced a non-finite state; stopped.", true);
      return false;
    }
    sim.x = next;
    sim.accumulator -= sim.dt;
    substeps += 1;
  }
  sim.elapsedS += deltaS;
  const pose = flightSimPose(sim.x);
  aircraft.visible = true;
  aircraft.position.copy(pose.position);
  aircraft.quaternion.copy(pose.quaternion);
  updateAircraftControls(aircraft, [stick[0], stick[2], stick[1], stick[3]], deltaS);
  appendFlightTrail(playback, aircraft.position);
  updateControlHud([stick[0], stick[2], stick[1], stick[3]], sim.safeEnabled ? 1 : 0);
  updateAircraftCamera(playback, aircraft);
  return true;
}

function renderDatasets() {
  const body = document.querySelector("#dataset-body");
  body.innerHTML = "";
  for (const dataset of datasetsForModel()) {
    const tr = document.createElement("tr");
    const values = [
      dataset.title || dataset.id,
      dataset.status || "--",
      dataset.source_type || "--",
      dataset.local_data_dir || dataset.generator || "--",
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value;
      tr.append(cell);
    }
    body.append(tr);
  }
}

function renderManeuver() {
  const maneuver = matchingManeuver();
  const list = document.querySelector("#maneuver-list");
  list.innerHTML = "";
  const rows = maneuver
    ? [
        ["Max |alpha|", `${formatNumber(maneuver.max_abs_alpha_deg)} deg`],
        ["Max |theta|", `${formatNumber(maneuver.max_abs_theta_deg)} deg`],
        ["Speed", `${formatNumber(maneuver.min_speed_mps)}-${formatNumber(maneuver.max_speed_mps)} m/s`],
        ["Vertical", `${formatNumber(maneuver.vertical_extent_m)} m`],
      ]
    : [["Envelope", "--"]];
  for (const [term, detail] of rows) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = term;
    dd.textContent = detail;
    list.append(dt, dd);
  }
}

function enuToThree(position) {
  return new THREE.Vector3(position[0], position[2], -position[1]);
}

function attitudeToThree(quaternionWxyz) {
  const bodyToEnu = new THREE.Quaternion(
    quaternionWxyz[1],
    quaternionWxyz[2],
    quaternionWxyz[3],
    quaternionWxyz[0],
  ).normalize();
  return ENU_TO_THREE_QUAT.clone().multiply(bodyToEnu).multiply(MESH_TO_BODY_FRD_QUAT).normalize();
}

function quaternionToEulerDeg(quaternionWxyz) {
  if (!quaternionWxyz?.length) return [0, 0, 0];
  const q = new THREE.Quaternion(
    quaternionWxyz[1],
    quaternionWxyz[2],
    quaternionWxyz[3],
    quaternionWxyz[0],
  ).normalize();
  const matrix = new THREE.Matrix4().makeRotationFromQuaternion(q).elements;
  const forward = new THREE.Vector3(matrix[0], matrix[1], matrix[2]).normalize();
  const right = new THREE.Vector3(matrix[4], matrix[5], matrix[6]).normalize();
  const pitch = Math.asin(clamp(forward.z, -1, 1));
  const yaw = Math.atan2(forward.y, forward.x);
  const rightLevel = new THREE.Vector3(Math.sin(yaw), -Math.cos(yaw), 0).normalize();
  const downLevel = new THREE.Vector3().crossVectors(forward, rightLevel).normalize();
  const roll = Math.atan2(right.dot(downLevel), right.dot(rightLevel));
  const radToDeg = 180 / Math.PI;
  return [roll * radToDeg, pitch * radToDeg, yaw * radToDeg];
}

function playbackDuration(trackOrSegment) {
  const segment = trackOrSegment?.segments ? activeSegment(trackOrSegment) : trackOrSegment;
  return Math.max(segment?.time_s?.at(-1) || 1, 1);
}

function seekPlayback(timeS) {
  const segment = activeSegment();
  state.playbackTimeS = clamp(timeS, 0, playbackDuration(segment));
  state.playbackLastMs = null;
  updatePlaybackScrub(segment);
}

function makeTaperedControlGeometry(chord, span, thickness, spanAxis = "z") {
  const halfSpan = span / 2;
  const halfThickness = thickness / 2;
  const positions = spanAxis === "z"
    ? [
        0, -halfThickness, -halfSpan, 0, -halfThickness, halfSpan, 0, halfThickness, halfSpan, 0, halfThickness, -halfSpan,
        -chord, 0, -halfSpan, -chord, 0, halfSpan,
      ]
    : [
        0, -halfSpan, -halfThickness, 0, halfSpan, -halfThickness, 0, halfSpan, halfThickness, 0, -halfSpan, halfThickness,
        -chord, -halfSpan, 0, -chord, halfSpan, 0,
      ];
  const indices = [
    0, 1, 2, 0, 2, 3,
    0, 4, 5, 0, 5, 1,
    3, 2, 5, 3, 5, 4,
    0, 3, 4,
    1, 5, 2,
  ];
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function makeWingPanelGeometry(rootChord, tipChord, span, thickness, side) {
  const zRoot = side * 0.18;
  const zTip = side * (0.18 + span);
  const leadingRoot = rootChord / 2;
  const trailingRoot = -rootChord / 2;
  const leadingTip = rootChord / 2 - 0.1;
  const trailingTip = leadingTip - tipChord;
  const yTop = thickness / 2;
  const yBottom = -thickness / 2;
  const positions = [
    leadingRoot, yTop, zRoot, trailingRoot, yTop, zRoot, trailingTip, yTop, zTip, leadingTip, yTop, zTip,
    leadingRoot, yBottom, zRoot, trailingRoot, yBottom, zRoot, trailingTip, yBottom, zTip, leadingTip, yBottom, zTip,
  ];
  const indices = [
    0, 1, 2, 0, 2, 3,
    4, 7, 6, 4, 6, 5,
    0, 4, 5, 0, 5, 1,
    1, 5, 6, 1, 6, 2,
    2, 6, 7, 2, 7, 3,
    3, 7, 4, 3, 4, 0,
  ];
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

// Detailed Sport Cub model: loaded once, cloned per aircraft instance, with
// the procedural mesh as a fallback until (or in case) the asset loads.
const aircraftInstances = [];
let aircraftTemplate = null;
new GLTFLoader().load(
  "./public/assets/airplane.glb",
  (gltf) => {
    const scene = gltf.scene;
    // The asset's Right* meshes contain an unremoved duplicate of the left
    // side's geometry (RightAileron spans X = [-6.27, +6.27]); strip the
    // wrong-side triangles so the duplicate never renders or articulates.
    for (const name of ["RightAileron", "RightFlap", "RightWheel"]) {
      const node = findNamedPart(scene, name);
      let mesh = null;
      if (node) node.traverse((child) => { if (!mesh && child.isMesh) mesh = child; });
      if (!mesh) continue;
      const source = mesh.geometry.index ? mesh.geometry.toNonIndexed() : mesh.geometry;
      const pos = source.getAttribute("position");
      const materialIndexOf = (tri) => {
        for (const group of source.groups) {
          if (tri >= group.start && tri < group.start + group.count) return group.materialIndex || 0;
        }
        return 0;
      };
      const kept = [];
      for (let tri = 0; tri < pos.count; tri += 3) {
        const cx = (pos.getX(tri) + pos.getX(tri + 1) + pos.getX(tri + 2)) / 3;
        if (cx < 0) kept.push(tri);
      }
      const attributes = {};
      for (const [attrName, attr] of Object.entries(source.attributes)) {
        const itemSize = attr.itemSize;
        const out = new Float32Array(kept.length * 3 * itemSize);
        let w = 0;
        for (const tri of kept) {
          for (let v = 0; v < 3; v++) {
            // getComponent handles interleaved and normalized attributes;
            // raw array indexing corrupted the UVs (white tires).
            for (let c = 0; c < itemSize; c++) out[w++] = attr.getComponent(tri + v, c);
          }
        }
        attributes[attrName] = new THREE.BufferAttribute(out, itemSize, attr.normalized);
      }
      const cleaned = new THREE.BufferGeometry();
      for (const [attrName, attr] of Object.entries(attributes)) cleaned.setAttribute(attrName, attr);
      // Multi-material meshes (the wheels: black tire + light hub) address
      // their materials through geometry groups; rebuild them for the kept
      // triangles or everything renders with material 0.
      if (source.groups && source.groups.length) {
        let runStart = 0;
        let runMaterial = kept.length ? materialIndexOf(kept[0]) : 0;
        kept.forEach((tri, idx) => {
          const m = materialIndexOf(tri);
          if (m !== runMaterial) {
            cleaned.addGroup(runStart * 3, (idx - runStart) * 3, runMaterial);
            runStart = idx;
            runMaterial = m;
          }
        });
        if (kept.length) cleaned.addGroup(runStart * 3, (kept.length - runStart) * 3, runMaterial);
      }
      mesh.geometry = cleaned;
    }
    // The asset authors pivots as hinge-location empties that are siblings of
    // the surface meshes; re-parent each mesh under its pivot (preserving
    // world transforms) so rotating the pivot articulates the surface.
    for (const [meshName, pivotName] of [
      ["Elevator", "ElevatorPivot"],
      ["Rudder", "RudderPivot"],
      ["LeftAileron", "LeftAileronPivot"],
      ["RightAileron", "RightAileronPivot"],
      ["LeftFlap", "LeftFlapPivot"],
      ["RightFlap", "RightFlapPivot"],
      ["Prop", "PropPivot"],
      ["LeftWheel", "LeftWheelPivot"],
      ["RightWheel", "RightWheelPivot"],
      ["NoseWheel", "NoseWheelPivot"],
    ]) {
      const mesh = findNamedPart(scene, meshName);
      const pivot = findNamedPart(scene, pivotName);
      if (mesh && pivot) pivot.attach(mesh);
    }
    const box = new THREE.Box3().setFromObject(scene);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    scene.position.sub(center);
    const wrapper = new THREE.Group();
    wrapper.add(scene);
    // glTF assets face +Z with +Y up; the playback body frame is x-forward.
    wrapper.rotation.y = Math.PI / 2;
    // Normalize the span to the procedural model's 2.2 units so the existing
    // AIRCRAFT_MODEL_SCALE still yields the Sport Cub's 0.6 m wingspan.
    wrapper.scale.setScalar(2.2 / Math.max(size.x, size.z, 1e-6));
    aircraftTemplate = wrapper;
    for (const instance of aircraftInstances) {
      if (instance.group.parent) refreshAircraftInstance(instance);
    }
  },
  undefined,
  (error) => console.warn("airplane.glb unavailable, keeping procedural aircraft", error),
);

function findNamedPart(root, name) {
  let found = null;
  root.traverse((node) => {
    if (!found && node.name === name) found = node;
  });
  return found;
}

function refreshAircraftInstance(instance) {
  if (!aircraftTemplate) return;
  instance.model.clear();
  const clone = aircraftTemplate.clone(true);
  instance.model.add(clone);
  if (instance.color != null) {
    clone.traverse((child) => {
      if (!child.isMesh || !child.material) return;
      const material = child.material.clone();
      if (material.color) material.color.set(instance.color);
      material.transparent = true;
      material.opacity = 0.34;
      material.depthWrite = false;
      child.material = material;
      child.renderOrder = 10;
    });
  }
  const parts = {
    glb: true,
    leftAileron: findNamedPart(clone, "LeftAileronPivot"),
    rightAileron: findNamedPart(clone, "RightAileronPivot"),
    elevator: findNamedPart(clone, "ElevatorPivot"),
    rudder: findNamedPart(clone, "RudderPivot"),
    prop: findNamedPart(clone, "PropPivot"),
  };
  // Pivots carry authored base orientations (the hinge alignment); store them
  // so control deflections compose with the base instead of overwriting it.
  for (const part of Object.values(parts)) {
    if (part && part.isObject3D) part.userData.baseQuat = part.quaternion.clone();
  }
  // The authored aileron pivot axes are slightly misaligned with the actual
  // hinge lines on the tapered wing, so large deflections swing the surface
  // out of the wing plane. Derive each aileron's hinge axis from its own
  // geometry: the surface's longest dimension is the hinge line.
  let referenceWorldAxis = null;
  for (const key of ["leftAileron", "rightAileron"]) {
    const pivot = parts[key];
    if (!pivot) continue;
    let mesh = null;
    pivot.traverse((node) => {
      if (!mesh && node.isMesh) mesh = node;
    });
    if (!mesh) continue;
    // The aileron is tapered, so its bounding-box long axis is skewed off
    // the hinge line. The pivot empty sits ON the hinge, so the hinge-edge
    // vertices are those closest to the pivot origin in the cross-hinge
    // plane: bin the vertices spanwise, take each bin's nearest-the-origin
    // vertex, and fit the hinge direction through those edge points.
    mesh.updateMatrix();
    const posAttr = mesh.geometry.getAttribute("position");
    const pts = [];
    for (let i = 0; i < posAttr.count; i++) {
      pts.push(new THREE.Vector3(posAttr.getX(i), posAttr.getY(i), posAttr.getZ(i)).applyMatrix4(mesh.matrix));
    }
    const bb = new THREE.Box3().setFromPoints(pts);
    const ext = bb.getSize(new THREE.Vector3());
    const spanDim = ext.x >= ext.y && ext.x >= ext.z ? "x" : ext.y >= ext.z ? "y" : "z";
    const others = ["x", "y", "z"].filter((d) => d !== spanDim);
    const bins = 8;
    const lo = bb.min[spanDim];
    const step = Math.max(ext[spanDim] / bins, 1e-9);
    const best = new Array(bins).fill(null);
    for (const point of pts) {
      const bin = Math.min(bins - 1, Math.max(0, Math.floor((point[spanDim] - lo) / step)));
      const cross = point[others[0]] ** 2 + point[others[1]] ** 2;
      if (!best[bin] || cross < best[bin].cross) best[bin] = { point, cross };
    }
    const edge = best.filter(Boolean).map((b) => b.point);
    let axis;
    if (edge.length >= 2) {
      axis = edge[edge.length - 1].clone().sub(edge[0]).normalize();
    } else {
      const axisLocal = new THREE.Vector3(spanDim === "x" ? 1 : 0, spanDim === "y" ? 1 : 0, spanDim === "z" ? 1 : 0);
      axis = axisLocal;
    }
    // Sign convention must mirror across the wings: align both hinge axes to
    // the same WORLD spanwise direction (a local-Z test cannot see a mirrored
    // pivot, which made the right aileron deflect backwards).
    const worldQuat = new THREE.Quaternion();
    pivot.getWorldQuaternion(worldQuat);
    const worldAxis = axis.clone().applyQuaternion(worldQuat);
    if (referenceWorldAxis === null) {
      if (axis.dot(SPIN_Z) < 0) {
        axis.negate();
        worldAxis.negate();
      }
      referenceWorldAxis = worldAxis;
    } else if (worldAxis.dot(referenceWorldAxis) < 0) {
      axis.negate();
    }
    pivot.userData.hingeAxis = axis;
  }
  instance.group.userData = parts;
}

const HINGE_X = new THREE.Vector3(1, 0, 0);
const HINGE_Y = new THREE.Vector3(0, 1, 0);
const SPIN_Z = new THREE.Vector3(0, 0, 1);

function setHinge(part, axis, angle) {
  if (!part || !part.userData.baseQuat) return;
  part.quaternion.copy(part.userData.baseQuat).multiply(new THREE.Quaternion().setFromAxisAngle(axis, angle));
}

function makeAircraftMesh(tintColor = null) {
  const group = new THREE.Group();
  const model = new THREE.Group();
  model.scale.setScalar(AIRCRAFT_MODEL_SCALE);
  group.add(model);
  const instance = { group, model, color: tintColor };
  aircraftInstances.push(instance);
  if (aircraftTemplate) {
    refreshAircraftInstance(instance);
    return group;
  }
  const bodyMaterial = new THREE.MeshStandardMaterial({ color: 0x2f5f9f, roughness: 0.46, metalness: 0.08 });
  const wingMaterial = new THREE.MeshStandardMaterial({ color: 0xd9e2ef, roughness: 0.55, metalness: 0.04 });
  const accentMaterial = new THREE.MeshStandardMaterial({ color: 0xd97706, roughness: 0.5, metalness: 0.02 });
  const cockpitMaterial = new THREE.MeshStandardMaterial({ color: 0x8aa5bf, roughness: 0.28, metalness: 0.04, transparent: true, opacity: 0.78 });
  const controlMaterial = new THREE.MeshStandardMaterial({ color: 0xff8a1f, roughness: 0.42, metalness: 0.02, side: THREE.DoubleSide });
  const propMaterial = new THREE.MeshStandardMaterial({ color: 0x2b3442, roughness: 0.35, metalness: 0.08 });
  const propDiskMaterial = new THREE.MeshBasicMaterial({ color: 0x7fb4ff, transparent: true, opacity: 0.16, side: THREE.DoubleSide });

  const fuselage = new THREE.Mesh(new THREE.CylinderGeometry(0.115, 0.115, 1.35, 18), bodyMaterial);
  fuselage.rotation.z = Math.PI / 2;
  model.add(fuselage);
  const cockpit = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.15, 0.15), cockpitMaterial);
  cockpit.position.set(0.28, 0.14, 0);
  model.add(cockpit);
  const nose = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.105, 0.28, 24), wingMaterial);
  nose.rotation.z = Math.PI / 2;
  nose.position.x = 0.82;
  model.add(nose);
  const wingCenter = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.035, 0.42), wingMaterial);
  wingCenter.position.set(0.2, -0.035, 0);
  model.add(wingCenter);
  const leftWing = new THREE.Mesh(makeWingPanelGeometry(0.48, 0.32, 0.78, 0.035, -1), wingMaterial);
  leftWing.position.set(0.2, -0.035, 0);
  model.add(leftWing);
  const rightWing = new THREE.Mesh(makeWingPanelGeometry(0.48, 0.32, 0.78, 0.035, 1), wingMaterial);
  rightWing.position.set(0.2, -0.035, 0);
  model.add(rightWing);

  const leftAileron = new THREE.Group();
  leftAileron.position.set(0.02, -0.05, -0.72);
  const leftAileronPanel = new THREE.Mesh(makeTaperedControlGeometry(0.18, 0.46, 0.032), controlMaterial);
  leftAileron.add(leftAileronPanel);
  model.add(leftAileron);
  const rightAileron = new THREE.Group();
  rightAileron.position.set(0.02, -0.05, 0.72);
  const rightAileronPanel = new THREE.Mesh(makeTaperedControlGeometry(0.18, 0.46, 0.032), controlMaterial);
  rightAileron.add(rightAileronPanel);
  model.add(rightAileron);

  const tail = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.03, 0.96), wingMaterial);
  tail.position.x = -0.64;
  model.add(tail);
  const elevator = new THREE.Group();
  elevator.position.set(-0.78, -0.01, 0);
  const elevatorPanel = new THREE.Mesh(makeTaperedControlGeometry(0.16, 0.96, 0.038), controlMaterial);
  elevator.add(elevatorPanel);
  model.add(elevator);
  const fin = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.52, 0.05), wingMaterial);
  fin.position.set(-0.66, 0.31, 0);
  model.add(fin);
  const rudder = new THREE.Group();
  rudder.position.set(-0.79, 0.31, 0);
  const rudderPanel = new THREE.Mesh(makeTaperedControlGeometry(0.15, 0.52, 0.045, "y"), controlMaterial);
  rudder.add(rudderPanel);
  model.add(rudder);

  const prop = new THREE.Group();
  prop.position.x = 1.03;
  const bladeA = new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.72, 0.045), propMaterial);
  const bladeB = new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.045, 0.72), propMaterial);
  const disk = new THREE.Mesh(new THREE.CircleGeometry(0.42, 48), propDiskMaterial);
  disk.rotation.y = Math.PI / 2;
  prop.add(bladeA, bladeB, disk);
  model.add(prop);
  group.userData = { leftAileron, rightAileron, elevator, rudder, prop, propDisk: disk };
  return group;
}

function makeTransparentAircraftMesh(color) {
  const aircraft = makeAircraftMesh(color);
  if (aircraft.userData.glb) return aircraft;
  aircraft.traverse((child) => {
    if (!child.isMesh || !child.material) return;
    const material = child.material.clone();
    if (material.color) material.color.set(color);
    material.transparent = true;
    material.opacity = child.geometry?.type === "CircleGeometry" ? 0.08 : 0.34;
    material.depthWrite = false;
    material.side = THREE.DoubleSide;
    child.material = material;
    child.renderOrder = 10;
  });
  return aircraft;
}

function updateAircraftControls(aircraft, controls, deltaS) {
  const parts = aircraft.userData || {};
  const thrust = clamp(controls[0] ?? 0.45, 0, 1);
  const aileron = clamp(controls[1] ?? 0, -1, 1);
  const elevator = clamp(controls[2] ?? 0, -1, 1);
  const rudder = clamp(controls[3] ?? 0, -1, 1);
  if (parts.glb) {
    // The GLB pivots map local Z to the spanwise hinge line, local Y to the
    // fin line, and local X to the fuselage axis (verified from the authored
    // pivot orientations in the asset).
    // Deflections exaggerated 2x for visibility in the small viewport.
    setHinge(parts.leftAileron, parts.leftAileron?.userData.hingeAxis || SPIN_Z, -1.2 * aileron);
    setHinge(parts.rightAileron, parts.rightAileron?.userData.hingeAxis || SPIN_Z, 1.2 * aileron);
    setHinge(parts.elevator, SPIN_Z, 1.4 * elevator);
    setHinge(parts.rudder, HINGE_Y, -1.4 * rudder);
    if (parts.prop && parts.prop.userData.baseQuat) {
      parts.prop.userData.spin = (parts.prop.userData.spin || 0) + deltaS * (22 + 90 * thrust);
      setHinge(parts.prop, HINGE_X, parts.prop.userData.spin);
    }
    return;
  }
  if (parts.leftAileron) parts.leftAileron.rotation.z = -0.95 * aileron;
  if (parts.rightAileron) parts.rightAileron.rotation.z = 0.95 * aileron;
  if (parts.elevator) parts.elevator.rotation.z = -1.0 * elevator;
  if (parts.rudder) parts.rudder.rotation.y = 0.95 * rudder;
  if (parts.prop) parts.prop.rotation.x += deltaS * (22 + 90 * thrust);
  if (parts.propDisk) parts.propDisk.material.opacity = 0.08 + 0.22 * thrust;
}

function updateControlHud(controls, mode = null) {
  document.querySelector(".control-hud")?.classList.toggle("flight-active", state.flightSim.active);
  const modeLabel = document.querySelector("#control-mode-value");
  if (modeLabel) {
    if (mode === 1) {
      modeLabel.textContent = "SAFE";
      modeLabel.style.color = "#5c7cfa";
    } else if (mode === 0) {
      modeLabel.textContent = "Manual";
      modeLabel.style.color = "#f08c00";
    } else {
      modeLabel.textContent = "--";
      modeLabel.style.color = "";
    }
  }
  const ids = ["thrust", "aileron", "elevator", "rudder"];
  const values = [
    clamp(controls[0] ?? 0, 0, 1),
    clamp(controls[1] ?? 0, -1, 1),
    clamp(controls[2] ?? 0, -1, 1),
    clamp(controls[3] ?? 0, -1, 1),
  ];
  ids.forEach((id, index) => {
    const fill = document.querySelector(`#control-${id}`);
    const marker = document.querySelector(`#control-${id}-trim`);
    const label = document.querySelector(`#control-${id}-value`);
    if (fill) {
      const value = values[index];
      if (id === "thrust") {
        fill.style.left = "0";
        fill.style.width = `${100 * value}%`;
        fill.classList.remove("negative");
      } else {
        const magnitude = 50 * Math.abs(value);
        fill.style.left = value < 0 ? `${50 - magnitude}%` : "50%";
        fill.style.width = `${magnitude}%`;
        fill.classList.toggle("negative", value < 0);
      }
    }
    if (marker) {
      const trim = state.flightSim.trim;
      const trimValue = id === "thrust" ? trim.throttle : id === "aileron" ? trim.roll : id === "elevator" ? trim.pitch : trim.yaw;
      marker.style.left = id === "thrust" ? `${100 * trimValue}%` : `${50 + 50 * trimValue}%`;
      marker.title = `${id} trim ${trimValue.toFixed(2)}`;
    }
    if (label) label.textContent = values[index].toFixed(2);
  });
}

function disposeMaterial(material) {
  if (Array.isArray(material)) {
    for (const item of material) item.dispose();
  } else if (material) {
    material.dispose();
  }
}

function disposeLine(line) {
  line.geometry.dispose();
  disposeMaterial(line.material);
}

function disposeObject3D(object) {
  object.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      for (const material of materials) {
        if (material.map) material.map.dispose();
        material.dispose();
      }
    }
  });
}

function niceTickSpacing(size) {
  const target = Math.max(size / 5, 1);
  const exponent = Math.floor(Math.log10(target));
  const fraction = target / 10 ** exponent;
  const nice = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
  return nice * 10 ** exponent;
}

function makeAxisLabel(text, color = "#334155") {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 80;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = "600 28px system-ui, -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(1.6, 0.5, 1);
  return sprite;
}

function addAxisLine(group, origin, direction, length, color, label, tickSpacing) {
  const material = new THREE.LineBasicMaterial({ color });
  const end = origin.clone().addScaledVector(direction, length);
  group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([origin, end]), material));

  const tickHalf = Math.max(length * 0.012, 0.08);
  const tickMaterial = new THREE.LineBasicMaterial({ color });
  const tickDirection = Math.abs(direction.y) > 0.5 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
  for (let tick = tickSpacing; tick < length + tickSpacing * 0.25; tick += tickSpacing) {
    const center = origin.clone().addScaledVector(direction, Math.min(tick, length));
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
      center.clone().addScaledVector(tickDirection, -tickHalf),
      center.clone().addScaledVector(tickDirection, tickHalf),
    ]), tickMaterial));
    const tickLabel = makeAxisLabel(`${Math.round(tick)} m`);
    tickLabel.position.copy(center).addScaledVector(tickDirection, tickHalf * 3);
    tickLabel.scale.set(1.1, 0.34, 1);
    group.add(tickLabel);
  }

  const axisLabel = makeAxisLabel(label, `#${color.toString(16).padStart(6, "0")}`);
  axisLabel.position.copy(end).addScaledVector(direction, tickHalf * 5);
  group.add(axisLabel);
}

function makePlaybackAxes(center, size, floorY) {
  const group = new THREE.Group();
  const length = Math.max(niceTickSpacing(size) * 2, Math.min(size * 0.45, 40));
  const tickSpacing = niceTickSpacing(length);
  const origin = new THREE.Vector3(center.x - size * 0.48, floorY + 0.04, center.z + size * 0.48);
  addAxisLine(group, origin, new THREE.Vector3(1, 0, 0), length, 0xdc2626, "East", tickSpacing);
  addAxisLine(group, origin, new THREE.Vector3(0, 0, -1), length, 0x16a34a, "North", tickSpacing);
  addAxisLine(group, origin, new THREE.Vector3(0, 1, 0), Math.max(length * 0.45, tickSpacing), 0x2563eb, "Up", tickSpacing);
  return group;
}

function updatePlaybackCamera(playback) {
  const controls = playback.controls;
  const distance = controls.distance;
  const pitch = controls.pitch;
  const yaw = controls.yaw;
  const offset = new THREE.Vector3(
    distance * Math.cos(pitch) * Math.sin(yaw),
    distance * Math.sin(pitch),
    distance * Math.cos(pitch) * Math.cos(yaw),
  );
  playback.camera.position.copy(controls.target).add(offset);
  playback.camera.lookAt(controls.target);
}

function updateAircraftCamera(playback, aircraft) {
  if (!playback || !aircraft) return;
  if (state.playbackCameraMode === "observer") return;
  if (state.playbackCameraMode === "first") {
    const forward = new THREE.Vector3(1, 0, 0).applyQuaternion(aircraft.quaternion).normalize();
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(aircraft.quaternion).normalize();
    playback.camera.position.copy(aircraft.position).addScaledVector(forward, 0.28).addScaledVector(up, 0.16);
    playback.camera.up.copy(up);
    playback.camera.lookAt(aircraft.position.clone().addScaledVector(forward, 8).addScaledVector(up, 0.08));
    return;
  }
  if (state.playbackCameraMode === "chase") {
    const forward = new THREE.Vector3(1, 0, 0).applyQuaternion(aircraft.quaternion);
    forward.y = 0;
    if (forward.lengthSq() < 1e-8) forward.set(1, 0, 0);
    forward.normalize();
    const target = aircraft.position.clone().addScaledVector(forward, 0.8);
    playback.camera.up.set(0, 1, 0);
    playback.camera.position
      .copy(aircraft.position)
      .addScaledVector(forward, -Math.max(playback.controls.distance * 0.7, 5))
      .add(new THREE.Vector3(0, Math.max(playback.controls.distance * 0.28, 2), 0));
    playback.camera.lookAt(target);
    return;
  }
  playback.camera.up.set(0, 1, 0);
  playback.controls.target.copy(aircraft.position);
  updatePlaybackCamera(playback);
}

function bindOrbitControls(host, playback) {
  const controls = playback.controls;
  host.addEventListener("contextmenu", (event) => event.preventDefault());
  host.addEventListener("pointerdown", (event) => {
    controls.pointer = { x: event.clientX, y: event.clientY, button: event.button, pan: event.shiftKey || event.button === 2 };
    host.setPointerCapture(event.pointerId);
  });
  host.addEventListener("pointermove", (event) => {
    if (!controls.pointer) return;
    const dx = event.clientX - controls.pointer.x;
    const dy = event.clientY - controls.pointer.y;
    controls.pointer.x = event.clientX;
    controls.pointer.y = event.clientY;
    if (controls.pointer.pan) {
      const panScale = controls.distance * 0.0015;
      const right = new THREE.Vector3().subVectors(playback.camera.position, controls.target).cross(playback.camera.up).normalize();
      const up = playback.camera.up.clone().normalize();
      controls.target.addScaledVector(right, dx * panScale).addScaledVector(up, dy * panScale);
    } else {
      controls.yaw += dx * 0.006;
      controls.pitch = clamp(controls.pitch - dy * 0.006, -1.42, 1.42);
    }
    updatePlaybackCamera(playback);
  });
  host.addEventListener("pointerup", () => {
    controls.pointer = null;
  });
  host.addEventListener("wheel", (event) => {
    event.preventDefault();
    controls.distance = clamp(controls.distance * Math.exp(event.deltaY * 0.001), controls.minDistance, controls.maxDistance);
    updatePlaybackCamera(playback);
  }, { passive: false });
}

function ensurePlaybackScene() {
  if (state.playbackScene) return state.playbackScene;
  const host = document.querySelector("#aircraft-playback");
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  } catch (_error) {
    host.textContent = "3D playback requires WebGL. The plots and dataset comparisons are still available.";
    host.classList.add("playback-unavailable");
    return null;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(host.clientWidth || 900, host.clientHeight || 360);
  renderer.setClearColor(0x262b31, 1);
  host.append(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 1000);
  scene.add(new THREE.HemisphereLight(0xdfe6ee, 0x3a4148, 1.0));
  const sun = new THREE.DirectionalLight(0xffffff, 1.4);
  sun.position.set(6, 10, 8);
  scene.add(sun);
  const grid = new THREE.GridHelper(18, 18, 0x59636f, 0x39414b);
  scene.add(grid);
  const aircraft = makeAircraftMesh();
  scene.add(aircraft);

  state.playbackScene = {
    renderer,
    scene,
    camera,
    aircraft,
    flightAircraft: null,
    flightTrail: null,
    flightTrailPoints: [],
    grid,
    axes: null,
    trackLine: null,
    methodLines: [],
    methodAircraft: [],
    track: null,
    methodSignature: "",
    controls: {
      target: new THREE.Vector3(),
      yaw: 0.78,
      pitch: 0.45,
      distance: 12,
      minDistance: 1,
      maxDistance: 200,
      pointer: null,
    },
    lastRenderMs: performance.now(),
  };

  updatePlaybackCamera(state.playbackScene);
  bindOrbitControls(host, state.playbackScene);
  window.addEventListener("resize", () => resizePlayback());
  requestAnimationFrame(tickPlayback);
  return state.playbackScene;
}

function resizePlayback() {
  const playback = state.playbackScene;
  if (!playback) return;
  const host = document.querySelector("#aircraft-playback");
  const width = host.clientWidth || 900;
  const height = host.clientHeight || 360;
  playback.renderer.setSize(width, height);
  playback.camera.aspect = width / Math.max(height, 1);
  playback.camera.updateProjectionMatrix();
}

function setPlaybackTrack(track, force = false) {
  const playback = ensurePlaybackScene();
  if (!playback) {
    renderPlaybackControls(track);
    return;
  }
  const segment = activeSegment(track);
  const segmentName = segment?.name || "segment";
  const methodSignature = Array.from(state.selectedMethods).sort().join("|");
  const overlaySignature = state.explorerOverlay?.stamp || "";
  const trackChanged = playback.track?.id !== track?.id || playback.segmentName !== segmentName;
  if (!force && !trackChanged && playback.methodSignature === methodSignature && playback.overlaySignature === overlaySignature) return;
  playback.track = track;
  playback.segmentName = segmentName;
  playback.methodSignature = methodSignature;
  playback.overlaySignature = overlaySignature;
  if (force || trackChanged) {
    state.playbackTimeS = 0;
    state.playbackLastMs = null;
  }
  if (playback.trackLine) {
    playback.scene.remove(playback.trackLine);
    disposeLine(playback.trackLine);
    playback.trackLine = null;
  }
  for (const line of playback.methodLines) {
    playback.scene.remove(line);
    disposeLine(line);
  }
  playback.methodLines = [];
  for (const overlay of playback.methodAircraft) {
    playback.scene.remove(overlay.mesh);
    disposeObject3D(overlay.mesh);
  }
  playback.methodAircraft = [];
  if (!track || !segment) return;
  const points = segment.position_enu_m.map(enuToThree);
  if (!state.explorerOverlay) {
    // The segmentation-colored overlay replaces the plain blue track line.
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color: 0x1f6feb, linewidth: 2 });
    playback.trackLine = new THREE.Line(geometry, material);
    const segTimes = segment.time_s;
    playback.trackLine.userData.times = {
      t0: segTimes[0],
      dt: (segTimes[segTimes.length - 1] - segTimes[0]) / Math.max(segTimes.length - 1, 1),
      count: points.length,
    };
    playback.scene.add(playback.trackLine);
  }
  if (state.explorerOverlay) {
    // Full-flight track colored by segmentation class, plus the explorer's
    // on-the-fly free-run predictions, published by the flight explorer.
    // The playback track is re-zeroed to its flight start while the overlay
    // is in absolute facility coordinates: shift by the flight origin.
    const overlay = state.explorerOverlay;
    const shift = overlay.origin || [0, 0, 0];
    const shifted = (p) => enuToThree([p[0] - shift[0], p[1] - shift[1], p[2] - shift[2]]);
    const labelColors = [0x8d6e63, 0x26a69a, 0x5c7cfa, 0xf08c00];
    const trackedFlags = overlay.tracked || overlay.labels.map(() => 1);
    let runStart = 0;
    for (let k = 1; k <= overlay.labels.length; k++) {
      const boundary =
        k === overlay.labels.length || overlay.labels[k] !== overlay.labels[runStart] || trackedFlags[k] !== trackedFlags[runStart];
      if (!boundary) continue;
      if (trackedFlags[runStart]) {
        const runPoints = overlay.track.slice(runStart, Math.min(k + 1, overlay.track.length)).map(shifted);
        if (runPoints.length > 1) {
          const runLine = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(runPoints),
            new THREE.LineBasicMaterial({ color: labelColors[overlay.labels[runStart]] ?? 0x999999, linewidth: 2 }),
          );
          runLine.userData.times = { t0: runStart * 0.1, dt: 0.1, count: runPoints.length };
          playback.methodLines.push(runLine);
          playback.scene.add(runLine);
        }
      } else if (k < overlay.track.length) {
        // Mocap dropout: bridge the gap with a dashed gray connector instead
        // of plotting interpolated samples as if they were measurements.
        const gapLine = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([shifted(overlay.track[Math.max(0, runStart - 1)]), shifted(overlay.track[k])]),
          new THREE.LineDashedMaterial({ color: 0x888888, dashSize: 0.25, gapSize: 0.18, transparent: true, opacity: 0.8 }),
        );
        gapLine.userData.times = { t0: Math.max(0, runStart - 1) * 0.1, dt: Math.max((k - runStart + 1) * 0.1, 0.1), count: 2 };
        gapLine.computeLineDistances();
        playback.methodLines.push(gapLine);
        playback.scene.add(gapLine);
      }
      runStart = k;
    }
    for (const prediction of overlay.predictions) {
      const predPoints = prediction.points.map(shifted);
      if (predPoints.length < 2) continue;
      const predLine = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(predPoints),
        new THREE.LineBasicMaterial({ color: prediction.color, linewidth: 2, transparent: true, opacity: 0.85 }),
      );
      if (prediction.times?.length > 1) {
        const predTimes = prediction.times;
        predLine.userData.times = {
          t0: predTimes[0],
          dt: (predTimes[predTimes.length - 1] - predTimes[0]) / (predTimes.length - 1),
          count: predPoints.length,
        };
      }
      playback.methodLines.push(predLine);
      playback.scene.add(predLine);
      if (prediction.times && prediction.quats) {
        // Ghost aircraft flies the on-the-fly free-run prediction.
        const ghost = makeTransparentAircraftMesh(prediction.color);
        playback.methodAircraft.push({
          mesh: ghost,
          segment: {
            time_s: prediction.times,
            position_enu_m: prediction.points.map((p) => [p[0] - shift[0], p[1] - shift[1], p[2] - shift[2]]),
            quaternion_wxyz: prediction.quats,
          },
        });
        playback.scene.add(ghost);
      }
    }
  }
  const methodColors = [0x7c3aed, 0x059669, 0xb45309, 0xbe123c];
  // The full-flight explorer computes free-run predictions on the fly, so the
  // exported benchmark chunk traces would only duplicate them as stray dark
  // lines and ghosts; they remain the overlay for non-explorer playback.
  const traceOverlays = state.explorerOverlay ? [] : selectedTraceSegments();
  traceOverlays.forEach((trace, index) => {
    if (!trace.position_enu_m?.length) return;
    const color = methodColors[index % methodColors.length];
    const tracePoints = trace.position_enu_m.map(enuToThree);
    const traceGeometry = new THREE.BufferGeometry().setFromPoints(tracePoints);
    const traceMaterial = new THREE.LineBasicMaterial({
      color,
      linewidth: 2,
      transparent: true,
      opacity: 0.78,
    });
    const traceLine = new THREE.Line(traceGeometry, traceMaterial);
    if (trace.time_s?.length > 1) {
      const traceTimes = trace.time_s;
      traceLine.userData.times = {
        t0: (trace.flightOffsetS || 0) + traceTimes[0],
        dt: (traceTimes[traceTimes.length - 1] - traceTimes[0]) / (traceTimes.length - 1),
        count: tracePoints.length,
      };
    }
    playback.methodLines.push(traceLine);
    playback.scene.add(traceLine);
    const traceAircraft = makeTransparentAircraftMesh(color);
    playback.methodAircraft.push({ mesh: traceAircraft, segment: trace });
    playback.scene.add(traceAircraft);
  });
  let framePoints = points;
  if (state.explorerOverlay) {
    const overlay = state.explorerOverlay;
    const shift = overlay.origin || [0, 0, 0];
    framePoints = overlay.track.map((p) => enuToThree([p[0] - shift[0], p[1] - shift[1], p[2] - shift[2]]));
  }
  const box = new THREE.Box3().setFromPoints(framePoints);
  const center = box.getCenter(new THREE.Vector3());
  const extents = box.getSize(new THREE.Vector3());
  const size = Math.max(extents.x, extents.y, extents.z, 1);
  if (force || trackChanged) {
    // Only re-home the camera when the displayed flight actually changes;
    // overlay updates (Predict here, method toggles) keep the current view.
    // Fit the whole flight in view (fov 42 deg => distance ~1.3x extent).
    playback.controls.target.copy(center);
    playback.controls.distance = clamp(size * 1.3, 4, 80);
  }
  playback.controls.minDistance = 0.25;
  playback.controls.maxDistance = Math.max(size * 8, 20);
  if (playback.grid) {
    playback.scene.remove(playback.grid);
    playback.grid.geometry.dispose();
    disposeMaterial(playback.grid.material);
  }
  const gridSize = Math.max(10, Math.ceil(size * 1.5));
  playback.grid = new THREE.GridHelper(gridSize, 20, 0x59636f, 0x39414b);
  playback.grid.position.copy(center);
  playback.grid.position.y = Math.min(...points.map((point) => point.y));
  playback.scene.add(playback.grid);
  if (playback.axes) {
    playback.scene.remove(playback.axes);
    disposeObject3D(playback.axes);
  }
  playback.axes = makePlaybackAxes(center, gridSize, playback.grid.position.y);
  playback.scene.add(playback.axes);
  updatePlaybackCamera(playback);
  resizePlayback();
  renderPlaybackControls(track);
}

function sampleTrack(trackOrSegment, elapsedS, options = {}) {
  const track = trackOrSegment?.segments ? activeSegment(trackOrSegment) : trackOrSegment;
  if (!track) return null;
  const times = track.time_s;
  const duration = Math.max(times[times.length - 1] || 1, 1);
  if (!options.loop && (elapsedS < times[0] || elapsedS > duration)) return null;
  const t = options.loop ? ((elapsedS % duration) + duration) % duration : clamp(elapsedS, times[0], duration);
  let index = 0;
  while (index < times.length - 2 && times[index + 1] < t) index += 1;
  const t0 = times[index];
  const t1 = times[index + 1] ?? t0 + 1;
  const ratio = Math.max(0, Math.min(1, (t - t0) / Math.max(t1 - t0, 1e-9)));
  const p0 = enuToThree(track.position_enu_m[index]);
  const p1 = enuToThree(track.position_enu_m[index + 1] || track.position_enu_m[index]);
  const q0 = track.quaternion_wxyz[index];
  const q1 = track.quaternion_wxyz[index + 1] || q0;
  const quat0 = attitudeToThree(q0);
  const quat1 = attitudeToThree(q1);
  const c0 = track.control_meas?.[index] || [0.45, 0, 0, 0];
  const c1 = track.control_meas?.[index + 1] || c0;
  const controls = c0.map((value, controlIndex) => value + (c1[controlIndex] - value) * ratio);
  const nominalDt = (times[times.length - 1] - times[0]) / Math.max(times.length - 1, 1);
  const inGap =
    (track.tracked && (!track.tracked[index] || !track.tracked[index + 1])) ||
    t1 - t0 > 3 * Math.max(nominalDt, 1e-6);
  const mode = track.mode ? track.mode[index] : null;
  return { position: p0.lerp(p1, ratio), quaternion: quat0.slerp(quat1, ratio), controls, tracked: !inGap, mode };
}

function applyTrailWindow(playback, segment) {
  // Qualisys-style trail: every line (measured track, free-run predictions,
  // chunk traces) is windowed to [t - past, t + future] of playback time.
  const tNow = state.playbackTimeS;
  const a = tNow - state.trailPastS;
  const b = tNow + state.trailFutureS;
  const overlay = state.explorerOverlay;
  // With no future span the line must end exactly at the aircraft: the
  // static geometry stops at the last sample behind it and a dynamic head
  // segment bridges to the aircraft's interpolated position every frame.
  const headExact = Boolean(overlay) && state.trailFutureS < 0.1;
  const lines = playback.trackLine ? [playback.trackLine, ...playback.methodLines] : playback.methodLines;
  for (const line of lines) {
    const meta = line.userData?.times;
    if (!meta) continue;
    const lo = clamp(Math.floor((a - meta.t0) / meta.dt), 0, meta.count - 1);
    const hi = headExact
      ? clamp(Math.floor((tNow - meta.t0) / meta.dt), 0, meta.count - 1)
      : clamp(Math.ceil((b - meta.t0) / meta.dt), 0, meta.count - 1);
    if (hi <= lo) {
      line.visible = false;
      continue;
    }
    line.visible = true;
    line.geometry.setDrawRange(lo, hi - lo + 1);
  }
  if (!playback.trailHead) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(6), 3));
    playback.trailHead = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: 0x5c7cfa, linewidth: 2 }));
    playback.scene.add(playback.trailHead);
  }
  const head = playback.trailHead;
  if (headExact && playback.aircraft.visible && overlay.track?.length) {
    const labelColors = [0x8d6e63, 0x26a69a, 0x5c7cfa, 0xf08c00];
    const index = clamp(Math.floor(tNow / 0.1), 0, overlay.track.length - 1);
    const shift = overlay.origin || [0, 0, 0];
    const point = overlay.track[index];
    const v0 = enuToThree([point[0] - shift[0], point[1] - shift[1], point[2] - shift[2]]);
    const positions = head.geometry.attributes.position;
    positions.setXYZ(0, v0.x, v0.y, v0.z);
    positions.setXYZ(1, playback.aircraft.position.x, playback.aircraft.position.y, playback.aircraft.position.z);
    positions.needsUpdate = true;
    head.geometry.computeBoundingSphere();
    head.material.color.setHex(labelColors[overlay.labels?.[index]] ?? 0x5c7cfa);
    head.visible = true;
  } else {
    head.visible = false;
  }
}

function updateTrailHandles(duration) {
  const wrap = document.querySelector(".scrub-wrap");
  if (!wrap || !duration) return;
  const tNow = state.playbackTimeS;
  const left = clamp((tNow - state.trailPastS) / duration, 0, 1) * 100;
  const right = clamp((tNow + state.trailFutureS) / duration, 0, 1) * 100;
  const past = document.querySelector("#trail-past-handle");
  const future = document.querySelector("#trail-future-handle");
  const windowBar = document.querySelector("#trail-window");
  if (past) past.style.left = `calc(${left}% - 5px)`;
  if (future) future.style.left = `calc(${right}% - 5px)`;
  if (windowBar) {
    windowBar.style.left = `${left}%`;
    windowBar.style.width = `${Math.max(right - left, 0)}%`;
  }
}

const SCRUB_LABEL_COLORS = ["#8d6e63", "#26a69a", "#5c7cfa", "#f08c00"];

function updatePlaybackScrub(track) {
  const scrub = document.querySelector("#playback-scrub");
  if (!scrub) return;
  paintScrubSegmentation(scrub, track);
  paintScrubLegend(track);
  updateTrailHandles(playbackDuration(track));
  if (state.playbackScrubbing) return;
  scrub.value = String(clamp(state.playbackTimeS / playbackDuration(track), 0, 1));
}

function paintScrubSegmentation(scrub, track) {
  const signature = track?.name || "";
  if (scrub.dataset.paintedFor === signature) return;
  scrub.dataset.paintedFor = signature;
  if (!track?.labels || !track.time_s?.length) {
    scrub.style.background = "";
    return;
  }
  const total = track.time_s[track.time_s.length - 1] || 1;
  const stops = [];
  let runStart = 0;
  const colorAt = (k) => (track.tracked && !track.tracked[k] ? "#4a5159" : SCRUB_LABEL_COLORS[track.labels[k]] || "#666");
  for (let k = 1; k <= track.labels.length; k += 1) {
    if (k === track.labels.length || colorAt(k) !== colorAt(runStart)) {
      const a = ((track.time_s[runStart] / total) * 100).toFixed(2);
      const b = ((track.time_s[Math.min(k, track.time_s.length - 1)] / total) * 100).toFixed(2);
      stops.push(`${colorAt(runStart)} ${a}% ${b}%`);
      runStart = k;
    }
  }
  scrub.style.background = `linear-gradient(to right, ${stops.join(", ")})`;
}

const SCRUB_LEGEND = [
  ["#8d6e63", "ground"],
  ["#26a69a", "ground effect"],
  ["#5c7cfa", "stabilized"],
  ["#f08c00", "manual"],
  ["#4a5159", "dropout"],
];

function paintScrubLegend(track) {
  const legend = document.querySelector("#scrub-legend");
  if (!legend) return;
  const showing = Boolean(track?.labels && track.time_s?.length);
  legend.hidden = !showing;
  if (!showing || legend.dataset.painted) return;
  legend.dataset.painted = "1";
  legend.innerHTML = SCRUB_LEGEND
    .map(([color, label]) => `<span><i style="background:${color}"></i>${label}</span>`)
    .join("");
}

function renderPlaybackControls(track) {
  const toggle = document.querySelector("#playback-toggle");
  const camera = document.querySelector("#playback-camera");
  const speed = document.querySelector("#playback-speed");
  const segmentSelect = document.querySelector("#playback-segment");
  if (toggle) {
    toggle.innerHTML = state.playbackPlaying ? "&#10074;&#10074;" : "&#9654;";
    toggle.setAttribute("aria-label", state.playbackPlaying ? "Pause" : "Play");
    toggle.title = state.playbackPlaying ? "Pause" : "Play";
  }
  if (camera) camera.value = state.playbackCameraMode;
  if (speed) speed.value = String(state.playbackSpeed);
  if (segmentSelect) {
    // The flight explorer is the single flight selector while it drives the
    // playback; the playback's own picker only appears for datasets without
    // full-flight records (synthetic trials, 4/17 windows).
    const wrapper = segmentSelect.closest("label");
    if (wrapper) wrapper.style.display = state.playbackTrackOverride ? "none" : "";
  }
  const predict = document.querySelector("#playback-predict");
  if (predict) predict.style.display = state.playbackTrackOverride ? "" : "none";
  if (segmentSelect && track && !state.playbackTrackOverride) {
    const segments = track.segments?.length ? track.segments : [track];
    segmentSelect.innerHTML = "";
    segments.forEach((segment, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = segment.name || `flight ${index + 1}`;
      segmentSelect.append(option);
    });
    state.playbackSegmentIndex = clamp(state.playbackSegmentIndex, 0, segments.length - 1);
    segmentSelect.value = String(state.playbackSegmentIndex);
    segmentSelect.disabled = segments.length <= 1;
  }
  updatePlaybackScrub(activeSegment(track));
}

function tickPlayback(nowMs) {
  const playback = state.playbackScene;
  const deltaS = Math.min(((nowMs || performance.now()) - (state.playbackLastMs || nowMs || performance.now())) / 1000, 0.08);
  state.playbackLastMs = nowMs || performance.now();
  if (playback?.track) {
    const segment = activeSegment(playback.track);
    if (state.playbackPlaying && !state.playbackScrubbing) {
      state.playbackTimeS = (state.playbackTimeS + deltaS * state.playbackSpeed) % playbackDuration(segment);
    }
    const flightActive = state.flightSim.active;
    const sample = sampleTrack(segment, state.playbackTimeS, { loop: true });
    if (sample) {
      // During a mocap dropout there is no data for where the aircraft is:
      // hide it and leave it (and therefore the follow camera) at the last
      // measured pose instead of flying interpolated positions.
      playback.aircraft.visible = sample.tracked !== false;
      if (sample.tracked !== false) {
        playback.aircraft.position.copy(sample.position);
        playback.aircraft.quaternion.copy(sample.quaternion);
      }
      updateAircraftControls(playback.aircraft, sample.controls, deltaS);
      if (!flightActive) updateControlHud(sample.controls, sample.mode);
      for (const overlay of playback.methodAircraft) {
        const methodSample = sampleTrack(overlay.segment, state.playbackTimeS - (overlay.segment.flightOffsetS || 0));
        // Ghosts hide during dropouts too: a prediction with no measurement
        // to compare against just looks like a stray aircraft.
        overlay.mesh.visible = Boolean(methodSample) && sample.tracked !== false;
        if (!methodSample) continue;
        overlay.mesh.position.copy(methodSample.position);
        overlay.mesh.quaternion.copy(methodSample.quaternion);
        updateAircraftControls(overlay.mesh, methodSample.controls, deltaS);
      }
      if (!flightActive) {
        // Follow the (frozen-during-dropout) aircraft, not the raw sample:
        // the sample path is interpolation during gaps and the camera would
        // glide along it with no aircraft in view.
        updateAircraftCamera(playback, playback.aircraft);
      }
    }
    if (flightActive) updateFlightSim(playback, deltaS);
    applyTrailWindow(playback, segment);
    updatePlaybackScrub(segment);
    playback.renderer.render(playback.scene, playback.camera);
  }
  requestAnimationFrame(tickPlayback);
}

function renderPlayback() {
  const track = selectedPlayback();
  const status = document.querySelector("#playback-status");
  if (!track) {
    status.textContent = "No trajectory available";
    return;
  }
  const segment = activeSegment(track);
  status.textContent = [track.title, track.source, segment?.name].filter(Boolean).join(" | ");
  setPlaybackTrack(track);
  renderPlaybackControls(track);
}

function linearExtent(values) {
  const finite = values.filter((value) => finiteNumber(value));
  if (!finite.length) return [-1, 1];
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (Math.abs(max - min) < 1e-9) {
    min -= 1;
    max += 1;
  }
  const pad = 0.08 * (max - min);
  return [min - pad, max + pad];
}

function renderMiniSeries(title, series, traces, bands = []) {
  const width = 520;
  const height = 170;
  const margin = { top: 28, right: 18, bottom: 34, left: 58 };
  const values = series.values.concat(...traces.map((trace) => trace.values));
  const xExtent = [series.time[0] || 0, series.time.at(-1) || 1];
  const yExtent = linearExtent(values);
  const xScale = (value) => margin.left + ((value - xExtent[0]) / Math.max(xExtent[1] - xExtent[0], 1e-9)) * (width - margin.left - margin.right);
  const yScale = (value) => height - margin.bottom - ((value - yExtent[0]) / Math.max(yExtent[1] - yExtent[0], 1e-9)) * (height - margin.top - margin.bottom);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const add = (tag, attrs, text) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    svg.append(node);
    return node;
  };
  add("text", { x: margin.left, y: 16, class: "series-title" }, title);
  for (const band of bands) {
    const x0 = xScale(Math.max(band.start, xExtent[0]));
    const x1 = xScale(Math.min(band.stop, xExtent[1]));
    if (x1 <= x0) continue;
    add("rect", {
      x: x0.toFixed(2),
      y: margin.top,
      width: (x1 - x0).toFixed(2),
      height: height - margin.top - margin.bottom,
      fill: band.color,
      opacity: band.opacity ?? 0.85,
    });
  }
  for (let i = 0; i <= 3; i += 1) {
    const fraction = i / 3;
    const value = yExtent[1] - fraction * (yExtent[1] - yExtent[0]);
    const y = margin.top + fraction * (height - margin.top - margin.bottom);
    add("line", { x1: margin.left, y1: y, x2: width - margin.right, y2: y, class: "grid-line" });
    add("text", {
      x: margin.left - 8,
      y: y + 4,
      "text-anchor": "end",
      class: "tick",
    }, formatNumber(value));
  }
  add("line", { x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, class: "axis-line" });
  add("line", { x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, class: "axis-line" });
  const path = (time, data) => time.map((t, index) => `${index ? "L" : "M"}${xScale(t).toFixed(2)},${yScale(data[index]).toFixed(2)}`).join(" ");
  if (series.tracked) {
    // Break the measured line at mocap dropouts instead of plotting the
    // interpolated span as if it were data.
    let chunk = { time: [], values: [] };
    const flushChunk = () => {
      if (chunk.time.length > 1) add("path", { d: path(chunk.time, chunk.values), class: "truth-series" });
      chunk = { time: [], values: [] };
    };
    for (let index = 0; index < series.time.length; index += 1) {
      if (!series.tracked[index]) {
        flushChunk();
        continue;
      }
      chunk.time.push(series.time[index]);
      chunk.values.push(series.values[index]);
    }
    flushChunk();
  } else {
    add("path", { d: path(series.time, series.values), class: "truth-series" });
  }
  for (const trace of traces) {
    const pairs = trace.time
      .map((timeValue, index) => [timeValue, trace.values[index]])
      .filter(([timeValue]) => timeValue >= xExtent[0] && timeValue <= xExtent[1]);
    if (pairs.length > 1) {
      add("path", { d: path(pairs.map((row) => row[0]), pairs.map((row) => row[1])), class: "method-series" });
    }
  }
  add("text", { x: margin.left, y: height - 17, class: "tick" }, formatNumber(xExtent[0]));
  add("text", { x: width - margin.right, y: height - 17, "text-anchor": "end", class: "tick" }, formatNumber(xExtent[1]));
  add("text", { x: (margin.left + width - margin.right) / 2, y: height - 7, "text-anchor": "middle", class: "axis-label" }, "time [s]");
  return svg;
}

function renderTimeseries() {
  const host = document.querySelector("#timeseries-plots");
  const status = document.querySelector("#timeseries-status");
  host.innerHTML = "";
  const track = selectedPlayback();
  const segment = activeSegment(track);
  if (!segment) {
    status.textContent = "No time history data available";
    return;
  }
  const time = segment.time_s;
  const pose = segment.position_enu_m;
  const quat = segment.quaternion_wxyz || [];
  const controls = segment.control_meas || [];
  const eulerDeg = quat.map(quaternionToEulerDeg);
  const traceSegments = selectedTraceSegments();
  const definitions = [
    ["East position [m]", pose.map((row) => row[0]), (trace) => trace.position_enu_m?.map((row) => row[0])],
    ["North position [m]", pose.map((row) => row[1]), (trace) => trace.position_enu_m?.map((row) => row[1])],
    ["Up position [m]", pose.map((row) => row[2]), (trace) => trace.position_enu_m?.map((row) => row[2])],
    ["Roll [deg]", eulerDeg.map((row) => row[0]), (trace) => trace.quaternion_wxyz?.map((row) => quaternionToEulerDeg(row)[0])],
    ["Pitch [deg]", eulerDeg.map((row) => row[1]), (trace) => trace.quaternion_wxyz?.map((row) => quaternionToEulerDeg(row)[1])],
    ["Yaw [deg]", eulerDeg.map((row) => row[2]), (trace) => trace.quaternion_wxyz?.map((row) => quaternionToEulerDeg(row)[2])],
    ["Thrust command [-]", controls.map((row) => row[0] ?? 0), (trace) => trace.control_meas?.map((row) => row[0] ?? 0)],
    ["Aileron command [-]", controls.map((row) => row[1] ?? 0), (trace) => trace.control_meas?.map((row) => row[1] ?? 0)],
    ["Elevator command [-]", controls.map((row) => row[2] ?? 0), (trace) => trace.control_meas?.map((row) => row[2] ?? 0)],
    ["Rudder command [-]", controls.map((row) => row[3] ?? 0), (trace) => trace.control_meas?.map((row) => row[3] ?? 0)],
  ].filter((item) => item[1].length);
  const labelBandColors = ["#8d6e63", "#26a69a", "#5c7cfa", "#f08c00"];
  const tracked = segment.tracked || null;
  const bands = [];
  if (segment.labels) {
    let runStart = 0;
    for (let k = 1; k <= segment.labels.length; k += 1) {
      const runTracked = tracked ? tracked[runStart] : 1;
      const boundary =
        k === segment.labels.length ||
        segment.labels[k] !== segment.labels[runStart] ||
        (tracked ? tracked[k] : 1) !== runTracked;
      if (!boundary) continue;
      bands.push(
        runTracked
          ? { start: time[runStart], stop: time[Math.min(k, time.length - 1)], color: labelBandColors[segment.labels[runStart]] || "#666" }
          : { start: time[runStart], stop: time[Math.min(k, time.length - 1)], color: "#6b7280", opacity: 0.25 },
      );
      runStart = k;
    }
  }
  for (const [title, values, traceAccessor] of definitions) {
    const traces = traceSegments
      .map((trace) => ({ method: trace.method, time: trace.time_s, values: traceAccessor(trace) }))
      .filter((trace) => trace.values?.length === trace.time?.length);
    host.append(renderMiniSeries(title, { time, values, tracked }, traces, bands));
  }
  const selected = state.selectedMethods.size ? `${Array.from(state.selectedMethods)[0]} selected` : "select one method to overlay its exported model trajectory";
  const available = traceSegments.length ? "model trajectories shown" : "no exported model trajectories for this dataset yet";
  status.textContent = `${segment.name || "flight"} | ${selected} | ${available}`;
}

function render() {
  setDefaultScenario();
  renderPlaybackTabs();
  renderModelTabs();
  renderScenarioSelect();
  const rows = selectedRows();
  renderSummary(rows);
  renderTradeoff(rows);
  renderLeaderboard(rows);
  renderDatasets();
  renderManeuver();
  renderPlayback();
  renderFlightSimControls();
  renderTimeseries();
}

async function init() {
  state.manifest = await loadJson("manifest.json");
  state.rows = await loadJson("method_results.json");
  state.maneuvers = await loadJson("maneuver_summary.json");
  state.playback = await loadJson("playback.json");
  state.methodTraces = await loadJson("method_traces.json");
  state.externalFlightModels = await loadExternalFlightModelEntries();
  if (!state.manifest.model_families.includes(state.modelFamily)) {
    state.modelFamily = state.manifest.model_families[0] || "aircraft6dof";
  }
  setDefaultScenario();
  bindControls();
  renderMeta();
  render();
  // The explorer module may have announced its flight tracks and overlay
  // before our listeners existed (module load race); ask it to re-publish.
  window.dispatchEvent(new CustomEvent("playback-ready"));
}

init().catch((error) => {
  console.error(error);
  document.querySelector("#run-meta").textContent = error.message;
});
