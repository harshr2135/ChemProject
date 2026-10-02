import { useEffect, useRef, useState } from "react";

function inkFor(hex) {
  const red = parseInt(hex.slice(1, 3), 16);
  const green = parseInt(hex.slice(3, 5), 16);
  const blue = parseInt(hex.slice(5, 7), 16);
  const luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue;
  return luminance > 170 ? "#1c1c1c" : "#ffffff";
}

function FlaskIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M9 3h6M10 3v5.2L5.4 18.2A2 2 0 0 0 7.2 21h9.6a2 2 0 0 0 1.8-2.8L14 8.2V3"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M8.2 14h7.6" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

function CameraIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M4 8.5h3l1.4-2h7.2L17 8.5h3v9.2a1.3 1.3 0 0 1-1.3 1.3H5.3A1.3 1.3 0 0 1 4 17.7V8.5Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinejoin="round"
      />
      <circle cx="12" cy="13" r="2.6" fill="none" stroke="currentColor" strokeWidth="1.7" />
    </svg>
  );
}

function DatabaseIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <ellipse cx="12" cy="6" rx="7" ry="3" fill="none" stroke="currentColor" strokeWidth="1.7" />
      <path d="M5 6v6c0 1.7 3.1 3 7 3s7-1.3 7-3V6" fill="none" stroke="currentColor" strokeWidth="1.7" />
      <path d="M5 12v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6" fill="none" stroke="currentColor" strokeWidth="1.7" />
    </svg>
  );
}

async function readError(response) {
  try {
    const body = await response.json();
    return body.detail || "Request failed.";
  } catch {
    return "Request failed.";
  }
}

// Same centre crop as analyze_beaker_colour in lab_store.py.
const SAMPLE_X0 = 0.3;
const SAMPLE_X1 = 0.7;
const SAMPLE_Y0 = 0.32;
const SAMPLE_Y1 = 0.78;

export default function App() {
  const videoRef = useRef(null);
  const frameRef = useRef(null);
  const streamRef = useRef(null);
  const photoRef = useRef(null);
  const [config, setConfig] = useState(null);
  const [concentration, setConcentration] = useState("1.50");
  const [prediction, setPrediction] = useState(null);
  const [analysed, setAnalysed] = useState(null);
  const [records, setRecords] = useState([]);
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [savedId, setSavedId] = useState(null);
  const [guide, setGuide] = useState(null);

  async function loadRecords() {
    const response = await fetch("/api/analyses");
    if (!response.ok) return;
    const body = await response.json();
    setRecords(body.records || []);
  }

  useEffect(() => {
    fetch("/api/config")
      .then((response) => response.json())
      .then((body) => {
        setConfig(body);
        setConcentration(Number(body.default_concentration).toFixed(2));
      })
      .catch(() => setError("The prediction API is not running."));
    loadRecords().catch(() => {});
  }, []);

  useEffect(() => {
    const handle = setTimeout(async () => {
      const conc = Number(concentration);
      if (!Number.isFinite(conc)) return;
      try {
        const response = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ concentration: conc }),
        });
        if (!response.ok) {
          setError(await readError(response));
          return;
        }
        setError("");
        setPrediction(await response.json());
        setSavedId(null);
      } catch {
        setError("Could not reach the prediction API.");
      }
    }, 200);
    return () => clearTimeout(handle);
  }, [concentration]);

  async function startCamera() {
    setCameraError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraReady(true);
    } catch {
      setCameraReady(false);
      setCameraError("Allow the camera, then point it at the beaker.");
    }
  }

  useEffect(() => {
    startCamera();
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    const frame = frameRef.current;
    if (!video || !frame) return undefined;

    function placeGuide() {
      const width = frame.clientWidth;
      const height = frame.clientHeight;
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;
      if (!width || !height || !videoWidth || !videoHeight) {
        setGuide(null);
        return;
      }
      const scale = Math.max(width / videoWidth, height / videoHeight);
      const displayedWidth = videoWidth * scale;
      const displayedHeight = videoHeight * scale;
      const offsetX = (width - displayedWidth) / 2;
      const offsetY = (height - displayedHeight) / 2;
      setGuide({
        boxLeft: offsetX + displayedWidth * SAMPLE_X0,
        boxTop: offsetY + displayedHeight * SAMPLE_Y0,
        boxWidth: displayedWidth * (SAMPLE_X1 - SAMPLE_X0),
        boxHeight: displayedHeight * (SAMPLE_Y1 - SAMPLE_Y0),
        cx: offsetX + displayedWidth * 0.5,
        cy: offsetY + displayedHeight * 0.5,
      });
    }

    placeGuide();
    video.addEventListener("loadedmetadata", placeGuide);
    const observer = new ResizeObserver(placeGuide);
    observer.observe(frame);
    return () => {
      video.removeEventListener("loadedmetadata", placeGuide);
      observer.disconnect();
    };
  }, [cameraReady]);

  async function capture() {
    const video = videoRef.current;
    if (!video || !video.videoWidth) {
      setError("The camera is not ready yet.");
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
    photoRef.current = blob;
    setBusy(true);
    setError("");
    setStatus("");
    try {
      const body = new FormData();
      body.append("image", blob, "capture.jpg");
      const response = await fetch("/api/analyze", { method: "POST", body });
      if (!response.ok) {
        setError(await readError(response));
        return;
      }
      setAnalysed(await response.json());
      setSavedId(null);
    } finally {
      setBusy(false);
    }
  }

  async function store() {
    if (!photoRef.current || !prediction) return;
    setBusy(true);
    setError("");
    try {
      const body = new FormData();
      body.append("concentration", concentration);
      body.append("image", photoRef.current, "capture.jpg");
      const response = await fetch("/api/analyses", { method: "POST", body });
      if (!response.ok) {
        setError(await readError(response));
        return;
      }
      const saved = await response.json();
      setSavedId(saved.id);
      setStatus(`Stored analysis #${saved.id}.`);
      await loadRecords();
    } finally {
      setBusy(false);
    }
  }

  const predictedStyle = prediction
    ? { background: prediction.predicted_hex, color: inkFor(prediction.predicted_hex) }
    : undefined;
  const analysedStyle = analysed
    ? { background: analysed.analyzed_hex, color: inkFor(analysed.analyzed_hex) }
    : undefined;

  return (
    <main className="page">
      <section className="card">
        {config?.schema_error && (
          <div className="banner">
            <p>{config.schema_error}</p>
            {config.sql_editor_url && (
              <a href={config.sql_editor_url} target="_blank" rel="noreferrer">
                Open the SQL Editor
              </a>
            )}
          </div>
        )}

        <div className="inputs">
          <label className="field">
            <span>Enter Concentration</span>
            <div className="control">
              <FlaskIcon />
              <input
                inputMode="decimal"
                value={concentration}
                placeholder="Enter concentration (e.g., 1.5)"
                onChange={(event) => setConcentration(event.target.value)}
              />
            </div>
          </label>
        </div>
        <p className="hint">Concentration in the training sheet runs from 0.2 to 10.</p>

        <div className="camera">
          <div className="live">
            <span className="dot" />
            LIVE CAMERA RECORDING
          </div>
          <div className="camera-frame" ref={frameRef}>
            <video ref={videoRef} autoPlay playsInline muted />
            <div className="aim" aria-hidden="true">
              <div
                className="aim-box"
                style={
                  guide
                    ? {
                        left: guide.boxLeft,
                        top: guide.boxTop,
                        width: guide.boxWidth,
                        height: guide.boxHeight,
                      }
                    : undefined
                }
              />
              <div
                className="aim-cross"
                style={guide ? { left: guide.cx, top: guide.cy } : undefined}
              >
                <svg viewBox="0 0 72 72">
                  <circle cx="36" cy="36" r="16" />
                  <path d="M36 6v12M36 54v12M6 36h12M54 36h12" />
                  <circle className="aim-dot" cx="36" cy="36" r="2.2" />
                </svg>
              </div>
              <span
                className="aim-label"
                style={guide ? { left: guide.cx, top: guide.cy } : undefined}
              >
                Place beaker here
              </span>
            </div>
          </div>
          {!cameraReady && (
            <button className="enable" type="button" onClick={startCamera}>
              Enable camera
            </button>
          )}
          <button className="capture" type="button" onClick={capture} disabled={busy || !cameraReady}>
            <CameraIcon />
            Capture and analyse colour
          </button>
        </div>
        <p className="hint camera-hint">
          Line the coloured solution up with the centre mark. Colour is read from inside the box.
        </p>
        {cameraError && <p className="hint">{cameraError}</p>}
        {analysed?.low_colour_signal && (
          <p className="warn">The photo looks mostly blank. Centre the coloured solution and take another photo.</p>
        )}

        <div className="swatches">
          <article className="swatch" style={predictedStyle}>
            <h2>Predicted<br />Colour</h2>
            <p>{prediction ? prediction.predicted_hex : "—"}</p>
            {prediction && <small>Abs {prediction.predicted_absorbance.toFixed(4)}</small>}
          </article>
          <article className={`swatch ${analysed ? "" : "empty"}`} style={analysedStyle}>
            <h2>Analyzed<br />Colour</h2>
            <p>{analysed ? analysed.analyzed_hex : "Take a photo"}</p>
          </article>
        </div>

        {prediction && (
          <p className="hint algorithms">
            Absorbance = {prediction.abs_algorithm}, colour = {prediction.color_algorithm}.
            {analysed && prediction.predicted_rgb && analysed.analyzed_rgb
              ? ` Camera colour is ${Math.round(
                  Math.hypot(
                    prediction.predicted_rgb[0] - analysed.analyzed_rgb[0],
                    prediction.predicted_rgb[1] - analysed.analyzed_rgb[1],
                    prediction.predicted_rgb[2] - analysed.analyzed_rgb[2],
                  ),
                )} RGB units from the prediction.`
              : ""}
          </p>
        )}

        {error && <p className="warn">{error}</p>}
        {(status || savedId) && <p className="ok">{status}</p>}

        <div className="actions">
          <button className="store" type="button" disabled={!analysed || !prediction || busy || savedId} onClick={store}>
            <DatabaseIcon />
            Store Analysis to Database
          </button>
        </div>
      </section>

      <section className="history">
        <h3>Saved analyses</h3>
        {records.length === 0 ? (
          <p className="hint">Nothing stored yet. Capture a photo, check the two colours, then store the analysis.</p>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Saved</th>
                  <th>Conc</th>
                  <th>Abs</th>
                  <th>Predicted</th>
                  <th>Analyzed</th>
                </tr>
              </thead>
              <tbody>
                {records.map((row) => (
                  <tr key={row.id}>
                    <td>{row.id}</td>
                    <td>{row.created_at}</td>
                    <td>{row.concentration}</td>
                    <td>{Number(row.predicted_absorbance).toFixed(4)}</td>
                    <td>
                      <span className="chip" style={{ background: row.predicted_hex }} />
                      {row.predicted_hex}
                    </td>
                    <td>
                      <span className="chip" style={{ background: row.analyzed_hex }} />
                      {row.analyzed_hex}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </main>
  );
}
