import { useState, useRef } from 'react';
import './App.css';

/* ─── SVG Icons ─── */
const IconCamera = () => (
  <svg viewBox="0 0 24 24">
    <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"/>
    <circle cx="12" cy="13" r="4"/>
  </svg>
);

const IconShield = () => (
  <svg viewBox="0 0 24 24">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
  </svg>
);

const IconUpload = () => (
  <svg viewBox="0 0 24 24">
    <polyline points="16 16 12 12 8 16"/>
    <line x1="12" y1="12" x2="12" y2="21"/>
    <path d="M20.39 18.39A5 5 0 0018 9h-1.26A8 8 0 103 16.3"/>
  </svg>
);

const IconFilm = () => (
  <svg viewBox="0 0 24 24">
    <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"/>
    <line x1="7" y1="2" x2="7" y2="22"/>
    <line x1="17" y1="2" x2="17" y2="22"/>
    <line x1="2" y1="12" x2="22" y2="12"/>
    <line x1="2" y1="7" x2="7" y2="7"/>
    <line x1="2" y1="17" x2="7" y2="17"/>
    <line x1="17" y1="17" x2="22" y2="17"/>
    <line x1="17" y1="7" x2="22" y2="7"/>
  </svg>
);

const IconDownload = () => (
  <svg viewBox="0 0 24 24">
    <polyline points="8 17 12 21 16 17"/>
    <line x1="12" y1="21" x2="12" y2="3"/>
    <path d="M20 21H4"/>
  </svg>
);

const IconArrow = () => (
  <svg viewBox="0 0 24 24">
    <line x1="5" y1="12" x2="19" y2="12"/>
    <polyline points="12 5 19 12 12 19"/>
  </svg>
);

const IconX = () => (
  <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" fill="none" strokeWidth="2">
    <line x1="18" y1="6" x2="6" y2="18"/>
    <line x1="6" y1="6" x2="18" y2="18"/>
  </svg>
);

/* ─── Upload Zone Component ─── */
function UploadZone({ file, onFile, onRemove }) {
  const [dragging, setDragging] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped && dropped.type === 'video/mp4') onFile(dropped);
  };

  return (
    <>
      <div
        className={`upload-zone ${dragging ? 'dragging' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept="video/mp4"
          onChange={(e) => e.target.files[0] && onFile(e.target.files[0])}
        />
        <div className="upload-icon">
          <IconUpload />
        </div>
        <div className="upload-text">
          <strong>Drop .mp4 here or click to browse</strong>
          Supported: MP4 • Max 500MB
        </div>
      </div>

      {file && (
        <div className="file-selected">
          <div className="file-selected-icon"><IconFilm /></div>
          <span className="file-selected-name">{file.name}</span>
          <button className="file-selected-remove" onClick={onRemove}>
            <IconX />
          </button>
        </div>
      )}
    </>
  );
}



/* ─── Clone Card ─── */
function CloneCard() {
  const [file, setFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [downloadReady, setDownloadReady] = useState(false);
  const pollRef = useRef(null);

  const handleProcess = async () => {
    if (!file) return;
    setProcessing(true);
    setDownloadReady(false);
    setJobId(null);

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://localhost:5000/generate", {
      method: "POST",
      body: formData,
    });
    const { job_id } = await res.json();
    setJobId(job_id);

    // Poll until done
    pollRef.current = setInterval(async () => {
      const r = await fetch(`http://localhost:5000/status/${job_id}`);
      const data = await r.json();
      
   
      
      
      if (data.status === "done") {
        clearInterval(pollRef.current);
        setDownloadReady(true);
        setProcessing(false);
      } else if (data.status === "error") {
        clearInterval(pollRef.current);
        setProcessing(false);
        alert("Pipeline error: " + data.error);
      }
    }, 3000);
  };

  const handleRemove = () => {
    setFile(null);
    setDownloadReady(false);
    setJobId(null);
    clearInterval(pollRef.current);
  };

  return (
  <div className="card">
    <div className="card-corner-br" />
    <div className="card-number">01</div>

    <div className="card-icon"><IconCamera /></div>
    <div className="card-label">Module Alpha</div>
    <h2 className="card-title">CREATE YOUR CLONE</h2>
    <p className="card-desc">
      Upload a video of yourself and our neural engine synthesizes a digital replica.
      A perfect deepfake — rendered in seconds, indistinguishable from reality.
    </p>

    <UploadZone file={file} onFile={setFile} onRemove={handleRemove} />

    <button className="btn" onClick={handleProcess} disabled={!file || processing}>
      {processing ? <>Synthesizing...</> : <><IconArrow /> Generate Clone</>}
    </button>

    {processing && (
      <div className="processing">
        <div className="processing-bar">
          <div className="processing-bar-fill" style={{ width: '100%' }} />
        </div>
        <div className="processing-text">NEURAL NET PROCESSING...</div>
      </div>
    )}

    {downloadReady && !processing && (
      <div className="result-clone">
        <div className="result-clone-label">Clone Generated Successfully</div>

        <a
          className="btn-download"
          href={`http://localhost:5000/download/${jobId}`}
          download
        >
          <IconDownload /> Download Clone Now!
        </a>

      </div>
    )}
  </div>
);
}







/* ─── Detect Card ─── */
function DetectCard() {
  const [file, setFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [verdict, setVerdict] = useState(null);
  const [distance, setDistance] = useState(null);
  const [framesInfo, setFramesInfo] = useState(null);

  const handleAnalyze = async () => {
    if (!file) return;
    setProcessing(true);
    setVerdict(null);
    setDistance(null);
    setFramesInfo(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:5000/final_discriminate", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (data.status === "done") {
        setFramesInfo(data.features);       // features object store karo
        setVerdict(data.verdict);           // model ka actual verdict
        setDistance(data.distance);
        
      } else {
        alert("Error: " + data.error);
      }
    } catch (err) {
      alert("Server se connect nahi ho pa raha: " + err.message);
    } finally {
      setProcessing(false);
    }
  };

  const handleRemove = () => {
    setFile(null);
    setVerdict(null);
    setDistance(null);
    setFramesInfo(null);
  };

  //const confidence = verdict
  //  ? (78 + Math.floor(Math.random() * 19)).toString()
  //  : "";

  return (
    <div className="card">
      <div className="card-corner-br" />
      <div className="card-number">02</div>

      <div className="card-icon"><IconShield /></div>
      <div className="card-label">Module Beta</div>
      <h2 className="card-title">DETECT FAKE / REAL</h2>
      <p className="card-desc">
        Feed any video into our forensic AI — it dissects pixel patterns, compression
        artifacts, and temporal inconsistencies to expose the truth hiding in every frame.
      </p>

      <UploadZone file={file} onFile={setFile} onRemove={handleRemove} />

      <button
        className="btn"
        onClick={handleAnalyze}
        disabled={!file || processing}
      >
        {processing ? <>Scanning...</> : <><IconArrow /> Analyze Video</>}
      </button>

      {processing && (
        <div className="processing">
          <div className="processing-bar">
            <div className="processing-bar-fill" style={{ width: "100%" }} />
          </div>
          <div className="processing-text">FORENSIC ANALYSIS RUNNING...</div>
        </div>
      )}

      {verdict && !processing && (
        <div className={`result-verdict ${verdict}`}>
          <div className="verdict-label">Forensic Verdict</div>
          <div className={`verdict-result ${verdict}`}>
            {verdict.toUpperCase()}
          </div>
          {framesInfo && (
            <div className="verdict-confidence">
              Sharpness STD: {framesInfo.sharpness_std?.toFixed(3)} &nbsp;|&nbsp;
              Confidence: {distance} 
            </div>
          )}
        </div>
      )}
    </div>
  );
}


/* ─── App ─── */
export default function App() {
  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="logo">
          <div className="logo-icon" />
          <div className="logo-text">DEEP<span>FAKE</span></div>
        </div>
        <div className="header-badge">SYSTEM ONLINE</div>
      </header>

      {/* Hero */}
      <section className="hero">
        <div className="hero-eyebrow">AI-Powered Video Intelligence</div>
        <h1 className="hero-title">
          DEEPFAKE
          <span className="accent">DETECTION</span>
        </h1>
        <p className="hero-tagline">
          <em>See Through the Lies.</em> Trust Nothing. Verify Everything.
        </p>
        <div className="hero-divider" />
      </section>

      {/* Cards */}
      <main className="cards-section">
        <CloneCard />
        <DetectCard />
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-copy">© 2025 DEEPFAKE DETECTION — ALL RIGHTS RESERVED</div>
        <div className="footer-links">
          <a href="#">Privacy</a>
          <a href="#">Docs</a>
          <a href="#">API</a>
          <a href="#">Contact</a>
        </div>
      </footer>
    </div>
  );
}
