import React, { useState } from 'react';

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const onFileChange = (e) => {
    const f = e.target.files[0];
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const upload = async () => {
    if (!file) return;
    setLoading(true);
    const fd = new FormData();
    fd.append('image', file);

    try {
      // call Node backend
      const res = await fetch('http://localhost:3000/predict', {
        method: 'POST',
        body: fd
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setResult({ error: err.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{padding:20, fontFamily:'sans-serif', maxWidth:800, margin:'0 auto'}}>
      <h1>Food Recognition</h1>

      <input type="file" accept="image/*" onChange={onFileChange} />
      {preview && <div style={{marginTop:10}}><img src={preview} alt="preview" style={{maxWidth:'100%', maxHeight:320}}/></div>}

      <div style={{marginTop:10}}>
        <button onClick={upload} disabled={loading}>{loading ? 'Predicting...' : 'Predict'}</button>
      </div>

      {result && (
        <div style={{marginTop:12, padding:10, border:'1px solid #ddd', borderRadius:6}}>
          {result.error ? (
            <div style={{color:'red'}}>Error: {result.error}</div>
          ) : (
            <>
              <div><b>Label:</b> {result.label}</div>
              {result.confidence !== null && result.confidence !== undefined && <div><b>Confidence:</b> {(result.confidence*100).toFixed(2)}%</div>}
            </>
          )}
        </div>
      )}
    </div>
  );
}
