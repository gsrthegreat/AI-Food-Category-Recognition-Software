// node_backend/index.js
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const UPLOAD_DIR = '/tmp/uploads';
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR);

const upload = multer({ dest: UPLOAD_DIR });
const app = express();

app.use(cors());

const MODEL_SERVICE = process.env.MODEL_SERVICE_URL || 'http://python_model:5000/predict';

console.log('='.repeat(50));
console.log('NODE BACKEND STARTING');
console.log('='.repeat(50));
console.log('Upload directory:', UPLOAD_DIR);
console.log('Model service URL:', MODEL_SERVICE);
console.log('='.repeat(50));

app.get('/', (req, res) => {
  res.json({
    message: 'Node backend running',
    modelService: MODEL_SERVICE,
    uploadDir: UPLOAD_DIR
  });
});

app.post('/predict', upload.single('image'), async (req, res) => {
  console.log('\n--- New prediction request ---');
  console.log('File received:', req.file ? req.file.originalname : 'none');
  
  if (!req.file) {
    console.log('ERROR: No file uploaded');
    return res.status(400).json({ error: 'no file uploaded' });
  }

  console.log('File details:', {
    originalname: req.file.originalname,
    mimetype: req.file.mimetype,
    size: req.file.size,
    path: req.file.path
  });

  try {
    const form = new FormData();
    form.append('image', fs.createReadStream(req.file.path), req.file.originalname);

    console.log('Forwarding to:', MODEL_SERVICE);
    console.log('Form headers:', form.getHeaders());

    const r = await axios.post(MODEL_SERVICE, form, {
      headers: {
        ...form.getHeaders()
      },
      maxBodyLength: 50 * 1024 * 1024,
      timeout: 30000 // 30 second timeout
    });

    console.log('Success! Response:', r.data);

    // cleanup
    fs.unlinkSync(req.file.path);
    console.log('Temp file cleaned up');
    console.log('--- Request complete ---\n');

    return res.json(r.data);
    
  } catch (err) {
    console.error('\n!!! ERROR forwarding to model !!!');
    console.error('Error type:', err.constructor.name);
    console.error('Error message:', err.message);
    
    if (err.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      console.error('Response status:', err.response.status);
      console.error('Response data:', err.response.data);
      console.error('Response headers:', err.response.headers);
    } else if (err.request) {
      // The request was made but no response was received
      console.error('No response received from model service');
      console.error('Request details:', err.request);
    } else {
      // Something happened in setting up the request that triggered an Error
      console.error('Error setting up request:', err.message);
    }
    
    if (err.stack) {
      console.error('Stack trace:', err.stack);
    }
    console.error('!!!\n');
    
    // cleanup
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
      console.log('Temp file cleaned up after error');
    }
    
    const errorResponse = {
      error: err.message || 'server error',
      details: err.response?.data || null
    };
    
    return res.status(err.response?.status || 500).json(errorResponse);
  }
});

app.listen(3000, () => {
  console.log('\n✓ Node backend listening on port 3000');
  console.log('✓ Ready to forward requests to', MODEL_SERVICE);
  console.log('\n');
});