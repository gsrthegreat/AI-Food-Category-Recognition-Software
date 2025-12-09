// import React from 'react'
// import { createRoot } from 'react-dom/client'
// import App from "App"
// import './style.css' // optional

// createRoot(document.getElementById('root')).render(<App />)

import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./style.css"; // make sure this exists

const container = document.getElementById("root");
const root = createRoot(container);
root.render(<App />);
