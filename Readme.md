# 🍽️ AI Food Category Recognition

A full-stack, production-ready web application that identifies food items from images using a custom machine-learning model.

🔥 Built with **Node.js**, **Python (SIFT + Bag-of-Visual-Words)**, **Docker**, and a modern **Vite + React** frontend.

---

## 🚀 Features

* 🧠 **AI-powered food classification**
* ⚡ Fast **REST API** (Node.js backend)
* 🐍 Python microservice for ML inference
* 📦 Fully **Dockerized** (frontend + backend + model)
* 🌐 Clean and simple UI for uploading images
* 📊 Shows prediction + confidence score

---

## 🏗️ Tech Stack

**Frontend:** React + Vite
**Backend:** Node.js + Express
**AI Model:** Python, OpenCV, SIFT, Scikit-Learn
**Infrastructure:** Docker + Docker Compose
**Storage:** Local model files (mounted volumes)

---

## 📂 Project Structure

```
/frontend
/backend
/python_model
/models
```

---

## 🐳 Run with Docker

```
docker-compose up --build
```

* Frontend: **[http://localhost/](http://localhost/)**
* Backend API: **[http://localhost:3000/](http://localhost:3000/)**
* ML Model API: **[http://localhost:5000/predict](http://localhost:5000/predict)**

---

## 📸 How it Works

1. User uploads a food image
2. Node backend sends the file to the Python model
3. Python predicts using SIFT + histogram classifier
4. Result returned to frontend instantly

---

## 📦 Production Deployment

Uses a **3-container production setup**:

* `food-classifier-frontend`
* `node-backend`
* `python-model`

Orchestrated via `docker-compose.prod.yml`.

---

## 🙌 Contribution

PRs are welcome. Feel free to open issues or suggest new features.

---

## ⭐ Support

If you like this project, please give it a ⭐ on GitHub!
