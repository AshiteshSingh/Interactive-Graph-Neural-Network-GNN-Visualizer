# Interactive Graph Neural Network (GNN) Visualizer

A highly interactive, real-time visualizer for Differentiable Graph Neural Network (GNN) Optimization. This application bridges a highly optimized pure **JAX** backend with a futuristic, glassmorphism-styled **React** frontend. 

It allows you to visualize neural network topologies (Residual, Sequential Chain, Dense) and watch an optimizer compute fusion speedups and FLOP costs in real-time.

![Interactive UI Preview](frontend/src/assets/react.svg)

## Features
- **JAX-powered Compute Backend**: Real-time FLOP calculations and differentiable architecture search optimizations.
- **Interactive DAG Layout**: Clean Left-to-Right auto-layout powered by `Dagre.js`.
- **Node Inspector**: Click on any node to view its Operation Type, Compute Load, and exact connections.
- **Real-Time Fusion Animations**: Watch as the JAX optimizer finds optimal operation fusions, lighting up the graph with animated cyan paths.
- **Premium Glassmorphism UI**: Built with React Flow, Lucide icons, and modern typography (Outfit/Inter).

## Tech Stack
- **Frontend**: React, Vite, React Flow, Dagre.js
- **Backend**: Python, JAX, FastAPI, Uvicorn

## Live Demo
Check out the live web interface hosted on Vercel:
[Interactive GNN Visualizer](https://interactive-graph-neural-network-gn.vercel.app/)

*(Note: The live demo uses mock inference data as the pure JAX backend cannot run on edge static hosting).*

## Running Locally

To run the full stack locally with the Python JAX backend:

### Prerequisites
- Python 3.9+
- Node.js (v18+)

### 1. Install Backend Dependencies
```bash
pip install jax jaxlib fastapi uvicorn pydantic numpy
```

### 2. Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### 3. Launch the Application
You can start both the backend and frontend simultaneously by running the included batch script from the root directory:
```bash
.\start.bat
```
Or, you can run them via npm:
```bash
npm start
```

The React UI will automatically open at `http://localhost:5173`, communicating with the FastAPI backend running on `http://localhost:8000`.
