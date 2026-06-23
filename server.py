import os
import jax
import jax.numpy as jnp
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gnn_compiler import init_params, make_graph, forward, N_NODES, OP_NAMES, _FLOP, oracle

app = FastAPI(title="GNN Compiler API")

# Allow CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
class GnnState:
    def __init__(self):
        self.key = jax.random.PRNGKey(42)
        self.key, ik = jax.random.split(self.key)
        self.params = init_params(ik)
        
        ckpt = 'gnn_compiler_checkpoint.npy'
        if os.path.exists(ckpt):
            raw = np.load(ckpt, allow_pickle=True).item()
            self.params = jax.tree_util.tree_map(jnp.array, raw)
            print(f"Loaded checkpoint: {ckpt}")
        
        self.nf = None
        self.adj = None

state = GnnState()

@app.get("/api/graph")
def get_graph(topology: str = "residual"):
    state.key, gk = jax.random.split(state.key)
    state.nf, state.adj = make_graph(gk, topology, N=N_NODES)
    
    n = N_NODES
    
    nodes = []
    for i in range(n):
        op_idx = int(jnp.argmax(state.nf[i, :len(OP_NAMES)]))
        op_name = OP_NAMES[op_idx]
        nodes.append({
            "id": str(i),
            "data": {
                "label": f"{op_name}",
                "flop": float(_FLOP[op_idx] * state.nf[i, len(OP_NAMES)])
            },
            # Position will be set by dagre on frontend, but we can pass a dummy
            "position": {"x": 0, "y": 0}
        })
        
    edges = []
    for i in range(n):
        for j in range(n):
            if state.adj[i, j] > 0:
                edges.append({
                    "id": f"e{i}-{j}",
                    "source": str(i),
                    "target": str(j),
                    "fused": False
                })
                
    base_cost = float(oracle(state.nf, state.adj, jnp.zeros_like(state.adj)))
    
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "cost": f"{base_cost:.2f} (Base)",
            "speedup": "1.00x",
            "fusions": "0"
        }
    }

@app.post("/api/infer")
def run_inference():
    if state.nf is None or state.adj is None:
        return {"error": "No graph generated yet"}
        
    state.key, fk = jax.random.split(state.key)
    costs, fusion, perm, mem, fl = forward(state.params, state.nf, state.adj, fk)
    fh = (fusion > 0.5).astype(jnp.float32)
    
    fused_edges = []
    n_fusions = 0
    n = N_NODES
    for i in range(n):
        for j in range(n):
            if state.adj[i, j] > 0 and fh[i, j] > 0.5:
                fused_edges.append(f"e{i}-{j}")
                n_fusions += 1
                
    oracle_no_fuse = float(oracle(state.nf, state.adj, jnp.zeros_like(state.adj)))
    oracle_fused   = float(oracle(state.nf, state.adj, fh))
    speedup = oracle_no_fuse / (oracle_fused + 1e-6)
    
    return {
        "fused_edges": fused_edges,
        "stats": {
            "cost": f"{oracle_fused:.2f}",
            "speedup": f"{speedup:.2f}x",
            "fusions": str(n_fusions)
        }
    }
