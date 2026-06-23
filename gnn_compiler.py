import time
import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np

N_NODES     = 32
D_NODE      = 14
D_MODEL     = 128
D_FF        = 256
N_HEADS     = 8
D_HEAD      = D_MODEL // N_HEADS
N_LAYERS    = 4
N_OPS       = 12
SINKHORN_K  = 20
GUMBEL_T    = 0.5
BATCH       = 8
LR_PEAK     = 3e-4
WARMUP      = 300
TOTAL_STEPS = 5000
GRAD_CLIP   = 1.0

OP_NAMES = ['MatMul','Conv2D','ReLU','GeLU','Softmax','LayerNorm',
            'Add','Mul','Transpose','Reshape','Gather','Scatter']

_FLOP = jnp.array([10., 20., 0.2, 0.4,  2.,  1.5, 0.1, 0.1, 0.2, 0.05, 1.5, 1.5])
_MEM  = jnp.array([ 0.5, 1.0, 2.0, 2.0, 3.0,  3.0, 4.0, 4.0, 3.5, 2.0,  4.5, 4.5])
_FUSE = jnp.array([[float(i in [2,3,6,7] or j in [2,3,6,7]
                          or (i == 0 and j == 6) or (i == 4 and j == 0))
                    for j in range(N_OPS)] for i in range(N_OPS)])
_NORM = 25.0


def glorot(key, shape):
    return jax.random.normal(key, shape) * jnp.sqrt(2.0 / (shape[0] + shape[-1]))


def init_params(key):
    ks = jax.random.split(key, 60)
    k  = iter(ks)
    return {
        'proj': {
            'w': glorot(next(k), (D_NODE, D_MODEL)),
            'b': jnp.zeros(D_MODEL),
        },
        'gat': {
            'wq':    jnp.stack([glorot(next(k), (D_MODEL, D_MODEL)) for _ in range(N_LAYERS)]),
            'wk':    jnp.stack([glorot(next(k), (D_MODEL, D_MODEL)) for _ in range(N_LAYERS)]),
            'wv':    jnp.stack([glorot(next(k), (D_MODEL, D_MODEL)) for _ in range(N_LAYERS)]),
            'wo':    jnp.stack([glorot(next(k), (D_MODEL, D_MODEL)) for _ in range(N_LAYERS)]),
            'ff1_w': jnp.stack([glorot(next(k), (D_MODEL, D_FF))   for _ in range(N_LAYERS)]),
            'ff1_b': jnp.zeros((N_LAYERS, D_FF)),
            'ff2_w': jnp.stack([glorot(next(k), (D_FF,   D_MODEL)) for _ in range(N_LAYERS)]),
            'ff2_b': jnp.zeros((N_LAYERS, D_MODEL)),
            'ln1_g': jnp.ones( (N_LAYERS, D_MODEL)),
            'ln1_b': jnp.zeros((N_LAYERS, D_MODEL)),
            'ln2_g': jnp.ones( (N_LAYERS, D_MODEL)),
            'ln2_b': jnp.zeros((N_LAYERS, D_MODEL)),
        },
        'heads': {
            'cost_w1':  glorot(next(k), (D_MODEL + D_NODE, D_MODEL)),
            'cost_b1':  jnp.zeros(D_MODEL),
            'cost_w2':  glorot(next(k), (D_MODEL, 1)),
            'cost_b2':  jnp.zeros(1),
            'fuse_w1':  glorot(next(k), (D_MODEL * 2, D_MODEL)),
            'fuse_b1':  jnp.zeros(D_MODEL),
            'fuse_w2':  glorot(next(k), (D_MODEL, 1)),
            'fuse_b2':  jnp.zeros(1),
            'order_w':  glorot(next(k), (D_MODEL, 1)),
            'order_b':  jnp.zeros(1),
            'mem_w1':   glorot(next(k), (D_MODEL + D_NODE, 64)),
            'mem_b1':   jnp.zeros(64),
            'mem_w2':   glorot(next(k), (64, 1)),
            'mem_b2':   jnp.zeros(1),
        },
    }


def layer_norm(x, g, b, eps=1e-5):
    mu  = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x,  axis=-1, keepdims=True)
    return g * (x - mu) / jnp.sqrt(var + eps) + b


def gat_layer(lp, h, mask):
    N = h.shape[0]
    Q = jnp.dot(h, lp['wq']).reshape(N, N_HEADS, D_HEAD)
    K = jnp.dot(h, lp['wk']).reshape(N, N_HEADS, D_HEAD)
    V = jnp.dot(h, lp['wv']).reshape(N, N_HEADS, D_HEAD)
    s = jnp.einsum('ihd,jhd->hij', Q, K) / float(D_HEAD) ** 0.5 + mask[None]
    a = jax.nn.softmax(s, axis=-1)
    o = jnp.einsum('hij,jhd->ihd', a, V).reshape(N, D_MODEL)
    o = jnp.dot(o, lp['wo'])
    h = layer_norm(h + o, lp['ln1_g'], lp['ln1_b'])
    f = jax.nn.gelu(jnp.dot(h, lp['ff1_w']) + lp['ff1_b'])
    f = jnp.dot(f, lp['ff2_w']) + lp['ff2_b']
    h = layer_norm(h + f, lp['ln2_g'], lp['ln2_b'])
    return h


def encode(params, nf, adj):
    mask = jnp.where(adj > 0, 0.0, -1e9)
    h = jax.nn.gelu(jnp.dot(nf, params['proj']['w']) + params['proj']['b'])
    g = params['gat']
    for i in range(N_LAYERS):
        lp = {k: g[k][i] for k in g}
        h  = gat_layer(lp, h, mask)
    return h


def sinkhorn(logits, n_iter=SINKHORN_K):
    for _ in range(n_iter):
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-2, keepdims=True)
    return jnp.exp(logits)


def gumbel_sig(logits, key, temp=GUMBEL_T):
    u    = jax.random.uniform(key, logits.shape, minval=1e-6, maxval=1.0 - 1e-6)
    g    = -jnp.log(-jnp.log(u))
    soft = jax.nn.sigmoid((logits + g) / temp)
    hard = (soft > 0.5).astype(jnp.float32)
    return hard - jax.lax.stop_gradient(soft) + soft


def forward(params, nf, adj, key):
    h  = encode(params, nf, adj)
    hp = params['heads']
    N  = nf.shape[0]

    cat   = jnp.concatenate([h, nf], axis=-1)
    c     = jax.nn.relu(jnp.dot(cat, hp['cost_w1']) + hp['cost_b1'])
    costs = jax.nn.sigmoid(jnp.squeeze(jnp.dot(c, hp['cost_w2']) + hp['cost_b2'], -1))

    src   = jnp.tile(h[:, None, :], (1, N, 1))
    dst   = jnp.tile(h[None, :, :], (N, 1, 1))
    fl    = jnp.squeeze(
                jnp.dot(jax.nn.relu(jnp.dot(jnp.concatenate([src, dst], axis=-1),
                                            hp['fuse_w1']) + hp['fuse_b1']),
                        hp['fuse_w2']) + hp['fuse_b2'], -1) * adj
    fusion = gumbel_sig(fl, key) * adj

    os   = jnp.squeeze(jnp.dot(h, hp['order_w']) + hp['order_b'], -1)
    os_n = (os - jnp.min(os)) / (jnp.max(os) - jnp.min(os) + 1e-6) * float(N - 1)
    pos  = jnp.arange(N, dtype=jnp.float32)
    perm = sinkhorn(-(os_n[:, None] - pos[None, :]) ** 2)

    m   = jax.nn.relu(jnp.dot(cat, hp['mem_w1']) + hp['mem_b1'])
    mem = jax.nn.sigmoid(jnp.squeeze(jnp.dot(m, hp['mem_w2']) + hp['mem_b2'], -1))

    return costs, fusion, perm, mem, fl


def oracle(nf, adj, fusion_hard):
    oi        = jnp.argmax(nf[:, :N_OPS], axis=-1)
    sizes     = nf[:, N_OPS]
    flop_c    = _FLOP[oi] * sizes
    mem_c     = _MEM[oi]  * sizes
    fused_row = jnp.any(fusion_hard * adj > 0.5, axis=1).astype(jnp.float32)
    fused_col = jnp.any(fusion_hard * adj > 0.5, axis=0).astype(jnp.float32)
    fused     = jnp.maximum(fused_row, fused_col)
    fuse_ok   = jnp.sum(fusion_hard * adj * _FUSE[oi[:, None], oi[None, :]])
    return jnp.sum(flop_c) + jnp.sum(mem_c * (1.0 - 0.5 * fused)) - fuse_ok * 0.5


def loss_fn(params, nf, adj, key):
    costs, fusion, perm, mem, fl = forward(params, nf, adj, key)
    N  = nf.shape[0]
    oi = jnp.argmax(nf[:, :N_OPS], axis=-1)

    node_oracle  = (_FLOP[oi] * nf[:, N_OPS] + _MEM[oi] * nf[:, N_OPS]) / _NORM
    cost_loss    = jnp.mean((costs - node_oracle) ** 2)

    target_fuse  = adj * _FUSE[oi[:, None], oi[None, :]]
    fuse_probs   = jax.nn.sigmoid(fl)
    eps          = 1e-6
    fusion_bce   = -jnp.mean(
                     target_fuse * jnp.log(fuse_probs + eps) +
                     (1 - target_fuse) * jnp.log(1 - fuse_probs + eps))

    order_target = jnp.argsort(node_oracle).astype(jnp.float32) / float(N)
    order_pred   = jnp.sum(perm * jnp.arange(N, dtype=jnp.float32)[None, :], axis=-1) / float(N)
    order_loss   = jnp.mean((order_pred - order_target) ** 2)

    prow         = jnp.sum(perm, axis=-1)
    pcol         = jnp.sum(perm, axis=-2)
    ds_loss      = jnp.mean((prow - 1) ** 2) + jnp.mean((pcol - 1) ** 2)

    l2           = 1e-5 * sum(jnp.sum(p ** 2) for p in jax.tree_util.tree_leaves(params))

    return cost_loss + fusion_bce + order_loss + 0.01 * ds_loss + l2


def lr_schedule(step):
    step = jnp.float32(step)
    warm = LR_PEAK * step / WARMUP
    cos  = 0.5 * LR_PEAK * (1 + jnp.cos(jnp.pi * (step - WARMUP) / (TOTAL_STEPS - WARMUP)))
    return jnp.where(step < WARMUP, warm, cos)


def adam_init(params):
    z = jax.tree_util.tree_map(jnp.zeros_like, params)
    return z, z


def clip_grads(grads):
    norm  = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
    scale = jnp.minimum(1.0, GRAD_CLIP / (norm + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * scale, grads)


def adam_update(params, grads, m, v, step, b1=0.9, b2=0.999, eps=1e-8):
    lr  = lr_schedule(step)
    sf  = jnp.float32(step)
    m   = jax.tree_util.tree_map(lambda mi, g: b1 * mi + (1 - b1) * g, m, grads)
    v   = jax.tree_util.tree_map(lambda vi, g: b2 * vi + (1 - b2) * g ** 2, v, grads)
    mh  = jax.tree_util.tree_map(lambda mi: mi / (1 - jnp.float32(b1) ** sf), m)
    vh  = jax.tree_util.tree_map(lambda vi: vi / (1 - jnp.float32(b2) ** sf), v)
    p   = jax.tree_util.tree_map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
                                  params, mh, vh)
    return p, m, v


def batch_loss(params, nf_batch, adj_batch, keys):
    losses = vmap(loss_fn, in_axes=(None, 0, 0, 0))(params, nf_batch, adj_batch, keys)
    return jnp.mean(losses)


@jit
def train_step(params, m, v, nf_batch, adj_batch, keys, step):
    loss, grads = jax.value_and_grad(batch_loss)(params, nf_batch, adj_batch, keys)
    grads       = clip_grads(grads)
    params, m, v = adam_update(params, grads, m, v, step)
    return loss, params, m, v


def make_graph(key, topo='chain', N=N_NODES):
    k1, k2, k3 = jax.random.split(key, 3)
    ops = jax.random.randint(k1, (N,), 0, N_OPS)
    nf  = jnp.concatenate([
            jax.nn.one_hot(ops, N_OPS),
            jax.random.uniform(k2, (N, 1)),
            jax.random.uniform(k3, (N, 1)),
          ], axis=-1)
    adj = np.zeros((N, N), np.float32)
    for i in range(N - 1):
        adj[i, i + 1] = 1.0
    if topo == 'residual':
        for i in range(0, N - 4, 4):
            adj[i, i + 4] = 1.0
    elif topo == 'dense':
        for i in range(N):
            for j in range(i + 1, min(i + 4, N)):
                adj[i, j] = 1.0
    return nf, jnp.array(adj)


def make_batch(key, N=N_NODES, B=BATCH):
    topos = ['chain', 'residual', 'dense']
    nfs, adjs = [], []
    for i, k in enumerate(jax.random.split(key, B)):
        nf, adj = make_graph(k, topos[i % len(topos)], N)
        nfs.append(nf)
        adjs.append(adj)
    return jnp.stack(nfs), jnp.stack(adjs)


def evaluate(params, key, n=32):
    cost_errs, fuses, fuse_accs = [], [], []
    for _ in range(n):
        key, gk, fk = jax.random.split(key, 3)
        nf, adj     = make_graph(gk, 'residual')
        costs, fusion, perm, mem, fl = forward(params, nf, adj, fk)
        fh          = (fusion > 0.5).astype(jnp.float32)
        oi          = jnp.argmax(nf[:, :N_OPS], axis=-1)
        node_oracle = (_FLOP[oi] * nf[:, N_OPS] + _MEM[oi] * nf[:, N_OPS]) / _NORM
        cost_errs.append(float(jnp.mean(jnp.abs(costs - node_oracle))))
        fuses.append(float(jnp.sum(fh * adj)))
        target = adj * _FUSE[oi[:, None], oi[None, :]]
        pred   = (jax.nn.sigmoid(fl) > 0.5).astype(jnp.float32)
        acc    = float(jnp.mean((pred == target).astype(jnp.float32)))
        fuse_accs.append(acc)
    return float(np.mean(cost_errs)), float(np.mean(fuses)), float(np.mean(fuse_accs))


def save_params(params, path='gnn_compiler_checkpoint.npy'):
    np_params = jax.tree_util.tree_map(np.array, params)
    np.save(path, np_params, allow_pickle=True)
    print(f"  Checkpoint saved -> {path}")


if __name__ == '__main__':
    print("=" * 68)
    print("  DIFFERENTIABLE GNN COMPILER  --  Milestone Build")
    print("=" * 68)
    print(f"  JAX version  : {jax.__version__}")
    print(f"  Devices      : {jax.devices()}")
    print(f"  Device kind  : {jax.devices()[0].device_kind}")
    print("=" * 68)

    key = jax.random.PRNGKey(0)
    key, ik = jax.random.split(key)
    params = init_params(ik)
    m, v   = adam_init(params)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"\n  Architecture : {N_LAYERS}-layer Multi-Head Graph Attention")
    print(f"  Model dim    : {D_MODEL}  |  Heads: {N_HEADS}  |  FF: {D_FF}")
    print(f"  Graph size   : {N_NODES} nodes  |  {N_OPS} op types")
    print(f"  Parameters   : {n_params:,}")
    print(f"  Batch        : {BATCH} graphs/step (vmap)")
    print(f"  Optimizer    : Adam + cosine-warmup LR (peak={LR_PEAK})")
    print(f"  Loss         : CostMSE + FusionBCE + OrderMSE + SinkhornDS")
    print(f"  Training     : {TOTAL_STEPS} steps\n")

    print("-" * 68)
    print(f"  {'Step':>6}  {'Loss':>9}  {'CostMAE':>8}  {'FuseAcc':>8}  {'AvgFuse':>8}  {'LR':>8}")
    print("-" * 68)

    t0 = time.time()
    for step in range(1, TOTAL_STEPS + 1):
        key, bk  = jax.random.split(key)
        nf_b, adj_b = make_batch(bk)
        bkeys    = jax.random.split(key, BATCH)
        loss, params, m, v = train_step(params, m, v, nf_b, adj_b, bkeys, jnp.int32(step))

        if step % 500 == 0 or step == 1:
            cost_err, avg_fuse, fuse_acc = evaluate(params, key)
            lr      = float(lr_schedule(jnp.int32(step)))
            elapsed = time.time() - t0
            print(f"  {step:>6}  {float(loss):>9.5f}  {cost_err:>8.4f}  "
                  f"{fuse_acc:>8.4f}  {avg_fuse:>8.1f}  {lr:>8.2e}  [{elapsed:.0f}s]")

    total_time = time.time() - t0
    print("-" * 68)
    print(f"\n  Training complete: {total_time:.1f}s  ({total_time/TOTAL_STEPS*1000:.1f} ms/step)")
    save_params(params)

    print("\n" + "=" * 68)
    print("  COMPILATION BENCHMARK")
    print("=" * 68)
    key, gk, fk = jax.random.split(key, 3)
    nf, adj = make_graph(gk, 'dense')

    t_c = time.time()
    costs, fusion, perm, mem, fl = forward(params, nf, adj, fk)
    fusion.block_until_ready()
    compile_ms = (time.time() - t_c) * 1000

    fh     = (fusion > 0.5).astype(jnp.float32)
    order  = np.array(jnp.argmax(perm, axis=-1))
    op_seq = [OP_NAMES[int(jnp.argmax(nf[i, :N_OPS]))] for i in range(N_NODES)]
    fused  = [(op_seq[i], op_seq[j])
              for i in range(N_NODES) for j in range(N_NODES)
              if float(fh[i, j]) > 0.5]
    n_fuse = len(fused)
    oracle_no_fuse = float(oracle(nf, adj, jnp.zeros_like(adj)))
    oracle_fused   = float(oracle(nf, adj, fh))
    speedup        = oracle_no_fuse / (oracle_fused + 1e-6)

    print(f"\n  Input graph     : {N_NODES} nodes, dense topology")
    print(f"  Op sequence     : {op_seq}")
    print(f"  Execution order : {order}")
    print(f"  Fused pairs     : {fused[:6]}{'...' if n_fuse > 6 else ''} ({n_fuse} total)")
    print(f"  Per-node costs  : {np.array(costs).round(3)}")
    print(f"  Oracle (no fuse): {oracle_no_fuse:.2f}")
    print(f"  Oracle (fused)  : {oracle_fused:.2f}")
    print(f"  True speedup    : {speedup:.2f}x")
    print(f"  Compile time    : {compile_ms:.3f} ms  (single forward pass)")
    print(f"  vs. Brute-force : 2^{N_NODES} = {2**N_NODES:.2e} combinations")
    print("=" * 68)
