"""Smoke-test Sprint 1 on synthetic bundles so logic is verified before the real run."""
import numpy as np, subprocess, sys, tempfile, pathlib
rng = np.random.default_rng(1)
d = pathlib.Path(tempfile.mkdtemp())
N, T, V = 120, 900, 40
for tag, strength in (("subind", 0.06), ("none", 0.02)):
    for w in ["w2017","w2018","w2019","w2020","w2021","w2022","w2024"]:
        for s in [42,123,7]:
            rets = rng.normal(0, 0.012, size=(T, N))
            hist = np.array([f"2016-{1+(i//28)%12:02d}-{1+(i%28):02d}" for i in range(T)])
            vidx = np.arange(200, 200+V*5, 5)
            vdates = hist[vidx]
            # signal correlated with next-period return, strength differs by tag
            fut = np.array([rets[i:i+20].sum(axis=0) for i in vidx])
            z = (fut - fut.mean(axis=1, keepdims=True)) / (fut.std(axis=1, keepdims=True)+1e-9)
            latent = strength*z + rng.normal(0, 1, size=(V, N))
            p_up = 1/(1+np.exp(-latent)); p_dn = 1/(1+np.exp(latent))
            p_ne = np.full_like(p_up, 0.9)
            tot = p_up+p_dn+p_ne
            probs = np.stack([p_dn/tot, p_ne/tot, p_up/tot], axis=-1)
            np.savez_compressed(d/f"{tag}_{w}_s{s}.npz",
                tickers=np.array([f"T{i}" for i in range(N)], dtype=object),
                val_dates=vdates, val_probs=probs.astype(np.float32),
                val_fwd_ret=np.array([rets[i:i+5].sum(axis=0) for i in vidx]).astype(np.float32),
                rets_hist=rets.astype(np.float32), hist_dates=hist)
print("synthetic bundles:", len(list(d.glob('*.npz'))))
r = subprocess.run([sys.executable, "scripts/run_sprint1_portfolio_engine.py", str(d), "/tmp/s1.json"],
                   capture_output=True, text=True)
print(r.stdout[-3000:])
if r.returncode != 0:
    print("STDERR:", r.stderr[-2500:])
sys.exit(r.returncode)
