"""Скачки позы /odom за тик против истины. jump.py record.npz [t1 t2 ...] (sim, моменты переключений)"""
import sys, numpy as np
r = np.load(sys.argv[1], allow_pickle=True)
ts, p, q = r['odom__stamp'], r['odom__p'], r['odom__q']
gs, gp, gq = r['gt__stamp'], r['gt__p'], r['gt__q']
yaw = lambda q: np.arctan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]), 1-2*(q[:,1]**2+q[:,2]**2))
o = np.argsort(ts); ts, p, q = ts[o], p[o], q[o]
gy = np.unwrap(yaw(gq)); oy = np.unwrap(yaw(q))
gpi = np.stack([np.interp(ts, gs, gp[:, k]) for k in range(3)], 1); gyi = np.interp(ts, gs, gy)
dp = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1); dg = np.linalg.norm(np.diff(gpi[:, :2], axis=0), axis=1)
dyo = np.abs(np.diff(oy)); dyg = np.abs(np.diff(gyi)); tt = ts[1:]
def rep(m, name):
    if not m.any(): print(f'{name}: нет тиков'); return
    i = np.argmax(dp[m]); j = np.argmax(np.abs(dp[m]-dg[m]))
    print(f'{name}: тиков {m.sum()}, макс |Δp| одом {dp[m].max()*1000:.2f} мм (истина в том же тике {dg[m][i]*1000:.2f} мм, макс истины {dg[m].max()*1000:.2f}), '
          f'макс | |Δp|од−|Δp|ист | {np.abs(dp[m]-dg[m]).max()*1000:.2f} мм @ sim {tt[m][j]:.2f}; макс |Δψ| одом {np.degrees(dyo[m].max()):.3f}° (истина макс {np.degrees(dyg[m].max()):.3f}°)')
mv = dg > 1e-4
rep(np.ones_like(dp, bool), 'весь прогон')
for t in map(float, sys.argv[2:]):
    rep(np.abs(tt - t) <= 1.0, f'±1 с вокруг sim {t:.2f}')
# ошибка положения относительно истины (после совмещения старта) — до/после
e0 = p[0, :2] - gpi[0, :2]
err = np.linalg.norm(p[:, :2] - gpi[:, :2] - e0, axis=1)
for t in map(float, sys.argv[2:]):
    a = err[(ts > t-1.5) & (ts < t-0.5)]; b = err[(ts > t+0.5) & (ts < t+1.5)]
    if len(a) and len(b): print(f'  ошибка положения (без совмещения курса) до/после sim {t:.2f}: {a.mean()*1000:.1f} → {b.mean()*1000:.1f} мм')
