"""Стенд T4: повторы header.stamp в /odom записи judge_t4 (record.npz) и разница поз в повторах. stand_t4_dup_stamps.py <record.npz>..."""
import sys, numpy as np
for f in sys.argv[1:]:
    r=np.load(f); s=r['odom__stamp']; p=r['odom__p']; rv=r['odom__recv']
    o=np.argsort(s,kind='stable'); s,p,rv=s[o],p[o],rv[o]
    d=np.where(np.abs(np.diff(s))<=1e-6)[0]
    dp=np.linalg.norm(p[d+1]-p[d],axis=1) if len(d) else np.array([0.])
    print(f.split('/')[-3], 'дублей stamp', len(d), 'макс Δp в дубле %.4f м'%dp.max(), ('пример stamp %.3f'%s[d[np.argmax(dp)]]) if len(d) else '')
