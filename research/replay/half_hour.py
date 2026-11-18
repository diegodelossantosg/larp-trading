import dataclasses, numpy as np
import research.replay.sim as sim
D=sim.Data(); P=sim.Params()
def runs(Q): return np.array([sim.summary(sim.run(D,dataclasses.replace(Q,keep=0.5),seed=s))["extra_per_year_pct"] for s in range(30)])
# same 30-min data, looking only at the full hours: a fair hourly baseline from the same prices
H=dataclasses.replace(P,half_hours=True,hours=tuple(range(600,960,60)))
A=dataclasses.replace(P,half_hours=True)
b=runs(H); fb=sim.summary(sim.run(D,H))
print(f"cada hora (datos 30 min): full {fb['extra_per_year_pct']:+.2f}, compras/año {fb['buys_per_year']}, 30 historias {b.mean():+.2f}, ganan {(b>0).sum()}/30",flush=True)
for name,Q in {"cada media hora":A,"cada media hora, lee 1h de noticias":dataclasses.replace(A,window_h=1.0)}.items():
    f=sim.summary(sim.run(D,Q)); v=runs(Q); d=v-b
    print(f"{name:36s} full {f['extra_per_year_pct']:+.2f}, compras/año {f['buys_per_year']}, 30 historias {v.mean():+.2f}, ganan {(v>0).sum()}/30, vs cada hora {d.mean():+.2f}, mejor en {(d>0).sum()}/30",flush=True)
