import dataclasses, numpy as np
import research.replay.sim as sim
D=sim.Data(); P=sim.Params()
def runs(Q): return np.array([sim.summary(sim.run(D,dataclasses.replace(Q,keep=0.5),seed=s))["extra_per_year_pct"] for s in range(30)])
base=runs(P); f=sim.summary(sim.run(D,P))
print(f"cada hora (6 veces/dia): full {f['extra_per_year_pct']:+.2f}  30 historias {base.mean():+.2f}, ganan al S&P {(base>0).sum()}/30",flush=True)
for name,ch in {"cada 2 horas (10,12,14)":{"hours":(10,12,14)},"cada 2 horas (11,13,15)":{"hours":(11,13,15)},"una vez (11)":{"hours":(11,)},"una vez (13)":{"hours":(13,)},"solo mananas (10,11)":{"hours":(10,11)},"solo tardes (14,15)":{"hours":(14,15)}}.items():
    Q=dataclasses.replace(P,**ch); f=sim.summary(sim.run(D,Q)); v=runs(Q); d=v-base
    print(f"{name:26s} full {f['extra_per_year_pct']:+.2f}  30 historias {v.mean():+.2f}, ganan {(v>0).sum()}/30, vs cada hora {d.mean():+.2f} mejor en {(d>0).sum()}/30",flush=True)
