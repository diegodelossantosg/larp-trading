import pandas as pd, numpy as np
from trader.lab import Lab
L=Lab()
def px(s): return L.series(s).dropna()
spy=px("SPY"); shy=px("SHY"); ief=px("IEF"); gld=px("GLD"); tlt=px("TLT")
df=pd.concat({"SPY":spy,"SHY":shy,"IEF":ief,"GLD":gld,"TLT":tlt},axis=1).dropna()
r=df.pct_change().fillna(0)
cost=0.0005   # 0.05% per switch
def stats(ret,name):
    ret=ret.loc["2017-01-01":]; eq=(1+ret).cumprod(); yrs=len(ret)/252
    cagr=eq.iloc[-1]**(1/yrs)-1; dd=(eq/eq.cummax()-1).min(); sh=ret.mean()/ret.std()*np.sqrt(252)
    by={y:(1+g).prod()-1 for y,g in ret.groupby(ret.index.year)}
    return f"{name:44} CAGR {cagr*100:5.1f}%  maxDD {dd*100:6.1f}%  Sharpe {sh:4.2f}  | " + " ".join(f"{y%100}:{v*100:+.0f}" for y,v in by.items())
bench=r.SPY
print(stats(bench,"SPY buy and hold"))
sma=df.SPY.rolling(200).mean()
for alt in ("SHY","IEF","GLD"):
    sig=(df.SPY>sma).shift(1).fillna(False)
    ret=np.where(sig,r.SPY,r[alt]); ret=pd.Series(ret,index=r.index)-cost*sig.astype(int).diff().abs().fillna(0)
    print(stats(ret,f"SPY above 200d else {alt}"))
sma100=df.SPY.rolling(100).mean(); sig=(df.SPY>sma100).shift(1).fillna(False)
ret=pd.Series(np.where(sig,r.SPY,r.SHY),index=r.index)-cost*sig.astype(int).diff().abs().fillna(0); print(stats(ret,"SPY above 100d else SHY"))
# vol target 12%
rv=r.SPY.rolling(20).std()*np.sqrt(252)
w=(0.12/rv).clip(upper=1.0).shift(1).fillna(1.0)
ret=w*r.SPY+(1-w)*r.SHY-cost*w.diff().abs().fillna(0); print(stats(ret,"vol-target 12% (cap 100%) rest SHY"))
# drawdown guard: cut to SHY after -10% from 252d high until new high in 20d
dd=df.SPY/df.SPY.rolling(252,min_periods=60).max()-1
sig=(dd>-0.10).shift(1).fillna(True)
ret=pd.Series(np.where(sig,r.SPY,r.SHY),index=r.index)-cost*sig.astype(int).diff().abs().fillna(0); print(stats(ret,"out of SPY while >10% below its high"))
