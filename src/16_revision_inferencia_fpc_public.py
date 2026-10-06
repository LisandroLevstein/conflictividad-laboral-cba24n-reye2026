#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce la inferencia final del artículo sin requerir el texto de Cba24n.

Entrada pública: data/public/evaluation_compact.csv
Salida: results/inferencia_fpc_revision/

Implementa estimadores ponderados por diseño, linealización de Taylor bajo
muestreo aleatorio estratificado sin reposición con corrección por población
finita (FPC), y bootstrap percentil estratificado original como sensibilidad.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data/public/evaluation_compact.csv"
OUT = ROOT / "results/inferencia_fpc_revision"
OUT.mkdir(parents=True, exist_ok=True)
Z = 1.959963984540054
B = 2000
SEED = 20162017

MAIN = {
    "proto_estricta": "Regla protocolar estricta",
    "proto_amplia": "Regla protocolar amplia",
    "ens_ge1": "Ensamble >= 1",
    "ens_ge2": "Ensamble >= 2 (punto de operación)",
    "ens_ge3": "Ensamble >= 3",
    "ens_ge4": "Ensamble = 4",
    "m_lda": "LDA anclado",
    "m_nmf": "NMF anclado",
    "m_kmeans": "K-medias anclado",
    "m_prototipo": "Prototipo (umbral GMM)",
}

def load():
    d=pd.read_csv(IN)
    if len(d)!=1400: raise SystemExit(f"Se esperaban 1400 casos; hay {len(d)}")
    d['dom']=(d.etiqueta!='DUDOSO').astype(float)
    d['y']=(d.etiqueta=='SI').astype(float)
    nh=d.groupby('votos').size().astype(float).sort_index()
    Nh=d.groupby('votos').N_h_eval.first().astype(float).sort_index()
    if int(Nh.sum())!=26605: raise SystemExit('Marco de evaluación inesperado')
    d['w']=d.N_h_eval / d.votos.map(nh)
    return d,nh,Nh

def ht(d,x): return float(np.sum(d.w.to_numpy(float)*np.asarray(x,float)))

def varlin(d,u,nh,Nh):
    u=np.asarray(u,float); h=d.votos.to_numpy(); out=0.0
    for k in nh.index:
        m=h==k; n=float(nh.loc[k]); N=float(Nh.loc[k]); f=n/N
        s2=float(np.var(u[m],ddof=1)) if n>1 else 0.0
        out += N*N*(1-f)*s2/n
    return out

def ratio(d,a,b,nh,Nh):
    A,Bb=ht(d,a),ht(d,b); th=A/Bb
    u=np.asarray(a,float)/Bb - A*np.asarray(b,float)/(Bb**2)
    se=math.sqrt(max(varlin(d,u,nh,Nh),0.0))
    return th,se,(th-Z*se,th+Z*se),u

def f1(d,c,nh,Nh):
    dom=d.dom.to_numpy(float); y=d.y.to_numpy(float); c=np.asarray(c,float)
    tp=dom*y*c; pred=dom*c; pos=dom*y
    TP,M,Y=ht(d,tp),ht(d,pred),ht(d,pos); den=M+Y; th=2*TP/den
    u=2*tp/den - 2*TP*(pred+pos)/(den**2)
    se=math.sqrt(max(varlin(d,u,nh,Nh),0.0))
    return th,se,(th-Z*se,th+Z*se),u

def prec(d,c,nh,Nh):
    dom=d.dom.to_numpy(float); y=d.y.to_numpy(float); c=np.asarray(c,float)
    return ratio(d,dom*y*c,dom*c,nh,Nh)

def rec(d,c,nh,Nh):
    dom=d.dom.to_numpy(float); y=d.y.to_numpy(float); c=np.asarray(c,float)
    return ratio(d,dom*y*c,dom*y,nh,Nh)

def diff(d,a,b,nh,Nh):
    fa=f1(d,d[a],nh,Nh); fb=f1(d,d[b],nh,Nh)
    th=fa[0]-fb[0]; u=fa[3]-fb[3]
    se=math.sqrt(max(varlin(d,u,nh,Nh),0.0))
    return th,se,(th-Z*se,th+Z*se)

def f1point(z,col):
    dom=(z.etiqueta!='DUDOSO').to_numpy(float); y=(z.etiqueta=='SI').to_numpy(float)
    c=z[col].to_numpy(float); w=z.N_h_eval.to_numpy(float)/z.votos.map(z.groupby('votos').size()).to_numpy(float)
    TP=np.sum(w*dom*y*c); M=np.sum(w*dom*c); Y=np.sum(w*dom*y)
    return float(2*TP/(M+Y))

def bootstrap_reps(d):
    rng=np.random.default_rng(SEED); hs=d.votos.to_numpy(); groups=[np.where(hs==h)[0] for h in range(5)]
    return [np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]) for _ in range(B)]

def bootstrap_ci(d,reps,fn):
    vals=np.array([fn(d.iloc[ix]) for ix in reps],float)
    return tuple(np.percentile(vals,[2.5,97.5]))

def main():
    d,nh,Nh=load(); reps=bootstrap_reps(d)
    # prevalence
    p=ratio(d,d.dom*d.y,d.dom,nh,Nh)
    prev=pd.DataFrame([{'estimacion':p[0],'se_taylor_fpc':p[1],'ic95_lo':p[2][0],'ic95_hi':p[2][1]}])
    prev.to_csv(OUT/'prevalencia.csv',index=False)
    # metrics
    rows=[]
    cols=list(MAIN.items()) + [(f'beto_{s}',f'BETO semilla {s}') for s in (20162017,20162018,20162019)]
    for col,name in cols:
        P=prec(d,d[col],nh,Nh); R=rec(d,d[col],nh,Nh); F=f1(d,d[col],nh,Nh)
        rows.append({'clasificador':name,'precision':P[0],'exhaustividad':R[0],'f1':F[0],
                     'f1_se_taylor_fpc':F[1],'f1_ic95_lo':F[2][0],'f1_ic95_hi':F[2][1]})
    pd.DataFrame(rows).to_csv(OUT/'metricas.csv',index=False)
    # paired comparisons
    pairs=[
      ('Regla estricta - Ensamble >=2','proto_estricta','ens_ge2'),
      ('Regla amplia - Ensamble >=2','proto_amplia','ens_ge2'),
      ('BETO 20162017 - Ensamble >=2','beto_20162017','ens_ge2'),
      ('BETO 20162017 - Regla amplia','beto_20162017','proto_amplia'),
      ('BETO 20162018 - Ensamble >=2','beto_20162018','ens_ge2'),
      ('BETO 20162019 - Ensamble >=2','beto_20162019','ens_ge2'),
    ]
    out=[]
    for label,a,b in pairs:
        D=diff(d,a,b,nh,Nh)
        boot=bootstrap_ci(d,reps,lambda z,aa=a,bb=b:f1point(z,aa)-f1point(z,bb))
        out.append({'comparacion':label,'diferencia':D[0],'taylor_fpc_lo':D[2][0],'taylor_fpc_hi':D[2][1],
                    'bootstrap_original_lo':boot[0],'bootstrap_original_hi':boot[1]})
    pd.DataFrame(out).to_csv(OUT/'comparaciones_pareadas.csv',index=False)
    diag=pd.DataFrame({'votos':nh.index.astype(int),'n_h':nh.astype(int).values,'N_h':Nh.astype(int).values,
                       'fraccion_muestreo':(nh/Nh).values,'fpc_varianza':(1-nh/Nh).values})
    diag.to_csv(OUT/'diagnostico_estratos.csv',index=False)
    print('OK: inferencia pública reproducida en',OUT)
    print(pd.DataFrame(out).to_string(index=False))
if __name__=='__main__': main()
