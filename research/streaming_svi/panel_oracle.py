"""Bounded 8-unit control: exact Gaussian expectations, analytic ELBO gradient.

Model: a~N(0,2), l=log(tau)~N(log(.7),.4), z_i~N(0,1),
y_ij~N(a+exp(l)*z_i,.5). Independent Gaussian q over these ten scalars.
This is a diagnostic reference, not an inference implementation.
Run: python3 research/streaming_svi/panel_oracle.py OUTPUT_DIR
For the checked synthetic snapshot, use research/streaming_svi/results.
"""
import csv
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
import numpy as np

labels=[f's{i}' for i in range(1,9)]
stats=np.zeros((6,8,3))
with open(output_dir / 'panel-stats.tsv') as f:
    for w,label,n,s,ss in csv.reader(f,delimiter='\t'):
        stats[int(w)-1,labels.index(label)]=[float(n),float(s),float(ss)]

def loss_gradient(x, data, prior):
    m,r=x[:10],x[10:]
    v=np.exp(2*r)
    m0,v0=prior
    p=1/v0
    a,l=m[:2]
    z=m[2:]
    n,sy,sy2=data.T
    e1=np.exp(l+v[1]/2)
    e2=np.exp(2*l+2*v[1])
    res=sy-n*a
    cross=np.dot(res,z)
    zsecond=np.dot(n,z*z+v[2:])
    llquad=np.sum(sy2-2*sy*(a+e1*z)+n*(a*a+v[0]+2*a*e1*z+e2*(z*z+v[2:])))
    value=.5*np.dot(p,(m-m0)**2+v)+2*llquad-r.sum()
    dm=p*(m-m0)
    dm[0]+=4*(n.sum()*a+e1*np.dot(n,z)-sy.sum())
    dm[1]+=4*(-e1*cross+e2*zsecond)
    dm[2:]+=4*(-e1*res+n*e2*z)
    dr=p*v-1
    dr[0]+=4*n.sum()*v[0]
    dr[1]+=4*(-e1*cross+2*e2*zsecond)*v[1]
    dr[2:]+=4*n*e2*v[2:]
    return value,np.r_[dm,dr]

initial=(np.r_[0.,np.log(.7),np.zeros(8)],np.r_[4.,.16,np.ones(8)])
probe=np.r_[initial[0]+np.linspace(-.1,.1,10),.5*np.log(initial[1])-.2]
val,grad=loss_gradient(probe,stats[0],initial)
h=1e-5
numeric=np.array([(loss_gradient(probe+h*np.eye(20)[i],stats[0],initial)[0]-loss_gradient(probe-h*np.eye(20)[i],stats[0],initial)[0])/(2*h) for i in range(20)])
error=np.max(np.abs(grad-numeric))
print('ORACLE_GRADIENT max_error=',error)
assert error<1e-5

def optimize(data, prior):
    x=np.r_[prior[0],.5*np.log(prior[1])]
    p=1/prior[1]
    n,sy,_=data.T
    for it in range(20000):
        # Given q(l), solve jointly for a and z means using the scalar Schur
        # complement; their independent variances are reciprocal precisions.
        vl=np.exp(2*x[11]); e1=np.exp(x[1]+vl/2); e2=np.exp(2*x[1]+2*vl)
        pa=p[0]+4*n.sum(); pz=p[2:]+4*n*e2; c=4*n*e1
        ha=p[0]*prior[0][0]+4*sy.sum(); hz=p[2:]*prior[0][2:]+4*e1*sy
        x[0]=(ha-np.dot(c,hz/pz))/(pa-np.dot(c,c/pz))
        x[2:10]=(hz-c*x[0])/pz
        x[10]=-.5*np.log(pa); x[12:]=-.5*np.log(pz)
        # A two-coordinate Newton step for log-scale mean and log SD, using
        # its analytic Hessian with damping and an objective line search.
        value,g=loss_gradient(x,data,prior)
        if np.max(np.abs(g))<1e-7: break
        z=x[2:10]; vz=np.exp(2*x[12:]); cross=np.dot(sy-n*x[0],z)
        second=np.dot(n,z*z+vz)
        h00=p[1]-4*e1*cross+8*e2*second
        h01=(-4*e1*cross+16*e2*second)*vl
        h11=2*vl*(p[1]-4*e1*cross+8*e2*second)+4*vl*vl*(-e1*cross+8*e2*second)
        H=np.array([[h00,h01],[h01,h11]])
        H+=np.eye(2)*max(0.,1e-6-np.linalg.eigvalsh(H)[0])
        gg=g[[1,11]]; step=-np.linalg.solve(H,gg)
        step*=min(1.,1/max(np.linalg.norm(step),1e-100))
        rate=1.
        for _ in range(60):
            proposal=x.copy(); proposal[[1,11]]+=rate*step
            newvalue,_=loss_gradient(proposal,data,prior)
            if newvalue<=value+1e-4*rate*np.dot(gg,step)+1e-12:
                x=proposal; break
            rate*=.5
        else: raise AssertionError('line search failed')
    _,g=loss_gradient(x,data,prior)
    print('ORACLE_OPT iterations=',it+1,'max_gradient=',np.max(np.abs(g)))
    assert np.max(np.abs(g))<1e-4
    return x[:10],np.exp(2*x[10:])

oracle=[]
prior=initial
for i in range(6):
    prior=optimize(stats[i],prior)
    oracle.append(prior)
    print('ORACLE_WINDOW',i+1,'globals=',prior[0][:2],'sd=',np.sqrt(prior[1][:2]))
batch=optimize(stats.sum(axis=0),initial)
print('ORACLE_BATCH globals=',batch[0][:2],'sd=',np.sqrt(batch[1][:2]))
print('ORACLE_STREAM_BATCH mean_error=',np.abs(prior[0]-batch[0]),'sd_ratio=',np.sqrt(prior[1]/batch[1]))

factors={}
with open(output_dir / 'panel-factors.tsv') as f:
    for mode,w,name,m,v in csv.reader(f,delimiter='\t'):
        factors.setdefault((mode,int(w)),{})[name]=(float(m),float(v))
def factor(mode,i):
    fs=factors[mode,i]
    globals_=[k for k in fs if k not in labels]
    assert 'a' in globals_ and len(globals_)==2,globals_
    scale=next(k for k in globals_ if k!='a')
    return np.array([fs[k] for k in ['a',scale]+labels]).T
for mode in ['short','long']:
    for i in [1,6,0]:
        q=factor(mode,i)
        ref=batch if i==0 else oracle[i-1]
        print('SVI_ORACLE',mode,i,'max_mean_error=',np.max(np.abs(q[0]-ref[0])),
            'global_mean_error=',q[0][:2]-ref[0][:2],
            'sd_ratios=',np.sqrt(q[1]/ref[1]))
    q=factor(mode,6)
    direct=optimize(stats[5],factor(mode,5))
    print('SVI_LAST_WINDOW_TARGET',mode,'max_mean_error=',np.max(np.abs(q[0]-direct[0])),
        'sd_ratios=',np.sqrt(q[1]/direct[1]))
