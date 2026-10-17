import math
import json
import numpy as np
from scipy import ndimage
from skimage import measure
from euv_stochastic_car_vs_mor import model

EXPOSURE_DOSE_GRID=np.linspace(20.,140.,13)
RESIST_TYPE_GRID=("CAR","MOR")
PATTERN_GRID=("lines","contacts")
CHEMISTRY_GRID=("on","off")

def exposure_dose_grid(): return EXPOSURE_DOSE_GRID.copy()
def trial_grid(): return [{"resist_type":r,"pattern":p,"chemistry":c} for r in RESIST_TYPE_GRID for p in PATTERN_GRID for c in CHEMISTRY_GRID]

def _parameters(resist_type,sensitivity):
    if resist_type=="CAR":
        alpha={"low_absorption":.00375,"high_absorption":.00625}.get(sensitivity,.005)
        y={"low_yield":1.5,"high_yield":2.5}.get(sensitivity,2.)
        e={"low_blur":1.5,"high_blur":2.5}.get(sensitivity,2.)
        q={"no_quencher":0.,"high_quencher":.40}.get(sensitivity,.20)
        return alpha,35.,y,e,5.,q
    return .020,20.,1.,2.,0.,0.

def _aerial(pattern,n,grid_nm,sensitivity="nominal"):
    shape=(n,n) if np.isscalar(n) else tuple(n); ny,nx=shape
    y,x=np.mgrid[:ny,:nx]*grid_nm; x-=nx*grid_nm/2; y-=ny*grid_nm/2
    sr={"low_source_radius":.18,"high_source_radius":.32}.get(sensitivity,.25); sc=.65
    if pattern=="lines":
        mask=np.abs((x+10.)%20.-10.)>=5.
        pts=[(sc-sr/2,0),(sc+sr/2,0),(-sc+sr/2,0),(-sc-sr/2,0)]
    else:
        cx=np.arange(-nx*grid_nm/2+16.,nx*grid_nm/2,32.); cy=np.arange(-ny*grid_nm/2+16.,ny*grid_nm/2,32.)
        dx=np.min(np.abs(x[...,None]-cx),axis=-1); dy=np.min(np.abs(y[...,None]-cy),axis=-1)
        mask=np.hypot(dx,dy)<=8.
        pts=[(r*np.cos(a),r*np.sin(a)) for a in np.linspace(0,2*np.pi,12,endpoint=False) for r in (sc-sr/2,sc+sr/2)]
    cutoff=.55/13.5; fy=np.fft.fftfreq(ny,d=grid_nm); fx=np.fft.fftfreq(nx,d=grid_nm)
    fyy,fxx=np.meshgrid(fy,fx,indexing="ij"); spec=np.fft.fft2(mask.astype(float)); out=np.zeros(shape)
    for sx,sy in pts:
        pupil=(fxx-sx*cutoff)**2+(fyy-sy*cutoff)**2<=cutoff**2
        out+=np.abs(np.fft.ifft2(spec*pupil))**2
    return out/out.max()/len(pts)

def _line_metrics(openings,grid_nm,minimum_nm):
    ny,nx=openings.shape; line=~openings; center=nx//2; pitch=int(round(20/grid_nm)); half=int(round(5/grid_nm)); seg=int(round(1000/grid_nm)); nseg=ny//seg
    if not nseg:return float("nan"),float("nan"),0.,0.
    def sl(c):return slice(max(0,c-half),min(nx,c+half+1))
    lers=[];lwrs=[];bridges=[];breaks=[]
    for i in range(nseg):
        a=i*seg; s=line[a:a+seg]; lab=measure.label(s,connectivity=2)
        top=set(lab[0,sl(center)]); bot=set(lab[-1,sl(center)]); top.discard(0);bot.discard(0); brk=not bool(top&bot)
        cc=set(lab[:,sl(center)].ravel());cc.discard(0); bridge=False
        for nc in (center-pitch,center+pitch):
            if 0<=nc<nx:
                z=set(lab[:,sl(nc)].ravel());z.discard(0)
                if cc&z:bridge=True;break
        w=[];le=[];re=[]
        for row in range(seg):
            if not s[row,center]:brk=True;continue
            l=r=center
            while l>0 and s[row,l-1]:l-=1
            while r<nx-1 and s[row,r+1]:r+=1
            w.append((r-l+1)*grid_nm);le.append(l*grid_nm);re.append(r*grid_nm)
        w=np.asarray(w);bridges.append(float(bridge));breaks.append(float(brk))
        if len(w)==seg and not np.any(w<=grid_nm) and not bridge and not brk:
            lers.append(1.5*(np.std(le)+np.std(re)));lwrs.append(3*np.std(w))
    return (float(np.mean(lers)) if lers else float("nan"),float(np.mean(lwrs)) if lwrs else float("nan"),float(np.mean(bridges)),float(np.mean(breaks)))

def _contact_metrics(openings,grid_nm,minimum_nm):
    ny,nx=openings.shape; labels=measure.label(openings,connectivity=2)
    xs=np.rint((np.arange(-nx*grid_nm/2+16,nx*grid_nm/2,32)+nx*grid_nm/2)/grid_nm).astype(int)
    ys=np.rint((np.arange(-ny*grid_nm/2+16,ny*grid_nm/2,32)+ny*grid_nm/2)/grid_nm).astype(int)
    yy,xx=np.ogrid[:ny,:nx]; rad=8/grid_nm; pitch=int(round(32/grid_nm)); d={}
    for cy in ys:
        for cx in xs:
            z=set(labels[(yy-cy)**2+(xx-cx)**2<=rad**2]);z.discard(0);d[cy,cx]=z
    missing=merged=total=0
    for cy in ys:
        for cx in xs:
            total+=1; z=d[cy,cx]
            if not z:missing+=1;continue
            for dy,dx in ((-pitch,0),(pitch,0),(0,-pitch),(0,pitch)):
                q=(cy+dy,cx+dx)
                if q in d and z&d[q]:merged+=1;break
    return missing/total,merged/total

def run_trial(cell,trial_id,seed):
    rng=np.random.default_rng(seed); resist=cell.get("resist_type","CAR");dose=float(cell.get("exposure_dose",80.));pattern=cell.get("pattern","lines");chemistry=cell.get("chemistry","on");sensitivity=cell.get("sensitivity_setting","nominal");minimum_nm=float(cell.get("classifier_minimum_nm",.5))
    alpha,thickness,yield_mean,electron_blur,diffusion_blur,quencher=_parameters(resist,sensitivity); absorption=1-math.exp(-alpha*thickness)
    if "region_area_nm2" in cell:
        area=float(cell["region_area_nm2"])
        n_trials=int(cell.get("n_trials",1))
        if n_trials < 1:
            raise ValueError("n_trials must be at least one")
        # E1
        phi=model.photon_density(dose)
        # E2
        expected=model.expected_absorbed_count(phi,area,alpha,thickness)
        incident=rng.poisson(phi*area,size=n_trials)
        absorbed=rng.binomial(incident,absorption)
        transmitted=incident-absorbed
        partition_error=np.abs(incident-absorbed-transmitted)
        mean_absorbed=float(np.mean(absorbed))
        var_absorbed=float(np.var(absorbed,ddof=1)) if n_trials > 1 else 0.0
        mean_incident=float(np.mean(incident))
        return {"mean_absorbed_photons":mean_absorbed,"var_absorbed_photons":var_absorbed,"mean_incident_photons":mean_incident,"absorbed_count":float(absorbed.sum()),"incident_count":float(incident.sum()),"transmitted_count":float(transmitted.sum()),"photon_partition_error":float(np.max(partition_error)),"fraction_zero_absorption":float(np.mean(absorbed==0)),"max_abs_incident_minus_absorbed_minus_transmitted":float(np.max(partition_error)),"expected_absorbed":expected}
    grid_nm=.5;shape=(2000,96) if pattern=="lines" else (128,128);aerial=_aerial(pattern,shape,grid_nm,sensitivity)
    # E1
    photon_mean=model.photon_density(dose)*aerial*grid_nm*grid_nm;incident=rng.poisson(photon_mean);absorbed=rng.binomial(incident,absorption)
    reference_dose=float(cell["exposure_dose"])
    reference_photon_mean=model.photon_density(reference_dose)*aerial*grid_nm*grid_nm
    if chemistry=="off":proxy=absorbed.astype(float);threshold=float(np.quantile(reference_photon_mean*absorption,.62))
    elif resist=="CAR":
        acid=rng.poisson(absorbed*yield_mean).astype(float);proxy=ndimage.gaussian_filter(acid,electron_blur/grid_nm,mode="nearest");proxy=ndimage.gaussian_filter(proxy,diffusion_blur/grid_nm,mode="nearest")
        qrng=np.random.default_rng(np.random.SeedSequence([int(cell.get("quencher_field_seed",0)),trial_id]));q=qrng.poisson(quencher*reference_photon_mean*absorption*yield_mean).astype(float);proxy-=q;threshold=.55*reference_dose*absorption*yield_mean*grid_nm*grid_nm
    else:proxy=ndimage.gaussian_filter(absorbed.astype(float),electron_blur/grid_nm,mode="nearest");threshold=.48*dose*absorption*grid_nm*grid_nm
    developed=proxy>threshold;out={"absorbed_count":float(absorbed.sum()),"incident_count":float(incident.sum()),"transmitted_count":float(incident.sum()-absorbed.sum()),"photon_partition_error":0.,"mean_absorbed_photons":float(absorbed.sum()),"mean_incident_photons":float(incident.sum()),"absorption_fraction":absorption,"printed_fraction":float(developed.mean())}
    if pattern=="lines":
        ler,lwr,b,br=_line_metrics(developed,grid_nm,minimum_nm);pr=float(np.isfinite(ler));out.update({"LER":ler,"LWR":lwr,"p_bridge":float(b),"p_break":float(br),"roughness_defined":pr,"printable_fraction":pr,"excluded_defect_count":int(not pr),"roughness_segment_length_nm":1000.})
    else:
        missing,merged=_contact_metrics(developed,grid_nm,minimum_nm);out.update({"p_missing":float(missing),"p_merged":float(merged)})
    return out

if __name__=="__main__":
    print("RESULT_JSON: "+json.dumps({"status":"ready","exposure_dose_grid":exposure_dose_grid().tolist()}))
