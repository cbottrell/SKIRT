import os,sys,string,time,glob,h5py
import illustris_python as il
import numpy as np
import pandas as pd

def Create_TNG_Catalogue(basePath,snap,mstar_lower=0,mstar_upper=np.inf,little_h=0.6774):
    '''
    mstar_lower and mstar_upper have log10(Mstar/Msun) units and are physical (i.e. Msun, not Msun/h).
    '''
    # convert to units used by TNG (1e10 Msun/h)
    mstar_lower = 10**(mstar_lower)/1e10*little_h
    mstar_upper = 10**(mstar_upper)/1e10*little_h
    
    fields = ['SubhaloMassType','SubhaloFlag','SubhaloSFR','SubhaloSFRinHalfRad','SubhaloSFRinRad','SubhaloHalfmassRadType']

    ptNumStars = il.snapshot.partTypeNum('stars')
    ptNumGas = il.snapshot.partTypeNum('gas')
    ptNumDM = il.snapshot.partTypeNum('dm')

    cat = il.groupcat.loadSubhalos(basePath,snap,fields=fields)
    hdr = il.groupcat.loadHeader(basePath,snap)
    redshift,a2 = hdr['Redshift'],hdr['Time']
    subs = np.arange(cat['count'],dtype=int)
    flags = cat['SubhaloFlag']
    mstar = cat['SubhaloMassType'][:,ptNumStars]
    cat['SubhaloID'] = subs
    cat['Snapshot'] = [snap for sub in subs]
    subs = subs[(flags!=0)*(mstar>=mstar_lower)*(mstar<=mstar_upper)]

    cat['SubhaloHalfmassRadType_stars'] = cat['SubhaloHalfmassRadType'][:,ptNumStars]*a2/little_h
    cat['SubhaloHalfmassRadType_gas'] = cat['SubhaloHalfmassRadType'][:,ptNumGas]*a2/little_h
    cat['SubhaloHalfmassRadType_dm'] = cat['SubhaloHalfmassRadType'][:,ptNumDM]*a2/little_h
    cat['SubhaloMassType_stars'] = np.log10(cat['SubhaloMassType'][:,ptNumStars]*1e10/little_h)
    cat['SubhaloMassType_gas'] = np.log10(cat['SubhaloMassType'][:,ptNumGas]*1e10/little_h)
    cat['SubhaloMassType_dm'] = np.log10(cat['SubhaloMassType'][:,ptNumDM]*1e10/little_h)
    del cat['SubhaloMassType'],cat['SubhaloHalfmassRadType'],cat['count']
    cat = pd.DataFrame.from_dict(cat)
    cat = cat.loc[subs]
    return cat
    
def main():
    sim_tag='TNG50-1'
    snaps = [91,84,78,72,67,63,59]
    mstar_lower=9
    basePath = f'/lustre/work/connor.bottrell/Simulations/IllustrisTNG/{sim_tag}/output'
    catPath = f'/lustre/work/connor.bottrell/Simulations/IllustrisTNG/{sim_tag}/postprocessing/Catalogues'
    if not os.access(catPath,0):
        os.system(f'mkdir -p {catPath}')
    for snap in snaps:
        cat = Create_TNG_Catalogue(basePath,snap=snap,mstar_lower=mstar_lower)
        cat.to_csv(f'{catPath}/{sim_tag}_{snap:03}_Catalogue.csv')

if __name__=='__main__':
    main()
