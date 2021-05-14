import os,sys,re
import scipy.spatial as spt
import numpy as np
import h5py
import illustris_tools as it
import illustris_python as il

def nn_dist(args):
    smoothlength,point,tree = args
    return tree.query(point,k=smoothlength)[0][-1]

def skirt_requirements(snap,sub,tmp_path,sim_tag='TNG50-1',
                       pixelsize=100.,fovsize_hmr=15.,smoothlength=32):
    '''
    pixel_size: output pixel size in [pc]
    fov_hmr: size of fov in stellar half mass radius units
    smooth_length: kernel smoothing length of stellar particles (nn distance)
    '''

    #basepath where TNG data is stored
    basePath = f'/home/bottrell/scratch/Simulations/{sim_tag}/output'

    #load current scale factor
    header = dict(h5py.File(f'{basePath}/snapdir_{snap:03d}/snap_{snap:03d}.0.hdf5',"r")["Header"].attrs.items())
    a2 = header["Time"]

    #take fov_size times stellar halfmassradius in proper pc as boxsize for dust calcualtion in SKIRT 
    fovsize = it.convertlength(il.groupcat.loadSingle(basePath,snap,subhaloID=sub)["SubhaloHalfmassRadType"][4])
    fovsize *= a2*fovsize_hmr*1000 # pc

    npix = np.ceil(fovsize/pixelsize).astype(int)

    if fovsize == 0:
        print("Not a valid subhalo, I will stop now and running SKIRT makes no sense")
        raise SystemExit
    
    # save f_stars,f_mappings,f_gas to file
    f_stars = f'shalo_{snap:03d}-{sub}_stars.dat'
    f_mappings = f'shalo_{snap:03d}-{sub}_mappings.dat'
    f_gas = f'shalo_{snap:03d}-{sub}_gas.dat'
    
    #############
    ### Stars ###
    #############
    if not (os.access(f'{tmp_path}/{f_stars}',0) and os.access(f'{tmp_path}/{f_mappings}',0)):
        fields = ["Coordinates","GFM_StellarFormationTime","GFM_InitialMass","GFM_Metallicity", "Velocities", "Masses"]
        #load star particles
        stars = il.snapshot.loadSubhalo(basePath,snap,sub,4,fields)
        #ignore all wind particles
        if (stars["count"]!=0):
            starindex = np.where(stars["GFM_StellarFormationTime"] > 0)

        #Transform coordinates to center
        centerGal=il.groupcat.loadSingle(basePath,snap,subhaloID=sub)["SubhaloPos"]
        old_coordinates,center = it.periodicfix(stars["Coordinates"], centerGal)
        old_coordinates = (it.convertlength(old_coordinates[starindex])*1000*a2).T
        center = (it.convertlength(center)*1000*a2)
        x,y,z = old_coordinates[0] - center[0],old_coordinates[1] - center[1],old_coordinates[2] - center[2]
        del old_coordinates, center

        #smoothing length in proper pc (distance to smoothlength closest neighbour)
        if (stars["count"]!=0):
            #create array with locations
            loc = np.array([x,y,z]).T
            #setup KDTree
            tree = spt.cKDTree(loc,leafsize = 100)

            environ = os.environ
            nthreads = int(environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in environ else 1
            if nthreads>1:
                from multiprocessing import Pool
                argList = [(smoothlength,point,tree) for point in loc]
                with Pool(nthreads) as pool:
                    h = np.asarray(pool.map(nn_dist,argList))
            else:
                h = np.zeros(len(x))
                for i,point in enumerate(loc):
                    dist = tree.query(point,k=smoothlength)[0][-1]
                    h[i] = dist
                h = np.asarray(h)
        else:
            h = []

        #initial mass in solar masses
        if (stars["count"]!=0):
            M = it.convertmass(stars["GFM_InitialMass"][starindex])
        else:
            M = []

        #metallicity (actual metallicity, not in solar units)
        if (stars["count"]!=0):
            Z = stars["GFM_Metallicity"][starindex]
        else:
            Z = []

        #age of the stars in years
        if (stars["count"]!=0):
            t = it.age(stars["GFM_StellarFormationTime"][starindex],a2)
        else:
            t = []

        ### Old stars ####
        #only consider stars older than 10Myr for bruzual charlot spectra
        xb = x[t > 1e7]
        yb = y[t > 1e7]
        zb = z[t > 1e7]
        hb = h[t > 1e7]
        Mb = M[t > 1e7]
        Zb = Z[t > 1e7]
        tb = t[t > 1e7]

        #create 2D-array and write into data file
        totalb = np.column_stack((xb,yb,zb,hb,Mb,Zb,tb))
        np.savetxt(f'{tmp_path}/{f_stars}',totalb,delimiter = " ")

        ### Young stars ###
        xm = x[t <= 1e7]
        ym = y[t <= 1e7]
        zm = z[t <= 1e7]
        hm = h[t <= 1e7]
        Mm = M[t <= 1e7]
        Zm = Z[t <= 1e7]
        tm = t[t <= 1e7]

        #calculate SFR (assuming SFR has been constant in the last 10Myr)
        SFRm = Mm/1e7
        #for compactness, take typical value of 5:
        logC = np.full(len(SFRm),5)
        #for ISM pressure, take typical value log(P/kB / cm^-3K) = 5, P = 1.38e-12 N/m2
        pISM = np.full(len(SFRm),1.38e-12)
        #for PDR covering fraction, take f = 0.2
        fPDR = np.full(len(SFRm),0.2)
        #create 2D-array and write into data file
        totalm = np.column_stack((xm,ym,zm,hm,SFRm,Zm,logC,pISM,fPDR))
        np.savetxt(f'{tmp_path}/{f_mappings}',totalm,delimiter = " ")

    #############
    #### Gas ####
    #############
    fields = ["Coordinates","Density","GFM_Metallicity","InternalEnergy","ElectronAbundance","StarFormationRate"]
    #load gas particles
    gas= il.snapshot.loadSubhalo(basePath,snap,sub,0,fields=fields)
    if gas["count"] != 0:
        #Transform coordinates to center
        centerGal=il.groupcat.loadSingle(basePath,snap,subhaloID=sub)["SubhaloPos"]
        old_coordinates,center = it.periodicfix(gas["Coordinates"], centerGal)
        old_coordinates = (it.convertlength(old_coordinates)*1000*a2).T
        center = (it.convertlength(center)*1000*a2)
        xg,yg,zg = old_coordinates[0] - center[0], old_coordinates[1] - center[1], old_coordinates[2] - center[2]
        del old_coordinates, center

        rhog = it.convertdensity(gas["Density"])/((1000*a2)**3)

        #gas particle metallicity (not in solar units)
        #This will actually become the dust abundance
        Zg = gas["GFM_Metallicity"]

        #gas temperature in K
        T = it.temp(gas["InternalEnergy"],gas["ElectronAbundance"])

        #SFR of gas cell
        SFRg = gas["StarFormationRate"]

        #Calculate the dust abunndance
        #set density to zero where Temperature is above threshold value and where there is no SFR
        #Dust properties
        Tthreshold = 75000 #define temperature threshold for dust formation
        Zg[((T > Tthreshold) & (SFRg == 0))] = 0.0
        DTM = it.DTM(Zg/0.014) #gas phase metallicity in solar units
        dustAbundance = Zg * DTM 
        # set lower bound for dust abundance
        dustAbundance[np.isnan(dustAbundance)]=0.0
        del Zg, T, SFRg
    else:
        xg,yg,zg,rhog,dustAbundance = [],[],[],[],[]

    ncells = len(xg)

    #create 2D-array and write into data file
    totalg = np.column_stack((xg,yg,zg,rhog,dustAbundance))
    np.savetxt(f'{tmp_path}/{f_gas}',totalg,delimiter = " ")

    return fovsize,npix,ncells,f_stars,f_mappings,f_gas

def skirt_run(snap,sub,sim_tag='TNG50-1',tmp_path=None):
    
    snap,sub = int(snap),int(sub)
    
    ppath = '/home/bottrell/projects/def-simardl/bottrell/Simulations/IllustrisTNG'
    spath = f'{ppath}/Scripts/SKIRT'
    ski_in = f'{spath}/shalo.ski'
    opath = f'{ppath}/{sim_tag}/postprocessing/SKIRT/{snap:03d}'

    if not tmp_path:
        environ = os.environ
        if 'SLURM_TMPDIR' in environ:
            tmp_path=environ['SLURM_TMPDIR']
        else:
            # run locally in temprorary directory
            tmp_path = f'./tmp_{snap:03d}-{sub}'
            os.system(f'mkdir -p {tmp_path}')
    else:
        os.system(f'mkdir -p {tmp_path}')
    os.chdir(tmp_path)
            
    # prepare skirt output and place in tmpdir
    ski_out = f'FineGrid_{snap:03d}-{sub}.ski'
    
    f_wavelengths = 'FineWavelength_Grid.dat'
    os.system(f'cp {spath}/{f_wavelengths} {tmp_path}/{f_wavelengths}')
    
    fovsize,npix,ncells,f_stars,f_mappings,f_gas = skirt_requirements(snap,sub,tmp_path)
    
    instrument_names = ['v0','v1','v2','v3']
    incls = [109.5,109.5,109.5,0.]
    azims = [-30.,30.,90.,0.]
    
    environ = os.environ
    if 'SLURM_NTASKS' in environ:
        nthreads = int(environ['SLURM_CPUS_PER_TASK'])
    else:
        nproc = 1
        nthreads = 1
    
    for instrument_name,incl,azim in zip(instrument_names,incls,azims):
        
        with open(ski_in,'r') as f:
            filedata = f.read()
    
        # Replace the target string
        filedata = re.sub('_FILENAME_STARS_',f_stars,filedata)
        filedata = re.sub('_FILENAME_MAPPINGS_',f_mappings,filedata)
        filedata = re.sub('_FILENAME_GAS_',f_gas,filedata)
        filedata = re.sub('_FILENAME_WAVELENGTHS_',f_wavelengths,filedata)
        filedata = re.sub('_FOVSIZE_',str(fovsize),filedata)
        filedata = re.sub('_PIXELNUM_',str(npix),filedata)
        filedata = re.sub('_SIZE_',str(fovsize/2.),filedata)
        filedata = re.sub('_NUMCELLS_',str(ncells),filedata)
        filedata = re.sub('_INSTRUMENT_NAME_',instrument_name,filedata)
        filedata = re.sub('_INCLINATION_',str(incl),filedata)
        filedata = re.sub('_AZIMUTH_',str(azim),filedata)

        with open(ski_out, 'w') as f:
            f.write(filedata)

        os.system(f'srun skirt -t {nthreads} {ski_out}')
        os.system(f'mkdir -p {opath}')
        os.system(f'cp {ski_out.replace(".ski","_"+instrument_name+"_total.fits")} {opath}')
        #os.system(f'rm -rf {tmp_path}')
    
if __name__=='__main__':
    
    snap = 51
    sub = 4

    tmp_base=f'/scratch/bottrell/tmpdir'
    tmp_path=f'{tmp_base}/tmp_{snap:03d}-{sub}'
    skirt_run(snap,sub,tmp_path=tmp_path)