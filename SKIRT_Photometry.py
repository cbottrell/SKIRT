import os,sys,re,string,time,glob
import numpy as np
import scipy.spatial as spt
from astropy.io import fits
import h5py
import illustris_tools as it
import illustris_python as il

def get_subhalos(basePath,snap,mstar_lower=0,mstar_upper=np.inf):
    '''
    mstar_lower and mstar_upper have log10(Mstar/Msun) units and are physical (i.e. Msun, not Msun/h).
    '''
    from astropy.cosmology import Planck15 as cosmo
    little_h = cosmo.H0.value/100.
    
    ptNumStars = il.snapshot.partTypeNum('stars') 
    fields = ['SubhaloMassType','SubhaloFlag']
    subs = il.groupcat.loadSubhalos(basePath,snap,fields=fields)

    mstar = subs['SubhaloMassType'][:,ptNumStars]
    flags = subs['SubhaloFlag']
    subs = np.arange(subs['count'],dtype=int)
    
    # convert to units used by TNG (1e10 Msun/h)
    mstar_lower = 10**(mstar_lower)/1e10*little_h
    mstar_upper = 10**(mstar_upper)/1e10*little_h
    subs = subs[(flags!=0)*(mstar>=mstar_lower)*(mstar<=mstar_upper)]
    return subs,mstar[subs]

def apply_filter(data_cube,wavelengths,filter_data,redshift=0.0,airmass=0.):
    '''
    Apply filter transmission function to a rest-frame SKIRT data cube which is redshifted 
    by `redshift`. Airmass is currently a place-holder. Output image is in AB mag/arcsec2.
    '''
    # useful constants / quantities
    speed_of_light = 2.998e8 # [m/s]
    speed_of_light = speed_of_light*1e10 # [Angstrom/s]
    
    # convert data cube from [W/m2/micron/arcsec2] to [Jy*Hz/Angstrom/arcsec2]
    data_cube=data_cube*1e22

    # wavelengths of datacube [microns] (expand by redshift factor)
    wl=wavelengths*1e4*(1+redshift) # [Angstrom]

    # filter data
    filter_wl = filter_data[:,0]
    filter_res = filter_data[:,1]

    # filter response interpolated onto image wavelength grid
    filter_res = np.interp(xp=filter_wl,x=wl,fp=filter_res,left=0,right=0) # unitless

    # filter-specific pivot wavelength squared [Angstrom2]
    wl_pivot2 = np.trapz(filter_res*wl,wl)/np.trapz(filter_res/wl,wl)
    
    # now the mean photon rate density in the filter [Jy*Hz/Angstrom/arcsec2]
    f_wl = np.trapz((wl*filter_res).reshape(-1,1,1)*data_cube,wl,axis=0)/np.trapz(wl*filter_res,wl)

    # multiplying by wl_pivot2/speed_of_light gives [Jy/arcsec2]
    # convert to maggies/arcsec2 (w/ AB zeropoint) using 1 maggy ~ 3631 Jy
    # apply (1+z)**-5 redshift degradation to wavelength specific intensities
    f_mgys = f_wl*wl_pivot2/speed_of_light/3631.*(1+redshift)**-5 # [maggies/arcsec2]

    # convert to mag/arcsec2 surface brightness for numerical ease
    with np.errstate(divide='ignore'):
        f_sb = -2.5*np.log10(f_mgys) # [mag/arcsec2] AB system
        f_sb[f_sb==np.inf]=99.

    return f_sb


def photometry(cube_dir,snap,sub,cam,sim_tag,skirt_path,sim_path,out_path,
               filters=['CFHT_MegaCam.u','CFHT_MegaCam.r',
                        'Subaru_HSC.g','Subaru_HSC.r','Subaru_HSC.i','Subaru_HSC.z','Subaru_HSC.Y',
                        'UKIRT_UKIDSS.J','UKIRT_UKIDSS.H','UKIRT_UKIDSS.K',
                        'Spitzer_IRAC.I1','Spitzer_IRAC.I2','Spitzer_IRAC.I3','Spitzer_IRAC.I4']):
    '''
    Using an existing datacube, compute photometry for snap,sub,cam in the provided filters.
    An image is only created if the redshifted spectrum of the galaxy completely covers the 
    filter transmission function (i.e. like a boxcar).
    '''

    filter_path = f'{skirt_path}/Photometry/Filters'
    if not os.access(out_path,0):
        os.system(f'mkdir -p {out_path}')
        
    redshift = il.groupcat.loadHeader(sim_path,snap)['Redshift']
    file_name = f'{cube_dir}/shalo_{snap:03}-{sub}_{cam}_total.fits'
    photo_name = f'{out_path}/shalo_{snap:03}-{sub}_{cam}_photo.fits'

    if not os.access(file_name,0):
        print(f'Data cube file: {file_name} not found. Skipping...')
        return
    # read IFU cube header and data
    with fits.open(file_name,mode='readonly') as hdul:
        # IFU header
        header = hdul[0].header
        # image spectral flux density in [W/m2/micron/arcsec2]
        data_cube = hdul[0].data
    
    wavelengths = np.loadtxt(f'{skirt_path}/Wavelength_Grid.dat',skiprows=1).astype(float)
    
    hdul = fits.HDUList()
    for i,filter_name in enumerate(filters):
        filter_data = np.loadtxt(f'{filter_path}/{filter_name}.dat')
        if not (filter_data[-1,0])<=(wavelengths[-1]*1e4*(1+redshift)):
            continue
        image = apply_filter(data_cube,wavelengths,filter_data,redshift=redshift)
        hdu = fits.ImageHDU(image.astype('float32'),name=filter_name)
        hdr = hdu.header
        hdr['SIMTAG'] = (sim_tag,'Simulation name')
        hdr['SNAPNUM'] = (snap,'Simulation snapshot')
        hdr['SUBHALO'] = (sub,'Subhalo ID')
        hdr['ORIGIN'] = ('SKIRT 8 Simulation','Image origin')
        hdr['REDSHIFT'] = (float(f'{redshift:.4f}'),'Redshift')
        hdr['FILTER'] = (filter_name,'Filter name')
        hdr['BUNIT'] = ('AB mag/arcsec2','Image units')
        hdr['CRPIX1'] = (header['CRPIX1'], 'X-axis coordinate system reference pixel')
        hdr['CRVAL1'] = (header['CRVAL1'], 'Coordinate system value at X-axis reference pix')
        hdr['CDELT1'] = (header['CDELT1'], 'Coordinate increment along X-axis')
        hdr['CTYPE1'] = (header['CTYPE1'], 'Physical units of the X-axis increment')
        hdr['CRPIX2'] = (header['CRPIX2'], 'Y-axis coordinate system reference pixel')
        hdr['CRVAL2'] = (header['CRVAL2'], 'Coordinate system value at Y-axis reference pix')
        hdr['CDELT2'] = (header['CDELT2'], 'Coordinate increment along Y-axis')
        hdr['CTYPE2'] = (header['CTYPE2'], 'Physical units of the Y-axis increment')
        hdul.append(hdu)
        
    hdul.writeto(photo_name,overwrite=True) 
    
    
    
def create_wl_grid(filename='Wavelength_Grid.dat'):
    wl = np.linspace(0.1,1.,181,endpoint=True)
    wl = np.append(wl,np.linspace(1.,5.,21)[1:])
    np.savetxt(filename,wl,header=str(len(wl)),comments='')
    
    
def nn_dist(args):
    smoothlength,point,tree = args
    return tree.query(point,k=smoothlength)[0][-1]


def prepare_skirt(snap,sub,sim_path,tmp_path,
                  pixelsize=100.,fovsize_hmr=15.,
                  smoothlength=32,fof_factor=3):
    '''
    pixel_size: output pixel size in [pc]
    fov_hmr: size of fov in stellar half mass radius units
    smooth_length: kernel smoothing length of stellar particles (nn distance)
    fof_factor: factor by which fovsize_hmr is multiplied to determine which FOF halo particles are used
    '''

    #load current scale factor
    header = dict(h5py.File(f'{sim_path}/snapdir_{snap:03d}/snap_{snap:03d}.0.hdf5',"r")["Header"].attrs.items())
    a2 = header["Time"]

    sub_cat = il.groupcat.loadSingle(sim_path,snap,subhaloID=sub)
    # GPM position and parent FOF halo of subhalo
    group_idx = sub_cat['SubhaloGrNr']
    centerGal = sub_cat['SubhaloPos']
    
    fovsize_ckpch = fovsize_hmr*sub_cat["SubhaloHalfmassRadType"][4] # ckpc/h
    fovsize = it.convertlength(fovsize_ckpch)*a2*1000 # pc
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
    if not (os.access(f'{tmp_path}/{f_stars}',0) or os.access(f'{tmp_path}/{f_mappings}',0)):
        fields = ["Coordinates","GFM_StellarFormationTime","GFM_InitialMass",
                  "GFM_Metallicity", "Velocities", "Masses"]
        #load star particles
        stars = il.snapshot.loadHalo(sim_path,snap,group_idx,4,fields)
        coords,cntr = it.periodicfix(stars['Coordinates'],centerGal) # ckpc/h
        coords = it.convertlength(coords)*a2*1000 # pc
        cntr = it.convertlength(cntr)*a2*1000 # pc

        #ignore all wind particles 
        star_idx = stars['GFM_StellarFormationTime'] > 0
        #include only particles within certain distance of target galaxy
        star_idx *= np.prod(np.abs(coords-cntr)<=fovsize*fof_factor,axis=1,dtype=bool)
        #coords of fof stars within fovsize*fof_factor from SubhaloPos
        coords = coords[star_idx]-cntr

        #smoothing length in proper pc (distance to smoothlength closest neighbour)
        #create array with locations
        #setup KDTree
        tree = spt.cKDTree(coords,leafsize = 100)
        environ = os.environ
        #nthreads = int(environ['SKIRT_CPUS_PER_TASK']) if 'SKIRT_CPUS_PER_TASK' in environ else 1
        nthreads = 52
        if nthreads>1:
            from multiprocessing import Pool
            argList = [(smoothlength,point,tree) for point in coords]
            with Pool(nthreads) as pool:
                h = np.asarray(pool.map(nn_dist,argList))
        else:
            h = np.zeros(len(coords))
            for i,point in enumerate(coords):
                dist = tree.query(point,k=smoothlength)[0][-1]
                h[i] = dist
            h = np.asarray(h)

        #initial mass in solar masses
        M = it.convertmass(stars["GFM_InitialMass"][star_idx])
        #metallicity (actual metallicity, not in solar units)
        Z = stars["GFM_Metallicity"][star_idx]
        #age of the stars in years
        if nthreads>1:
            from multiprocessing import Pool
            argList = [(a1,a2) for a1 in stars["GFM_StellarFormationTime"][star_idx]]
            with Pool(nthreads) as pool:
                t = np.asarray(pool.starmap(it.age,argList))
        else:
            t = it.age(stars["GFM_StellarFormationTime"][star_idx],a2)
        #parameters for all stars
        total_stars = np.column_stack((coords,h,M,Z,t))

        ### Old stars ####
        age_idx = t>1e7 #only consider stars older than 10Myr for bruzual charlot spectra
        np.savetxt(f'{tmp_path}/{f_stars}',total_stars[age_idx],delimiter = " ")

        ### Young stars ###
        age_idx = t<=1e7 #only consider stars younger than 10Myr for MappingsIII spectra
        #calculate SFR (assuming SFR has been constant in the last 10Myr)
        SFRm = total_stars[age_idx,4]/1e7
        #for compactness, take typical value of 5:
        logC = np.full(len(SFRm),5)
        #for ISM pressure, take typical value log(P/kB / cm^-3K) = 5, P = 1.38e-12 N/m2
        pISM = np.full(len(SFRm),1.38e-12)
        #for PDR covering fraction, take f = 0.2
        fPDR = np.full(len(SFRm),0.2)
        #create 2D-array and write into data file
        young_stars = np.column_stack((coords[age_idx],h[age_idx],
                                       SFRm,Z[age_idx],logC,pISM,fPDR))
        np.savetxt(f'{tmp_path}/{f_mappings}',young_stars,delimiter = " ")
    else:
        #load young star properties for gas/dust calculations
        young_stars = np.loadtxt(f'{tmp_path}/{f_mappings}')

    #############
    #### Gas ####
    #############
    fields = ["Coordinates","Density","GFM_Metallicity","InternalEnergy","ElectronAbundance","StarFormationRate","Masses"]
    #load gas particles
    gas = il.snapshot.loadHalo(sim_path,snap,group_idx,0,fields)
    # allow possibility of no gas
    if gas["count"] != 0:
        coords,cntr = it.periodicfix(gas['Coordinates'],centerGal) # ckpc/h
        coords = it.convertlength(coords)*a2*1000 # pc
        cntr = it.convertlength(cntr)*a2*1000 # pc
        #include only particles within certain distance of target galaxy
        gas_idx = np.prod(np.abs(coords-cntr)<=fovsize*fof_factor,axis=1,dtype=bool)
        # coords of fof gas within fovsize*fof_factor from SubhaloPos
        coords = coords[gas_idx]-cntr
        # density
        rhog = it.convertdensity(gas["Density"][gas_idx])/((1000*a2)**3)
        #gas particle metallicity (not in solar units)
        #This will actually become the dust abundance
        Zg = gas["GFM_Metallicity"][gas_idx]
        #gas temperature in K
        T = it.temp(gas["InternalEnergy"][gas_idx],gas["ElectronAbundance"][gas_idx])
        #SFR of gas cell
        SFRg = gas["StarFormationRate"][gas_idx]
        #metal mass of each gas cells
        metal_mass = Zg*it.convertmass(gas["Masses"][gas_idx])
        #Correct for double accounting of dust in PDR regions
        #Calculate mass of metals in the PDR and take this from the stellar particles within the smoothing length
        for young_star in young_stars:
            #assume metallicity of star equals metallicity of parent PDR
            #factor of 10 assumes Mpdr = 10 x Mstar
            metal_mass_PDR = 10 * young_star[5] * young_star[4]*1e7                  
            metal_mass_excess, n_grow = -1, 1.
            #grow N until the enclosed dust mass in h*N exceeds PDR mass
            while metal_mass_excess < 0:
                loc_idx = ( np.sum((coords-young_star[:3])**2,axis=1) < (n_grow*young_star[3])**2 )
                #Metal mass in the gas cells within that smoothing length
                loc_metal_mass = np.sum(metal_mass[loc_idx])
                #Metal mass left over in the gas cells after subtracting pdr contribution
                metal_mass_excess = loc_metal_mass - metal_mass_PDR
                n_grow *= 2.
            norm = loc_metal_mass/metal_mass_excess
            Zg[loc_idx] = Zg[loc_idx]/norm
        #Calculate the dust abunndance
        #Set density to zero where Temperature is above threshold value and where there is no SFR
        #Dust properties
        Tthreshold = 75000 #define temperature threshold for dust formation
        Zg[((T > Tthreshold) & (SFRg == 0))] = 0.0
        DTM = it.DTM(Zg/0.014) #gas phase metallicity in solar units
        dustAbundance = Zg * DTM 
        # set lower bound for dust abundance
        dustAbundance[np.isnan(dustAbundance)]=0.0
    else:
        coords,rhog,dustAbundance = [[],[],[]],[],[]

    ncells = len(coords)
    #create 2D-array and write into data file
    total_gas = np.column_stack((coords,rhog,dustAbundance))
    np.savetxt(f'{tmp_path}/{f_gas}',total_gas,delimiter = " ")

    return fovsize,npix,ncells,f_stars,f_mappings,f_gas


def run_skirt(snap,sub,sim_tag,sim_path,tmp_path,skirt_path):
    '''
    sim_path: base path to simulation snapshot data
    tmp_path: working directory for job 
    skirt_path: path to skirt template files (like shalo.ski, wavelength file)
    
    Paths default to working directory if not specified.
    '''
    
    snap,sub = int(snap),int(sub)
    ski_in = f'{skirt_path}/shalo.ski'

    if not os.access(tmp_path,0):
        os.system(f'mkdir -p {tmp_path}')
    os.chdir(tmp_path)
            
    # prepare skirt output and place in tmpdir
    ski_out = f'shalo_{snap:03d}-{sub}.ski'
    
    f_wavelengths = 'Wavelength_Grid.dat'
    os.system(f'cp {skirt_path}/{f_wavelengths} {tmp_path}/{f_wavelengths}')
    
    fovsize,npix,ncells,f_stars,f_mappings,f_gas = prepare_skirt(snap,sub,sim_path,tmp_path)
    
    instrument_names = ['v0','v1','v2','v3']
    incls = [109.5,109.5,109.5,0.]
    azims = [-30.,30.,90.,0.]
    
    environ = os.environ
    if 'SKIRT_NTASKS' in environ:
        nprocesses = int(environ['SKIRT_NTASKS'])
        nthreads = int(environ['SKIRT_CPUS_PER_TASK'])
    else:
        nprocesses = 1
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

        os.system(f'mpirun -n {nprocesses} skirt -t {nthreads} {ski_out}')
        
        
def spectroscopy(cube_dir,snap,sub,cam,sim_tag,out_path):
    return


if __name__=='__main__':
    
    snap = int(sys.argv[1])
    sim_tag = 'TNG50-1'
    
    # base path where TNG data is stored
    sim_path = f'/lustre/work/connor.bottrell/Simulations/IllustrisTNG/{sim_tag}/output'
    # base path of output directory
    project_path = '/lustre/work/connor.bottrell/Simulations/IllustrisTNG'
    # photometry path
    phot_path = f'{project_path}/{sim_tag}/postprocessing/Photometry/{snap:03}'
    # spectroscopy path
    spec_path = f'{project_path}/{sim_tag}/postprocessing/Spectroscopy/{snap:03}'
    
    subs,mstar = get_subhalos(sim_path,snap=snap,mstar_lower=9)
    sub = subs[int(sys.argv[2])]

#     sub = 4
    
    cams = ['v0','v1','v2','v3']
    # If all files are found, skip. If only some are found, delete and redo.
    outfiles = glob.glob(f'{phot_path}/shalo_{snap:03}-{sub}_*_photo.fits')
    if len(outfiles)!=len(cams):
        for outfile in outfiles:
            if os.access(outfile,0): 
                os.remove(outfile)
    else:
        sys.exit(f'All output files already exist for {sim_tag}-{snap:03}-{sub}.')
    
        
    # working directory for job
    tmp_path=f'/lustre/work/connor.bottrell/tmpdir/tmp_{snap:03d}-{sub}'
    skirt_path = f'{project_path}/Scripts/SKIRT'
    
    # produce cube
    run_skirt(snap,sub,sim_tag,
             sim_path=sim_path,
             tmp_path=tmp_path,
             skirt_path=skirt_path)

    # photometry and spectroscopy
    for cam in cams:
        photometry(cube_dir=tmp_path,snap=snap,sub=sub,cam=cam,sim_tag=sim_tag,
                   skirt_path=skirt_path,sim_path=sim_path,out_path=phot_path)
    
    # clean up
    os.system(f'rm -rf {tmp_path}')
