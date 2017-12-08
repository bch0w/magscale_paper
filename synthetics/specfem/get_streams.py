import os
import glob
import pyasdf
import obspy

path = '/import/como-data/bchow/specfem_data/'
outpath = '/home/bchow/Documents/post-grad/magscale_paper/synthetics/specfem/output/'
filelist = glob.glob(path+'*')

for file in filelist:
    basefile = os.path.basename(file)
    ds = pyasdf.ASDFDataSet(file)
    rlas = ds.waveforms.BW_RLAS.synthetic
    fur = ds.waveforms.GR_FUR.synthetic
    
    rlas_name_out = os.path.join(outpath,basefile+'_RLAS.mseed')
    fur_name_out = os.path.join(outpath,basefile+'_FUR.mseed')

    rlas.write(rlas_name_out,format='MSEED')
    fur.write(fur_name_out,format='MSEED')





