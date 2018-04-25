"""get waveforms for magscale paper for comparison against synthetic events 
"""

import os
import glob
from obspy import read
from obspy.clients.fdsn import Client

c_lmu = Client("LMU")
c_bgr = Client("BGR")

event_folder = '/Users/chowbr/Documents/magscale_paper/output/mseeds/'
event_files = glob.glob(event_folder + '*')
prepath = os.path.dirname(event_files[0])
for event in event_files:
    basename = os.path.basename(event).split('_')[-1].split('.')[0]
    st_old = read(event)
    starttime = st_old[0].stats.starttime
    endtime = st_old[0].stats.endtime
    st_RLAS = c_lmu.get_waveforms(network = 'BW',
                                    station = 'RLAS',
                                    location = '*',
                                    channel = 'BJZ',
                                    starttime = starttime,
                                    endtime = endtime,
                                    attach_response = True)
    st_outname = os.path.join(prepath,basename+'_RLAS.pickle') 
    st_RLAS.write(st_outname,format='pickle')                           
    
    st_WET = c_bgr.get_waveforms(network = 'GR',
                                    station = 'WET',
                                    location = '*',
                                    channel = 'HH?',
                                    starttime = starttime,
                                    endtime = endtime,
                                    attach_response = True)
    st_outname = os.path.join(prepath,basename+'_WET.pickle')
    st_WET.write(st_outname,format='pickle')

