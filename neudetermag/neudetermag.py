"""11.09.17 PAPER VERSION LMU 
extremely paired down version of determag
"""
import os
import sys
import glob
import numpy as np

from obspy import read, read_events, Catalog, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics.base import gps2dist_azimuth, locations2degrees
from xml.dom.minidom import parseString
from collections import OrderedDict

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def __pretty_grids(input_ax):
    """grid formatting
    """
    input_ax.set_axisbelow(True)
    input_ax.tick_params(which='major',
                         direction='in',
                         top=True,
                         right=True,
                         width=1,
                         length=3.5)
    input_ax.tick_params(which='minor',
                          direction='in',
                          top=True,
                          right=True,
                          width=0.25,
                          length=1)
    input_ax.minorticks_on()
    input_ax.grid(which='minor',
                    linestyle=':',
                    linewidth='0.5',
                    color='k',
                    alpha=0.25)
    input_ax.grid(which='major',
                    linestyle='-',
                    linewidth='0.5',
                    color='k',
                    alpha=0.15)


def event_info_data(event):
    """set all event information
    event: obspy event 
    """
    origin = event.origins[0]
    event_info_dict = {'lat':origin.latitude,
                       'lon':origin.longitude,
                       'starttime':origin.time,
                       'depth':origin.depth * 0.001}

    # set station and channel information
    pfo_code = "II.PFO.00.BHZ"
    wet_code = "GR.WET..BHZ"
    station_code = wet_code
    net,sta,loc,cha = station_code.split('.')
    
    pfo_lon = -116.456
    pfo_lat = 33.611
    wet_lat = 12.8782
    wet_lon = 49.144001
    lat,lon = wet_lat,wet_lon
    

    # broadband station signal
    st = download_data(origin_time=event_info_dict['starttime'],
                       net=net, 
                       sta=sta, 
                       loc=loc,
                       cha=cha)
    if not st:
        return (None, None, None)

    # theoretical event backazimuth and distance
    baz = gps2dist_azimuth(event_info_dict['lat'],event_info_dict['lon'],
                                                                lat,lon)
    for tr in st:
        tr.stats.back_azimuth = baz
    
    # great circle distance
    gcdist = locations2degrees(event_info_dict['lat'],event_info_dict['lon'],
                                                                lat,lon)

    return st, event_info_dict, gcdist
    
def download_data(origin_time, net, sta, loc, cha):
    """download data from FDSN
    """
    try:
        c = Client("BGR")
        st = c.get_waveforms(network=net, 
                             station=sta, 
                             location=loc,  
                             channel=cha, 
                             starttime=origin_time-180,
                             endtime=origin_time+3*3600,
                             attach_response=True)
    except:
        print("\tFailed")
        return None
        
    return st

def preprocess(st,bounds):
    """preprocess waveform data:
    resample, demean, detrend, taper, remv. resp. (if applicable)
    """

    st_manipulate = st.copy()
    st_manipulate.detrend("demean")
    st_manipulate.detrend("linear")
    st_manipulate.taper(max_percentage=0.05)
    st_manipulate.remove_response(output='VEL',
                                  # pre_filt=pre_filt,
                                  water_level=60,
                                  plot=False)
    st_manipulate.filter('bandpass',freqmin=bounds[0],freqmax=bounds[1])
    st_manipulate.taper(max_percentage=0.05)

                                  
    return st_manipulate

def process_save(event):
    """main processing
    """
    st_original, event_info_dict, gcdist = event_info_data(event)
    if not st_original:
        return (None,None,None,None)
    st = preprocess(st_original,bounds=[1/60,1/3])

    # set sampling rate and copy traces for filtering, rotate to NEZ->TRZ
    sampling_rate = int(st[0].stats.sampling_rate) 
    search = int(20 * sampling_rate)

    AT = st.select(component='Z')[0]

    # choosing amplitudes/indices/zerocrossing for each trace
    peak = max(AT.data)
    trough = min(AT.data)
    peak_ind = np.where(AT.data == peak)[0][0]
    trough_ind = np.where(AT.data == trough)[0][0]
    
    # look for adjacent peaks and troughs
    adj_trough = min(AT.data[peak_ind-search:peak_ind+search])
    adj_peak = max(AT.data[trough_ind-search:trough_ind+search])
    adj_trough_ind = np.where(AT.data == adj_trough)[0][0]
    adj_peak_ind = np.where(AT.data == adj_peak)[0][0]

    # determine peak-to-trough values
    p2at = peak - adj_trough
    ap2t = adj_peak - trough

    if p2at >= ap2t:
        peak2troughs = (0.5*p2at)
        if peak_ind < adj_trough_ind:
            peak_to_peak = AT.data[peak_ind:adj_trough_ind]
            zero_ind = np.where(np.diff(np.signbit(peak_to_peak)))[0][0]
            zero_crossing = (peak_ind + zero_ind) 
        else:
            peak_to_peak = AT.data[adj_trough_ind:peak_ind]
            zero_ind = np.where(np.diff(np.signbit(peak_to_peak)))[0][0]
            zero_crossing = (adj_trough_ind + zero_ind)
    else:
        peak2troughs = (0.5*ap2t)
        periods = (2*abs(adj_peak_ind-trough_ind)/sampling_rate)
        if adj_peak_ind < trough_ind:
            peak_to_peak = AT.data[adj_peak_ind:trough_ind]
            zero_ind = np.where(np.diff(np.signbit(peak_to_peak)))[0][0]
            zero_crossing = (adj_peak_ind + zero_ind) 
        else:
            peak_to_peak = AT.data[trough_ind:adj_peak_ind]
            zero_ind = np.where(np.diff(np.signbit(peak_to_peak)))[0][0]
            zero_crossing = (trough_ind + zero_ind) 

    return st, gcdist, peak2troughs, zero_crossing

def quickplot(st,event,peak2trough,zero_crossing):
    import matplotlib.pyplot as plt
    f = plt.figure(1,figsize=(10,5),dpi=100)
    ax = plt.subplot(111)
    t = np.linspace(0,len(st[0].data),len(st[0].data))
    ax.plot(t,st[0].data,'k')
    ax.scatter(zero_crossing,peak2trough,c='r',marker='o',zorder=5)
    __pretty_grids(ax)
    plt.title('BHZ {} {}'.format(event.origins[0].time,event.magnitudes[0].mag))
    plt.xlabel('Samples')
    plt.ylabel('Velocity ($\mu$ m s$^{-1}$)')
    title = str(event.origins[0].resource_id).split('=')[-1]
    plt.savefig('/Users/chowbr/Documents/magscale_paper/neudetermag/{}.png'.format(title))

    plt.close()
            
def runthrough():
    event_list = glob.glob(
                '/Users/chowbr/Documents/magscale_paper/output/fur/xmls/*.xml')
    starttimes,magnitudes,amplitudes,distances = [],[],[],[]
    for event_path in event_list:
        try:
            cat = read_events(event_path)
            st,dist,A,Z = process_save(cat[0])
            if not st:
                continue
            quickplot(st,cat[0],A,Z)
            amplitudes.append(A)
            distances.append(dist)
            starttimes.append(cat[0].origins[0].time)
            magnitudes.append(cat[0].magnitudes[0].mag)
        except KeyboardInterrupt:
            print('keyboard interrupt')
            sys.exit()
        except Exception as e:
            print('exception ',e)
            continue
    import ipdb;ipdb.set_trace()
    save_dict = {'starttimes':starttimes,'magnitudes':magnitudes,
                'amplitudes':amplitudes,'distances':distances}
    np.savez('/Users/chowbr/Documents/magscale_paper/neudetermag/WETlist',
                                                                **save_dict)
    
if __name__ == "__main__":
    runthrough()
