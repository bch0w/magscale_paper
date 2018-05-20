"""11.09.17 PAPER VERSION LMU 

Script for FUERSTENFELDBRUCK - All rotation information removed
Waveform comparison code used to determine magnitude scales from rotational and
translation motion, outputs all processed information into JSON, XML and PNG's.
Files tagged as the event information and magnitudes

+variable 'ac' refers only to translation components, used to be acceleration 
and too much work to change all variable names
+ spaces not tabs
+ added an error log
+ removed correlation coefficients
+ add time of max amplitude
"""
from __future__ import division
import os
import sys
import json
import obspy
import heapq
import shutil
import urllib2
import argparse
import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
from obspy.core import read
from obspy.taup import TauPyModel
from obspy import read_events, Catalog
from obspy.core.stream import Stream
from obspy.core.utcdatetime import UTCDateTime
from obspy.clients.fdsn import Client as fdsnClient
from obspy.clients.arclink.client import Client as arclinkClient
from obspy.signal.rotate import rotate_ne_rt
from obspy.signal.cross_correlation import xcorr
from obspy.core.util.attribdict import AttribDict
from obspy.geodetics.base import gps2dist_azimuth, locations2degrees
from xml.dom.minidom import parseString
from collections import OrderedDict

mpl.rcParams.update({'font.size': 6.5})


def download_data(origin_time, net, sta, loc, chan):
    """^
    Downloads the data from seismic stations for the desired event(s).
    Direct data fetching from LMU servers

    :type origin_time: :class: `~obspy.core.utcdatetime.UTCDateTime`
    :param origin_time: origin time of the event.
    :type net: str
    :param net: Network code, e.g. ``'BW'``.
    :type sta: str
    :param sta: Station code, e.g. ``'WET'``.
    :type loc: str
    :param loc: Location code, e.g. ``'01'``. Location code may
        contain wild cards.
    :type chan: str
    :param chan: Channel code, e.g. ``'EHE'``. Channel code may
        contain wild cards.
    :return: Stream object :class: `~obspy.core.stream.Stream`
    """
    
    dataDir_get = '/import/netapp-m-02-bay200/mseed_online/archive/'
    
    fileName = ".".join((net, sta, "." + chan + ".D",
                             origin_time.strftime("%Y.%j")))
    filePath = os.path.join(dataDir_get, origin_time.strftime("%Y"),
                                net, sta, chan + '.D', fileName)
    o_time2 = origin_time + 86400
    fileName2 = ".".join((net, sta, "." + chan + ".D",
                             o_time2.strftime("%Y.%j")))
    filePath2 = os.path.join(dataDir_get, o_time2.strftime("%Y"),
                                net, sta, chan + '.D', fileName2)

    if os.path.isfile(filePath):
        if origin_time.hour > 21:
            st = Stream()
            st.extend(read(filePath, starttime = origin_time - 180,
                      endtime = origin_time + 3 * 3600))
            st.extend(read(filePath2, 
                      starttime = UTCDateTime(o_time2.year, o_time2.month, 
                                                            o_time2.day, 0, 0),
                      endtime = origin_time + 3 * 3600))
            st.merge(method=-1)
        else:
            st = read(filePath, starttime = origin_time - 180,
                      endtime = origin_time + 3 * 3600)
    else:
        print "++++ cannot find the following file: \n %s \n++++" % filePath

    if not st:
        raise RotationalProcessingException('Data not available for this'
                                                ' event...')
    st.trim(starttime=origin_time-180, endtime=origin_time+3*3600)

    print 'Download of', st[0].stats.station, st[0].stats.channel, \
        'data successful!'

    return st


def event_info_data(event, station):
    """
    Extracts information from the event and generates variables containing
    the event latitude, longitude, depth, and origin time.
    Ringlaser (RLAS) and broadband signals (WET) are received from the
    download_data function.
    The great circle distance (in m and deg) between event location and station
    in Wetzell, as well as the theoretical backazimuth are computed.

    :type event: :class: `~obspy.core.event.Event`
    :param event: Contains the event information.
    :type station: str
    :param station: Station from which data are fetched ('WET' or 'PFO').
    :rtype latter: float
    :return latter: Latitude of the event in degrees.
    :rtype lonter: float
    :return lonter: Longitude of the event in degrees.
    :rtype depth: float
    :return depth: Hypocenter depth in km
    :type startev: :class: `~obspy.core.utcdatetime.UTCDateTime`
    :return startev: Origin time of the event.
    :rtype ac: :class: `~obspy.core.stream.Stream`
    :return ac: Three component broadband station signal.
    :rtype baz: tuple
    :return baz: [0] great circle distance in m, [1] theoretical azimuth,
        [2] theoretical backazimuth.
    :rtype gcdist: float
    :return gcdist: Great circle distance in degrees.

    """
    origin = event.preferred_origin() or event.origins[0]
    latter = origin.latitude
    lonter = origin.longitude
    startev = origin.time
    depth = origin.depth * 0.001

    # set station and channel information
    if station == 'FUR':
        net_s = 'GR'
        sta_s = 'FUR'
        loc_s = ''
        chan2 = 'BHE'
        chan3 = 'BHN'
        chan4 = 'BHZ'

        # broadband station signal
        acE = download_data(startev, net_s, sta_s, loc_s, chan2)
        acN = download_data(startev, net_s, sta_s, loc_s, chan3)
        acZ = download_data(startev, net_s, sta_s, loc_s, chan4)
        ac = Stream(traces=[acE[0], acN[0], acZ[0]])

        for ca in [ac[0], ac[1], ac[2]]:
            ca.stats.coordinates = AttribDict()
            ca.stats.coordinates['longitude'] = 11.275
            ca.stats.coordinates['latitude'] = 48.163
            ca.stats['starttime'] = startev - 180
            ca.stats['sampling_rate'] = 20.

        # theoretical event backazimuth and distance
        baz = gps2dist_azimuth(latter, lonter, ac[0].stats.coordinates.latitude,
                              ac[0].stats.coordinates.longitude)
        # great circle distance
        gcdist = locations2degrees(latter, lonter,
                                   ac[0].stats.coordinates.latitude,
                                   ac[0].stats.coordinates.longitude)

    return latter, lonter, depth, startev, ac, baz, gcdist, \
            net_s, chan2, chan3, chan4, sta_s, loc_s


def is_local(baz):
    """
    Checks whether the event is close (< 333.33 km), local (< 1111.1 km) or
    non-local.
    :type baz: tuple
    :param baz: Great circle distance in m, azimuth A->B in degrees,
        azimuth B->A in degrees.
    :rtype: str
    :return: Self-explaining string for event distance.
    """
    if 0.001 * baz[0] / 111.11 < 10.0:
        if 0.001 * baz[0] / 111.11 < 3.0:
            is_local = 'close'
        else:
            is_local = 'local'
    else:
        is_local = 'non-local'

    return is_local


def resample(is_local, baz, ac):
    """
    Resamples signal accordingly with sampling rates and cut-off frequencies
    dependent on the location of the event (5 sec and 2Hz for local events,
    60 sec and 1 Hz for non-local events).

    :type is_local: str
    :param is_local: Self-explaining string for event distance.
    :type baz: tuple
    :param baz: Great circle distance in m, azimuth A->B in degrees,
        azimuth B->A in degrees.
    :type ac: :class: `~obspy.core.stream.Stream`
    :param ac: Three component broadband station signal.
    :rtype ac: :class: `~obspy.core.stream.Stream`
    :return ac: Decimated three component broadband station signal.
    :rtype sec: int
    :return sec: Sampling rate.
    :rtype cutoff: float
    :return cutoff: Cut-off frequency for the lowpass filter.
    :rtype cutoff_pc: float
    :return cutoff_pc: Cut-off frequency for the highpass filter in P-coda.

    """
    if is_local == 'local':
        ac.data = ac.data[0: 1800 * ac[0].stats.sampling_rate]
        ac.decimate(factor=2)
        sec = 5
        cutoff = 2.0  # local events
    elif is_local == 'non-local':
        ac.decimate(factor=4)
        sec = 120
        cutoff = 1.0  # nonlocal events
    else:
        ac.data = trr.data[0: 1800 * ac[0].stats.sampling_rate]
        ac.decimate(factor=2)
        sec = 3
        cutoff = 4.0  # close events
    
    return ac, sec, cutoff


def remove_instr_resp(ac, station, startev):
    """
    Remove instrument response, detrend data, trim if unequal lengths
    sensitivity controls order of magnitude of units

    NOTE:
    + 9.44*10**8 gives m/s, 944.xxx gives um/s, 0.944xxx for nm/s (w/ 2 zeros)
    + 2 zeros velocity, 1 zero acceleration, 3 zeroes displacement

    :type ac: :class: `~obspy.core.stream.Stream`
    :param ac: Three component broadband station signal.
    :type station: str
    :param station: Station from which data are fetched ('WET' or 'PFO').
    :type startev: :class: `~obspy.core.utcdatetime.UTCDateTime`
    :param startev: Origin time of the event.
    :rtype ac: :class: `~obspy.core.stream.Stream`
    :return ac: Detrended and trimmed three component broadband station signal.
    """      
    
    if station == 'FUR':
        ac.detrend(type='linear')
        ac.taper(max_percentage=0.05) 

        paz_sts2_vel = {'poles': [(-0.0367429 + 0.036754j),
                              (-0.0367429 - 0.036754j)],
                        'sensitivity': 0.944019640, 
                        'zeros': [0j,0j], 
                        'gain': 1.0}
       
        ac.simulate(paz_remove=paz_sts2_vel, remove_sensitivity=True)  

    else:
        print 'Incorrect station call'

    return ac


def process_save(ac,baz,cutoff,station,is_local,min_lwi,max_lwf,
                                                folder_name,tag_name,event_ID):
    """
    Filter traces for 10s to 60s bandpass, rotate translation components to 
    radial/transverse coordinate system, determine max peak-to-trough trace 
    amplitudes and quickplot to assess waveform quality
    + Amplitude is given as 1/2 maximum peak to adjacent trough 
    + Associated period is 2x time interval seperating peak and adjacent trough
    *be careful with the copy statement


    :type ac: :class: `~obspy.core.stream.Stream`
    :param ac: Three component broadband station signal.
    :type baz: tuple
    :param baz: Great circle distance in m, azimuth A->B in degrees,
        azimuth B->A in degrees.
    :type cutoff: float
    :param cutoff: Cut-off frequency for the lowpass filter.
    :type station: str
    :param station: Station from which data are fetched ('WET' or 'PFO').
    :type min_lwi/max_lwf: float
    :param min_lwi/max_lwf: theoretical surface wave start and end in sample #
    :type folder_name/tag_name/event_ID: str
    :param folder_name: event folder for storing all figures
    :param tag_name: identifier containing event info for figure filename
    :param event_ID: number identifier corresponding to quakeML event ID #
    :rtype peak2troughs: list of floats
    :return peak2troughs: maximum peak to trough deflection for all traces
    :rtype periods: list of floats
    :return periods: associated periods for peak2trough deflections
    :rtype zero_crossings_abs: list of floats
    :return zero_crossing_abs: associated arrival times of max amplitudes in 
                                seconds from trace start
    """

    # set sampling rate and copy traces for filtering, rotate to NEZ->TRZ
    sampling_rate = int(ac[0].stats.sampling_rate) 
    velocity_nez = ac.copy()
    velocity_rtz = ac.copy()
    velocity_rtz.rotate(method='NE->RT',back_azimuth=baz[2])

    # surface wave train
    #surf_big = int(sampling_rate * min_lwi)
    #surf_end = int(sampling_rate * max_lwf)
    surf_big = 0
    surf_end = len(ac[0].data)-1

    # bandpass all traces for 3s to 60s
    f_start = 1/60
    f_end = 1/3

    for traces in [velocity_rtz,velocity_nez]:
        traces.filter('bandpass', freqmin=f_start, freqmax=f_end, corners=3,
                  zerophase=True)

    # seperate streams into traces and create list for ease of processing 
    z_vel = velocity_nez.select(component='Z')
    n_vel = velocity_nez.select(component='N')
    e_vel = velocity_nez.select(component='E')
    r_vel = velocity_rtz.select(component='R')
    t_vel = velocity_rtz.select(component='T')
    alltraces = [z_vel[0],n_vel[0],e_vel[0],r_vel[0],t_vel[0]]

    # search criteria for adjacent peaks and troughs, 20s period (trial&error)
    search = int(20 * sampling_rate)

    # choosing amplitudes/indices/zerocrossing for each trace
    peak2troughs,periods,zero_crossings = [],[],[]
    plot_lists = [[] for _ in range(11)]
    for AT in alltraces:
        channel = str(AT.stats.channel)

        # determine peaks/troughs and corresponding sample number
        peak = max(AT.data[surf_big:surf_end])
        trough = min(AT.data[surf_big:surf_end])
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
        
        # 1) choose largest p2t, take 0.5*max deflection, 
        # 2) find associated period
        # 3) find zero crossing for arrival time
        # * case by case basis depending on which maximum we take

        if p2at >= ap2t:
            peak2troughs.append(0.5*p2at)
            periods.append(2*abs(peak_ind-adj_trough_ind)/sampling_rate)
            if peak_ind < adj_trough_ind:
                peak_to_peak = AT.data[peak_ind:adj_trough_ind]
                zero_ind = np.where(np.diff(np.signbit(peak_to_peak)))[0][0]
                zero_crossings.append(peak_ind + zero_ind) 
            else:
                peak_to_peak = AT.data[adj_trough_ind:peak_ind]
                zero_ind = np.where(np.diff(np.signbit(peak_to_peak)))[0][0]
                zero_crossings.append(adj_trough_ind + zero_ind)
        else:
            peak2troughs.append(0.5*ap2t)
            periods.append(2*abs(adj_peak_ind-trough_ind)/sampling_rate)
            if adj_peak_ind < trough_ind:
                peak_to_peak = AT.data[adj_peak_ind:trough_ind]
                zero_ind = np.where(np.diff(np.signbit(peak_to_peak)))[0][0]
                zero_crossings.append(adj_peak_ind + zero_ind) 
            else:
                peak_to_peak = AT.data[trough_ind:adj_peak_ind]
                zero_ind = np.where(np.diff(np.signbit(peak_to_peak)))[0][0]
                zero_crossings.append(trough_ind + zero_ind) 
        
        # append parameters to use in plotting all traces together
        plot_params = [peak_ind,peak,adj_trough_ind,adj_trough,adj_peak_ind,
                        adj_peak,trough_ind,trough,channel,p2at,ap2t]
        
        for i in range(len(plot_lists)):
            plot_lists[i].append(plot_params[i])

    # divy up lists
    peak_indS,peakS,adj_trough_indS,adj_troughS,adj_peak_indS,adj_peakS,\
                        trough_indS,troughS,channelS,p2atS,ap2tS = plot_lists[:]

    # change units of zero crossing from samples to seconds from trace start
    zero_crossings_abs = [_/sampling_rate for _ in zero_crossings]

    # PLOT data to see waveform quality and peak-trough behavior
    f,axes = plt.subplots(nrows=5,ncols=2,sharex='col',sharey='row')
    j = 0
    for i in range(5):
        # trace overviews
        axes[i][0].plot(alltraces[j].data,'k')
        axes[i][0].plot(peak_indS[j],peakS[j],'go',adj_trough_indS[j],
                                                        adj_troughS[j],'go')
        axes[i][0].plot(adj_peak_indS[j],adj_peakS[j],'ro',trough_indS[j],
                                                            troughS[j],'ro')
        axes[i][0].plot(zero_crossings[j],0,'bo',zorder=8)
        #axes[i][0].plot((surf_big,surf_big),(peakS[j],troughS[j]),'b')
        #axes[i][0].plot((surf_end,surf_end),(peakS[j],troughS[j]),'b')
        axes[i][0].annotate(p2atS[j], xy=(peak_indS[j],peakS[j]), 
            xytext=(peak_indS[j],peakS[j]),zorder=10,color='g')
        axes[i][0].annotate(ap2tS[j], xy=(trough_indS[j],troughS[j]), 
            xytext=(trough_indS[j],troughS[j]),zorder=10,color='r')
        axes[i][0].grid()    
        axes[i][0].set_ylabel('{} vel.(nm/s)'.format(channelS[j]))

        # zoomed in plot
        axes[i][1].plot(alltraces[j].data,'k')
        axes[i][1].plot(peak_indS[j],peakS[j],'go',adj_trough_indS[j],
                                                        adj_troughS[j],'go')
        axes[i][1].plot(adj_peak_indS[j],adj_peakS[j],'ro',trough_indS[j],
                                                            troughS[j],'ro')
        axes[i][1].plot(zero_crossings[j],0,'bo',zorder=8) 
        axes[i][1].annotate(p2atS[j], xy=(peak_indS[j],peakS[j]), 
            xytext=(peak_indS[j],peakS[j]),zorder=10,color='g')
        axes[i][1].annotate(ap2tS[j], xy=(trough_indS[j],troughS[j]), 
            xytext=(trough_indS[j],troughS[j]),zorder=10,color='r')
        axes[i][1].grid()

        # only label specific axes
        if j == 0:
            axes[i][0].set_title(tag_name)
            axes[i][1].set_title(event_ID)
        elif j == len(channelS)-1:
            axes[i][0].set_xlabel('sample')
            axes[i][1].set_xlabel('sample')
            axes[i][1].set_xlim([min(zero_crossings)-2000,
                                        max(zero_crossings)+2000])
        j+=1

    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0.1)

    plt.savefig('OUTPUT/FUR/imgs/'+'FUR_'+tag_name+'.png',dpi=150)
    plt.savefig(folder_name+'FUR_'+tag_name+'.png',dpi=150)
    #plt.show() 
    plt.close()

    return peak2troughs,periods,zero_crossings_abs
                

def ps_arrival_times(distance, depth, init_sec):
    """
    Obtains the arrival times (in seconds after the start time of the fetched
    data) of the first P an S waves of the event. The inputs are the
    epicentral distance in degrees, the depth in km and the initial time in
    seconds (starttime_of_the_event - data_starttime)

    :type distance: float
    :param distance: Great circle distance between earthquake source and
        receiver station.
    :type depth: float
    :param depth: Hypocenter depth in km.
    :type init_sec: float
    :param init_sec: Initial time of the event in sec in the fetched data.
    :rtype arriv_p: float
    :return arriv_p: Arrival time of the first P-wave.
    :rtype arriv_s: float
    :return arriv_s: Arrival time of the first S-wave.
    """

    # use taup to get the theoretical arrival times for P & S
    TauPy_model = TauPyModel('iasp91')
    tt = TauPy_model.get_travel_times(
        distance_in_degree=0.001 * distance / 111.11, source_depth_in_km=depth)
    tiemp = []
    tiems = []
    # from all possible P arrivals select the earliest one
    for i2 in xrange(0, len(tt)):
        if tt.__getitem__(i2).__dict__['name'] == 'P' or \
                tt.__getitem__(i2).__dict__['name'] == 'p' or \
                tt.__getitem__(i2).__dict__['name'] == 'Pdiff' or \
                tt.__getitem__(i2).__dict__['name'] == 'PKiKP' or\
                tt.__getitem__(i2).__dict__['name'] == 'PKIKP' or \
                tt.__getitem__(i2).__dict__['name'] == 'PP' or \
                tt.__getitem__(i2).__dict__['name'] == 'Pb' or \
                tt.__getitem__(i2).__dict__['name'] == 'Pn' or \
                tt.__getitem__(i2).__dict__['name'] == 'Pg':
                    tiem_p = tt.__getitem__(i2).__dict__['time']
                    tiemp.append(tiem_p)
    arriv_p = np.floor(init_sec + min(tiemp))

    # from all possible S arrivals select the earliest one
    for i3 in xrange(0, len(tt)):
        if tt.__getitem__(i3).__dict__['name'] == 'S' or \
            tt.__getitem__(i3).__dict__['name'] == 's' or \
            tt.__getitem__(i3).__dict__['name'] == 'Sdiff' or \
            tt.__getitem__(i3).__dict__['name'] == 'SKiKS' or \
            tt.__getitem__(i3).__dict__['name'] == 'SKIKS' or \
            tt.__getitem__(i3).__dict__['name'] == 'SS' or \
            tt.__getitem__(i3).__dict__['name'] == 'Sb' or \
            tt.__getitem__(i3).__dict__['name'] == 'Sn' or \
            tt.__getitem__(i3).__dict__['name'] == 'Sg':
                    tiem_s = tt.__getitem__(i3).__dict__['time']
                    tiems.append(tiem_s)
    arriv_s = np.floor(init_sec + min(tiems))
    return arriv_p, arriv_s

def surf_tts(distance, start_time):

    """
    Uses arrival times for different epicentral distances based on the IASP91
    travel times model to estimate a curve of travel times for surface waves
    and get the arrival time of the surface waves of the event. Inputs are the
    epicentral distance in degrees and the time in seconds at which the event
    starts in the fetched data.
    """
    deltas = np.arange(0., 140., 5.)
    tts = 60. * np.array(
        [0., 2., 4., 6.2, 8.4, 11., 13., 15.2, 17.8, 19.4, 22., 24.1, 26.6,
         28.6, 30.8, 33., 35.6, 37.4, 39.8, 42., 44.2, 46.4, 48.8, 50.9, 53.6,
         55.2, 57.8, 60.])
    (mval, nval) = np.polyfit(deltas, tts, 1)
    # calculate surface wave travel times for degrees 1 to 180 ?
    surftts = mval * np.arange(0., 180.1, 0.01)
    difer = []
    for i4 in xrange(0, len(surftts)):
        dife_r = abs(0.001 * distance / 111.11 - np.arange(0., 180.1, 0.01)
                     [i4])
        difer.append(dife_r)
    # love wave arrival: event time + surftts for closest degree??
    # (smallest difference between distance for surftts and actual distance of
    #  event)
    arriv_lov = np.floor(start_time + surftts[np.asarray(difer).argmin()])
    diferans = []
    for i1 in xrange(len(deltas)):
        dif2 = abs(np.arange(0., 180.1, 0.01)[np.asarray(difer).argmin()] -
                   deltas[i1])
        diferans.append(dif2)
    # arrival = love wave arrival - p arrival?
    peq = surftts[np.asarray(difer).argmin()] - \
        tts[np.asarray(diferans).argmin()]
    arrival = arriv_lov + peq

    return arrival


def time_windows(baz, arriv_p, arriv_s, init_sec, is_local):
    """
    Determines time windows for arrivals and subplots for P-waves,
    S-waves, initial and latter surface waves.

    """

    # TIME WINDOWS (for arrivals and subplots)
    # Window lengths dependent on event distance
    if is_local == 'non-local':
        min_pw = arriv_p
        max_pw = min_pw + (arriv_s - arriv_p) // 4
        min_sw = arriv_s - 0.001 * (arriv_s - arriv_p)
        max_sw = arriv_s + 150
        min_lwi = surf_tts(baz[0], init_sec) - 20
        t1 = (baz[0]/1000000) * 50
        # window length grows 50 sec per 1000 km.
        max_lwi = min_lwi + t1
        min_lwf = max_lwi
        t2 = (baz[0]/1000000) * 60
        # window length grows 60 sec per 1000 km.
        max_lwf = min_lwf + t2
    elif is_local == 'local':
        min_pw = arriv_p
        max_pw = min_pw + 20
        min_sw = arriv_s - 5
        max_sw = min_sw + 20
        min_lwi = surf_tts(baz[0], init_sec) + 20
        max_lwi = min_lwi + 50
        min_lwf = max_lwi
        max_lwf = min_lwf + 80
    else:
        min_pw = arriv_p
        max_pw = min_pw + 7
        min_sw = arriv_s
        max_sw = min_sw + 7
        min_lwi = surf_tts(baz[0], init_sec) + 5
        max_lwi = min_lwi + 12
        min_lwf = max_lwi
        max_lwf = min_lwf + 80

    return min_pw, max_pw, min_sw, max_sw, min_lwi, max_lwi, min_lwf, max_lwf


def store_info_json(tag_name, folder_name, ac, baz, peak2troughs, 
                    periods, zero_crossings_abs, station, startev, event,
                    net_s, chan2, chan3, chan4, sta_s,
                    loc_s, event_source, depth, magnitude):
    """
    Generates a human readable .json file that stores data for each event,
    like peak values (acceleration, rotation rate, signal-to-noise ratio, 
    backazimuths.

    :type folder_name/tag_name/event_ID: str
    :param tag_name: identifier containing event info for figure filename
    :param folder_name: event folder for storing all figures
    :type ac: :class: `~obspy.core.stream.Stream`
    :param ac: Three component broadband station signal.
    :type baz: tuple
    :param baz: Great circle distance in m, azimuth A->B in degrees,
        azimuth B->A in degrees.
    :type peak2troughs: list of floats
    :param peak2troughs: maximum peak to trough deflection for all traces
    :type periods: list of floats
    :param periods: associated periods for peak2trough deflections
    :type zero_crossings_abs: list of floats
    :param zero_crossing_abs: associated arrival times of max amplitudes in 
                                seconds from trace start
    :type station: str
    :param station: Station from which data are fetched ('WET' or 'PFO').
    """
    zv_max, nv_max, ev_max, rv_max, tv_max = peak2troughs[:]
    zv_per, nv_per, ev_per, rv_per, tv_per = periods[:]
    zv_zca, nv_zca, ev_zca, rv_zca, tv_zca = zero_crossings_abs[:]
    
    sampl_rate = ac[0].stats.sampling_rate
    TBA = baz[2]  # Theoretical backazimuth [deg]
    distance = 0.001*baz[0]
    dic = OrderedDict([
            ('data', OrderedDict([
                ('translational', OrderedDict([
                    ('network', net_s),
                    ('station', sta_s),
                    ('loc', loc_s),
                    ('channel_N', chan3),
                    ('channel_E', chan2),
                    ('channel_Z', chan4)]))
                ])),
            ('event_id', event.resource_id.id),
            ('event_source', event_source),
            ('starttime', str(startev-180)),
            ('endtime', str(startev+3*3600)),
            ('station_latitude', str(ac[0].stats.coordinates['latitude'])),
            ('station_longitude', str(ac[0].stats.coordinates['longitude'])),
            ('event_latitude', event.preferred_origin().latitude),
            ('event_longitude', event.preferred_origin().longitude),
            ('magnitude', magnitude),
            ('depth', depth),
            ('depth_unit', 'km'),
            ('epicentral_distance', distance),
            ('epicentral_distance_unit', 'km'),
            ('theoretical_backazimuth', TBA),
            ('theoretical_backazimuth_unit', 'degree'),
            ('peak_filtered_vertical_vel', zv_max),
            ('dominant_period_vertical_vel',zv_per),
            ('vertical_vel_zero_crossing',zv_zca),
            ('peak_filtered_north_vel', nv_max),
            ('dominant_period_north_vel',nv_per),
            ('north_vel_zero_crossing',nv_zca),
            ('peak_filtered_east_vel', ev_max),
            ('dominant_period_east_vel',ev_per),
            ('east_vel_zero_crossing',ev_zca),
            ('peak_filtered_radial_vel',rv_max),
            ('dominant_period_radial_vel',rv_per),
            ('radial_vel_zero_crossing',rv_zca),
            ('peak_filtered_transverse_vel',tv_max),
            ('dominant_period_transverse_vel',tv_per),
            ('transverse_vel_zero_crossing',tv_zca),
            ('peak_filtered_vel_unit', 'nm/s'),
            ('zero_crossing_unit','sec. from trace start')
        ])

    outfile = open(folder_name+station+'_'+tag_name+'.json', 'wt')
    json.dump(dic, outfile, indent=4)

    outfile.close()


# MAIN
# set catalog variables
# * could truncate catpath before reading to make faster
catpath = './catalogs/277_reeval_catalog.xml'
cat = read_events(catpath,format='QUAKEML')

station = 'FUR'
catalog= 'GCMT'
event_source = 'IRIS'
cat_start = int(raw_input('cat_start: '))
cat_end = int(raw_input('cat_end: '))

print 'Number of Events: {}'.format(len(cat)-cat_start)   
event_counter = 0
error_counter = 0
error_ID,error_log = [],[]
for event in cat[cat_start:cat_end]:
    try:
        print '____________________________________________________________'
        print str(event).split('\n')[0] 
        print '____________________________________________________________'
        latter,lonter,depth,startev,ac,baz,gcdist,net_s,chan2,chan3,\
                    chan4,sta_s,loc_s = event_info_data(event,station)
        
        ac,sec,cutoff, = resample(is_local(baz),baz,ac)


        if not os.path.exists('OUTPUT/{}'.format(station)):
            print 'Creating folders...'
            os.makedirs('OUTPUT/{}'.format(station))
        else:
            print 'Writing into existing folder...'
        # i.e. 'OUTPUT/FUR/GCMT_2011-03-11T
        #                            547_9.1_OFF_EAST_COAST_OF_HONSHU__JAPAN/'
        folder_name = os.path.join('OUTPUT',station, catalog + '_' 
                                + str(event.origins[0]['time'].date) + 'T' 
                                + str(event.origins[0]['time'].hour)   
                                + str(event.origins[0]['time'].minute) +'_' 
                                + str(event.magnitudes[0]['mag']) + '_' 
                                + str(event.event_descriptions[0]['text'].
                                    splitlines()[0].replace(' ', '_').
                                    replace(',', '_'))
                                + '/')

        # i.e. 'GCMT_2011-03-11T547_9.1_OFF_EAST_COAST_OF_HONSHU__JAPAN'
        tag_name = catalog + '_' + str(event.origins[0]['time'].date) + 'T'\
                                + str(event.origins[0]['time'].hour)\
                                + str(event.origins[0]['time'].minute) +'_'\
                                + str(event.magnitudes[0]['mag']) + '_'\
                                + str(event.event_descriptions[0]['text'].\
                                    splitlines()[0].replace(' ', '_').\
                                    replace(',', '_'))
        
        # event id for plotting and json file
        event_ID = event.resource_id.id[-7:] 
        
        # make event folder and write XML file if necessary
        if os.path.exists(str(folder_name)):
            print 'This event was already processed, overwriting data...'           
        elif not os.path.exists(str(folder_name)):  
            os.makedirs(str(folder_name))
            event.write(folder_name + tag_name+'.xml', format='QUAKEML')
        
        print 'Removing instrument response...'
        ac = remove_instr_resp(ac,station,startev)

        print 'Getting arrival times...'
        init_sec = startev - ac[0].stats.starttime
        # When the event starts in the fetched data
        arriv_p,arriv_s = ps_arrival_times(baz[0],depth,init_sec)

        min_pw,max_pw,min_sw,max_sw,min_lwi,max_lwi,min_lwf,max_lwf =\
                        time_windows(baz,arriv_p,arriv_s,init_sec,is_local(baz))
        
        print 'Processing data and saving figures, mseeds...'
        peak2troughs,periods,zero_crossings_abs = process_save(
                                                ac,baz,cutoff,station,
                                                is_local(baz),min_lwi,max_lwf,
                                                folder_name,tag_name,event_ID)
        
    #        error_check = raw_input('Error Check: ')
    #        if len(error_check) > 0:
    #            error_ID.append(event_ID)
    #            error_log.append(error_check)

        print 'Saving data...'
        store_info_json(tag_name,folder_name,ac,baz,peak2troughs,
                        periods,zero_crossings_abs,station,startev,event,
                        net_s,chan2,chan3,chan4,sta_s,
                        loc_s,event_source,depth,
                        event.magnitudes[0]['mag'])

        event_counter += 1
        print 'Done, {} Events left'.format(
                len(cat)-cat_start-event_counter-error_counter)
    
    except KeyboardInterrupt:
        sys.exit()

    except Exception as e:
       print e
       error_ID.append(event_ID)
       error_log.append(e)
       error_counter += 1
       pass

# create error log
timestamp = UTCDateTime()
H = timestamp.hour
M = timestamp.minute
if len(error_log) != 0:
    with open('./error_logs/{}/errorlog{}-{}_{}{}.txt'.format(station,
                                        cat_start,cat_end,H,M),'w') as f:
        for ii in range(len(error_log)):
            f.write('{}\t{}\n'.format(error_ID[ii],error_log[ii]))

print 'All done, {} events successful, {} events failed'.format(
                                            event_counter, error_counter)
print 'cat_start: {} cat_end: {}'.format(cat_start,cat_end)
#from IPython.core.debugger import Tracer; Tracer(colors="Linux")() 
