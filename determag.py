"""25.09.17 PAPER VERSION LMU 

Script for WETTZELL
Waveform comparison code used to determine magnitude scales from rotational and
translation motion, outputs all processed information into JSON, XML and PNG's.
Files tagged as the event information and magnitudes

+variable 'ac' refers only to translation components, used to be acceleration 
and too much work to change all variable names
+ spaces not tabs
+ added an error log
+ removed correlation coefficients
+ add time of max amplitude
+ 25.09.17 output rotation (integration of rotation rate) - 
    and plot, comment out other plots and create new figure, 
    overwrite .json in output folder
"""

from __future__ import division
import os
import sys
import json
import obspy
import heapq
import shutil
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

def download_data(origin_time, net, sta, loc, chan, source):
    """
    It downloads the data from seismic stations for the desired event(s).
    Inputs are the origin time (UTC), network, station, location and channel
    of the event. Returns a stream object fetched from Arclink. If Arclink
    does not work data is alternatively fetched from Seishub.

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

    :type st: Stream object :class: `~obspy.core.stream.Stream`
    :return st: fetched data stream
    """
    # arclink is deprecated, but call is kept for posterity? rip
    # try:
    #     c = arclinkClient(user='test@obspy.org')
    #     st = c.get_waveforms(network=net, station=sta, location='', 
    #                          channel=chan,
    #                          starttime=origin_time-190,
    #                          endtime=origin_time+3*3600+10)
    
    # check paths to see if running on FFB, LMU or neither

    st = None
    dataDir_get = '/bay200/mseed_online/archive/' #FFB
    if not os.path.exists(dataDir_get):
        dataDir_get = '/import/netapp-m-02-bay200/mseed_online/archive/'#LMU            
    if not os.path.exists(dataDir_get):
        dataDir_get = None
    
    # if data path exists, read in data from file
    if dataDir_get:
        print("Fetching {} data from file".format(net))
        fileName = '.'.join((net, sta, '.' + chan + '.D',
                             origin_time.strftime('%Y.%j')))
        filePath = os.path.join(dataDir_get, origin_time.strftime('%Y'),
                                net, sta, chan + '.D', fileName)
        o_time2 = origin_time + 86400
        fileName2 = '.'.join((net, sta, '.' + chan + '.D',
                             o_time2.strftime('%Y.%j')))
        filePath2 = os.path.join(dataDir_get, o_time2.strftime('%Y'),
                                net, sta, chan + '.D', fileName2)
        if os.path.isfile(filePath):
            if origin_time.hour > 21:
                st = Stream()
                st.extend(read(filePath, starttime = origin_time - 180,
                      endtime = origin_time + 3 * 3600))
                st.extend(read(filePath2, 
                      starttime = UTCDateTime(o_time2.year, 
                                        o_time2.month, o_time2.day, 0, 0),
                      endtime = origin_time + 3 * 3600))
                st.merge(method=-1)
            else:
                st = read(filePath, starttime = origin_time - 180,
                      endtime = origin_time + 3 * 3600)
            data_source = 'Archive'
        else:
            print("\tFile not found: \n\t %s \n" % filePath)    
    
    # if data/path does not exist, try querying FDSN webservices
    elif (not dataDir_get) or (not st):
        for S in source:
            try:
                print("Fetching {} data from FDSN ({})".format(net,S))
                c = fdsnClient(S)
                st = c.get_waveforms(network=net, station=sta, location=loc, 
                                    channel=chan, starttime=origin_time-190,
                                    endtime=origin_time+3*3600+10)
                break
            except:
                print("\tFailed")
                pass
        data_source = S 
    
    if not st:
        sys.exit('Data not available for this event')

    st.trim(starttime=origin_time-180, endtime=origin_time+3*3600)
    print("\tDownload of {!s} {!s} data successful".format(
              st[0].stats.station, st[0].stats.channel))

    
    return st, data_source



def event_info_data(event, station):

    """
    Extracts information from the event and generates variables containing
    the event latitude, longitude, depth, and origin time.
    Ringlaser (RLAS) and broadband signals (WET) are received from the
    download_data function.
    The great circle distance (in m and °) between event location and station
    in Wetzell, as well as the theoretical backazimuth are computed.

    :type event: :class: `~obspy.core.event.Event`
    :param event: Contains the event information.
    :type station: str
    :param station: Station from which data are fetched (i.e. 'RLAS').
    :type mode: str
    :param mode: Defines where data fetched from
    :rtype latter: float
    :return latter: Latitude of the event in degrees.
    :rtype lonter: float
    :return lonter: Longitude of the event in degrees.
    :rtype depth: float
    :return depth: Hypocenter depth in km
    :type startev: :class: `~obspy.core.utcdatetime.UTCDateTime`
    :return startev: Origin time of the event.
    :rtype rt: :class: `~obspy.core.stream.Stream`
    :return rt: Rotational signal from ringlaser.
    :rtype ac: :class: `~obspy.core.stream.Stream`
    :return ac: Three component broadband station signal.
    :rtype baz: tuple
    :return baz: [0] great circle distance in m, 
                 [1] theoretical azimuth,
                 [2] theoretical backazimuth.
    """
    origin = event.preferred_origin() or event.origins[0]
    latter = origin.latitude
    lonter = origin.longitude
    startev = origin.time
    depth = origin.depth * 0.001  # Depth in km

    if station == 'RLAS':
        source = ['http://eida.bgr.de', 
                  'http://erde.geophysik.uni-muenchen.de']
        net_r = 'BW'
        net_s = 'GR' 
        sta_r = 'RLAS'
        sta_s = 'WET'
        loc_r = ''
        loc_s = ''
        # RLAS channel code was changed after 16.4.2010
        if origin.time < UTCDateTime(2010, 4, 16):
            chan1 = 'BAZ'
        else: 
            chan1 = 'BJZ'
        chan2 = 'BHE'
        chan3 = 'BHN'
        chan4 = 'BHZ'

        # ringlaser signal, source LMU first
        rt,srcRT = download_data(
                            startev, net_r, sta_r, loc_r, chan1, source[::-1])

        # broadband station signal
        # assuming all translation data comes from same source
        acE,srcTR = download_data(startev, net_s, sta_s, loc_s, chan2, source)
        acN,srcTR = download_data(startev,  net_s, sta_s, loc_s, chan3, source)
        acZ,srcTR = download_data(startev,  net_s, sta_s, loc_s, chan4, source)
        ac = Stream(traces=[acE[0], acN[0], acZ[0]])
        for ca in [ac[0], ac[1], ac[2], rt[0]]:
            ca.stats.coordinates = AttribDict()
            ca.stats.coordinates['longitude'] = 12.8782
            ca.stats.coordinates['latitude'] = 49.144001
            ca.stats['starttime'] = startev - 180
            ca.stats['sampling_rate'] = 20.

    # theoretical event backazimuth and distance
    baz = gps2dist_azimuth(latter, lonter, rt[0].stats.coordinates.latitude,
                          rt[0].stats.coordinates.longitude)
    
    return latter, lonter, depth, startev, rt, ac, baz, net_r, net_s,\
        chan1, chan2, chan3, chan4, sta_r, sta_s, loc_r, loc_s, srcRT, srcTR


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


def resample(is_local, baz, rt, ac):
    """
    Resamples signal according to sampling rates and cut-off frequencies
    dependent on the location of the event (5 sec and 2Hz for local events,
    60 sec and 1 Hz for non-local events).

    :type is_local: str
    :param is_local: Self-explaining string for event distance.
    :type baz: tuple
    :param baz: Great circle distance in m, azimuth A->B in degrees,
        azimuth B->A in degrees.
    :type rt: :class: `~obspy.core.stream.Stream`
    :param rt: Rotational signal from ringlaser.
    :type ac: :class: `~obspy.core.stream.Stream`
    :param ac: Three component broadband station signal.
    :rtype rt: :class: `~obspy.core.stream.Stream`
    :return rt: Decimated rotational signal from ringlaser.
    :rtype ac: :class: `~obspy.core.stream.Stream`
    :return ac: Decimated three component broadband station signal.
    :rtype sec: int
    :return sec: time window length.
    :rtype cutoff: float
    :return cutoff: Cut-off frequency for the lowpass filter.
    :rtype cutoff_pc: float
    :return cutoff_pc: Cut-off frequency for the highpass filter in P-coda.

    """
    if is_local == 'local':
        for trr in (rt + ac):
            trr.data = trr.data[0: 1800 * rt[0].stats.sampling_rate]
        rt.decimate(factor=2)
        ac.decimate(factor=2)
        sec = 5
        cutoff = 2.0  # local events
    elif is_local == 'non-local':
        rt.decimate(factor=4)
        ac.decimate(factor=4)
        sec = 120
        cutoff = 1.0  # nonlocal events
    else:
        for trr in (rt + ac):
            trr.data = trr.data[0: 1800 * rt[0].stats.sampling_rate]
        rt.decimate(factor=2)
        ac.decimate(factor=2)
        sec = 3
        cutoff = 4.0  # close events

    return rt, ac, sec, cutoff

def resample(is_local, baz, rt, ac):
    """
    Resamples signal according to sampling rates and cut-off frequencies
    dependent on the location of the event (5 sec and 2Hz for local events,
    60 sec and 1 Hz for non-local events).

    :type is_local: str
    :param is_local: Self-explaining string for event distance.
    :type baz: tuple
    :param baz: Great circle distance in m, azimuth A->B in degrees,
        azimuth B->A in degrees.
    :type rt: :class: `~obspy.core.stream.Stream`
    :param rt: Rotational signal from ringlaser.
    :type ac: :class: `~obspy.core.stream.Stream`
    :param ac: Three component broadband station signal.
    :rtype rt: :class: `~obspy.core.stream.Stream`
    :return rt: Decimated rotational signal from ringlaser.
    :rtype ac: :class: `~obspy.core.stream.Stream`
    :return ac: Decimated three component broadband station signal.
    :rtype sec: int
    :return sec: Sampling rate.
    :rtype cutoff: float
    :return cutoff: Cut-off frequency for the lowpass filter.
    :rtype cutoff_pc: float
    :return cutoff_pc: Cut-off frequency for the highpass filter in P-coda.

    """

    cutoff_pc = 0.5  # Cut-off frequency for the highpass filter in the P-coda
    if is_local == 'local':
        for trr in (rt + ac):
            trr.data = trr.data[0: int(1800 * rt[0].stats.sampling_rate)]
        rt.decimate(factor=2)
        ac.decimate(factor=2)
        sec = 5
        cutoff = 2.0  # Cut-off freq for the lowpass filter for local events
    elif is_local == 'non-local':
        rt.decimate(factor=4)
        ac.decimate(factor=4)
        sec = 120
        cutoff = 1.0  # Cut-off freq for the lowpass filter for non-loc events
    elif is_local == 'close':
        for trr in (rt + ac):
            trr.data = trr.data[0: int(1800 * rt[0].stats.sampling_rate)]

        rt.decimate(factor=2)
        ac.decimate(factor=2)
        sec = 3
        cutoff = 4.0  # Cut-off freq for the lowpass filter for close events

    return rt, ac, sec, cutoff


def remove_instr_resp(rt, ac, station, startev):
    """
    Remove instrument response, detrend data, trim if unequal lengths
    sensitivity controls order of magnitude of units

    NOTE:
    + 9.44*10**8 gives m/s, 944.xxx gives um/s, 0.944xxx for nm/s (w/ 2 zeros)
    + 2 zeros velocity, 1 zero acceleration, 3 zeroes displacement

    :type rt: :class: `~obspy.core.stream.Stream`
    :param rt: Rotational signal from ringlaser.
    :type ac: :class: `~obspy.core.stream.Stream`
    :param ac: Three component broadband station signal.
    :type station: str
    :param station: Station from which data are fetched ('WET' or 'PFO').
    :type startev: :class: `~obspy.core.utcdatetime.UTCDateTime`
    :param startev: Origin time of the event.
    :rtype rt: :class: `~obspy.core.stream.Stream`
    :return rt: Detrended and trimmed rotational signal from ringlaser.
    :rtype ac: :class: `~obspy.core.stream.Stream`
    :return ac: Detrended and trimmed three component broadband station signal.
    """    

    if station == 'RLAS':
        rt[0].data = rt[0].data * 1. / 6.3191 * 1e-3  # Rotation rate in nrad/s
        
        ac.detrend(type='linear')
        rt.detrend(type='linear')
        ac.taper(max_percentage=0.05)
        rt.taper(max_percentage=0.05)
        
        paz_sts2_vel = {'poles': [(-0.0367429 + 0.036754j),
                                (-0.0367429 - 0.036754j)],
                        'sensitivity': 0.944019640, 
                        'zeros': [0j,0j], 
                        'gain': 1.0}

        ac.simulate(paz_remove=paz_sts2_vel, remove_sensitivity=True)  

    else:
        print('Incorrect station call')

    # make sure start and endtimes match for both instruments
    startaim = max([tr.stats.starttime for tr in (ac + rt)])
    endtaim = min([tr.stats.endtime for tr in (ac + rt)])

    ac.trim(startaim, endtaim, nearest_sample=True)
    rt.trim(startaim, endtaim, nearest_sample=True)

    return rt, ac


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
    pwave_list = ['P','p','Pdiff','PKiKP','PKIKP','PP','Pb','Pn','Pg']
    for i2 in range(0, len(tt)):
        if tt.__getitem__(i2).__dict__['name'] in pwave_list:
            tiem_p = tt.__getitem__(i2).__dict__['time']
            tiemp.append(tiem_p)

    arriv_p = np.floor(init_sec + min(tiemp))

    # from all possible S arrivals select the earliest one
    swave_list = ['S','s','Sdiff','SKiKS','SKIKS','SS','Sb','Sn','Sg']
    for i3 in range(0, len(tt)):
        if tt.__getitem__(i3).__dict__['name'] in swave_list:
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

    :type distance: float
    :param distance: Epicentral distance in degrees between earthquake source
        and receiver station.
    :type start_time: float
    :param start_time: Starttime of the event in the fetched seismogram.
    :rtype arrival: float
    :return arrival: Arrival time of the surface waves of the event.
    """
    deltas = np.arange(0., 140., 5.)
    tts = 60. * np.array(
                        [0.,2.,4.,6.2,8.4,11.,13.,15.2,17.8,19.4,22.,24.1,26.6,
                        28.6,30.8,33.,35.6,37.4,39.8,42.,44.2,46.4,48.8,50.9,
                        53.6,55.2,57.8,60.])
    (mval, nval) = np.polyfit(deltas, tts, 1)
    # calculate surface wave travel times for degrees 1 to 180 ?
    surftts = mval * np.arange(0., 180.1, 0.01)
    difer = []
    for i4 in range(0, len(surftts)):
        dife_r = abs(0.001 * distance / 111.11 - np.arange(0., 180.1, 0.01)
                     [i4])
        difer.append(dife_r)
    # love wave arrival: event time + surftts for closest degree??
    # (smallest difference between distance for surftts and actual distance of
    #  event)
    arriv_lov = np.floor(start_time + surftts[np.asarray(difer).argmin()])
    diferans = []
    for i1 in range(len(deltas)):
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

    :type baz: tuple
    :param baz: Great circle distance in m, azimuth A->B in degrees,
        azimuth B->A in degrees.
    :type arriv_p: float
    :param arriv_p: Arrival time of the first P-wave.
    :type arriv_s: float
    :param arriv_s: Arrival time of the first S-wave.
    :type init_sec: float
    :param init_sec: Initial time of the event in sec in the fetched data.
    :type is_local: str
    :param is_local: Self-explaining string for event distance.
    :rtype min_pw: float
    :return min_pw: Starttime for P-waves window.
    :rtype max_pw: Endtime for P-waves window.
    :return min_sw: Starttime for S-waves window.
    :rtype max_sw: Endtime for S-waves window.
    :return min_lwi: Starttime for initial surface-waves window.
    :rtype max_lwi: Endtime for initial surface-waves window.
    :return min_lwf: Starttime for latter surface-waves window.
    :rtype max_lwf: Endtime for latter surface-waves window.

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

def peak_correlation(ac_ori,rt,sec,station):
   
    """find trace cross correlations
    :type ac_ori: :class: `~obspy.core.stream.Stream`
    :param ac_ori: Stream object, horizontal velocities
    :type rt: :class: `~obspy.core.stream.Stream`
    :param rt: Rotational signal from ringlaser.
    """

    ac = ac_ori.copy()
    ac.differentiate(method='gradient')
    ac.rotate(method='NE->RT',back_azimuth=baz[2])
    tr_acc = ac.select(component='T')

    rot_sr = int(rt[0].stats.sampling_rate * sec)
    tra_sr = int(tr_acc[0].stats.sampling_rate* sec)

    corrcoefs=[]
    for i in range(0, len(rt[0].data) // rot_sr):
        coeffs = xcorr(rt[0].data[i*rot_sr:(i+1)*rot_sr],
                                    tr_acc[0][i*tra_sr:(i+1)*tra_sr], 0)

        corrcoefs.append(coeffs[1])

    corrcoefs = np.asarray(corrcoefs)

    return max(corrcoefs)

def process_save(ac,rt,baz,cutoff,station,is_local,min_lwi,max_lwf,
                                                output_path,tag_name,event_ID):
    """
    Filter traces for 10s to 60s bandpass, rotate translation components to 
    radial/transverse coordinate system, determine max peak-to-trough trace 
    amplitudes and quickplot to assess waveform quality
    + Amplitude is given as 1/2 maximum peak to adjacent trough 
    + Associated period is 2x time interval seperating peak and adjacent trough
    *be careful with the copy statement


    :type rt: :class: `~obspy.core.stream.Stream`
    :param rt: Rotational signal from ringlaser.
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
    sampling_rate = int(rt[0].stats.sampling_rate)
    velocity_nez = ac.copy()
    velocity_rtz = ac.copy()
    rot_rate = rt.copy()
    rotation = rt.copy()
    velocity_rtz.rotate(method='NE->RT',back_azimuth=baz[2])
    rotation.integrate(method='cumtrapz')

    # surface wave train
    #surf_big = int(sampling_rate * min_lwi)
    #surf_end = int(sampling_rate * max_lwf)
    surf_big = 0
    surf_end = len(rt[0].data)-1

    # bandpass all traces for 3s to 60s
    f_start = 1/60
    f_end = 1/3
    for traces in [rot_rate,rotation,velocity_rtz,velocity_nez]:
        traces.filter('bandpass', freqmin=f_start, freqmax=f_end, corners=3,
                  zerophase=True)

    # seperate streams into traces and create list for ease of processing 
    z_vel = velocity_nez.select(component='Z')
    n_vel = velocity_nez.select(component='N')
    e_vel = velocity_nez.select(component='E')
    r_vel = velocity_rtz.select(component='R')
    t_vel = velocity_rtz.select(component='T')
    alltraces = [rot_rate[0],rotation[0],z_vel[0],t_vel[0],
                                    n_vel[0],e_vel[0],r_vel[0]]


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
    # CHANGED: only plotting rotations and Z and T, hacky ylabels
    axrow = 4
    f,axes = plt.subplots(nrows=axrow,ncols=2,sharex='col',sharey='row')
    j = 0
    for i in range(axrow):
        # trace overviews
        axes[i][0].plot(alltraces[j].data,'k')
        axes[i][0].plot(peak_indS[j],peakS[j],'go',adj_trough_indS[j],
                                                        adj_troughS[j],'go')
        axes[i][0].plot(adj_peak_indS[j],adj_peakS[j],'ro',trough_indS[j],
                                                            troughS[j],'ro')
        axes[i][0].plot(zero_crossings[j],0,'bo',zorder=8)

        axes[i][0].annotate(p2atS[j], xy=(peak_indS[j],peakS[j]), 
            xytext=(peak_indS[j],peakS[j]),zorder=10,color='g')
        axes[i][0].annotate(ap2tS[j], xy=(trough_indS[j],troughS[j]), 
            xytext=(trough_indS[j],troughS[j]),zorder=10,color='r')
        axes[i][0].grid()
        # if channelS[j] == 'BJZ' or channelS[j] == 'BAZ':
        #     axes[i][0].set_ylabel('rot. rate(nrad/s)'.format(channelS[j]))
        # else:
        #     axes[i][0].set_ylabel('{} vel.(nm/s)'.format(channelS[j]))
        
        # hacky ylabels
        if i == 0:
            axes[i][0].set_ylabel('rot. rate (nrad/s)')
        elif i == 1:
            axes[i][0].set_ylabel('rotation (nrad)')
        else:
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
        elif j == axrow-1:
            axes[i][0].set_xlabel('sample')
            axes[i][1].set_xlabel('sample')
            axes[i][1].set_xlim([min(zero_crossings)-2000,
                                                max(zero_crossings)+2000])
        j+=1

    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0.1)

    filename_png = os.path.join(output_path,'imgs',tag_name+'.png')
    plt.savefig(filename_png,dpi=150)
    plt.close()

    return peak2troughs,periods,zero_crossings_abs


def store_info_json(tag_name, output_path, ac, rt, baz, peak2troughs, 
                    periods, zero_crossings_abs, station, startev, event,
                    event_source, depth, PCC):
    """
    Generates a human readable .json file that stores data for each event,
    like peak values (acceleration, rotation rate, signal-to-noise ratio, 
    backazimuths.

    :type folder_name/tag_name/event_ID: str
    :param tag_name: identifier containing event info for figure filename
    :param folder_name: event folder for storing all figures
    :type ac: :class: `~obspy.core.stream.Stream`
    :param ac: Three component broadband station signal.
    :type rt: :class: `~obspy.core.stream.Stream`
    :param rt: Rotational signal from ringlaser.
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
    rr_max, rt_max, zv_max, tv_max, nv_max, ev_max, rv_max = peak2troughs[:]
    rr_per, rt_per, zv_per, tv_per, nv_per, ev_per, rv_per = periods[:]
    rr_zca, rt_zca, zv_zca, tv_zca, nv_zca, ev_zca, rv_zca \
                                                        = zero_crossings_abs[:]
    
    sampl_rate = rt[0].stats.sampling_rate
    TBA = baz[2]  # Theoretical backazimuth [deg]
    distance = 0.001*baz[0]
    dic = OrderedDict([
            ('event_id', event.resource_id.id),
            ('event_source', event_source),
            ('starttime', str(startev-180)),
            ('endtime', str(startev+3*3600)),
            ('station_latitude', rt[0].stats.coordinates['latitude']),
            ('station_longitude', rt[0].stats.coordinates['longitude']),
            ('event_latitude', event.preferred_origin().latitude),
            ('event_longitude', event.preferred_origin().longitude),
            ('magnitude', event.magnitudes[0]['mag']),
            ('depth_in_km', depth),
            ('epicentral_distance_in_km', distance),
            ('theoretical_backazimuth', TBA),
            ('peak_correlation_coefficient', PCC),
            ('zero_crossing_unit','sec. from trace start'),           
            ('vertical_rotation_rate', 
                OrderedDict([
                ('peak_amplitude',rr_max),
                ('dominant_period',rr_per),
                ('zero_crossing',rr_zca),
                ('unit','nrad/s')
                ])
            ),
            ('vertical_rotation', 
                OrderedDict([
                ('peak_amplitude',rt_max),
                ('dominant_period',rt_per),
                ('zero_crossing',rt_zca),
                ('unit','nrad')
                ])
            ),
            ('vertical_velocity', 
                OrderedDict([
                ('peak_amplitude',zv_max),
                ('dominant_period',zv_per),
                ('zero_crossing',zv_zca),
                ('unit','nm/s')
                ])
            ),
            ('transverse_velocity', 
                OrderedDict([
                ('peak_amplitude',tv_max),
                ('dominant_period',tv_per),
                ('zero_crossing',tv_zca),
                ('unit','nm/s')
                ])
            ),
            ('radial_velocity', 
                OrderedDict([
                ('peak_amplitude',rv_max),
                ('dominant_period',rv_per),
                ('zero_crossing',rv_zca),
                ('unit','nm/s')
                ])
            ),
            ('north_velocity', 
                OrderedDict([
                ('peak_amplitude',nv_max),
                ('dominant_period',nv_per),
                ('zero_crossing',nv_zca),
                ('unit','nm/s')
                ])
            ),
            ('east_velocity', 
                OrderedDict([
                ('peak_amplitude',nv_max),
                ('dominant_period',nv_per),
                ('zero_crossing',nv_zca),
                ('unit','nm/s')
                ])
            )
            ])

    filename_json = os.path.join(output_path,'jsons',tag_name + '.json')
    outfile = open(filename_json, 'wt')
    json.dump(dic, outfile, indent=4)

    outfile.close()


# MAIN
parser = argparse.ArgumentParser(description='Magscale')
parser.add_argument('--station', help='Choice of station: RLAS, FFB\
    (default is RLAS)', type=str, default='RLAS')
parser.add_argument('--mode', help='XML or IRIS', type=str,default='XML')
parser.add_argument('--min_magnitude', help='Minimum magnitude for \
    events (default is 3).', type=float or int, default=4.0)
parser.add_argument('--max_magnitude', help='Maximum magnitude for \
    events (default is 10).', type=float or int, default=10.0)
parser.add_argument('--min_datetime', help='Earliest date and time for \
    the search. Format is UTC: yyyy-mm-dd-[T hh:mm:ss]. \
    Example: 2010-02-27T05:00', type=str, default=str(
                    datetime.datetime.now()-datetime.timedelta(hours=168)))
parser.add_argument('--max_datetime', help='Latest date and time for \
    the search (default is today).',type=str, default=str(
                                                datetime.datetime.now()))

args = parser.parse_args()
station = args.station.upper()
mode = args.mode.upper()


# set catalog variables
if mode == 'XML':
    event_source = 'QUAKEML'
    catpath = './catalogs/final_events_244.xml'
    cat = read_events(catpath,format='QUAKEML')

elif mode == 'IRIS':
        print("\nDownloading events from IRIS")
        catalog = 'GCMT'
        event_source = 'IRIS'
        c = fdsnClient(event_source)
        cat = c.get_events(minmagnitude=args.min_magnitude,
                           maxmagnitude=args.max_magnitude,
                           starttime=args.min_datetime,
                           endtime=args.max_datetime,
                           catalog=catalog)

# set file output path
output_path = './output/processed_events/'
for extra_path in ['imgs','jsons','xmls']:
    extra_folder = os.path.join(output_path,extra_path)
    if not os.path.exists(extra_folder):
        os.makedirs(extra_folder)

print("%i event(s) downloaded, beginning processing..." % len(cat))
success_counter = fail_counter = already_processed = 0
bars = '='*79
error_list = []
for event in cat:
    try:
        # print event divider
        event_information = str(event).split('\n')[0][7:]
        flinn_engdahl = event.event_descriptions[0]['text'].upper()
        print('{}\n{}\n{}\n{}'.format(
                                bars,flinn_engdahl,event_information,bars))

        # create tags for standard filenaming
        # magnitude, always keep at 3 characters long, fill w/ 0's if not
        mag_tag = '{:0^4}'.format(str(event.magnitudes[0]['mag']))

        # Flinn Engdahl region, i.e. SOUTHEAST_OF_HONSHU_JAPAN
        substitutions = [(', ', '_'),(' ','_')]
        for search, replace in substitutions:
            flinn_engdahl = flinn_engdahl.replace(search,replace)

        # remove '.' from end of region name if necessary (i.e. _P.N.G.)
        if flinn_engdahl[-1] == '.':
            flinn_engdahl = flinn_engdahl[:-1]

        # ISO861 Time Format, i.e. '2017-09-23T125302Z'
        orig = event.preferred_origin() or event.origins[0]
        time_tag = orig['time'].isoformat()[:19].replace(':','')+'Z'

        # i.e. 'GCMT_2017-09-23T125302_6.05_OAXACA_MEXICO'
        tag_name = '_'.join((station,time_tag,mag_tag,flinn_engdahl))

        xml_tag = os.path.join(output_path,'xmls',tag_name + '.xml')
        if os.path.exists(xml_tag):
            print('Already Processed')
            continue
        
        # run processing function
        try:
            latter, lonter, depth, startev, rt, ac, baz, net_r, net_s,\
            chan1, chan2, chan3, chan4, sta_r, sta_s, loc_r, loc_s, srcRT, srcTR = \
                                                   event_info_data(event, station)
            
            rt,ac,sec,cutoff, = resample(is_local(baz),baz,rt,ac)

            rt,ac = remove_instr_resp(rt,ac,station,startev)

            print("Getting arrival times...")
            init_sec = startev - ac[0].stats.starttime
            # When the event starts in the fetched data
            arriv_p,arriv_s = ps_arrival_times(baz[0],depth,init_sec)

            min_pw,max_pw,min_sw,max_sw,min_lwi,max_lwi,min_lwf,max_lwf =\
                            time_windows(baz,arriv_p,arriv_s,init_sec,is_local(baz))
            
            print("Finding peak correlations...")
            PCC = peak_correlation(ac,rt,sec,station)

            print("Processing data and saving figures...")
            event_ID = event.resource_id.id[-7:] 
            peak2troughs,periods,zero_crossings_abs = process_save(
                                                ac,rt,baz,cutoff,station,
                                                is_local(baz),min_lwi,max_lwf,
                                                output_path,tag_name,event_ID)

            print("Saving data...")
            store_info_json(tag_name,output_path,ac,rt,baz,peak2troughs,
                            periods,zero_crossings_abs,station,startev,event,
                            event_source,depth,PCC)
            event.write(xml_tag,format='QUAKEML')
            
            success_counter += 1
        
        # if any error
        except Exception as e:
            fail_counter += 1
            print(e)
            error_list.append(tag_name)
       
        # if keyboard interrupt, quit
        except KeyboardInterrupt:
            fail_counter += 1
            sys.exit()


    # if any error
    except Exception as e:
        fail_counter += 1
        print(e)
        error_list.append(tag_name)
   
    # if keyboard interrupt, quit
    except KeyboardInterrupt:
        fail_counter += 1
        sys.exit()

# print end message
print('{}\n'.format('_'*79))
print("Catalog complete, no more events to show")
print("From a total of %i event(s):\n %i was/were successfully processed"
      "\n %i could not be processed \n %i already processed\n" % (
          len(cat), success_counter, fail_counter, already_processed))
 

#  # write error log to see events failed, named by search timeframe
if len(error_list) > 0:
  for i in error_list:  
        print(error_list)