import os
import sys
import glob
import obspy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from obspy import read, read_events
from obspy.geodetics.base import gps2dist_azimuth

mpl.rcParams['lines.linewidth']=1
mpl.rcParams.update({'font.size': 22.5})


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

# zv or rr
pick = sys.argv[1]

# paths and lists
# NOTE: observations - rotation rate stream in units of nrad/s
#                      velocity stream in nanometers/s
#       synthetics - MY? rotation stream in units of rad/s
#                    MX? velocity stream in m/s
inv_path = './synthetics/specfem/input/tenGoodCMTSolutions'
obs_list_path = './output/mseeds/'
syn_list_path = './synthetics/specfem/output/mseeds/'

obs_list = glob.glob(obs_list_path + '*')
syn_list = glob.glob(syn_list_path + '*RLAS.mseed')

# unique tags for each event
event_list = ['C201007181335A', #contains foreshock
            'C201109161926A', #great, aftershock at end of trace
            'C201301050858A', #great
            'C201304161044A', #starts too early, foreshock?
            'C201304191958A', #great
            'C201502131859A', #starts too early
            'C201504250611A', #long ringing, starts early
            'C201509130814A', #good
            'S201509162318A', #good
            'C201601250422A'] #great
event_ignore = ['C201007181335A', #contains foreshock
            'C201304161044A', #starts too early, foreshock?
            'C201502131859A', #starts too early
            'C201504250611A'] #long ringing, starts early


# get event information
event_info = read_events(inv_path)
identifier = []
for quakes in event_info:
    id_temp = str(quakes.resource_id).split('/')[2]
    index_temp = event_list.index(id_temp)
    identifier.append(index_temp)

# distances of each event in event_list to station RLAS
ds_list = [76.3802716176,124.01982398,57.5432613254,89.721876944,18.1306704563,
        71.9959629677,109.912276463,27.994777703,80.2856479882,42.8349093379]
mag_list = [6.63,7.92,6.42,6.1,7.36,7.56,7.14,7.11,6.7,7.77]

# iterate on tags
peak_differences = []
for i,(event,ident) in enumerate(zip(event_list,identifier)):
    # if event in event_ignore:
    #     continue

    # grab unique obs and syn by tag name
    for obs_tmp,syn_tmp in zip(obs_list,syn_list):
        if event in obs_tmp:
            obs_path = obs_tmp
        if event in syn_tmp:
            syn_path = syn_tmp
    obs = read(obs_path,format='mseed')
    syn = read(syn_path,format='mseed')

    # line up timing
    quakeml = event_info[ident]
    origin_time = quakeml.origins[0].time
    event_lat = quakeml.origins[0].latitude
    event_lon = quakeml.origins[0].longitude

    # filter values
    t_start = 10
    t_end = 60
    f_start = 1/t_end
    f_end = 1/t_start

    # narrow bandpass
    # t_central = 20
    # f_start = 0.9 * 1/t_central
    # f_end = 1.1 * 1/t_central

    # preprocessing, same for both observations and synthetics
    # NOTE: observations already have instrument response removed
    for streams in [obs,syn]:
        streams.trim(starttime=origin_time,endtime=origin_time+2*3600)
        streams.detrend(type='linear')
        streams.taper(max_percentage=0.05)
        streams.filter('bandpass',freqmin=f_start,freqmax=f_end,corners=3,
                                                                zerophase=True)
    # rotate traces for acceleration
    station_lat = 49.144001
    station_lon = 12.8782
    # BAz from station to source
    # can use client.distaz() to check baz values
    BAz = gps2dist_azimuth(lat1=event_lat,lon1=event_lon,
                            lat2=station_lat,lon2=station_lon)
    for tr in obs:
        tr.stats.back_azimuth = BAz[2]
    obs.rotate(method='NE->RT')

    # divy up traces - observations
    obs_velocityZ = obs.select(id='GR.WET..BHZ')
    obs_rot_rateZ = obs.select(id='BW.RLAS..BJZ')
    obs_accelT = obs.select(id='GR.WET..BHT').differentiate(method='gradient')

    # synthetics
    syn_velocityZ = syn.select(id='BW.RLAS.S3.MXZ').differentiate(
                                                            method='gradient')
    syn_rot_rateZ = syn.select(id='BW.RLAS.S3.MYZ').differentiate(
                                                            method='gradient')
    syn_accelT = syn.select(id='BW.RLAS.S3.MXT').differentiate(
        method='gradient').differentiate(method='gradient')

    # convert synthetic data from m/s to um/s and rad/s to nrad/s
    for tr_syn in syn:
        # rotation
        if tr_syn.get_id()[-2] == 'Y':
            tr_syn.data *= 10**9
        # velocity
        else:
            tr_syn.data *= 10**6

    # convert observation velocity from nm/s to um/s
    for tr_obs in obs:
        if tr_obs.get_id()[-2] == 'H':
            tr_obs.data *= 10**-3

    # isolate data (rad/s or m/s)
    if pick == 'zv':
        # obs_data = [_ * (10**-9) for _ in obs_velocityZ[0].data]
        obs_data = obs_velocityZ[0].data
        syn_data = syn_velocityZ[0].data
        units = 'um/s'
        choice = 'Vertical Velocity'
    elif pick == 'rr':
        # obs_data = [_ * (10**-9) for _ in obs_rot_rateZ[0].data]
        obs_data = obs_rot_rateZ[0].data
        syn_data = syn_rot_rateZ[0].data
        units = 'nrad/s'
        choice = 'Rotation Rate'

    # convert to time
    obs_SR = obs_rot_rateZ[0].stats.sampling_rate
    obs_maxtime = len(obs_data)/obs_SR
    obs_time = np.linspace(0,obs_maxtime,len(obs_data))

    # ======================= OBSERVATIONS =======================
    # grab peak value
    obs_peak = max(obs_data)
    obs_peak_ind = np.where(obs_data == obs_peak)[0][0]/obs_SR

    # plot
    f = plt.figure(1)
    ax1 = plt.subplot(111)
    a1= ax1.plot(obs_time,obs_data,'r',label='Obs.',zorder=2)
    ax1.plot(obs_peak_ind,obs_peak,'ro',zorder=3)
    # ax1.annotate('Peak Obs. Amplitude: {amp:.2E} {units}'.format(
    #                             amp=obs_peak,units=units),
    #                             xy=(obs_peak_ind,obs_peak*1.1),zorder=4)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Observation Rotation Rate [{units}]'.format(units=units))
    ax1.grid(zorder=0)

    # ax1.set_title('{} / Event: {} / Event-Station Distance: {} deg'.format(
    #                                         choice,event,round(ds_list[i],2)))
    #
    # # ======================= SYNTHETICS =======================
    syn_SR = syn_rot_rateZ[0].stats.sampling_rate
    syn_maxtime = len(syn_data)/syn_SR
    syn_time = np.linspace(0,syn_maxtime,len(syn_data))
    syn_peak = max(syn_data)
    syn_peak_ind = np.where(syn_data == syn_peak)[0][0]/syn_SR

    ax2 = ax1.twinx()
    a2 = ax2.plot(syn_time,syn_data,'k',label='Syn.',zorder=1)
    ax2.plot(syn_peak_ind,syn_peak,'ko',zorder=4)
    # ax2.annotate('Peak Syn. Amplitude: {amp:.2E} {units}'.format(
    #                             amp=syn_peak,units=units),
    #                             xy=(syn_peak_ind,syn_peak*.9),zorder=4)
    ax2.set_ylabel('Synthetic Rotation Rate [{units}]'.format(units=units))
    ax2.grid(zorder=0)

    # create legend from both axes
    lines = a1 + a2
    labels = [l.get_label() for l in lines]
    # ax1.legend(lines,labels,loc=0)
    align_yaxis(ax1, 0, ax2, 0)

    # set yticks equal for twinx
    ax1bound = abs(max([ax1.get_ybound()[0],ax1.get_ybound()[1]]))
    ax2bound = abs(max([ax2.get_ybound()[0],ax2.get_ybound()[1]]))
    ax1.set_yticks(np.linspace(-1*ax1bound,ax1bound, 7))
    ax2.set_yticks(np.linspace(-1*ax2bound,ax2bound, 7))

    figurename = os.path.join('./figures','waveforms',event+'.png')
    # plt.savefig(figurename,dpi=600,figsize=(11,7))
    plt.xlim([0,obs_maxtime])
    plt.show()

    # append to list of peak value differences
    peak_differences.append(syn_peak/obs_peak)

    # ==================== PHASE VELOCITY/CORRELATION ====================
    f2,(ax3,ax4) = plt.subplots(2,sharex=True)
    f2 = plt.figure(1)
    ax3 = plt.subplot(111)
    # rotation rate vs transverse acceleration - observations
    obs_TAdata = obs_accelT[0].data
    obs_RRdata = obs_rot_rateZ[0].data

    # find max values to scale by
    oTA_max = abs(max(obs_TAdata) and min(obs_TAdata))
    oRR_max = abs(max(obs_RRdata) and min(obs_RRdata))
    obs_c = oTA_max/(2*oRR_max)
    obs_TAdata_scaled = [_/obs_c for _ in obs_TAdata]

    ax3.plot(obs_time,obs_TAdata_scaled,'k',label='Transverse Acceleration')
    ax3.plot(obs_time,obs_RRdata,'r',label='Rotation Rate')
    ax3.set_xlabel('Time (sec)')
    ax3.set_ylabel('Rotation rate (rad/s) and scaled Transverse Acceleration')
    ax3.set_title('Observations, RR vs. TA | c = {}'.format(obs_c))
    ax3.grid(True)
    ax3.set_axisbelow(True)

    # synthetics
    syn_TAdata = syn_accelT[0].data
    syn_RRdata = syn_rot_rateZ[0].data

    # find max values to scale by
    sTA_max = abs(max(syn_TAdata) and min(syn_TAdata))
    sRR_max = abs(max(syn_RRdata) and min(syn_RRdata))
    syn_c = sTA_max/(2*sRR_max)
    syn_TAdata_scaled = [_/syn_c for _ in syn_TAdata]

    ax4.plot(syn_time,syn_TAdata_scaled,'k')
    ax4.plot(syn_time,syn_RRdata,'r')
    ax4.set_xlabel('Time (sec)')
    ax4.set_ylabel('Rotation rate (rad/s) and scaled Transverse Acceleration')
    ax4.set_title('Synthetics, RR vs. TA | c = {}'.format(syn_c))
    ax4.grid(True)
    ax4.set_axisbelow(True)

    # plt.legend()
    plt.show()
    print(obs_c,syn_c)

# plt.close()
# f3 = plt.figure()
# plt.scatter(ds_list,peak_differences,c='r',marker='x')
# plt.yscale('log')
# plt.xlabel('Distance (deg)')
# plt.ylabel('Observed Peak/Synthetic Peak (%)')
# plt.title('Peak Differences')
# plt.show()
