import os
import sys
import glob
import obspy
import numpy as np
import matplotlib.pyplot as plt
from obspy import read, read_events


# paths and lists
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

pick = 'zv'

# iterate on tags
for i,(event,ident) in enumerate(zip(event_list,identifier)):
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

    # filter values
    t_start = 10
    t_end = 30
    f_start = 1/t_end
    f_end = 1/t_start

    # preprocessing
    for streams in [obs,syn]:
        streams.trim(starttime=origin_time,endtime=origin_time+2*3600)
        streams.detrend(type='linear')
        streams.taper(max_percentage=0.05)
        streams.filter('bandpass',freqmin=f_start,freqmax=f_end,corners=3,
                                                                zerophase=True)

    # divy up traces
    obs_velocityZ = obs.select(id='GR.WET..BHZ')
    syn_velocityZ = syn.select(id='BW.RLAS.S3.MXZ').differentiate(
                                                            method='gradient')

    obs_rot_rateZ = obs.select(id='BW.RLAS..BJZ')
    syn_rot_rateZ = syn.select(id='BW.RLAS.S3.MYZ').differentiate(
                                                            method='gradient')

    syn_accelT = syn.select(id='BW.RLAS.S3.MXT').differentiate(
        method='gradient').differentiate(method='gradient')

    # isolate data (rad/s or m/s)
    if pick == 'zv':
        obs_data = [_ * (10**-9) for _ in obs_velocityZ[0].data]
        syn_data = syn_velocityZ[0].data
        units = 'm/s'
        choice = 'Vertical Velocity'
    elif pick == 'rr':
        obs_data = [_ * (10**-9) for _ in obs_rot_rateZ[0].data]
        syn_data = syn_rot_rateZ[0].data
        units = 'rad/s'
        choice = 'Rotation Rate'

    # convert to time
    obs_SR = obs_rot_rateZ[0].stats.sampling_rate
    syn_SR = syn_rot_rateZ[0].stats.sampling_rate

    obs_maxtime = len(obs_data)/obs_SR
    syn_maxtime = len(syn_data)/syn_SR
    obs_time = np.linspace(0,obs_maxtime,len(obs_data))
    syn_time = np.linspace(0,syn_maxtime,len(syn_data))

    # grab peak values
    obs_peak = max(obs_data)
    syn_peak = max(syn_data)
    obs_peak_ind = np.where(obs_data == obs_peak)[0][0]/obs_SR
    syn_peak_ind = np.where(syn_data == syn_peak)[0][0]/syn_SR
    # print(event,ds_list[i],syn_peak,obs_peak)

    tex_print = '{event} & {mag} & {ds:.2f} & {obs:.2E} & {syn:.2E} & {ratio:.2E} \\\ \hline'.format(
                event=event,mag=mag_list[i],ds=ds_list[i],obs=obs_peak,syn=syn_peak,ratio=syn_peak/obs_peak)
    print(tex_print)

    # plot
    plt.figure(1)
    ax1 = plt.subplot(111)
    a1= ax1.plot(obs_time,obs_data,'r',label='Obs.',zorder=2)
    ax1.plot(obs_peak_ind,obs_peak,'ro',zorder=3)
    ax1.annotate('Peak Obs. Amplitude: {amp:.2E} {units}'.format(
                                amp=obs_peak,units=units),
                                xy=(obs_peak_ind,obs_peak),zorder=4)

    ax2 = ax1.twinx()
    a2 = ax2.plot(syn_time,syn_data,'k',label='Syn.',zorder=1)
    ax2.plot(syn_peak_ind,syn_peak,'ko',zorder=3)
    ax2.annotate('Peak Syn. Amplitude: {amp:.2E} {units}'.format(
                                amp=syn_peak,units=units),
                                xy=(syn_peak_ind,syn_peak),zorder=4)
    ax2.grid(None)


    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Obs. Amplitude [{units}]'.format(units=units))
    # ax2.set_ylabel('Syn. Amplitude [{units}]'.format(units=units))
    ax1.grid()

    ax1.set_title('{} / Event: {} / Event-Station Distance: {} deg'.format(choice,event,round(ds_list[i],2)))

    lines = a1 + a2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines,labels,loc=0)

    figurename = os.path.join('./figures','waveforms',event+'.png')
    # plt.savefig(figurename,dpi=600,figsize=(11,7))
    plt.show()


    # compare zRR w/ tAC, keep in samples
    zRR_data = syn_rot_rateZ[0].data
    tAC_data = syn_accelT[0].data

    tAC_max = abs(max(tAC_data) and min(tAC_data))
    zRR_max = abs(max(zRR_data) and min(zRR_data))

    c = tAC_max/zRR_max

    tAC_data_scaled = [_/(2*c) for _ in tAC_data]
    print(c)
    # sys.exit()

    # plot
    # plt.figure(1)
    # ax = plt.subplot(111)
    # a1 = ax1.plot(zRR_data,'r',label='RR.',zorder=1)
    # a2 = ax1.plot(tAC_data_scaled,'k',label='ACC.',zorder=2)
    # ax1.set_xlabel('Samples')
    # ax1.set_ylabel('Amplitude')
    # ax1.grid()

    # ax1.set_title(event)
    # plt.legend()

    # plt.show()
