import os
import sys
import glob
import obspy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from obspy import read, read_events
from obspy.geodetics.base import gps2dist_azimuth

# plt.rc('text', usetex=True)
# plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['lines.linewidth']=1
mpl.rcParams.update({'font.size': 15})

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ================================ FUNCTIONS ==================================
def __pretty_grids(input_ax):
    """grid formatting
    """
    input_ax.set_axisbelow(True)
    input_ax.tick_params(which='both',
                         direction='in',
                         top=True,
                         right=True)
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
    input_ax.ticklabel_format(style='sci',
                            axis='y',
                            scilimits=(0,0))

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

def change_baz():
    from math import cos, sin, radians
    for ba in range(80,110,1):
        BAz = radians(ba)
        t = - e * cos(baz) + n * sin(baz)
        # print(ba,t)

# =========================== MAIN PROCESSING ==================================
def collect_data():
    """colect data for comparisons of observations and synthetics
    NOTE: observations - rotation rate stream in units of nrad/s
                             velocity stream in nanometers/s
          synthetics - MY? rotation stream in units of rad/s
                           MX? velocity stream in m/s
    """
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

    event_ignore = ['C201007181335A', #contains foreshock
                    'C201304161044A', #starts too early, foreshock?
                    'C201502131859A', #starts too early
                    'C201504250611A'] #long ringing, starts early


    # get event information
    event_info = read_events(inv_path)

        
    return obs_list,syn_list,event_list,event_info

def process_and_plot(t_start=5,t_end=60,pick='rr'):
    """preprocess data for plotting
    """
    obs_list,syn_list,event_list,event_info = collect_data()

    f_start = 1/t_end
    f_end = 1/t_start
    
    # necessary for trace rotation
    station_lat = 49.144001
    station_lon = 12.8782
    
    # fetch event identifiers
    identifier = []
    for quakes in event_info:
        id_temp = str(quakes.resource_id).split('/')[2]
        index_temp = event_list.index(id_temp)
        identifier.append(index_temp)
    
    for i,(event,ident) in enumerate(zip(event_list,identifier)):
        print(event)
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

        quakeml = event_info[ident]
        origin_time = quakeml.origins[0].time
        event_lat = quakeml.origins[0].latitude
        event_lon = quakeml.origins[0].longitude

        # preprocessing, same for both observations and synthetics
        # NOTE: observations already have instrument response removed, in units
        #       of micrometers/second (from determag.py)
        common_sampling_rate = min([obs[0].stats.sampling_rate,
                                   syn[0].stats.sampling_rate])
        for streams in [obs,syn]:
            streams.resample(common_sampling_rate)
            streams.detrend("demean")
            streams.detrend("linear")
            streams.trim(starttime=origin_time,
                         endtime=origin_time+2*3600)
            streams.detrend(type='linear')
            streams.taper(max_percentage=0.05)
            streams.filter('lowpass',freq=1)
            streams.filter('highpass',freq=1/100)
            streams.filter('bandpass',freqmin=f_start,freqmax=f_end,
                                                corners=3,zerophase=True)


        # rotate horizontal trace to theoretical backazimuth
        # can use client.distaz() to check baz values
        BAz = gps2dist_azimuth(lat1=event_lat,
                               lon1=event_lon,
                               lat2=station_lat,
                               lon2=station_lon)
        for tr in obs:
            tr.stats.back_azimuth = BAz[2]

        obs_velocityN = obs.select(id='GR.WET..BHN')
        obs_velocityE = obs.select(id='GR.WET..BHE')
        # print('N: ',obs_velocityN[0].data.max())
        # print('E: ',obs_velocityE[0].data.max())
        obs.rotate(method='NE->RT')

        # divy up traces 
        # observations
        obs_velocityZ = obs.select(id='GR.WET..BHZ')
        obs_rot_rateZ = obs.select(id='BW.RLAS..BJZ')
        obs_velocityT = obs.select(id='GR.WET..BHT')
        obs_accelT = obs.select(id='GR.WET..BHT').differentiate(
                                                            method='gradient')

        # synthetics
        syn_velocityZ = syn.select(id='BW.RLAS.S3.MXZ').differentiate(
                                                            method='gradient')
        syn_rot_rateZ = syn.select(id='BW.RLAS.S3.MYZ').differentiate(
                                                            method='gradient')
        syn_accelT = syn.select(id='BW.RLAS.S3.MXT').differentiate(
                                  method='gradient').differentiate(
                                  method='gradient')

        if pick == "rr":
            obs_data = obs_rot_rateZ[0].data
            syn_data = syn_rot_rateZ[0].data
        elif pick == "zv":
            obs_data = obs_velocityZ[0].data
            syn_data = syn_velocityZ[0].data
            
        # synthetic data from m/s to um/s and rad/s to nrad/s
        for tr_syn in syn:
            typelabel = tr_syn.get_id()[-2]
            if typelabel == "Y": # rotation (nrad/s)
                tr_syn.data *= 10E9
            elif typelabel == "X": # velocity (um/s)
                tr_syn.data *= 10E6
        
        # time axes and peak indices
        obs_SR = obs_rot_rateZ[0].stats.sampling_rate
        obs_maxtime = len(obs_data)/obs_SR
        obs_time = np.linspace(0,obs_maxtime,len(obs_data))        
        obs_peak = max(obs_data)
        obs_peak_ind = np.where(obs_data == obs_peak)[0][0]/obs_SR
        
        syn_SR = syn_rot_rateZ[0].stats.sampling_rate
        syn_maxtime = len(syn_data)/syn_SR
        syn_time = np.linspace(0,syn_maxtime,len(syn_data))
        syn_peak = max(syn_data)
        syn_peak_ind = np.where(syn_data == syn_peak)[0][0]/syn_SR
        
        plot_comparison(event=event,
                        x_obs=obs_time,x_syn=syn_time,
                        y_obs=obs_rot_rateZ[0],
                        y_syn=syn_rot_rateZ[0])
        
        plot_comparison(event=event,
                        x_obs=obs_time,x_syn=syn_time,
                        y_obs=obs_velocityZ[0],
                        y_syn=syn_velocityZ[0])
        
        # rotation rate vs transverse acceleration
        obs_TAdata = obs_accelT[0].data
        obs_RRdata = obs_rot_rateZ[0].data
        syn_TAdata = syn_accelT[0].data
        syn_RRdata = syn_rot_rateZ[0].data
        
        plot_phasematch(event,obs_time,syn_time,
                        obs_TAdata,obs_RRdata,
                        syn_TAdata,syn_RRdata)
        

def plot_comparison(event,x_obs,x_syn,y_obs,y_syn,
                                            twinax=True,save=False,show=True):
    """between observation and synthetic waveforms
    """
    # divy out data and determine trace properties
    filler = "Vertical velocity ($\mu$m s$^-1$)"
    if y_obs.stats.channel[1] == "J":
        filler = "Rotation Rate (nrad s$^-1$)"
    y_obs = y_obs.data
    y_syn = y_syn.data
    
    f,ax1 = plt.subplots(1,figsize=(11.69,8.27),dpi=100)
    
    # observations
    a1= ax1.plot(x_obs,y_obs,'r',label='Obs.',zorder=2)
    # ax1.plot(obs_peak_ind,obs_peak,'ro',zorder=3)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(filler)
    ax1.grid(zorder=0)

    # synthetics
    ax2 = ax1
    if twinax:
        ax2 = ax1.twinx()
    a2 = ax2.plot(x_syn,y_syn,'k',label='Syn.',zorder=1)
    # ax2.plot(syn_peak_ind,syn_peak,'ko',zorder=4)
    ax2.set_ylabel('Synthetic {F}'.format(F=filler))
    ax2.grid(zorder=0)

    # create legend
    lines = a1 + a2
    labels = [l.get_label() for l in lines]
    # ax1.legend(lines,labels,loc=0)
    align_yaxis(ax1, 0, ax2, 0)

    # set yticks equal for twinx
    ax1bound = abs(max([ax1.get_ybound()[0],ax1.get_ybound()[1]]))
    ax2bound = abs(max([ax2.get_ybound()[0],ax2.get_ybound()[1]]))
    ax1.set_yticks(np.linspace(-1*ax1bound,ax1bound, 7))
    ax2.set_yticks(np.linspace(-1*ax2bound,ax2bound, 7))
    __pretty_grids(ax1)
    __pretty_grids(ax2)
    # plt.xlim([0,obs_maxtime])
    
    plt.title(event)
    
    if save:
        figurename = os.path.join('./figures','waveforms',event+'_compare.png')
        plt.savefig(figurename,dpi=600,figsize=(11,7))
    if show:
        plt.show()

def plot_phasematch(event,x_obs,x_syn,y_TAobs,y_RRobs,y_TAsyn,y_RRsyn,
                                                        save=False,show=True):
    """comparison of transverse acceleration and rotation rate
    """
    f2,(ax3,ax4) = plt.subplots(2,sharex=True,figsize=(11.69,8.27),dpi=100)

    # normalize
    for DATA in [y_TAobs,y_RRobs,y_TAsyn,y_RRsyn]:
        DATA /= DATA.max()
        
    ax3.plot(x_obs,y_TAobs,'k',label='Transverse Acc.')
    ax3.plot(x_obs,y_RRobs,'r',label='Rotation Rate')
    ax4.plot(x_syn,y_TAsyn,'k',label="Transverse Acc.")
    ax4.plot(x_syn,y_RRsyn,'r',label="Rotation Rate")

    for ax in [ax3,ax4]:
        __pretty_grids(ax)
        ax.set_ylabel('Normalized amplitude')
        ax.legend(prop={"size":7.5})

    ax3.set_title(event)

    ax4.set_xlabel('Time (s)')
    plt.subplots_adjust(hspace=0)
    if save:
        figurename = os.path.join('./figures','waveforms',event+'_phmatch.png')
        plt.savefig(figurename,dpi=600,figsize=(11,7))
    if show:
        plt.show()
        
if __name__ == "__main__":
    process_and_plot(t_start=5,t_end=60,pick='zv')



