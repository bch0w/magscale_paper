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
mpl.rcParams['axes.linewidth']=2
mpl.rcParams.update({'font.size': 16.5})
from matplotlib.ticker import MaxNLocator

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



# ================================ FUNCTIONS ==================================
def __pretty_grids(input_ax):
    """grid formatting
    """
    input_ax.set_axisbelow(True)
    input_ax.tick_params(which='major',
                         direction='in',
                         top=True,
                         right=True,
                         width=1,
                         length=5)
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
    # input_ax.ticklabel_format(style='sci',
    #                         axis='y',
    #                         scilimits=(0,0))

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

    obs_wet_list = glob.glob(obs_list_path + '*WET.mseed')
    obs_rlas_list = glob.glob(obs_list_path + '*_RLAS.pickle')
    syn_list = glob.glob(syn_list_path + '*RLAS.mseed')
    pathlists = obs_wet_list  + obs_rlas_list + syn_list

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
    event_list = ['C201109161926A']

    event_ignore = ['C201007181335A', #contains foreshock
                    'C201304161044A', #starts too early, foreshock?
                    'C201502131859A', #starts too early
                    'C201504250611A'] #long ringing, starts early


    # get event information
    event_info = read_events(inv_path)
    
    return pathlists,event_list,event_info,event_ignore

def process_and_plot(t_start=5,t_end=60):
    """preprocess data for plotting
    """
    v = lambda m,b,d,c: 2*np.pi*10**(m-b*np.log10(d)-c)
    
    pathlists,event_list,event_info,event_ignore = collect_data()

    f_start = 1/t_end
    f_end = 1/t_start
    
    # necessary for trace rotation
    station_lat = 49.144001
    station_lon = 12.8782
    
    # BEGIN FORLOOP
    for i,event in enumerate(event_list):
        print(event)
        # ignore events
        if event in event_ignore:
            continue

        # grab unique obs and syn by tag name
        for tmp in pathlists:
            if (event in tmp) and ('RLAS.pickle' in tmp):
                obs_path_rlas = tmp
            elif (event in tmp) and ('WET.mseed' in tmp):
                obs_path_wet = tmp
            elif (event in tmp) and ('RLAS.mseed' in tmp):
                syn_path = tmp

        obs_wet = read(obs_path_wet,format='mseed')
        obs_rlas = read(obs_path_rlas,format='pickle')
        syn = read(syn_path,format='mseed')
        
        # remove instrument response from RLAS (counts to nrad/s)
        obs_rlas.remove_sensitivity()
        obs_rlas[0].data *= 1E9

        
        # convert data from m/s to um/s and rad/s to nrad/s
        for tr_obs in obs_wet:
            tr_obs.data *= 1E6
                
        obs = obs_wet + obs_rlas

        for tr_syn in syn:
            typelabel = tr_syn.get_id()[-2]
            if typelabel == "Y": # rotation (nrad/s)
                tr_syn.data *= 1E9
            elif typelabel == "X": # velocity (um/s)
                tr_syn.data *= 1E6
        
        # parse out earthquake information
        for quake in event_info:
            id_temp = str(quake.resource_id).split('/')[2]
            if event == id_temp:
                break

        origin_time = quake.origins[0].time
        event_lat = quake.origins[0].latitude
        event_lon = quake.origins[0].longitude
        magnitude = quake.magnitudes[0].mag

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
            streams.filter('lowpass',freq=1,zerophase=True)
            streams.filter('highpass',freq=1/100,zerophase=True)
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
        obs.rotate(method='NE->RT')

        # divy up traces (observations)

        obs_velocityZ = obs.select(id='GR.WET..?HZ')
        obs_rot_rateZ = obs.select(id='BW.RLAS..BJZ')
        obs_velocityT = obs.select(id='GR.WET..?HT')
        obs_accelT = obs.select(id='GR.WET..?HT').differentiate(
                                                            method='gradient')

        # synthetics
        syn_velocityZ = syn.select(id='BW.RLAS.S3.MXZ').differentiate(
                                                            method='gradient')
        syn_rot_rateZ = syn.select(id='BW.RLAS.S3.MYZ').differentiate(
                                                            method='gradient')
        syn_accelT = syn.select(id='BW.RLAS.S3.MXT').differentiate(
                                  method='gradient').differentiate(
                                  method='gradient')

            
        
        # +++++++++ collect peak amplitudes        
        distance = (BAz[0]/1E3)/111.19
        expected_amplitude = v(magnitude,1.66,distance,0.3)
        print("M: {M}\nD: {D}\nZ_obs: {O}\nZ_syn: {S}\nZ_exp: {E}".format(
                                                    M=magnitude,
                                                    D=distance,
                                                    O=obs_velocityZ.max()[0],
                                                    S=syn_velocityZ.max()[0],
                                                    E=expected_amplitude)
                                                    )
        # +++++++++ additional breakpoint
        
        # time axes and peak indices
        obs_SR = obs_rot_rateZ[0].stats.sampling_rate
        obs_maxtime = len(obs_rot_rateZ[0])/obs_SR
        obs_time = np.linspace(0,obs_maxtime,len(obs_rot_rateZ[0]))        
        # obs_peak = max(obs_rot_rateZ[0])
        # obs_peak_ind = np.where(obs_rot_rateZ[0] == obs_peak)[0][0]/obs_SR
        
        syn_SR = syn_rot_rateZ[0].stats.sampling_rate
        syn_maxtime = len(syn_rot_rateZ[0])/syn_SR
        syn_time = np.linspace(0,syn_maxtime,len(syn_rot_rateZ[0]))
        # syn_peak = max(syn_rot_rateZ[0])
        # syn_peak_ind = np.where(syn_rot_rateZ[0] == syn_peak)[0][0]/syn_SR
        
        # plot_comparison(event=event,
        #                 x_obs=obs_time,x_syn=syn_time,
        #                 y_obs=obs_rot_rateZ[0],
        #                 y_syn=syn_rot_rateZ[0],
        #                 save=True)
        # 
        # plot_comparison(event=event,
        #                 x_obs=obs_time,x_syn=syn_time,
        #                 y_obs=obs_velocityZ[0],
        #                 y_syn=syn_velocityZ[0],
        #                 save=True)
        
        # rotation rate vs transverse acceleration
        obs_TAdata = obs_accelT[0].data
        obs_RRdata = obs_rot_rateZ[0].data
        syn_TAdata = syn_accelT[0].data
        syn_RRdata = syn_rot_rateZ[0].data
        
        print("obs_c = {}".format(obs_TAdata.max()/(2*obs_RRdata.max())))
        print("syn_c = {}".format(syn_TAdata.max()/(2*syn_RRdata.max())))

        
        # plot_phasematch(event,obs_time,syn_time,
        #                 obs_TAdata,obs_RRdata,
        #                 syn_TAdata,syn_RRdata,
        #                 show=False,
        #                 save=True)
        plot_phasematch_single(event,obs_time,obs_TAdata,obs_RRdata,save=False,show=True)        
# =============================== PLOTTING FUNCTIONS ==========================
def plot_comparison(event,x_obs,x_syn,y_obs,y_syn,
                                            twinax=True,save=False,show=True):
    """between observation and synthetic waveforms
    """
    # divy out data and determine trace properties
    filler = "Vertical velocity ($\mu$m s$^{-1}$)"
    C_ = y_obs.stats.channel
    if C_[1] == "J":
        filler = "Rotation Rate (nrad s$^{-1}$)"
    y_obs = y_obs.data
    y_syn = y_syn.data
    
    # f,ax1 = plt.subplots(1,figsize=(11.69,8.27),dpi=100)
    f,ax1 = plt.subplots(1,figsize=(11.69,8.27),dpi=100)
    
    # observations
    a1= ax1.plot(x_obs,y_obs,'k',label='Obs.',zorder=4)
    # ax1.plot(obs_peak_ind,obs_peak,'ro',zorder=3)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(filler)
    ax1.grid(zorder=0)

    # synthetics
    ax2 = ax1
    if twinax:
        ax2 = ax1.twinx()
    a2 = ax2.plot(x_syn,y_syn,'r',label='Syn.',zorder=3)
    # ax2.plot(syn_peak_ind,syn_peak,'ko',zorder=4)
    ax2.set_ylabel('Synthetic {F}'.format(F=filler),rotation=-90,labelpad=20)
    ax2.grid(zorder=0)

    # create legend
    lines = a1 + a2
    labels = [l.get_label() for l in lines]
    # ax1.legend(lines,labels,loc=0)
    align_yaxis(ax1, 0, ax2, 0)
    for ax in [ax1,ax2]:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
        ax.set_xlim([x_obs.min(),x_obs.max()])
    
    # set yticks equal for twinx
    ax1bound = abs(max([ax1.get_ybound()[0],ax1.get_ybound()[1]]))
    ax2bound = abs(max([ax2.get_ybound()[0],ax2.get_ybound()[1]]))
    ax1.set_yticks(np.linspace(-1*ax1bound,ax1bound, 7))
    ax2.set_yticks(np.linspace(-1*ax2bound,ax2bound, 7))


    __pretty_grids(ax1)
    __pretty_grids(ax2)
    # plt.xlim([0,obs_maxtime])
    
    # plt.title(event)
    
    if save:
        figurename = os.path.join('./figures','waveforms',event+'_compare{}.png'.format(C_))
        plt.savefig(figurename,dpi=200,figsize=(11,8.75))
    if show:
        plt.show()

def plot_phasematch(event,x_obs,x_syn,y_TAobs,y_RRobs,y_TAsyn,y_RRsyn,
                                                        save=False,show=True):
    """comparison of transverse acceleration and rotation rate
    """
    f2,(ax3,ax4) = plt.subplots(2,sharex=True,figsize=(11,8.5),dpi=200)

    # normalize
    for DATA in [y_TAobs,y_RRobs,y_TAsyn,y_RRsyn]:
        DATA /= DATA.max()
        
    ax3.plot(x_obs,y_TAobs,'k',label='Transverse Acc.')
    ax3.plot(x_obs,y_RRobs,'r',label='Rotation Rate')
    ax4.plot(x_syn,y_TAsyn,'k',label='Transverse Acc.')
    ax4.plot(x_syn,y_RRsyn,'r',label="Rotation Rate")

    for ax in [ax3,ax4]:
        __pretty_grids(ax)
        ax.set_ylim([-1.1,1.1])
        ax.set_xlim([500,x_syn.max()])
        # ax.legend(prop={"size":7.5})
        
    f2.text(0.04, 0.5, 'Normalized amplitude', va='center', rotation='vertical')
    # ax3.set_title(event)
    ax4.set_xlabel('Time (s)')
    plt.subplots_adjust(hspace=0)
    if save:
        figurename = os.path.join('./figures','waveforms',event+'_phmatch.png')
        plt.savefig(figurename,dpi=200,figsize=(11,8.5))
    if show:
        plt.show()

def plot_phasematch_single(event,x_obs,y_TAobs,y_RRobs,save=False,show=True):
    """comparison of transverse acceleration and rotation rate
    """
    f2,(ax3) = plt.subplots(1,sharex=True,figsize=(11,8.5),dpi=100)

    # normalize
    for DATA in [y_TAobs,y_RRobs]:
        DATA /= DATA.max()
        
    ax3.plot(x_obs,y_TAobs,'k',label='Transverse Acc.')
    ax3.plot(x_obs,y_RRobs,'r',label='Rotation Rate')

    __pretty_grids(ax3)
    ax3.set_ylim([-1.1,1.1])
        
    ax3.set_ylabel('Normalized amplitude')
    # ax3.set_title(event)
    ax3.set_xlabel('Time (s)')
    if save:
        figurename = os.path.join('./figures','waveforms',event+'_phmatch.png')
        plt.savefig(figurename,dpi=200,figsize=(11,8.5))
    if show:
        plt.show()
        
if __name__ == "__main__":
    process_and_plot(t_start=5,t_end=60)



