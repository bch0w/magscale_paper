"""determine peak trace ampltidues from synthetic data (pyasdf format) 
outputted by specfem3d globe
"""

from __future__ import division
import os
import sys
import json
import glob
import obspy
import heapq
import shutil
import pyasdf
# import urllib2
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

def process_save(station,station_tag,event):
    
    # set sampling rate and copy traces for filtering
    rt = station.synthetic.select(channel='MY*') # rotation
    tr = station.synthetic.select(channel='MX*') # displacement

    sampling_rate = int(rt[0].stats.sampling_rate)
    
    velocity_rtz = tr.copy()
    velocity_rtz.differentiate(method='gradient')

    rotation = rt.copy()
    rot_rate = rt.copy()
    rot_rate.differentiate(method='gradient')

    # change channel naming convention for rotation rate
    rot_rate.select(component='Z')[0].stats.channel = 'MyZ'
    rot_rate.select(component='R')[0].stats.channel = 'MyR'
    rot_rate.select(component='T')[0].stats.channel = 'MyT'

    surf_big = 0
    surf_end = len(rt[0].data)-1

    # bandpass all traces for 3s to 60s
    f_start = 1/35
    f_end = 1/25
    for traces in [rot_rate,rotation,velocity_rtz]:
        traces.filter('bandpass', freqmin=f_start, freqmax=f_end, corners=3,
                  zerophase=True)

    # seperate streams into traces and create list for ease of processing 
    alltraces = []
    for traces in [rot_rate,rotation,velocity_rtz]:
        for comp in ['Z','R','T']:
            alltraces.append(traces.select(component=comp)[0])

    # search criteria for adjacent peaks and troughs, 20s period (trial&error)
    search = int(20 * sampling_rate)

    # choosing amplitudes/indices/zerocrossing for each trace
    # plot lists will contain different (11) parameters to be plotted
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
    axrow = len(alltraces)
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
        if channelS[i][:2] == 'My':
            axes[i][0].set_ylabel('{} rot. rate (rad/s)'.format(channelS[j][2]))
        elif channelS[i][:2] == 'MY':
            axes[i][0].set_ylabel('{} rotation (rad)'.format(channelS[j][2]))
        else:
            axes[i][0].set_ylabel('{} vel.(m/s)'.format(channelS[j][2]))


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
            axes[i][0].set_title(station_tag)
            axes[i][1].set_title(station.synthetic[0].stats.starttime)
        elif j == axrow-1:
            axes[i][0].set_xlabel('sample')
            axes[i][1].set_xlabel('sample')
            axes[i][1].set_xlim([min(zero_crossings)-2000,
                                                max(zero_crossings)+2000])
        j+=1

    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0.1)

    output_path = './output/{}/'.format(event)
    image_path = os.path.join(output_path,'imgs',station_tag+'.png')
    plt.savefig(image_path,dpi=150)
    plt.close()

    return peak2troughs,periods,zero_crossings_abs


def store_info_json(event,station,peak2troughs,periods,zero_crossings_abs,
                                                                station_tag):
    
    # create all info to be stored in json
    rrz_max,rrr_max,rrt_max,rtz_max,rtr_max,rtt_max,vlz_max,vlr_max,vlt_max \
                                                            = peak2troughs[:]
    rrz_per,rrr_per,rrt_per,rtz_per,rtr_per,rtt_per,vlz_per,vlr_per,vlt_per \
                                                                = periods[:]
    rrz_zca,rrr_zca,rrt_zca,rtz_zca,rtr_zca,rtt_zca,vlz_zca,vlr_zca,vlt_zca \
                                                        = zero_crossings_abs[:]

    station_coords = station.StationXML.networks[0].stations[0]                                                    

     dic = OrderedDict([
            ('event_id', event.resource_id.id),
            ('starttime', str(event.origins[0].time)),
            ('network', station.synthetic[0].stats.network),
            ('station', station.synthetic[0].stats.station),
            ('loc', station.synthetic[0].stats.location),
            ('station_latitude', station_coords.latitude),
            ('station_longitude', station_coords.longitude),
            ('station_elevation', station_coords.elevation),
            ('event_latitude', event.origins[0].latitude),
            ('event_longitude', event.origins[0].longitude),
            ('moment', event.focal_mechanisms[0].moment_tensor.scalar_moment),
            ('depth', event.origins[0].depth/1000),
            ('epicentral_distance_in_km', distance),
            ('zero_crossing_unit','sec. from trace start'),           
            ('vertical_rotation_rate', 
                OrderedDict([
                ('peak_amplitude',rrz_max),
                ('dominant_period',rrz_per),
                ('zero_crossing',rrz_zca),
                ('unit','nrad/s')
                ])
            ),
            ('radial_rotation_rate', 
                OrderedDict([
                ('peak_amplitude',rrr_max),
                ('dominant_period',rrr_per),
                ('zero_crossing',rrr_zca),
                ('unit','nrad/s')
                ])
            ),
            ('transverse_rotation_rate', 
                OrderedDict([
                ('peak_amplitude',rrt_max),
                ('dominant_period',rrt_per),
                ('zero_crossing',rrt_zca),
                ('unit','nrad/s')
                ])
            ),
            ('vertical_rotation', 
                OrderedDict([
                ('peak_amplitude',rtz_max),
                ('dominant_period',rtz_per),
                ('zero_crossing',rtz_zca),
                ('unit','nrad')
                ])
            ),
            ('radial_rotation', 
                OrderedDict([
                ('peak_amplitude',rtr_max),
                ('dominant_period',rtr_per),
                ('zero_crossing',rtr_zca),
                ('unit','nrad')
                ])
            ),
            ('transverse_rotation', 
                OrderedDict([
                ('peak_amplitude',rtt_max),
                ('dominant_period',rtt_per),
                ('zero_crossing',rtt_zca),
                ('unit','nrad')
                ])
            ),
            ('vertical_velocity', 
                OrderedDict([
                ('peak_amplitude',vlz_max),
                ('dominant_period',vlz_per),
                ('zero_crossing',vlz_zca),
                ('unit','nm/s')
                ])
            ),
            ('transverse_velocity', 
                OrderedDict([
                ('peak_amplitude',vlt_max),
                ('dominant_period',vlt_per),
                ('zero_crossing',vlt_zca),
                ('unit','nm/s')
                ])
            ),
            ('radial_velocity', 
                OrderedDict([
                ('peak_amplitude',vlr_max),
                ('dominant_period',vlr_per),
                ('zero_crossing',vlr_zca),
                ('unit','nm/s')
                ])
            )
            ])

    output_path = './output/{}/'.format(event)
    json_path = os.path.join(output_path,'jsons',station_tag+'.json')
    outfile = open(json_path, 'wt')
    json.dump(dic, outfile, indent=4)
    outfile.close()


# MAIN
# create catalog
filepath = '/import/netapp-m-02-terra/bernhard/RUNS_SEM3D/S40RTS/S40RTS_MAGSCALE/OUTPUT_FILES'
cat = glob.glob(filepath + 'synthetic*A') 

for event in cat:
    # make event specific folder
    output_path = './output/{}/'.format(event)
    os.makedirs(os.path.join(output_path,'imgs'))
    os.makedirs(os.path.join(output_path,'jsons'))

    # read in data
    ds = pyasdf.ASDFDataSet(event)
    i=-1
    error_list = []

    # process each station
    for station in ds.waveforms:
        # for now, skip over synthetic stations
        if station.synthetic[0].stats.network == 'GG':
            continue
        i+=1
        print(i)

        station_tag = station.synthetic[0].stats.network+'_'+\
                                        station.synthetic[0].stats.station

        print(station_tag)
        event = ds.events[0]
        peak2troughs,periods,zero_crossings_abs = process_save(station,station_tag,event)
        store_info_json(event,station,peak2troughs,periods,zero_crossings_abs,station_tag)

        # except Exception as e:
        #     error_list.append('{} {}\t{}'.format(i,station_tag,e))


 
