"""09.05.17 - rr_magscale
Used for plotting the rotation rate magnitude scale
Also does linear regression for a subset of events filtered to get a
magnitude scale equation, plots events, equation and projected lines
"""
import sys
import math
import glob
import json
import random
import warnings
import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from numpy.linalg import inv
from obspy import UTCDateTime
from mpl_toolkits.basemap import Basemap
from obspy.geodetics.base import locations2degrees


# ignoring polyval rank warning - look into
warnings.simplefilter('ignore', np.RankWarning)
matplotlib.rcParams.update({'font.size': 11})

# for matplotlib font change
# matplotlib.rcParams.update({'font.size': 25})
# hfont = {'fontname':'Times New Roman'}
# matplotlib.rc('font',family='Times New Roman')

# del matplotlib.font_manager.weight_dict['roman']
# matplotlib.font_manager._rebuild()


def normalize(list):
    """Normalize the amplitudes of a list
    :type list: list
    :param list: list to be normalized
    :rtype norm_list: list
    :return norm_list: normalized list
    """
    norm_list = []
    min_list = min(list)
    max_list = max(list)
    diff = max_list - min_list
    for ni in range(0,len(list)):
        norm_list.append((list[ni] - min_list) / diff)

    return norm_list


def epidist(lat1, lon1, lat2, lon2):
    """Converts lat and lon lists to epicentral distance, iterates on lat1/lon1
    :type ev_lat/ev_lon: float
    :param ev_lat/ev_lon: lat/lon of event
    :rtype: np.array of floats
    :return: epicentral distance in degrees
    """
    delta=[]        
    for i in range(0,len(lat1)):
        delta.append(locations2degrees(lat1[i],lon1[i],lat2,lon2))

    return delta


def filter_for(num_fil,parms,vmin,vmax):
    """Filters parameter list between two bounds, inclusive
    :type num_fil: integer
    :param num_fil: index of parameter in list that will be filtered
    :type parms: list
    :param parms: list of parameters arranged in np.arrays
    :type vmin/vmax: float
    :param vmin/vmax: lower and upper bounds
    :rtype filtd: list
    :return filtd: filtered param list in the same order given
    """
    filtd = [[] for _ in range(len(parms))]
    for ff_i in range(0,len(parms[num_fil])):
        if (vmin <= parms[num_fil][ff_i] < vmax):
            for ff_j in range(0,len(parms)):
                filtd[ff_j].append(parms[ff_j][ff_i])

    return filtd


def hold_filter(num_fil,parms,vmin,vmax,vhold,holdnum):
    """Filters parameter list between two bounds, for a specific parameter
    :type num_fil: integer
    :param num_fil: index of parameter in list that will be filtered
    :type parms: list
    :param parms: list of parameters arranged in np.arrays
    :type vmin/vmax: float
    :param vmin/vmax: lower and upper bounds
    :type vhold: list
    :param vhold: the parameter that you wish to hold constant
    :type holdnum: float
    :param holdnum: the value of vhold you wish to filter for
    :rtype hfiltd: list
    :return hfiltd: filtered param list in the same order given
    """
    hfiltd = [[] for _ in range(len(parms))]
    for hf_i in range(0,len(parms[num_fil])):
        if math.floor(parms[vhold][hf_i]) == holdnum:
            if (vmin <= parms[num_fil][hf_i] < vmax):
                for hf_j in range(0,len(parms)):
                    hfiltd[hf_j].append(parms[hf_j][hf_i])
        else:
            for hf_jj in range(0,len(parms)):
                    hfiltd[hf_jj].append(parms[hf_jj][hf_i])
    return hfiltd



def loglog_misfit(dist,amps,magnis,check_ll):
    """Plots power law fitted lines for the given data, requires distance,
    amplitudes and magnitudes
    :type magnis: float 
    :param magnis: list of magnitudes to check_ll
    :type check_ll: integer
    :param check_ll: choose rotation rate w/ 0 or vertical displacement w/ 1
    """
    log_x = np.logspace(0,2.3,50,base=10)
    for ms in np.arange(4,10):
        mf_del=[];mf_surf=[]
        for j in range(0,len(magnis)):
            if math.floor(magnis[j]) == ms and check_ll == 0:
                mf_del.append(dist[j])
                # mf_surf.append(amps[j]*10**(-9))
                mf_surf.append(amps[j])
            elif math.floor(magnis[j]) == ms and check_ll == 1:
                mf_del.append(dist[j])
                mf_surf.append(amps[j]*10**(-3))
        if not mf_del:
            continue    
        log_del = np.log10(mf_del)
        log_surf = np.log10(mf_surf)
        polly = np.polyfit(log_del,log_surf,1) 
        if polly[0] > 0: continue
        y = (10**polly[1]) * (log_x**polly[0]) 
        # plt.plot(log_x,y,'k--',linewidth=2.0)
        # plt.annotate(xy=(log_x[13],y[13]),s='M{}'.format(ms))

        return polly[0],polly[1]


def leasquares(amplitudes,distances,magnitudes):
    """ numpy matrices are confusing so to get into the correct order we need to 
    initially transpose, at initialization G and d are Nx2 and 1x2 matrices, 
    resp. We need to solve the equation m = (GTG)^-1GTd where GT is G tranpose

    For the equation Mr = log(A) + Blog(D) + C => (B,C) = (m[0],m[1])
    
    :type amplitudes: array
    :param amplitudes: amplitude values for mag scale
    :type distances: array
    :param distances: epicentral distances in degrees
    :type magnitudes: array
    :param magnitudes: magnitudes in Ms or Mw
    :rtype m: float
    :param m: coefficients in magnitude equation
    """
    G = [np.log10(ds),np.ones(len(distances))]
    G = (np.asmatrix(G)).transpose() 

    d_hold = []
    for i4 in range(0,len(magnitudes)):
        d_hold.append(magnitudes[i4]-np.log10(amplitudes[i4]))
    d = (np.asmatrix(d_hold)).transpose() 

    GTG = np.dot(G.transpose(),G)
    GTGi = inv(GTG)
    GTGiGT = np.dot(GTGi,G.transpose())
    m = np.dot(GTGiGT,d)

    return float(m[0]),float(m[1]),GTGi,G.shape


def confidence(data,mags,dists,GTGi,nxp,m0,m1):
    """calculate the confidence interval for the least squares regression.
    an estimator for the confidence interval is given for estimator m, and 
    for an n x p matrix G. We use a c value of 1.96 for 95 percent confidence.
    need to check as our sample size is quite small 

            (GTG)^-1 * (sum(residuals))/(n-p)
    """
    c = 1.96
    n = nxp[0]
    p = nxp[1]
    resi, resi_sq = [],[]
    y = lambda x,B,C,Mr: 10**(Mr-B*np.log10(x)-C)

    for j in range(len(data)):
        resi_sq.append((np.log10(data[j])-
                            np.log10(y(dists[j],m0,m1,mags[j])))**2)
        resi.append((np.log10(data[j])-np.log10(y(dists[j],m0,m1,mags[j]))))

    noise_varia = (1/(n-p)) * sum(resi_sq)
    sum_resi_sq = sum(resi_sq)
    print('Sum of squared residuals: ',sum_resi_sq)
    var_b1 = GTGi.item(0) * noise_varia
    var_b2 = GTGi.item(3) * noise_varia

    m0_nci = m0 - c*np.sqrt(var_b1)
    m0_pci = m0 + c*np.sqrt(var_b1)
    m1_nci = m1 - c*np.sqrt(var_b2)
    m1_pci = m1 + c*np.sqrt(var_b2)

    return m0_nci,m0_pci,m1_nci,m1_pci,resi,sum_resi_sq

def plot_magnitude_lines(magnitude):
    """plot comparisons of different magnitude values
    """
    ax.plot(x,y(x,1.66,0.3,magnitude),'k',zorder=5,linewidth=2.5,
                      label='$M_{S}^{BB}$',color='k')
    # ax.plot(x,y(x,1.823,1.0,M),'k',zorder=5,linewidth=2.5,
    #                     label='$M_{R}^{WET}$',color='r')
    ax.plot(x,y(x,1.084,1.093,magnitude),'k',zorder=5,linewidth=2.5,
                      label='$M^{WET}_{Z}$',color='g')
    ax.plot(x,y(x,1.095,1.09,magnitude),'k',zorder=5,linewidth=2.5,
                      label='$M_{Z}^{FUR}$',color='b')
    ax.plot(x,y(x,1.45,0.527,magnitude),'k',zorder=5,linewidth=2.5,
                      label='$M_{T}^{WET}$',color='m')
    ax.plot(x,y(x,1.442,0.447,magnitude),'k',zorder=5,linewidth=2.5,
                      label='$M_{T}^{FUR}$',color='c')
    ax.set_xlim([0,160])
    ax.set_ylim([10,10**7])
    ax.set_yscale("log", nonposx='clip')
    ax.set_xlabel('Distance ($\Delta$)')
    ax.set_ylabel('Peak Velocity (nm/s)')
    ax.set_title('M7 Velocity Scale Comparison')
    ax.grid(which='both')   

    plt.legend()
    plt.show()

def plot_histograms():
    """make histograms of data
    """
    baz
    f = plt.figure()
    ax = plt.subplot(111,polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    nbins = 16
    hist,bin_edge = np.histogram(baz,range=(0,360),bins=nbins)
    bin_edge = [_*(2*np.pi)/360 for _ in bin_edge]
    width = (2*np.pi)/nbins
    bars = ax.bar(bin_edge[:-1],hist,width,color='k',ecolor='w')

    for b in bars:
        b.set_facecolor("#%06x" % random.randint(0,0xFFFFFF))
        b.set_alpha(0.8)

    for be in range(len(hist)):
      plt.annotate(xy=(bin_edge[be]+width/2,hist[be]),s=hist[be],fontsize=10)

    ax.set_rmax(max(hist))    
    ax.set_yticklabels([])
    # ax.set_rlabel_position(118)
    ax.set_title('{} Events, BAZ histogram'.format(len(baz)))
    f.set_tight_layout(True)

    plt.show()

def plot_map(lons,lats,mags,depths):
    """plot events on a map, no shooting 
    """
    # draw base map
    plt.figure(figsize=(18, 9))
    map = Basemap(projection='eck4', lon_0=10,lat_0=30, resolution='l')
    map.drawmeridians(np.arange(0, 360, 30))
    map.drawparallels(np.arange(-90, 90, 30))
    map.drawcoastlines(linewidth=0.25)
    map.drawcountries(linewidth=0.25)
    map.fillcontinents(color='coral', lake_color='lightblue')
    map.drawmapboundary(fill_color='lightblue')
    
    # event lat and lon in map x y
    lat_coo,lon_coo = [],[]
    for i in range(0,len(lats)):
        lon_c, lat_c = map(lons[i],lats[i])
        lat_coo.append(lat_c)
        lon_coo.append(lon_c)

    # station lat and lon
    statlon, statlat = map(12.846, 49.145)
    map.scatter(statlon, statlat, 200, color='w', marker="*",
                        edgecolor="k", zorder=100)

    # use magnitudes for size
    for i in range(0,len(mags)): 
        mags[i] = (mags[i]**5)/50

    sct = map.scatter(lon_coo, lat_coo, s=mags, c=depths, marker=".",
                        edgecolor="k", zorder=100, cmap='viridis')
    
    cbar = map.colorbar(sct,location='bottom',pad="5%")

    # plot markers to show relative event size
    map.scatter(3.16202E6,9.41777E6,s=(6**5)/50,c='r',marker='.',edgecolor='k')
    map.scatter(3.16202E6,8.95919E6,s=(7**5)/50,c='c',marker='.',edgecolor='k')
    map.scatter(3.16202E6,8.46062E6,s=(8**5)/50,c='m',marker='.',edgecolor='k')
    plt.text(3.5E6,9.41777E6-0.2E6,'M6',fontsize=10)
    plt.text(3.5E6,8.95919E6-0.2E6,'M7',fontsize=10)
    plt.text(3.5E6,8.46062E6-0.2E6,'M8',fontsize=10)

    cbar.ax.set_xlabel('Depth (km)')

    plt.show()

# =================================== MAIN ====================================
parser = argparse.ArgumentParser(description='Magscale script.')
parser.add_argument('--scale',help='scale choice: rotation [rt], \
rotation rate [rr], vertical velocity [zv], transverse velocity [tv]', 
type=str, default=None)
parser.add_argument('--sta',help='station choice: wettzell [wet], \
fuerstenfeldbruck [fur], italy [ita], specfem [spec]', type=str, default='wet')
parser.add_argument('--log',help='log or loglog plot (default: log)',type=str,
    default='log')
parser.add_argument('--mag',help='magnitude choice for CI (default: 6)',
    type=int,default=6)

# parse out arguments
args = parser.parse_args()
pick = args.scale
if not pick: 
    parser.print_help()
    sys.exit()
sta = args.sta.lower()
log = args.log
mag = args.mag

# change rejected events to station specific?
reject_events = np.load('./output/reject_events.npz')

# make lists, though not all are used
event_IDs,ev_lats,ev_lons,sta_lats,sta_lons,pccs,file_IDs,depths,\
mags,Z_vel_max,T_vel_max,Z_rr_max,Z_rt_max,ds = [[] for _ in range(14)]

# load in data as numpy arrays, convert to lists of floats, calculate distance
if sta == 'wet':
    path = './output/processed_events/jsons/'
    filenames = glob.glob(path + '*')    

    for file in filenames:
        with open(file) as f:
            data = json.load(f)
            event_id = data['event_id'][-7:]
            if event_id in reject_events[sta]:
                continue
            file_IDs.append(file)
            event_IDs.append(event_id)
            ev_lats.append(data['event_latitude'])
            ev_lons.append(data['event_longitude'])
            depths.append(data['depth_in_km'])
            mags.append(data['magnitude'])
            pccs.append(data['peak_correlation_coefficient'])
            Z_vel_max.append(data['vertical_velocity']['peak_amplitude'])
            T_vel_max.append(data['transverse_velocity']['peak_amplitude'])
            Z_rr_max.append(data['vertical_rotation_rate']['peak_amplitude'])
            Z_rt_max.append(data['vertical_rotation']['peak_amplitude'])
            # zv_Tmax.append(data['vertical_vel']['dominant_period'])
            # tv_Tmax.append(data['transverse_vel']['dominant_period'])
            # rr_Tmax.append(data['vertical_rotation_rate']['dominant_period'])
            # rt_Tmax.append(data['vertical_rotation']['dominant_period'])


    sta_lat = float(data['station_latitude'])
    sta_lon = float(data['station_longitude'])
    ds = epidist(ev_lats,ev_lons,lat2=sta_lat,lon2=sta_lon)

# for synthetics made by specfem
elif sta == 'specfem':
    path = './specfem/output/jsons/'
    filenames = glob.glob(path + '*')
    for file in filenames:
        with open(file) as f:
            data = json.load(f)
            ID.append(data['network']+'_'+data['station'])
            sta_lats.append(data['station_latitude'])
            sta_lons.append(data['station_longitude'])
            Z_vel_max.append(data['peak_filtered_vertical_vel'])
            T_vel_max.append(data['peak_filtered_transverse_vel'])
            Z_rr_max.append(data['peak_vertical_rotation_rate'])
            Z_rt_max.append(data['peak_filtered_vertical_rotation'])

    ev_lat = data['event_latitude']
    ev_lon = data['event_longitude'] 
    mag = 2/3 * np.log10(data['moment']*10**7) - 10.7
    mags = [mag]*len(filenames)
    depth = data['depth']
    ds = epidist(sta_lats,sta_lons,lat2=ev_lat,lon2=ev_lon)
    
    # convert to units of nano radians/(m/s))
    Z_vel_max = [_*(10**9) for _ in Z_vel_max]
    T_vel_max = [_*(10**9) for _ in T_vel_max]
    Z_rt_max = [_*(10**9) for _ in Z_rt_max]
    Z_rr_max = [_*(10**9) for _ in Z_rr_max]



# pick amplitudes and divide by 2pi for mag equation, amplitudes should be 
# in units of nano by here
if pick == 'rr':
    label = 'Rotation Rate'
    units = 'nrad $\mathregular{s^{-1}}}$'
    psurf = [_/(2*np.pi) for _ in Z_rr_max]
elif pick == 'rt':
    label = 'Rotation'
    units = 'nrad'
    psurf = [_/(2*np.pi) for _ in Z_rt_max]
elif pick == 'zv':
    label = 'Vertical Velocity'
    units = 'nm/s'
    psurf = [_/(2*np.pi) for _ in Z_vel_max]
elif pick == 'tv':
    label = 'Transverse Velocity'
    units = 'nm/s'
    psurf = [_/(2*np.pi) for _ in T_vel_max]

# determine least squares and confidence intervals
x = np.linspace(1,200,201)
y = lambda x,B,C,Mr: x**(-B) * 10**(Mr-C) 
m0,m1,GTGi,nxp = leasquares(psurf,ds,mags)
m0n,m0p,m1n,m1p,residuals,resi_sq = confidence(psurf,mags,ds,GTGi,nxp,m0,m1)


# ========================== Plot events on map ================================
if sta != 'specfem':
    # zip together magnitudes and pcc's, sort by mags
    zipped_lists = zip(mags,pccs,event_IDs,file_IDs,ev_lats,ev_lons,depths)
    sorted_lists = sorted(zipped_lists,key=lambda x:x[0])
    m6, m65, m7, m75 = [],[],[],[]
    for groups in sorted_lists:
        if 6.0 <= groups[0] < 6.5:
            m6.append(groups)
        elif 6.5 <= groups[0] < 7.0:
            m65.append(groups)
        elif 7.0 <= groups[0] < 7.5:
            m7.append(groups)
        elif 7.5 <= groups[0] <= 8.0:
            m75.append(groups)

    # sort each new list by pcc
    m6_sorted = sorted(m6, key=lambda x:x[1], reverse=True)
    m65_sorted = sorted(m65, key=lambda x:x[1], reverse=True)
    m7_sorted = sorted(m7, key=lambda x:x[1], reverse=True)
    m75_sorted = sorted(m75, key=lambda x:x[1], reverse=True)

    plot_ev_lons,plot_ev_lats,plot_mags,plot_depths = [],[],[],[]
    for sorty in [m6_sorted,m65_sorted,m7_sorted,m75_sorted]:
        for top in range(0,10,1):
            plot_mags.append(sorty[top][0])
            plot_ev_lats.append(sorty[top][4])
            plot_ev_lons.append(sorty[top][5])
            plot_depths.append(sorty[top][6])
    import pdb;pdb.set_trace()

    plot_map(plot_ev_lons,plot_ev_lats,plot_mags,plot_depths)
    sys.exit()
# ============================== PLOT ==========================================
# plot Attributes and hacked up color choice
# f = plt.figure(1,dpi=150,figsize=(11,7))
f = plt.figure(1)
ax = plt.subplot(111)
major_ticks = np.arange(0, 170, 10)                                              
minor_ticks = np.arange(0, 170, 5)  
# colorhack = ['0','1','2','b','y','g','r','c','m']
colorhack = ['0','1','2','g','r','g','r','c','m','k']

# scatter plot points
delta=[];ps_scale=[];mag_filt=[]
MS = 40
for ix in range(len(psurf)):
    # if ID[ix] in manual_reject:
    #     continue
    if 3.0 <= mags[ix] < 4.0:
        ax.scatter(ds[ix],psurf[ix],c='#329932',marker='o',s=MS,zorder=10)
    elif 4.0 <= mags[ix] < 5.0:
        ax.scatter(ds[ix],psurf[ix],c='#ff0000',marker='o',s=MS,zorder=10)
    elif 5.0 <= mags[ix] < 6.0:
        ax.scatter(ds[ix],psurf[ix],c='#00ffff',marker='o',s=MS,zorder=10)
    elif 6.0 <= mags[ix] < 6.5:
        ax.scatter(ds[ix],psurf[ix],c='#ff0000',marker='o',s=MS,zorder=10)
    elif 6.5 <= mags[ix] < 7.0:
        ax.scatter(ds[ix],psurf[ix],c='#ff0000',marker='o',s=MS,zorder=10)
    elif 7.0 <= mags[ix] < 7.5:
        ax.scatter(ds[ix],psurf[ix],c='#00ffff',marker='o',s=MS,zorder=10)
    elif 7.5 <= mags[ix] <= 8.0:
        ax.scatter(ds[ix],psurf[ix],c='#00ffff',marker='o',s=MS,zorder=10)
    elif 8.0 <= mags[ix] <= 8.5:
        ax.scatter(ds[ix],psurf[ix],c='m',marker='o',s=MS,zorder=10)
    # plt.annotate(xy=(ds[ix]+.5,psurf[ix]),s='{}'.format(ID[ix]),
    #                 fontsize=7.5,zorder=10)
    # print('{} : {} , {}'.format(i,delta[ix],ps_scale[ix]))

# plot least squares, confidence intervals, magnitude lines
# ax.fill_between(x,y(x,m0n,m1n,mag),y(x,m0p,m1p,mag),
#                       facecolor=colorhack[mag],zorder=4,alpha=0.1)
# ax.plot(x,y(x,m0n,m1n,mag),'k--',linewidth=1.75)
# ax.plot(x,y(x,m0p,m1p,mag),'k--',linewidth=1.75)

for i in range(5,10):
    ax.plot(x,y(x,m0,m1,i),zorder=5,linewidth=2.5,label='M{}'.format(i),
                                                    color=colorhack[i])

# create magnitude equation and annotate
mag_eq_ano = 'M = log(V$_{max}$/2$\pi$) + '+\
                '({} $\pm$ {}) log($\Delta$) + ({} $\pm$ {})'.format(
                round(m0,3),round(m0-m0n,3),round(m1,3),round(m1-m1n,3))
mag_eq = 'M_{} = log(V/2pi) + ({} +/- {}) log(D) + ({} +/- {})'.format(
                pick,round(m0,3),round(m0-m0n,3),round(m1,3),round(m1-m1n,3))


# set figure parameters
plt.legend(prop={'size':10})
ax.grid(which='both')   
ax.set_title(' Magnitude Scale | N = {} \n{}'.format(len(ds),mag_eq_ano))
ax.set_yscale("log", nonposx='clip')
ax.set_ylabel('Peak {} ({})'.format(label,units))
ax.set_xlabel('Epicentral Distance ($^{\circ}$)')
if log == 'loglog': 
    ax.set_xscale("log", nonposx='clip')
    ax.set_xlim([9,165])
else:   
    ax.set_xlim([1.1,165])
  

# make a histogram behind to show number of events
ax2 = ax.twinx()
ax2.set_ylabel('Number of Events')
plt.hist(ds,bins=16,color='k',alpha=0.1,range=(0,160),zorder=1)
ax2.set_xlim(([1.1,165]))


# f.set_tight_layout(True)
print(mag_eq)
# plt.savefig('./figures/{}/{}_{}.png'.format(sta,sta,pick),
#                                                     dpi=600,figsize=(11,7))
plt.show()



# from IPython.core.debugger import Tracer; Tracer(colors="Linux")()
