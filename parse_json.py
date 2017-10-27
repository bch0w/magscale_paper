"""07-09-2017
Parse JSON for database parameters into .npy arrays
"""

import sys
import json
import glob
import numpy as np
from obspy.geodetics.base import locations2degrees

base_path = './specfem3d/output/'
filenames = glob.glob(file_path + 'json/*.json')



# ============================ FOR OBSERVATIONS ==============================
# command line argument
# choice = sys.argv[1].upper()
# if not (choice == 'WET' or choice == 'FUR' or choice == 'SPECFEM'):
#     print('Correct call: python sysargv[0] (wet or fur or specfem)')
#     sys.exit()

# # paths
# file_path = './OUTPUT/{}/'.format(choice)
# filenames = glob.glob(file_path + '*/*.json')
# output_path = './OUTPUT/'

# # set parameters to be imported from .json files
# # set event removal from manual inspection 17.09.17
# if choice == 'WET':
#     sta_remove = [2842826,2872453,2872804,2873223,2837840,2838027,2839297,
#                 2839302,2846505,2849756,2874049,2874057,2875581,3279346,3279347,
#                 3279513,3279951,3279952,32800997,3320509,3320758,3344359,
#                 4212442,4368448,4597426,5161913,5164443,5164812,5183078,
#                 5184776,5190904,5190988,5193839]
#     param_list = ['event_id','event_latitude','event_longitude','magnitude',
#                 'depth','epicentral_distance','theoretical_backazimuth',
#                 'peak_filtered_rotation_rate','dominant_period_rotation_rate',
#                 'rotation_rate_zero_crossing','peak_filtered_rotation',
#                 'dominant_period_rotation','rotation_zero_crossing',
#                 'peak_filtered_vertical_vel','dominant_period_vertical_vel',
#                 'vertical_vel_zero_crossing','peak_filtered_transverse_vel',
#                 'dominant_period_transverse_vel','transverse_vel_zero_crossing'
#                 ]
# elif choice == 'FUR':
#     sta_remove = [2872804,2837840,2873223,2839297,2839302,2849756,2874049,
#                 2874057,2875581,2880277,3279513,3280516,3319094,3319094,
#                 3320667,3320758,3320765,3324326,4212442,4220606,4597426,
#                 4598598,5111570,5113518,5150490,5160106,5164812,5173533,
#                 5183078]
#     param_list = ['event_id','event_latitude','event_longitude','magnitude',
#                 'depth','epicentral_distance','theoretical_backazimuth',
#                 'peak_filtered_vertical_vel','dominant_period_vertical_vel',
#                 'vertical_vel_zero_crossing','peak_filtered_transverse_vel',
#                 'dominant_period_transverse_vel','transverse_vel_zero_crossing']

# sta_remove = [str(_) for _ in sta_remove]

# # aggregate parameters into lists, save as .npy
# lists = [[] for _ in range(len(param_list))]
# rem = 1
# for fid in filenames:
#     with open(fid) as data_file:
#         data = json.load(data_file)
#         ID = str(data['event_id'][-7:])
#         if ID in sta_remove:
#             rem+=1
#             print('Removing {}'.format(ID)
#             sta_remove.remove(ID)
#         else:
#             for N in range(len(param_list)):
#                 lists[N].append(data[param_list[N]])

# print(sta_remove)
# print('Done, {} events collected'.format(len(lists[0])))

# np.save(output_path+'{}_{}_params.npy'.format(choice,len(lists[0])),lists)




