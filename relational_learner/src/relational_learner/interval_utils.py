#!/usr/bin/env python

"""interval_utils.py: Interval utilities used by Activity_Graph class."""

__author__      = "Krishna Dubba and Paul Duckworth"
__copyright__   = "Copyright 2015, University of Leeds"

import itertools
import logging
import os, sys
import warnings
import time

logger = logging.getLogger('redvine.interval_utils')

class Episodes():
    '''
    Episodes class:
    '''
    def __init__(self, episodes_list):
        self.episodes = episodes_list
        
def get_episodes_from_file(episodes_file):
    """ Get list of episodes from a file
    """
    episodes      = []
    episodes_text = open(episodes_file)
    
    # Line structure example: o1,rh,o2,lh,DC,1,116
    for line in episodes_text:
        line   = line.strip()
        fields = line.split(',') 
        episodes.append((fields[0],fields[1],fields[2],fields[3],fields[4],int(fields[5]),int(fields[6])))
    return episodes

def encode_episodes(episodes):
    """ Give unique code to an episode and return a list of 
    encoded episodes (tuple with start, end and episode code) and episode code map
    """
    episode_count    = 1
    episode_code     = {}
    encoded_episodes = []
    
    # Gather the episodes and interval data    
    for fields in episodes:
        start = fields[5]
        end   = fields[6]
        # Use codes for the episodes throughout the function
        # At the end we replace the codes with episodes
        episode_code[fields]        = episode_count
        episode_code[episode_count] = fields
        episode_count += 1
        episode_data = (start, end, episode_code[fields])
        encoded_episodes.append(episode_data)
    return (encoded_episodes, episode_code)
    
def get_temporal_chords_from_intervals(intervals):
    interval_data   = {}
    interval_breaks = []
    # For each time point in the combined interval, get the state of the 
    # system which is just a list of relations active in that time point.
    for (s,e,v) in intervals:
        for i in range(s,e+1):
            if i not in interval_data:
                interval_data[i] = []
            interval_data[i].append(v)
            
    keys = interval_data.keys()
    keys.sort()
    
    # Now based on the state changes, break the combined interval 
    # whenever there is a chage in the state

    start = keys[0]
    interval_value = interval_data[start]
    for i in keys:
        if interval_value == interval_data[i]:
            end = i
            continue
        else:
            interval_breaks.append([start,end,interval_value])
            start = i
            end   = i
            interval_value = interval_data[start]
    else:
        # Adding the final interval
        interval_breaks.append([start,end,interval_value])
    # Return    
    return interval_breaks


def window_condition(window):
    """
    Tests the start and end of the  Sliding Window's temporal duration  
    Also, the Window must be less than a max duration
    """
    return window[1] > window[0]


def get_valid_graphlets_from_activity_intervals(episodes, params):
    """ This function implements Krishna's validity criteria to select only valid 
    graphlets from an activity graph.
    PD: Added Windowed time points
    """
    logger.info('Computing valid_graphlets_from_activity_intervals')
    logger.info('Length of episodes:' + repr(len(episodes)))
       
    timepoints    = []
    episode_code  = {}
    episode_count = 1
    obj_pair_intervals = {}
    
    # Gather the episodes and interval data    
    for fields in episodes:
        start = fields[5]
        end   = fields[6]
        # Use codes for the episodes throughout the function
        # At the end we replace the codes with episodes
        episode_code[fields]        = episode_count
        episode_code[episode_count] = fields
        episode_count += 1
        episode_data = [start, end, episode_code[fields]]
        timepoints.append(episode_data)
        obj_pair = (fields[0], fields[2])
        
        if obj_pair not in obj_pair_intervals:
            obj_pair_intervals[obj_pair] = []
        obj_pair_intervals[obj_pair].append(episode_data)
   
    (min_rows, max_rows, max_episodes, cores) = params
    min_rows=1 if min_rows==None else min(max_rows, len(obj_pair_intervals))
    max_rows=len(obj_pair_intervals) if max_rows==None else min(max_rows, len(obj_pair_intervals))

    #TEMPORAL SLICING
    all_timepoints, time_starts, time_ends = [], [], []
    temporal_slicing_dict = {}    
    
    for i in timepoints:
        if i[0] not in all_timepoints:
            all_timepoints.append(i[0])
            time_starts.append(i[0])
        if i[1] not in all_timepoints:
            all_timepoints.append(i[1])
            time_ends.append(i[1])
    time_starts.sort()
    time_ends.sort(reverse=True)
    
    time_starts=[time_starts[0]]
    time_ends=[time_ends[0]]

    #Loop through all the temporal sliding windows
    for i in filter(window_condition, itertools.product(time_starts, time_ends)):
        window_start = i[0]
        window_end = i[1]
       
        graphlets          = {}
        graphlets_all_rows = {}
        graphlets_list     = []
        episode_window_copy = episode_code.copy()

        # First we iterate over combinations of rows and in each iteration the number rows 
        # is varied from 1 to num_of_rows.
        range_ = range(min_rows, max_rows+1)            
        for r in range_:
            episode_code_row_copy = episode_window_copy.copy()
            # Once we select the number of rows, find all combinations of rows of r. 
            for obj_pair_comb in itertools.combinations(obj_pair_intervals.keys(),r):
                # Collect intervals from episodes of relevant rows
                episode_intervals = []
                for obj_pair in obj_pair_comb:
                    episodes = obj_pair_intervals[obj_pair]
                    add_episodes=[]
                        
                    #Restrict the episodes to the temporal slice
                    for i in episodes:
                          
                        #window fully contains episode
                        if i[0] >= window_start and i[1] <= window_end:
                            add_episodes.append(i)
                            
                       #episode is fully before window starts  - set to zero,zero
                        elif i[1] < window_start:
                            episode_code_row_copy[i[2]] = episode_code_row_copy[i[2]][:-2] + (0,0)
                                
                        #episode starts after the window - set to zero,zero 
                        elif i[0] > window_end:
                            episode_code_row_copy[i[2]] = episode_code_row_copy[i[2]][:-2] + (0,0)
  
                        #window is smaller than single episode - set both start and end of episode to window     
                        elif i[0] < window_start and i[1] > window_end:
                            new_i = [window_start, window_end, i[2]]
                            add_episodes.append(new_i)
                            episode_code_row_copy[i[2]] = episode_code_row_copy[i[2]][:-2]+ (window_start, window_end)

                        #episode overlaps window start - set episode start to window start    
                        elif i[0] < window_start and i[1] > window_start: #and i[1] < window_end:
                            new_i = [window_start, i[1],i[2]]
                            add_episodes.append(new_i)
                            episode_code_row_copy[i[2]] = episode_code_row_copy[i[2]][:-2]+ (window_start, i[1])
                            
                        #episode overlaps window end - set episode end to window end
                        elif i[0] < window_end and i[1] > window_end: # and i[0] > window_start: 
                            new_i = [i[0],window_end,i[2]]
                            add_episodes.append(new_i)
                            episode_code_row_copy[i[2]] = episode_code_row_copy[i[2]][:-1]+ (window_end,)
                            
                    episode_intervals.extend(add_episodes)
                if episode_intervals != []: 
                    interval_breaks = get_temporal_chords_from_intervals(episode_intervals)
                else: 
                    #Covers the case where no episodes are added. i.e. episodes that do not start at frame 1.
                    interval_breaks = []
                            
                logger.info('Length of episodes:' + repr(len(interval_breaks)))
                # Loop through this broken timeline and find all 
                # combinations (r is from 1 to num_of_intervals)
                # of consecutive intervals (intervals in a stretch). 
                for k in xrange(1,len(interval_breaks)+1):
                    for l in xrange(len(interval_breaks)):
                        # Find the combined interval of this combination of intervals
                        selected_intervals = interval_breaks[l:l+k]
                        # Get the relations active in this active interval
                        selected_relations = [m[2] for m in selected_intervals]
                        # Some episodes are repeated as they are active in two or more intervals.
                        # So remove the duplicates .
                        selected_relations_set = tuple(set(itertools.chain.from_iterable(selected_relations)))
                        
                        #Only allow Graphlets of the specified number of Rows. Not all rows. 
                        if hash(selected_relations_set) not in graphlets:
                            graphlets[hash(selected_relations_set)] = selected_relations_set              
        
        # Replace the codes with the episodes and return as a list instead of dictionary                    
        for hash_key in graphlets:
            graphlet_episodes       = []
            graphlet_episodes_codes = graphlets[hash_key]        
            for epi_code in graphlet_episodes_codes:
                graphlet_episodes.append(episode_code_row_copy[epi_code]) 
                        
            if max_episodes == None:
                graphlets_list.append(graphlet_episodes)
            elif len(graphlet_episodes) <= max_episodes:
                graphlets_list.append(graphlet_episodes)
        
        temporal_slicing_dict[(window_start, window_end)] = graphlets_list
    return temporal_slicing_dict
              
                
def get_temporal_neighbourhood_graphlets_from_activity_intervals(episodes, \
                                                                 min_rows=None, max_rows=None, \
                                                                 max_rt=None, max_dt=None):
    
    print 'Computing NSPD valid_graphlets_from_activity_intervals'
    print 'Length of episodes:' + repr(len(episodes))
    
    pair_graphlets = []
    timepoints     = []
    episode_code   = {}
    episode_count  = 1
    obj_pair_intervals = {}
    
    # Gather the episodes and interval data    
    for fields in episodes:
        
        #Use below for Muhannads input - get him to change it.
        #fields=( fields[0], fields[0],fields[1], fields[1],fields[2], fields[3], fields[4])
        start = fields[5]
        end   = fields[6]
        # Use codes for the episodes throughout the function
        # At the end we replace the codes with episodes
        episode_code[fields]        = episode_count
        episode_code[episode_count] = fields
        episode_count += 1
        episode_data = [start, end, episode_code[fields]]
        timepoints.append(episode_data)
        obj_pair = (fields[0], fields[2])
        if obj_pair not in obj_pair_intervals:
            obj_pair_intervals[obj_pair] = []
        obj_pair_intervals[obj_pair].append(episode_data)
        
    if min_rows == None:
        min_rows = 1
    elif min_rows == sys.maxint:
        min_rows = len(obj_pair_intervals)
    if max_rows == None:
        max_rows = len(obj_pair_intervals) + 1
    else:
        max_rows = min(max_rows, len(obj_pair_intervals) + 1)
        
    print 'Number of obj_pair_intervals:' + repr(len(obj_pair_intervals))
    # First we iterate over combinations of rows and in each iteration the number rows 
    # is varied from 1 to num_of_rows.
    #PD: added "+1"
    for r in range(min_rows,max_rows+1):
        # Once we select the number of rows, find all combinations of rows of r. 
        for obj_pair_comb in itertools.combinations(obj_pair_intervals.keys(),r):
            # Collect intervals from episodes of relevant rows
            episode_intervals = []
            for obj_pair in obj_pair_comb:
                episode_intervals.extend(obj_pair_intervals[obj_pair])
            
            # Generate the chords from the set of intervals.
            interval_breaks = get_temporal_chords_from_intervals(episode_intervals)
            #interval_breaks = [1,2,3,4,5,6,7,8,9]
            #print 'Length of episodes:' + repr(len(interval_breaks))
            # Set max_rt and max_dt accordingly
            if max_rt == None or max_rt == 0 or max_rt >= len(interval_breaks)/2:
                max_rt = len(interval_breaks)/3
            if max_dt == None or max_dt == 0 or max_dt >= len(interval_breaks):
                max_dt = min((2 * len(interval_breaks))/3, len(interval_breaks) - 1)
                
            # Loop through this broken timeline and find all 
            # combinations (r is from 1 to num_of_intervals)
            # of consecutive intervals (intervals in a stretch). 
            for i in xrange(0,len(interval_breaks)-1):
                for j in xrange(i+1,len(interval_breaks)):
                    if j-i > len(interval_breaks)/3 or j-i < len(interval_breaks)/4:
                        continue
                    for k in xrange(0,max_rt):
                        # Find the combined interval of this combination of intervals
                        neigbourhood_start   = max(0, i-k)
                        neigbourhood_end     = min(len(interval_breaks), i+1+k)
                        selected_intervals_i = tuple(interval_breaks[neigbourhood_start:neigbourhood_end])
                        #print selected_intervals_i
                        neigbourhood_start   = max(0, j-k)
                        neigbourhood_end     = min(len(interval_breaks), j+1+k)                
                        selected_intervals_j = tuple(interval_breaks[neigbourhood_start:neigbourhood_end])
                        #print selected_intervals_j
                        # Get the relations active in this active interval
                        selected_relations_i = [m[2] for m in selected_intervals_i]
                        selected_relations_j = [m[2] for m in selected_intervals_j]
                        # Some episodes are repeated as they are active in two or more intervals.
                        # So remove the duplicates.
                        selected_relations_set_i = set(itertools.chain.from_iterable(selected_relations_i))
                        selected_relations_set_j = set(itertools.chain.from_iterable(selected_relations_j))
                        #pairwise_graphlet = selected_relations_set_i.union(selected_relations_set_j)
                        #pair_graphlets.append([episode_code[epi_code] for epi_code in pairwise_graphlet])
                        epi_i = [episode_code[epi_code] for epi_code in selected_relations_set_i]
                        epi_j = [episode_code[epi_code] for epi_code in selected_relations_set_j]                
                        pair_graphlets.append((Episodes(epi_i), Episodes(epi_j)))
                                
    return pair_graphlets

if __name__ == '__main__':
    import cPickle as pickle
    import os, sys
    
    #episodes = get_episodes_from_file('/home/socleeds/work/my_code/trunk/python/redvine/arranging_objects0510175411.txt')
    #episodes = [('Head', 'Head', 'RH', 'RH', 'touch', 0, 841), ('Head', 'Head', 'RH', 'RH', 'near', 842, 1000), ('Head', 'Head', 'LH', 'LH', 'touch', 0, 841), ('Head', 'Head', 'LH', 'LH', 'near', 842, 1000), ('Head', 'Head', 'remote', 'remote', 'touch', 0, 841), ('Head', 'Head', 'remote', 'remote', 'far', 842, 887), ('Head', 'Head', 'remote', 'remote', 'near', 888, 1000), ('RH', 'RH', 'LH', 'LH', 'touch', 0, 841), ('RH', 'RH', 'LH', 'LH', 'near', 842, 1000), ('RH', 'RH', 'remote', 'remote', 'touch', 0, 841), ('RH', 'RH', 'remote', 'remote', 'far', 842, 870), ('RH', 'RH', 'remote', 'remote', 'near', 871, 1000), ('LH', 'LH', 'remote', 'remote', 'touch', 0, 841), ('LH', 'LH', 'remote', 'remote', 'near', 842, 887), ('LH', 'LH', 'remote', 'remote', 'touch', 888, 1000)]
    #graphlets = get_valid_graphlets_from_activity_intervals(episodes, min_rows=sys.maxint)
    #graphlets = get_temporal_neighbourhood_graphlets_from_activity_intervals(episodes)
    
    epi_dir = '/usr/not-backed-up/data_sets/race/cornell_human_activities/CAD_120/online/test_case/'
    epi_file = os.path.join(epi_dir,'drink1__5.p')
    episodes_dict = pickle.load(open(epi_file))
    episodes      = list(itertools.chain.from_iterable(episodes_dict.values()))    
    #episodes = episodes_dict
    graphlets = get_valid_graphlets_from_activity_intervals(episodes, 10)

    print len(graphlets)
