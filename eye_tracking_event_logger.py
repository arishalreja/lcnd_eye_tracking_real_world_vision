"""
Name:       : eye_tracking_event_logger.py 
Description : Build eye-tracking event log using raw eye-tracking traces captured by SMI ETG 2 Eye Tracking Glasses
              and combine eye gaze location with computer vision bounding boxes for annotated egocentric video from
              the Eye Tracking Glasses to assign 'face fixation' or 'non-face fixations' to each fixation event.
Date Created: July 16th 2019
Author      : Arish Alreja (aalreja@andrew.cmu.edu), Laboratory for Cognitive Neurodynamics, University of Pittsburgh
"""

import os, numpy as np, pandas as pd


############################### Section 1: Read Raw data ############################################
#                                                                                                   #
#####################################################################################################

# Read eye-tracking data file with traces sampled at 60 Hz. We only need the columns specified below.
columns_to_read = ['RecordingTime [ms]','Video Time [h:m:s:ms]','Frame Number','Content','Category Binocular','Point of Regard Binocular X [px]','Point of Regard Binocular Y [px]']
column_data_types = {'Frame Number':int,'Video Time [h:m:s:ms]':str,'RecordingTime [ms]':np.float,'Content':str,'Category Binocular':str,'Point of Regard Binocular X [px]':str,'Point of Regard Binocular Y [px]':str}
metrics = pd.read_csv(os.path.join('.','data','eye_tracking_sample.txt'),usecols=columns_to_read,sep=',',dtype=column_data_types)

# Clean up the gaze location columns. These can be missing (i.e., '-') in case of 'User Defined Events' aka 'Triggers' and should be substituted with a 'invalidation' marker (-1) of the same data type (int) as valid data
metrics['Point of Regard Binocular X [px]'] = metrics['Point of Regard Binocular X [px]'].str.replace('-','-1')
metrics['Point of Regard Binocular X [px]'] = metrics['Point of Regard Binocular X [px]'].astype(np.float)
metrics['Point of Regard Binocular Y [px]'] = metrics['Point of Regard Binocular Y [px]'].str.replace('-','-1')
metrics['Point of Regard Binocular Y [px]'] = metrics['Point of Regard Binocular Y [px]'].astype(np.float)

# Load video annotations (bounding box information for objects found in each frame)
boxes = np.load(os.path.join('.','data','boxes.npy'),allow_pickle=True)
boxes = boxes[()]




############################### Section 2: Create Empty Eye Tracking Event Log ######################
#                                                                                                   #
##################################################################################################### 

# Specify event log fields
event_log_column_names = ['trigger_num', 'trigger_start', 'trigger_end','event_start','event_end','event_type','event_duration','gaze_loc_x','gaze_loc_y','face','objects','bad_trial','video_frame']
event_log_types = [np.int64, np.float, np.float, np.float, np.float, str, np.float, np.float,np.float, bool,str, bool,int]

# Make an empty event log and populate it with the fields specified above
event_log = pd.DataFrame(index=[0])
event_log_main = pd.DataFrame(index=None)
for i in range(len(event_log_column_names)):
    event_log[event_log_column_names[i]] = pd.Series(dtype=event_log_types[i])
    event_log[event_log_column_names[i]] = None
    event_log_main[event_log_column_names[i]] = pd.Series(dtype=event_log_types[i])





############################### Section 3: Construct the Eye Tracking Event Log #####################
#                                                                                                   #
#####################################################################################################


# initialize counters for gaze location, event type, and digital trigger numbers
start_tag = True
trigger_num,event_type = -1, None 
gaze_loc_x, gaze_loc_y= None, None

# loop through all the eye-tracking traces, extract contigious blocks of traces for each eye tracking event type
# i.e, 'Visual Intake'/Fixation, 'Saccade', or 'Blink' and treat those contigious blocks as a single event with
# a duration denoted by the time difference between the last and first trace of the contigious block.
for i in range(len(metrics)): 
    
    if 'Align' in metrics.iloc[i]['Content']: # trigger messages are of the form "Align #1", "Align #2" and so on
        trigger_num = int(metrics.iloc[i]['Content'].split('#')[1])
        trigger_start= metrics.iloc[i]['RecordingTime [ms]']
        trigger_end = trigger_start + 10000 # 10 sec is the inter-trigger interval

    if trigger_num>0: # Start constructing the event log after the first trigger only
        if metrics.iloc[i]['Category Binocular']!=event_type and metrics.iloc[i]['Category Binocular']!='User Event':
            if start_tag==False: # Event type changed while already processing an event => event ended with previous line
                event_log['event_end'] = metrics.iloc[i-1]['RecordingTime [ms]']
                event_log['bad_trial'] = event_log.loc[0]['event_end']>event_log.loc[0]['trigger_end']
                event_log_main = pd.concat([event_log_main,event_log],ignore_index=True)
                start_tag= not start_tag

            if start_tag==True:
                event_log['gaze_loc_x'] = metrics.iloc[i]['Point of Regard Binocular X [px]']
                event_log['gaze_loc_y'] = metrics.iloc[i]['Point of Regard Binocular Y [px]']
                event_type = metrics.iloc[i]['Category Binocular']
                event_log['event_type']  = event_type
                event_log['trigger_num'] = trigger_num
                event_log['trigger_start'] = trigger_start
                event_log['trigger_end'] = trigger_end
                event_log['event_start'] = metrics.iloc[i]['RecordingTime [ms]']
                event_log['video_frame'] = metrics.iloc[i]['Frame Number']
                start_tag = not start_tag

        else: # metrics.iloc[i]['Category']==event_type or metrics.iloc[i]['Category Binocular']=='User Event':
            if start_tag==True:
                if metrics.iloc[i]['Content']=='Align #1':
                    print('Info: First Align Found at ',i)
                else:
                    print('Error: The event category did not change but the start tag is true at ',i)
                print('Record:\n')
                print(metrics.iloc[i])
            elif start_tag==False:
                test=None #print('Event: continuing')# This was a debug statement

event_log_main = pd.concat([event_log_main,event_log],ignore_index=True) # This is for the last eye-tracking event to get added to the main log
event_log_main['event_duration'] = event_log_main['event_end']-event_log_main['event_start']




################## Section 4: Add video annotations to the eye tracking event log ###################
#                                                                                                   #
#####################################################################################################
# ow put the labels and annotations on for each event based on alignment of event_start timestamp
fixations = np.argwhere(event_log_main['event_type'].values=='Visual Intake').flatten()
annotation_missing_counter = 0
for i in fixations:
    all_objects = boxes['image'+str(event_log_main['video_frame'].values[i]).zfill(8)+'.jpg']
    object_list = ''
    event_log_main.loc[i,'face']=False # initialize to a non-face fixation, correct if needed
    for j in range(len(all_objects)):
        object_list = object_list+all_objects[j][2]+',' # make a list of all objects
        if all_objects[j][2]=='face':
            x1 = all_objects[j][0][0]
            y1 = all_objects[j][0][1]
            x2 = all_objects[j][1][0]
            y2 = all_objects[j][1][1]
            if (x1 < event_log_main.iloc[i]['gaze_loc_x'] < x2) and (y1 < event_log_main.iloc[i]['gaze_loc_y'] < y2):
                event_log_main.loc[i,'face']=True
        event_log_main.loc[i,'objects'] = object_list
    if len(object_list)==0:
        annotation_missing_counter = annotation_missing_counter+1
        print('WARNING: There were no bounding boxes/annotations available for '+'image'+str(event_log_main['video_frame'].values[i]).zfill(8)+'.jpg')    
    
event_log_main.to_csv(os.path.join('.','output','eye_tracking_event_log.txt'),sep=',')