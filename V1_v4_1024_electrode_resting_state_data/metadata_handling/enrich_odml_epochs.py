"""Enrich odml recording

Create metadata file with recording and trial information extending the
odml metadata template.

Usage:
    enrich_odml_epochs.py --csv=FILE \
                          --template=FILE \
                          --SNR-thresh=INT \
                          --ns6=FILE \
                          --out=FILE

Options:
  -h --help         Show this screen and terminate script.
  --csv=FILE        Trial information .csv file.
  --template=FILE   Path to .odml template metadata file.
  --SNR-thresh=INT  SNR threshold, under which noise is too high
  --ns6=FILE        Path to .ns6 data file.
  --out=FILE        Output file path.

"""
from docopt import docopt
import neo
import numpy as np
import pandas as pd
import odml

if __name__ == "__main__":

    # Get arguments
    vargs = docopt(__doc__)
    epochs_csv_path = vargs['--csv']
    template = vargs['--template']
    odml_template = odml.load(template)
    SNR_treshold = float(vargs['--SNR-thresh'])
    ns6_path = vargs['--ns6']
    target_path = vargs['--out']

    # Create an odml section for the recording
    rec_sec = odml.Section(name='Recording',
                           definition='Metadata about experiment recording ' +
                                      'and trials',
                           type='Recording',
                           parent=odml_template)

    # Import ns6 structure to read metadata
    io = neo.io.BlackrockIO(ns6_path)
    block = io.read_block(lazy=True)
    dur_secs = float(block.segments[0].analogsignals[0].duration.magnitude)
    dur_mins = dur_secs/60
    date = block.rec_datetime.strftime("%Y-%m-%d")
    time = block.rec_datetime.strftime("%H:%M:%S")

    # Annotate with recording specific stuff
    odml.Property(name="Date",
                  definition="Date of recording",
                  values=date,
                  dtype='date',
                  parent=rec_sec)
    odml.Property(name="Time",
                  definition="Time of the start of the recording",
                  values=time,
                  dtype='string',
                  parent=rec_sec)
    odml.Property(name="Duration_seconds",
                  definition="Duration of recording in seconds",
                  values=dur_secs,
                  unit='s',
                  dtype='float',
                  parent=rec_sec)
    odml.Property(name="Duration_minutes",
                  definition="Duration of recording in minutes",
                  values=dur_mins,
                  unit='min',
                  dtype='float',
                  parent=rec_sec)
    odml.Property(name="rec_pauses",
                  definition="Whether the recordins were paused at some point",
                  values=block.annotations['rec_pauses'],
                  dtype='bool',
                  parent=rec_sec)
    odml.Property(name="SNR_threshold",
                  definition="""SNR under which it is not recommended
                                to use data from a certain electrode""",
                  values=SNR_treshold,
                  dtype='float',
                  parent=rec_sec)

    if 'RS' in epochs_csv_path:
        epochs = pd.read_csv(epochs_csv_path)
        odml.Property(name="Trial_type",
                      definition="Type of trial performed by subject",
                      values='Resting state (RS) no task was performed.',
                      dtype='string',
                      parent=rec_sec)
        odml.Property(name="Trial_codename",
                      definition="Short code name for trial type",
                      values='RS',
                      dtype='string',
                      parent=rec_sec)
        states = ['Open_eyes', 'Closed_eyes']
        odml.Property(name="Behavioural_states",
                      definition="Considered behavioural states of the monkey",
                      values=states,
                      dtype='string',
                      parent=rec_sec)

        # Assign values
        for stt in states:
            st_epochs = epochs[epochs['state'] == stt]
            st_sec = odml.Section(name=stt,
                                  definition='Metadata about eyes',
                                  type=stt,
                                  parent=rec_sec)
            odml.Property(name="t_start",
                          definition="Start of the eye state period",
                          values=st_epochs['t_start'].to_list(),
                          unit='s',
                          dtype='float',
                          parent=st_sec)
            odml.Property(name="t_stop",
                          definition="End of the eye state period",
                          values=st_epochs['t_stop'].to_list(),
                          unit='s',
                          dtype='float',
                          parent=st_sec)
            odml.Property(name="duration",
                          definition="Duration of the eye state period",
                          values=st_epochs['dur'].to_list(),
                          unit='s',
                          dtype='float',
                          parent=st_sec)

    else:

        # Generate a pandas dataframe with all the trials and their info
        trials = pd.read_csv(epochs_csv_path)

        # Create an odml section for the trials
        trials_sec = odml.Section(name='Trials',
                                  definition='Metadata about experiment trials',
                                  type='Trials',
                                  parent=rec_sec)
        odml.Property(name="Number_of_trials",
                      definition="Total number of trials in session",
                      values=len(trials),
                      dtype='int',
                      parent=trials_sec)

        # About the successful trials
        success_trials = odml.Section(name='Successful_trials',
                                      definition='Metadata on successful trials',
                                      type='Trials',
                                      parent=trials_sec)
        odml.Property(name="Number_successful_trials",
                      definition="Total number of successful trials in session",
                      values=len(trials[trials['success']]),
                      dtype='int',
                      parent=success_trials)

        # About the failed trials
        failed_trials = odml.Section(name='Failed_trials',
                                     definition='Metadata on failed trials',
                                     type='Trials',
                                     parent=trials_sec)
        failed = trials[np.logical_not(trials['success'])]
        odml.Property(name="Number_failed_trials",
                      definition="Total number of failed trials in session",
                      values=len(failed),
                      dtype='int',
                      parent=failed_trials)
        odml.Property(name="t_stim_on",
                      definition="Stimulus onset for failed trials",
                      values=failed['t_stim_on'].astype(float).tolist(),
                      unit='s',
                      uncertainty=1/30000,
                      dtype='float',
                      parent=failed_trials)

        if 'RF' in epochs_csv_path:
            odml.Property(name="Trial_type",
                          definition="Type of trial performed by subject",
                          values='Receptive Field mapping (RF)',
                          dtype='string',
                          parent=rec_sec)
            odml.Property(name="Trial_codename",
                          definition="Short code name for trial type",
                          values='RF',
                          dtype='string',
                          parent=rec_sec)

            stim = odml.Section(name='Stimulus',
                                definition='Metadata on the stimuli in the trials',
                                type='Stimulus',
                                parent=trials_sec)
            directions = ['rightward', 'upward', 'leftward', 'downward']
            odml.Property(name="Directions",
                          definition="Possible sweeping bar directions",
                          values=directions,
                          dtype='string',
                          parent=stim)
            angles = [180, 270, 0, 90]
            odml.Property(name="Direction_angles",
                          definition="Angle of bar initial with respect to the center",
                          values=angles,
                          unit='deg',
                          dtype='float',
                          parent=stim)
            odml.Property(name='Displayed_stimulus',
                          definition='Type of image showed to the subject',
                          values='Sweeping bar, in different directions',
                          dtype='string',
                          parent=stim)
            odml.Property(name='Stimulus_duration',
                          definition='Duration of the displayed stimulus',
                          values=1000,
                          dtype='float',
                          unit='ms',
                          parent=stim)
            odml.Property(name='Pre_stim_duration',
                          definition='Length of pre-stimulus-onset period',
                          values=300,
                          dtype='float',
                          unit='ms',
                          parent=stim)
            odml.Property(name='Post_stim_duration',
                          definition='Length of post-stimulus-onset period',
                          values=300,
                          dtype='float',
                          unit='ms',
                          parent=stim)
            odml.Property(name='Pixels_per_degree',
                          definition='Pixels per degree in screen',
                          values=25.8601,
                          dtype='float',
                          unit='1/deg',
                          parent=stim)
            # Bar params
            # size and speed of large bars
            if '260617' in target_path or '280818' in target_path:
                x0 = 70  # pixels
                y0 = -70  # pixels
                speed = 500  # pixels / s
            # size and speed of small bars
            elif '280617' in target_path or '290818' in target_path:
                x0 = 30  # pixels
                y0 = -30  # pixels
                speed = 100  # pixels / s

            odml.Property(name='bar_x0',
                          definition='X coordinate of center point',
                          values=x0,
                          dtype='float',
                          parent=stim)
            odml.Property(name='bar_y0',
                          definition='Y coordinate of center point',
                          values=y0,
                          dtype='float',
                          parent=stim)
            odml.Property(name='bar_speed',
                          definition='Speed of sweeping bar',
                          values=speed,
                          dtype='float',
                          unit='1/s',
                          parent=stim)

            # Annotate what each code actually means
            codes = odml.Section(name='Event_codes',
                                 definition='Labels of the encoded events',
                                 type='Event_codes',
                                 parent=trials_sec)
            odml.Property(name="Stimulus_onset",
                          definition="""Event code for stimulus onset
                                        (start of trial)""",
                          values='2',
                          dtype='int',
                          parent=codes)
            odml.Property(name="Reward",
                          definition="""Event code for reward delivery time
                                        (successful trials)""",
                          values='4',
                          dtype='int',
                          parent=codes)
            odml.Property(name="Rightward_direction",
                          definition="Event code for sweeping bar direction",
                          values='8',
                          dtype='int',
                          parent=codes)
            odml.Property(name="Upward_direction",
                          definition="Event code for sweeping bar direction",
                          values='16',
                          dtype='int',
                          parent=codes)
            odml.Property(name="Leftward_direction",
                          definition="Event code for sweeping bar direction",
                          values='32',
                          dtype='int',
                          parent=codes)
            odml.Property(name="Downward_direction",
                          definition="Event code for sweeping bar direction",
                          values='64',
                          dtype='int',
                          parent=codes)

            # Further subsections for each trial type
            for dir in directions:
                dir_sec = odml.Section(name='Trials_' + dir,
                                       definition='Metadata of trials with ' +
                                                  'a bar sweeping ' + dir,
                                       type='Trials_' + dir,
                                       parent=success_trials)
                dir_trial = trials[trials['cond'] == dir]
                odml.Property(name='Number_of_trials',
                              definition='Number of successful ' + dir +
                                         ' trials',
                              values=len(dir_trial),
                              dtype='int',
                              parent=dir_sec)
                odml.Property(name='t_stim_on',
                              definition='Time of stimulus onset in ' +
                                         'successful trials',
                              values=dir_trial['t_stim_on'].astype(float).tolist(),
                              unit='s',
                              uncertainty=1/30000,
                              dtype='float',
                              parent=dir_sec)
                odml.Property(name='t_rew',
                              definition='Time of reward in successful trials',
                              values=dir_trial['t_rew'].astype(float).tolist(),
                              unit='s',
                              uncertainty=1/30000,
                              dtype='float',
                              parent=dir_sec)

        elif 'SNR' in epochs_csv_path:
            odml.Property(name="Trial_type",
                          definition="Type of trial performed by subject",
                          values='Signal to Noise Ratio (SNR)',
                          dtype='string',
                          parent=rec_sec)
            odml.Property(name="Trial_codename",
                          definition="Short code name for trial type",
                          values='SNR',
                          dtype='string',
                          parent=rec_sec)

            stim = odml.Section(name='Stimulus',
                                definition='Metadata on the stimuli in the trials',
                                type='Stimulus',
                                parent=trials_sec)
            odml.Property(name='Displayed_stimulus',
                          definition='Type of image showed to the subject',
                          values='Checkerboard',
                          dtype='string',
                          parent=stim)
            odml.Property(name='Stimulus_duration',
                          definition='Duration of the displayed stimulus',
                          values=400/1000,
                          dtype='float',
                          unit='s',
                          parent=stim)
            odml.Property(name='Pre_stim_duration',
                          definition='Length of pre-stimulus-onset period',
                          values=300/1000,
                          dtype='float',
                          unit='s',
                          parent=stim)
            odml.Property(name='Post_stim_duration',
                          definition='Length of post-stimulus-onset period',
                          values=300/1000,
                          dtype='float',
                          unit='s',
                          parent=stim)

            # Annotate what each code actually means
            codes = odml.Section(name='Event_codes',
                                 definition='Labels of the encoded events',
                                 type='Event_codes',
                                 parent=trials_sec)
            odml.Property(name="Stimulus_onset",
                          definition="Event code for stimulus onset (start of trial)",
                          values='2',
                          dtype='int',
                          parent=codes)
            odml.Property(name="Reward",
                          definition="Event code for reward delivery (successful trials only)",
                          values='4',
                          dtype='int',
                          parent=codes)

            # Information about the successful trials
            successful = trials[trials['success']]
            odml.Property(name='t_stim_on',
                          definition='''Time of stimulus onset in successful
                                        trials''',
                          values=successful['t_stim_on'].astype(float).tolist(),
                          unit='s',
                          uncertainty=1/30000,
                          dtype='float',
                          parent=success_trials)
            odml.Property(name='t_rew',
                          definition='Time of reward in successful trials',
                          values=successful['t_rew'].astype(float).tolist(),
                          unit='s',
                          uncertainty=1/30000,
                          dtype='float',
                          parent=success_trials)

    # Save output
    odml.tools.XMLWriter(odml_template).write_file(target_path,
                                                   local_style=True)
