import neo
import odml
import quantities as pq
import numpy as np


def odml_to_dict(odmlpath, dict_index='128-nsp',
                 other_keys=None, NSP=None, array=None):
    """
    Return a dictionary with the electrode, array and NSP ids for each
    electrode.

    Parameters
    ----------
    odmlpath : string
        path to the location of the corresponding odml file.
    dict_index : string
        defines which index will be given to the dictionary keys. Options are
        '64-array', '128-nsp' and '1024-global'.
    other_keys : list of string
        list of strings for other keys that should be extracted from odml too.
    NSP : int, optional
        the number ID of a single NSP. When specified only the metadata
        related to the electrodes connected to this NSP will be returned.
    array : int, optional
        the number ID of a single array. When specified only the metadata
        related to the electrodes in this array will be returned.

    Returns
    -------
    dict_of_IDs: dict
        dictionary with all the ids from the requested NSPs and arrays

    Raises
    ------
    ValueError
        if both keyword arguments NSP and array are specified. Only one can
        be specified.
        (or)
        if specified dict_index is not valid
    """

    if (NSP is not None) and (array is not None):
        raise ValueError('''Keyword arguments NSP and array: only one of the
                         two should be specified''')

    if dict_index not in ['64-array', '128-nsp', '1024-global']:
        raise ValueError("""dict_index: invalid argument. Options are
                        '64-array', '128-nsp' and '1024-global'.""")

    odmlfile = odml.load(odmlpath)

    # Extract relevant electrode sections from metadata file
    electrodes = []
    cortical_areas = []
    if (NSP is None) and (array is None):
        # This looks for all electrodes in metadata odml file
        for odmlarray in odmlfile['Arrays'].sections:
            for elec in odmlarray.sections:
                electrodes.append(elec)
                area = odmlarray.properties['CorticalArea'].values[0]
                cortical_areas.append(area)
    elif NSP is not None:
        for odmlarray in odmlfile['Arrays'].sections:
            if odmlarray.properties['NSP_ID'].values[0] == int(NSP):
                for elec in odmlarray.sections:
                    electrodes.append(elec)
                    area = odmlarray.properties['CorticalArea'].values[0]
                    cortical_areas.append(area)
    elif array is not None:
        for odmlarray in odmlfile['Arrays'].sections:
            if odmlarray.properties['Array_ID'].values[0] == int(array):
                for elec in odmlarray.sections:
                    electrodes.append(elec)
                    area = odmlarray.properties['CorticalArea'].values[0]
                    cortical_areas.append(area)

    # loop over electrodes and extract all indexing metadata
    dict_of_IDs = {}
    for el, area in zip(electrodes, cortical_areas):
        # Retrieve properties
        Electrode_ID = el.properties['Electrode_ID'].values[0]
        NSP_ID = el.properties['NSP_ID'].values[0]
        nsp_index = el.properties['within_NSP_electrode_ID'].values[0]
        Array_ID = el.properties['Array_ID'].values[0]
        arr_index = el.properties['within_array_electrode_ID'].values[0]
        x = el.properties['schematic_X_position'].values[0]
        y = el.properties['schematic_Y_position'].values[0]

        # Resolve index
        if dict_index == '64-array':
            index = arr_index
        elif dict_index == '128-nsp':
            index = nsp_index
        elif dict_index == '1024-global':
            index = Electrode_ID

        # Add entry to dictionary of identifiers
        dict_of_IDs[index] = {'Electrode_ID': Electrode_ID,
                              'NSP_ID': NSP_ID,
                              'within_NSP_electrode_ID': nsp_index,
                              'Array_ID': Array_ID,
                              'within_array_electrode_ID': arr_index,
                              'cortical_area': area,
                              'schematic_X_position': x,
                              'schematic_Y_position': y}

        if other_keys is not None:
            if not isinstance(other_keys, list):
                other_keys = [other_keys]
            for key in other_keys:
                extra_entry = {key: el.properties[key].values[0]}
                dict_of_IDs[index].update(extra_entry)

    return dict_of_IDs


def odml2quantity(property):
    """
    Convert an odml property into a quantity instance

    Parameters
    ----------
    property : odml.property.BaseProperty
        property that wants to be converted

    Returns
    -------
    list of pq.Quantity
    """
    return [pq.Quantity(value, property.unit) for value in property.values]


def anasig_from_nsp(ns6, odmlpath=None):
    """
    Generator of neo.Analogsignals contained in the specified array.

    Parameters
    ----------
    ns6 : string
        path to a .ns6 file that contains data for 2 or more utah arrays
    array : int, string
        ID of target array.
    odmlpath : string, optional
        path to odml file containing the corresponding metadata. Optional, no
        metadata will be attached if not provided.

    Returns
    -------
        neo.Block object with one segment and the analogsignals of the array
        specified in the argument. If odmlpath was specified the analogsignal
        is also annotated with electrode IDs and the cortical area.
    """

    # Neo IO
    io = neo.io.BlackrockIO(ns6)

    # Load data structure but not the data itself, may be larger than memory
    block = io.read_block(lazy=True)

    # Get analogsignal proxy for all blocks and channels
    anasig_array = block.segments[0].analogsignals[-1]

    # Get channel ids
    channel_ids = anasig_array.array_annotations['channel_ids'].copy()

    # Retrieve metadata from odml
    if odmlpath:
        dictofIDs = odml_to_dict(odmlpath,
                                 dict_index='128-nsp')

    # For all channels yield anasig, annotated with metadata if available
    for chx in channel_ids:
        print(chx-1)
        anasig = anasig_array.load(channel_indexes=[chx-1])
        if odmlpath is not None:
            # Annotate analog signals with IDs and other goodies
            ids = dictofIDs[chx+1]
            anasig.array_annotations.update(ids)

        yield anasig


def anasig_from_array(ns6, array, odmlpath=None):
    """
    Generator of neo.Analogsignals contained in the specified array.

    Parameters
    ----------
    ns6 : string
        path to a .ns6 file that contains data for 2 or more utah arrays
    array : int, string
        ID of target array.
    odmlpath : string, optional
        path to odml file containing the corresponding metadata. Optional, no
        metadata will be attached if not provided.

    Returns
    -------
        neo.Block object with one segment and the analogsignals of the array
        specified in the argument. If odmlpath was specified the analogsignal
        is also annotated with electrode IDs and the cortical area.
    """

    # Neo IO
    io = neo.io.BlackrockIO(ns6)

    # Load data structure but not the data itself, may be larger than memory
    block = io.read_block(lazy=True)

    # Get analogsignal proxy for all blocks and channels
    anasig_array = block.segments[0].analogsignals[-1]

    # Find which electrodes correspond to which array from blackrock file
    channel_names = anasig_array.array_annotations['channel_names'].copy()
    array_ids = list(set([name.split('-')[0] for name in channel_names]))
    channel_ids = anasig_array.array_annotations['channel_ids'].copy()

    # adjust id to name in blackrock file
    target_array = 'elec' + str(array)

    # Retrieve metadata from odml
    dictofIDs = odml_to_dict(odmlpath,
                             dict_index='128-nsp',
                             array=array)

    for array_str in array_ids:
        if array_str == target_array:
            # Find which electrodes correspond to the current array
            channel_indexes = []
            for name, ch in zip(channel_names, channel_ids):
                if array_str in name:
                    channel_indexes.append(int(ch)-1)

            for chx in channel_indexes:
                # load the analog signals of the current array
                anasig = anasig_array.load(channel_indexes=[chx])

                if odmlpath is not None:
                    # Annotate analog signals with IDs and other goodies
                    ids = dictofIDs[chx+1]
                    anasig.array_annotations.update(ids)

                yield anasig


def merge_anasiglist(anasiglist):
    """
    Merges neo.AnalogSignal objects into a single object.

    Units, sampling_rate, t_start, t_stop and signals shape must be the same
    for all signals. Otherwise a ValueError is raised.

    Parameters
    ----------
    anasiglist: list of neo.AnalogSignal
        list of analogsignals that will be merged

    Returns
    -------
    merged_anasig: neo.AnalogSignal
        merged output signal
    """
    # Check units, sampling_rate, t_start, t_stop and signal shape
    for anasig in anasiglist:
        if not anasiglist[0].units == anasig.units:
            raise ValueError('Units must be the same for all signals')
        if not anasiglist[0].sampling_rate == anasig.sampling_rate:
            raise ValueError('Sampling rate must be the same for all signals')
        if not anasiglist[0].t_start == anasig.t_start:
            raise ValueError('t_start must be the same for all signals')
        if not anasiglist[0].t_stop == anasig.t_stop:
            raise ValueError('t_stop must be the same for all signals')
        if not anasiglist[0].magnitude.shape == anasig.magnitude.shape:
            raise ValueError('All signals must have the same shape')

    # Initialize the arrays
    anasig0 = anasiglist.pop(0)
    data_array = anasig0.magnitude
    sr = anasig0.sampling_rate
    t_start = anasig0.t_start
    t_stop = anasig0.t_stop
    units = anasig0.units

    # Get the full array annotations
    for anasig in anasiglist:
        anasig0.array_annotations = anasig0._merge_array_annotations(anasig)

    array_annot = anasig0.array_annotations
    del anasig0

    while len(anasiglist) != 0:
        anasig = anasiglist.pop(0)
        data_array = np.concatenate((data_array, anasig.magnitude),
                                    axis=np.argmin(anasig.magnitude.shape))
        del anasig

    merged_anasig = neo.AnalogSignal(data_array,
                                     sampling_rate=sr,
                                     t_start=t_start,
                                     t_stop=t_stop,
                                     units=units,
                                     array_annotations=array_annot)
    return merged_anasig


def mark_epochs(block, odmlpath, eyepath=None):
    """
    Mark epochs in the neo structure.

    Create epochs for the successful trials in the nix files.

    Parameters
    ----------
    block : neo.core.block
        A neo block, where the epochs will be marked upon.
    odmlpath : string
        path to odml file containing the corresponding metadata.
    eyepath : string
        path to eye signals, required for RS sessions only.

    Returns
    -------
        neo.Block object with epochs marked for the trials found in the odml,
        for the experiments where trials were performed (RF, SNR). In the RS
        case epochs mark when the eyes were opened/closed.

    """

    # Load odml
    metadata = odml.load(odmlpath)
    trial_name = metadata['Recording'].properties['Trial_codename'][0]

    if 'SNR' == trial_name:
        # Get successful trials
        trials = metadata['Recording']['Trials']
        success = trials['Successful_trials']

        # Get stimulus duration information
        stim = trials['Stimulus']
        stim_dur = pq.Quantity(stim.properties['Stimulus_duration'].values[0],
                               stim.properties['Stimulus_duration'].unit)
        pre_dur = pq.Quantity(stim.properties['Pre_stim_duration'].values[0],
                              stim.properties['Pre_stim_duration'].unit)
        post_dur = pq.Quantity(stim.properties['Post_stim_duration'].values[0],
                               stim.properties['Post_stim_duration'].unit)
        duration = stim_dur + pre_dur + post_dur

        # Calculate start time
        t_stim_on = pq.Quantity(success.properties['t_stim_on'].values,
                                success.properties['t_stim_on'].unit)
        t0 = t_stim_on[0]
        t_start = [(t - pre_dur).rescale(t0.units).magnitude
                   for t in t_stim_on] * t0.units

        # Number of successful trials
        count = int(success.properties['Number_successful_trials'].values[0])

        # Resolve durations and labels
        durs = ([duration.magnitude]*count)*duration.units
        lbls = [trial_name]*count

        # Check if the last segment exceeds the actual recording length
        rec_dur_sec = metadata['Recording'].properties['Duration_seconds']
        session_duration = pq.Quantity(rec_dur_sec.values[0],
                                       rec_dur_sec.unit)
        while t_start[-1] + durs[-1] > session_duration:
            t_start = t_start[:-1]
            durs = durs[:-1]
            lbls = lbls[:-1]

        # Check if first trial starts before the recording itself
        while t_start[0] < 0*pq.s:
            t_start = t_start[1:]
            durs = durs[1:]
            lbls = lbls[1:]

        # Create the epoch object
        epc = neo.core.Epoch(times=t_start,
                             durations=durs,
                             labels=lbls
                             )

        block.segments[0].epochs.append(epc)

    elif 'RF' == trial_name:
        # Get successful trials
        trials = metadata['Recording']['Trials']
        success = trials['Successful_trials']

        # Get stimulus duration information
        stim = trials['Stimulus']
        stim_dur = pq.Quantity(stim.properties['Stimulus_duration'].values[0],
                               stim.properties['Stimulus_duration'].unit)
        pre_dur = pq.Quantity(stim.properties['Pre_stim_duration'].values[0],
                              stim.properties['Pre_stim_duration'].unit)
        post_dur = pq.Quantity(stim.properties['Post_stim_duration'].values[0],
                               stim.properties['Post_stim_duration'].unit)
        duration = stim_dur + pre_dur + post_dur

        # For each trial direction type
        directions = stim.properties['Directions'].values
        for dir in directions:
            # Calculate start time
            scs_dir = success['Trials_' + dir]
            t_stim_on = pq.Quantity(scs_dir.properties['t_stim_on'].values,
                                    scs_dir.properties['t_stim_on'].unit)
            t0 = t_stim_on[0]
            t_start = [(t - pre_dur).rescale(t0.units).magnitude
                       for t in t_stim_on] * t0.units

            # Number of successful trials in that particular direction
            count = scs_dir.properties['Number_of_trials'].values[0]

            # convenience variables
            durs = ([duration.magnitude]*count)*duration.units
            lbls = [dir]*count

            # Check if the last segment exceeds the actual recording length
            rec_dur_sec = metadata['Recording'].properties['Duration_seconds']
            session_duration = pq.Quantity(rec_dur_sec.values[0],
                                           rec_dur_sec.unit).rescale(pq.s)
            while (t_start[-1] + durs[-1]).rescale(pq.s) > session_duration:
                t_start = t_start[:-1]
                durs = durs[:-1]
                lbls = lbls[:-1]

            # Check if first trial starts before the recording itself
            while t_start[0] < 0*pq.s:
                t_start = t_start[1:]
                durs = durs[1:]
                lbls = lbls[1:]

            # Create epoch object with times and durations
            epc = neo.core.Epoch(times=t_start,
                                 name=dir,
                                 durations=durs,
                                 labels=lbls
                                 )

            block.segments[0].epochs.append(epc)

    elif 'RS' == trial_name:
        if eyepath is None:
            raise TypeError('No eye signal path was passed to epoch marking!')

        # Extract the already created epoch from the eye signals and copy here
        with neo.NixIO(eyepath, mode='ro') as io:
            eye = io.read_block()
        epc = eye.segments[0].epochs[0]
        block.segments[0].epochs.append(epc)

    return block
