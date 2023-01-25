import odml


def annotate_RF_elec(elec, df, directions=None):
    """
    Helper function to annotate the receptive field metadata to an electrode
    section in an odml file in-place.
    """
    row = df[df['Electrode_ID'] == elec.properties['Electrode_ID']]
    if directions:
        for d in directions:
            odml.Property(name=f"SNR_fromRF_{d}",
                          definition="Signal to noise ratio from RF" +
                                     f" experiment with {d} moving bars" +
                                     f" measured on {row['date'].iloc[0]}",
                          values=row[f"SNR_fromRF_{d}"].iloc[0],
                          dtype='float',
                          parent=elec)
            odml.Property(name=f"R2_{d}",
                          definition="R^2 of gaussian fit to signal from RF" +
                                     f" experiment with {d} moving bars" +
                                     f" measured on {row['date'].iloc[0]}." +
                                     " R^2 = 1 - (ss_res / ss_tot)",
                          values=row[f"SNR_fromRF_{d}"].iloc[0],
                          dtype='float',
                          parent=elec)

    for d in ['top', 'bottom', 'left', 'right']:
        odml.Property(name=f"RF_{d}_edge",
                      definition="Receptive field edge in pixels, from" +
                                 f" experiment with {d} moving bars" +
                                 f" measured on {row['date'].iloc[0]}",
                      values=row[f"RF_{d}_edge (pixels)"].iloc[0],
                      dtype='float',
                      parent=elec)

    odml.Property(name="RF_center_X",
                  definition="Receptive field center X coordinate" +
                             " measured on " + row['date'].iloc[0],
                  values=row['RF center X (degrees)'].iloc[0],
                  unit='degree',
                  dtype='float',
                  parent=elec)
    odml.Property(name="RF_center_Y",
                  definition="Receptive field center Y coordinate" +
                             " measured on " + row['date'].iloc[0],
                  values=row['RF center Y (degrees)'].iloc[0],
                  unit='degree',
                  dtype='float',
                  parent=elec)


def annotate_SNR_elec(elec, df):
    """
    Helper function to annotate the signal to noirse ratio metadata to an
    electrode section in an odml file in-place.
    """
    row = df[df['Electrode_ID'] == elec.properties['Electrode_ID']]
    odml.Property(name="SNR",
                  definition="Channel signal to noise ratio" +
                             " measured on " + row['date'].iloc[0],
                  values=row['SNR'].iloc[0],
                  dtype='float',
                  parent=elec)
    odml.Property(name="response_onset_timing",
                  definition="Response delay to checkerboard" +
                             " measured on " + row['date'].iloc[0],
                  values=row['response_onset_timing (ms)'].iloc[0],
                  unit='ms',
                  dtype='float',
                  parent=elec)
    odml.Property(name="peak_response",
                  definition="Peak response to stimulus",
                  values=row['peak_response (uV)'].iloc[0],
                  unit='uV',
                  dtype='float',
                  parent=elec)
    odml.Property(name="baseline_avg",
                  definition="""Baseline mean value
                                (before stimulus)""",
                  values=row['baseline_avg (uV)'].iloc[0],
                  unit='uV',
                  dtype='float',
                  parent=elec)
    odml.Property(name="baseline_std",
                  definition="""Baseline standard deviation
                                (before stimulus)""",
                  values=row['baseline_std (uV)'].iloc[0],
                  unit='uV',
                  dtype='float',
                  parent=elec)
