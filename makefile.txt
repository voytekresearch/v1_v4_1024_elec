pipeline:
	python scripts/pipeline/step1_epoch_lfp.py
	python scripts/pipeline/step2a_compute_spectra.py
	python scripts/pipeline/step2b_compute_spectrogram.py
	python scripts/pipeline/step3a_fit_spectra.py
	python scripts/pipeline/step3b_fit_spectrogram.py

all: pipeline
