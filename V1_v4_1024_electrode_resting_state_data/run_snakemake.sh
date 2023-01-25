#!/bin/bash

# We recommend using a high performance computing system to run the workflow
# See cluster.json for general guidelines of computing resources needed by
# each workflow rule

# Activating the virtual environment (also possible from source)
conda activate rs_data

# If the workflow stops unexpectedly (some error)
# the working directory will need to be unlocked
snakemake --unlock\
	        --cores 1\
	        --rerun-incomplete\
          --use-conda

# This creates the workflow diagrams, can be commented out
snakemake --rulegraph | dot -T png -o "tmp_rulegraph.png"
snakemake --dag | dot -T pdf -o "tmp_dag.pdf"

# Adjust the cluster submission script to your needs
snakemake --jobs 100\
          --cluster-config cluster.json\
          --cluster "sbatch -n {cluster.n} \
                            -o {LOG_DIR}/%j_{rule}.out \
                            --cpus-per-task {cluster.nCPUs} \
                            --mem {cluster.mem} \
                            --time {cluster.time}"\
          --latency-wait 120\
	        --rerun-incomplete\
          --use-conda \
          --jobname "{cluster.name}.{jobid}"
