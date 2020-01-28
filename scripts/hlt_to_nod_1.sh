#!/bin/bash

module load python/3.6
source $HOME/torchmed/bin/activate
cp /scratch/obriaint/CT_Editor/data/healthy_segments.h5 $SLURM_TMPDIR
cp /scratch/obriaint/CT_Editor/data/healthy_segments.h5 $SLURM_TMPDIR
cp /scratch/obriaint/CT_Editor/data/nodule_segments.h5 $SLURM_TMPDIR
cp /scratch/obriaint/CT_Editor/data/nodule_segments.h5 $SLURM_TMPDIR
python /scratch/obriaint/CT_Editor/train_network.py hlt_to_nod_1 -v 2000 -dd $SLURM_TMPDIR/