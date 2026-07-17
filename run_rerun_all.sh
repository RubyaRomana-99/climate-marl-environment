#!/bin/bash
#BSUB -J rerun_all
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=shared"
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o rerun_%J.out
#BSUB -e rerun_%J.err
source ~/cicero-env/bin/activate
module load python3/3.10.16
module load cuda/12.8.1
cd ~/climate-marl-environment
for SEED in 1 2 3 4 5
do
sed -i "s/random_seed: .*/random_seed: $SEED/" config/marl.yaml
sed -i "s/damage_seed: .*/damage_seed: $SEED/" config/marl.yaml
sed -i "s/use_negotiation: true/use_negotiation: false/" config/marl.yaml
sed -i "s/marl_run_name: .*/marl_run_name: v2_empirical_ds$SEED/" config/marl.yaml
python src/marl_experiment.py
rm -rf data/*/marl_results/*v2_empirical_ds$SEED/None
rm -rf ~/ray_results
sed -i "s/use_negotiation: false/use_negotiation: true/" config/marl.yaml
sed -i "s/marl_run_name: .*/marl_run_name: v2_nego_ds$SEED/" config/marl.yaml
python src/marl_experiment.py
rm -rf data/*/marl_results/*v2_nego_ds$SEED/None
rm -rf ~/ray_results
done
