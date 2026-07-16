#!/bin/bash
#BSUB -J marl_nego_seeds
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=shared"
#BSUB -n 4
#BSUB -W 12:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o negoseeds_%J.out
#BSUB -e negoseeds_%J.err
source ~/cicero-env/bin/activate
module load python3/3.10.16
module load cuda/12.8.1
cd ~/climate-marl-environment
for SEED in 1 2 3 4 5
do
sed -i "s/random_seed: .*/random_seed: $SEED/" config/marl.yaml
sed -i "s/damage_seed: .*/damage_seed: $SEED/" config/marl.yaml
sed -i "s/use_empirical_damage: false/use_empirical_damage: true/" config/marl.yaml
sed -i "s/use_negotiation: false/use_negotiation: true/" config/marl.yaml
sed -i "s/marl_run_name: .*/marl_run_name: nego_ds$SEED/" config/marl.yaml
python src/marl_experiment.py
rm -rf data/*/marl_results/*nego_ds$SEED/None
rm -rf ~/ray_results
done
