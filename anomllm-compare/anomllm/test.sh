#!/bin/bash

# data=("trend" "range" "point" "freq" "noisy-point" "noisy-freq" "noisy-trend")

# data=("trend" "range" "point" "freq")
# variants=(
#   "0shot-text-s0.3-calc"
#   "0shot-text-s0.3-dyscalc"
#   "1shot-vision-calc"
#   "1shot-vision-dyscalc"
#   "0shot-vision-calc"
#   "0shot-vision-dyscalc"
# )

data=("flat-trend")
variants=(
  "1shot-vision" "0shot-vision" "1shot-text-s0.3" "0shot-text-s0.3" "0shot-text" \
  "0shot-text-s0.3-cot" "1shot-text-s0.3-cot" "0shot-vision-cot" "1shot-vision-cot"
  "0shot-text-s0.3-csv" "0shot-text-s0.3-cot-csv"
  "0shot-text-s0.3-tpd" "0shot-text-s0.3-cot-tpd"
  "0shot-text-s0.3-pap" "0shot-text-s0.3-cot-pap"
)

# models=("internvlm-76b")

models=("gpt-4o-mini")
for model in "${models[@]}"; do
  for datum in "${data[@]}"; do
    for variant in "${variants[@]}"; do
      session_name="${datum}_${model}_${variant}"
      command="python src/batch_api.py --data $datum --model $model --variant $variant"
      echo "Runing \`$command\` ..."
      tmux new-session -d -s "$session_name" "$command"
    done
  done
done

# variants=(
#   "0shot-vision" "1shot-vision" "0shot-text-s0.3" "0shot-text" \
#   "0shot-text-s0.3-cot" "1shot-text-s0.3-cot" "0shot-vision-cot" "1shot-vision-cot"
#   "0shot-text-s0.3-csv" "0shot-text-s0.3-cot-csv"
#   "0shot-text-s0.3-tpd" "0shot-text-s0.3-cot-tpd"
#   "0shot-text-s0.3-pap" "0shot-text-s0.3-cot-pap"
# )

# # variants=("0shot-vision-cot")

# for model in "${models[@]}"; do
#   for datum in "${data[@]}"; do
#     for variant in "${variants[@]}"; do
#       session_name="${datum}_${model}_${variant}"
#       command="python src/online_api.py --data $datum --model $model --variant $variant"
#       echo "Runing \`$command\` ..."
#       tmux new-session -d -s "$session_name" "$command"
#     done
#   done
# done


##############################################
# Baselines
##############################################

# for datum in "${data[@]}"; do
#   session_name="iso_${datum}"
#   # Kill the existing session if it exists
#   tmux has-session -t "$session_name" 2>/dev/null
#   if [ $? -eq 0 ]; then
#     tmux kill-session -t "$session_name"
#   fi
#   command="python src/baselines/isoforest.py --data $datum --model isolation-forest"
#   tmux new-session -d -s "$session_name" "$command"
# done





# tmux list-sessions -F '#{session_name}' | grep internvlm | xargs -I {} tmux kill-session -t {}
# find results/synthetic -type d -name "text" -exec rm -rf {} +