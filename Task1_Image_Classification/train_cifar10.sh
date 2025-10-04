MODELS=("ResNet50")
OPTIMIZERS=("KOALA-P")
SEEDS=(42)
EPOCHS=(100)
SIGMAS=(0.2)
QS=(0.2)
LRS=(2.0)

for model in "${MODELS[@]}"; do
  for opt in "${OPTIMIZERS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for epoch in "${EPOCHS[@]}"; do
        for sigma in "${SIGMAS[@]}"; do
          for q in "${QS[@]}"; do
            for lr in "${LRS[@]}"; do
              echo "Running: Model=$model, Optimizer=$opt, Seed=$seed, Epochs=$epoch, Sigma=$sigma, Q=$q, LR=$lr"
              python train.py \
                --model "$model" \
                --optimizer "$opt" \
                --dataset cifar10 \
                --epochs "$epoch" \
                --seed "$seed" \
                --sigma "$sigma" \
                --q "$q" \
                --lr "$lr"
            done
          done
        done
      done
    done
  done
done


