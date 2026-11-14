# Target2Route

This README.md is coming soon...

**检查点文件**

https://huggingface.co/CROWD-lpy/Target2Route

请将checkpoints完整下载，并放在主目录下

**测试检查点可用性**

预训练监督学习检查点

python run.py check-checkpoint --config configs/supervised.yaml --checkpoint checkpoints/supervised/model.pt

全局强化学习检查点

python run.py check-checkpoint --config configs/rl_global.yaml --checkpoint checkpoints/global/model.pt

单靶点强化学习检查点

python run.py check-checkpoint --config configs/rl_target.yaml --checkpoint checkpoints/target/HPK1/model.pt

python run.py check-checkpoint --config configs/rl_target.yaml --checkpoint checkpoints/target/RIPK1/model.pt

python run.py check-checkpoint --config configs/rl_target.yaml --checkpoint checkpoints/target/LRRK2/model.pt


**训练和评估**

预训练监督学习从头训练 (--max-batches 200 限制训练集数量)

python run.py train-sl --config configs/supervised.yaml --device cuda:0 --batch-size 64 --epochs 30 --seed 2026

预训练监督学习续算 (可使用--max-batches 300 限制训练集数量;  --epoch为累计总epoch数)

python run.py train-sl --model-config runs/supervised/xxxx/config.yaml --resume runs/supervised/xxxx/ckpt/xxxx.pt  --epochs 40  --device cuda:0

预训练监督学习检查点评估

python run.py evaluate-sl --config configs/supervised.yaml --checkpoint checkpoints/supervised/model.pt --split test --device cuda:0 --batch-size 32 --max-batches 50 --seed 2026

全局强化学习

python run.py train-global --config configs/rl_global.yaml --checkpoint checkpoints/supervised/model.pt --device cuda:0 --batch-size 64 --episodes 4000 --docking-workers 16 --seed 2026

单靶点强化学习（序列）

python run.py train-target --config configs/rl_target.yaml --checkpoint checkpoints/global/model.pt --sequence xxx --receptor xxx.pdbqt --box a b c --device cuda:0 --esm2-device cuda:0 --batch-size 64 --episodes 1000 --docking-workers 16 --seed 2026


**单靶点生成**

序列生成

python run.py generate --config configs/rl_target.yaml --checkpoint checkpoints/target/xxx/model.pt  --sequence xxx --no-docking --num-mols 100 --batch-size 64 --max-batches 100 --device cuda:0 --esm2-device cuda:0 --seed 2026 --output-dir runs/generation/xxx

序列生成+对接

python run.py generate --config configs/rl_target.yaml --checkpoint checkpoints/target/xxx/model.pt --sequence xxx --receptor xxx.pdbqt --box a b c --num-mols 20 --batch-size 64 --max-batches 100 --device cuda:0 --esm2-device cuda:0 --seed 2026 --output-dir runs/generation/xxx

fasta生成

python run.py generate --config configs/rl_target.yaml --checkpoint checkpoints/target/xxx/model.pt --fasta xxx.fasta --no-docking --num-mols 20 --batch-size 64 --max-batches 100 --device cuda:0 --esm2-device cuda:0 --seed 2026 --output-dir runs/generation/xxx

fasta生成+对接

python run.py generate --config configs/rl_target.yaml --checkpoint checkpoints/target/xxx/model.pt --fasta xxx.fasta --receptor xxx.pdbqt --box a b c --num-mols 20 --batch-size 64 --max-batches 100 --device cuda:0 --esm2-device cuda:0 --seed 2026 --output-dir runs/generation/xxx
