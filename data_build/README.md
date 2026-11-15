# Rebuild the TRACE dataset

Run the commands from the TRACE repository root. `prepare_data.py` writes the formats used by `run.py`. The examples keep source files, search results and model inputs under `dat/`.

## Evaluate the published checkpoint

Use the original split pickles, template CSV, FAISS index and its matching SMILES map, ESM-2 bank and training configuration. For the RL runs also use the original target IDs, allowlist, prepared receptors and docking centers. The source code archive contains the preparation code. Use the original data assets and their recorded checksums for the reported evaluation.

For the `dat/` layout, copy [the example paths](../configs/data_paths.example.yaml) to `configs/local_paths.yaml` and point each entry at the corresponding original file. Set `esm2_mean_path` if the original mean was retained.
The original `fp_pos_weights.pt` is needed for scratch training. The released checkpoint carries its fingerprint weights for evaluation.

```bash
python prepare_data.py verify --config configs/local_paths.yaml --hash-files
python run.py check-checkpoint --config configs/local_paths.yaml \
  --checkpoint checkpoints/supervised/model.pt
python run.py evaluate-sl --config configs/local_paths.yaml \
  --checkpoint checkpoints/supervised/model.pt --split test --device cuda:0 --seed 2026
```

Add `--model-config path/to/original/config.yaml` to each command if the checkpoint uses a different architecture. Compare the SHA-256 digests with the original assets. Strict loading checks parameter shapes and template arities. Two template tables can have the same shape and arities with different template IDs.

The manuscript reports 2,760 templates and 1,555,178 train, 194,388 validation and 194,259 test examples. Check the released files' counts and digests against those used for the checkpoint.

The following rebuild generates a new dataset from raw inputs. Even with the original sources, the deterministic ordering, route deduplication, stock checks and split seed can change counts, template order and split membership relative to the historical scripts. An exact replication of the reported evaluation uses the released split files and original preprocessing assets. To reconstruct samples while retaining the checkpoint's template IDs and feature scale, use `--frozen-templates` and `--reuse-embeddings` with the original assets.

## 1. Record the source files

| Input | What to retain |
| --- | --- |
| BindingDB | Exact 202602 `BindingDB_All.tsv` snapshot used for the study, plus its checksum. See [BindingDB downloads](https://bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp). |
| ZINC20 | Exact 2D tranche files used for the study, plus their filenames and checksums. The reader accepts `.smi`, `.smi.gz`, `.txt` and `.txt.gz`, with the SMILES in the first field. See the [ZINC20 2D cache](https://cache.docking.org/2D/). |
| AiZynthFinder | Version 4.4.1, the original matching USPTO expansion ONNX model and template table, and the same ZINC20 stock. The supplied [config.yml](aizynth/config.yml) uses only the USPTO expansion policy by default. [AiZynthFinder public assets and docs](https://molecularai.github.io/aizynthfinder/) |
| Study targets | The exact FASTA sequence, numeric ID and split or RL membership used for each target. Include BindingDB construct variants when they were excluded from the study. |
| Receptor data | Prepared receptor PDBQT, docking-box center in Å, and receptor chain, structure, preparation and pocket provenance. Use the original files for reproducing docking scores. |

The archive does not carry the paper's raw snapshots, tranche selection, target list, structures or exact split assignments. [Zenodo record 11430881](https://zenodo.org/records/11430881) provides the two public USPTO files named in the example YAML. Use the pair from the original run to reproduce the study. The commands below use the configured ZINC20 stock.

Install the model and data dependencies in the TRACE environment. Keep AiZynthFinder 4.4.1 in a separate Python 3.10–3.12 environment because its RDKit/NumPy requirements differ from those of TRACE.

```bash
pip install -e '.[dataset,esm2,docking,training]'

python3.12 -m venv .venv-aizynth
.venv-aizynth/bin/pip install 'aizynthfinder==4.4.1'
mkdir -p dat/raw/aizynth dat/work/retro
cp data_rebuild/aizynth/config.yml dat/raw/aizynth/trace.yml
```

Place the USPTO files next to `trace.yml`, or edit the paths in that YAML. Paths are relative to the YAML's directory. Set `stock.ZINC20` to the original `ZINC20.hdf5` when available. The build-stock command below provides a text InChIKey stock for a fresh run.

The three case-study structures are [GLP-1R, 7S15](https://www.rcsb.org/structure/7S15), [RIPK1, 8I2N](https://www.rcsb.org/structure/8I2N) and [HPK1, 7M0M](https://www.rcsb.org/structure/7M0M). Use the original conditioning FASTAs for exclusion and ESM-2; crystallized or engineered chains can differ from the sequences used for conditioning.

## 2. Filter BindingDB and prepare ZINC20

The manuscript uses a threshold of 1000 nM. The old standalone step-1 script defaulted to 100. The new default is 1000. The filter keeps rows with one nonempty chain and a numeric Ki, Kd or IC50 below the threshold, using the legacy convention of stripping `<`, `>` and `~` before comparison. In particular `>900` passes a 1000 nM threshold under that convention. Output pairs are deduplicated by normalized protein sequence and canonical isomeric SMILES.

Repeat `--exclude-fasta` once per held-out sequence. For the paper case studies, supply the matching GLP-1R, RIPK1 and HPK1 sequences and any relevant construct variants as separate single-record FASTAs. Exclusion is by exact normalized sequence.

```bash
EXCLUSIONS=(--exclude-fasta dat/raw/case_targets/GLP1R.fasta
            --exclude-fasta dat/raw/case_targets/RIPK1.fasta
            --exclude-fasta dat/raw/case_targets/HPK1.fasta)

python prepare_data.py filter-bindingdb \
  --bindingdb dat/raw/BindingDB_All.tsv --output dat/work/pairs.tsv \
  "${EXCLUSIONS[@]}" > dat/work/bindingdb_manifest.json

python prepare_data.py build-stock \
  --smiles dat/raw/zinc20 --output dat/raw/aizynth/ZINC20.txt \
  > dat/work/stock_manifest.json

python prepare_data.py build-faiss \
  --smiles dat/raw/zinc20 --output dat/building_blocks \
  --seed 2026 --threads 8
```

Skip the corresponding ZINC20 command when reusing the original stock or the exact FAISS pair. The default indexing command uses ECFP4 at 2048 bits and an IVFPQ index. It writes a SMILES entry for each FAISS ID and a manifest with source digests and index settings. Use `--flat` for a small smoke-test library. For a custom original index, pass its configuration through `--config` and keep its paired SMILES map. FAISS training can require several GB of RAM with the default sample size.

## 3. Search routes with AiZynthFinder

```bash
.venv-aizynth/bin/python prepare_data.py retrosynthesis \
  --pairs dat/work/pairs.tsv --aizynth-config dat/raw/aizynth/trace.yml \
  --output dat/work/retro/routes-1.jsonl --workers 8 --seed 2026
```

The default search uses at most six transforms, 100 iterations, 120 seconds and ten exported routes per ligand-target pair. Use `--limit 100` to exercise a small subset in a separate output file. For a full search distributed across jobs, give each job the same `--parts N`, a different `--part` in `1..N` and its own output filename. Run all parts with the same config, options and pair file. Completed shard files are written atomically and can be kept when restarting the other jobs. The assembler checks that all parts of each tagged search are present. Each worker loads its own policy and stock. Choose the worker count based on RAM. Wall-clock limits can change search results across hardware. Keep the route shards for repeatable downstream builds.

You can start at the next step if you already have JSONL route shards or the old `routes_threshold1000*.pkl` files. Use only trusted original pickles.

## 4. Write model inputs

```bash
python prepare_data.py build-supervised \
  --routes dat/work/retro/routes-*.jsonl \
  --uspto-templates dat/raw/aizynth/uspto_templates.csv.gz \
  --blocks-dir dat/building_blocks --output dat/rebuilt \
  "${EXCLUSIONS[@]}" --seed 2026 --esm2-device cuda:0

python prepare_data.py verify \
  --config dat/rebuilt/data_paths.yaml --manifest dat/rebuilt/manifest.json

python run.py train-sl --config dat/rebuilt/data_paths.yaml \
  --device cuda:0 --batch-size 8 --epochs 1 --max-batches 2 --seed 2026
```

Use a new output directory for each build. The emitted `data_paths.yaml` uses paths relative to the repository root unless absolute paths were supplied on the command line. Pass it after any model architecture overrides. It sets the observed template count and all model-data paths, including the FAISS pair and RL novelty dataset.

The route builder walks each route in synthesis order. It retains continuous suffixes of 3–6 mono- or bimolecular reactions with an initial molecule of at most 350 Da and in-stock side reactants. It merges duplicate routes for the same target and ligand. It emits a scaffold-conditioned example for each retained route and a de novo example for the longest eligible suffix when its starting molecule is in stock. A de novo record begins with `R`, and each subsequent `R` supplies the second reactant of a preceding bimolecular template. `run.py` adds EOS when collating. The default core template selection covers 90% of template occurrences counted in the longest usable suffixes, then prunes templates unused by surviving routes. Ties are sorted by template code.

ESM-2 encodes each unique retained sequence once using the last-layer CLS vector and the configured 1024-token limit. The rebuild subtracts the mean over those unique sequences. To add known target features for registration during the build, repeat `--registry-fasta path/to/target.fasta`. Those sequences receive the same centering and do not contribute to the mean, clustering or route splits unless already present in the supervised dataset. The optional `--raw-embeddings` expects a dictionary of uncentered CLS tensors for the same model. `--reuse-embeddings` instead preserves known vectors and calibrates new ones against the original bank or mean from the supplied configuration.

Splitting uses the historical **(ESM-2 sequence cluster, exact Bemis–Murcko scaffold)** key and tries to reach 80/10/10 while keeping observed templates represented in training. Sequence clustering uses average linkage over cosine distance with cutoff 0.75 on the centered embeddings. The same cluster can occur with different scaffolds across splits, and the same scaffold can occur with different clusters. Acyclic molecules are grouped using their canonical SMILES. The split manifest records the actual group membership. A small route subset may lack enough groups for three nonempty splits; increase the search subset in that case.

| Output | Format |
| --- | --- |
| `supervised/train.pkl`, `valid.pkl`, `test.pkl` | List of dictionaries with `target_seq`, `target_smi`, `start_mol` and `route`. Route actions are `("T", template_id, ...)` and `("R", reactant_smiles, ...)`. |
| `supervised/seqs_embeddings.pt`, `esm2_mean.pt` | Sequence-to-tensor dictionary and the ESM-2 centering vector. |
| `supervised/fp_pos_weights.pt` | Per-bit weights keyed by fingerprint type and length. From training `R` actions only, each observed bit uses the negative-to-positive ratio with a minimum of 1. Unobserved bits use 1. |
| `templates/trace_core_templates_top90_update.csv` | Zero-based template row IDs, forward SMARTS, and reactant arities. Existing weights require the original row order. |
| `targets/` | JSON receptor registry, docking centers and explicit global-RL allowlist, initially empty. |
| `manifest.json`, `split_groups.json`, `sequence_clusters.json` | Input/output hashes, settings, counts and split assignment. |

The supervised protein vector is sequence-only (`pocket_dim: 0`). The FAISS pair stays in the building-block directory selected in step 2.

To keep historical template row IDs, replace `--uspto-templates ...` with `--frozen-templates path/to/original.csv`. The original must contain `template_code`, `retro_template`, `forward_template` and `num_reactants`. With the original ESM-2 and FAISS inputs, also pass your `--config configs/local_paths.yaml --reuse-embeddings`. This rebuilds labeled samples using those assets. Use the original pickles to reproduce the published held-out metrics.

## 5. Add or restore targets

Register a target using its single-record FASTA. The command preserves existing IDs, calculates missing ESM-2 features against the configured calibration and saves the receptor and coordinates when provided. Replace example ID and box coordinates with the values for your target. Existing `dat/` installations use `configs/local_paths.yaml`.

```bash
DATA_CFG=configs/local_paths.yaml

python prepare_data.py register-target --config "$DATA_CFG" \
  --target-id 2009 --fasta dat/raw/targets/2009.fasta \
  --receptor dat/raw/targets/2009.pdbqt --box 12.3 -4.5 8.9 \
  --esm2-device cuda:0

python run.py generate --config "$DATA_CFG" \
  --checkpoint checkpoints/global/model.pt --target-id 2009 \
  --num-mols 10 --max-batches 100 --device cuda:0 --seed 2026
```

A FASTA alone is sufficient for registration and generation with `--no-docking`. For docking, prepare the matching structure as PDBQT and supply the center of the chosen binding site in that structure's coordinate frame. For experimental structures, start from the appropriate [RCSB PDB](https://www.rcsb.org/) chain and record any preparation steps. For a protein PDB with cleaned coordinates and curated protonation, use `mk_prepare_receptor.py --read_pdb protein.pdb -p receptor.pdbqt`. See the [Meeko CLI documentation](https://apidocmeeko.readthedocs.io/en/latest/cli_rec_prep.html) for receptor preparation options. TRACE uses a 20 Å cube centered at the supplied point.

For batch registration, make a JSON array like the one below. The FASTA and receptor paths are relative to the JSON file. Set `"global_rl": true` in JSON or use `--global-rl` on the command line to include a target in the global training pool. Keep the paper's case-study targets out of that list. Registering a target makes it available for generation and target-specific RL. Adding labeled routes for a new protein requires a supervised dataset rebuild in a separate output directory with that protein's ligand and route records.

```json
[
  {
    "target_id": 2009,
    "fasta": "2009.fasta",
    "receptor": "2009.pdbqt",
    "box": [12.3, -4.5, 8.9],
    "global_rl": false
  }
]
```

```bash
python prepare_data.py register-target --config "$DATA_CFG" \
  --manifest dat/raw/targets/registry.json --esm2-device cuda:0

# With the docking extra installed, also check that Vina reads the registered receptors.
python prepare_data.py verify --config "$DATA_CFG" --require-global-rl
```

On Linux and macOS, target registration uses a file lock and rejects an ID that would replace a different sequence, receptor or box. Repeating the same command is safe. For a newly rebuilt dataset, set `DATA_CFG=dat/rebuilt/data_paths.yaml` and load a checkpoint trained with that dataset's template table. After an ESM-2 vector is appended, use `verify --manifest dat/rebuilt/manifest.json --allow-extended-bank` to check the original supervised vectors and the other file digests.

### Add supervised examples for a new protein

To train on labeled ligands for the new target, prepare an additional TSV with the same two columns as `filter-bindingdb`. Use one ligand and its full conditioning sequence per row. The fields below are tab-separated.

```tsv
target_seq	target_smi
MADEK	CCO
```

Run `retrosynthesis --pairs dat/work/new_target_pairs.tsv` using the same AiZynthFinder policy and stock in step 3 and an independent shard output. Include that shard alongside the original route shards in `build-supervised --routes` and choose a new output directory. Keep any study holdouts in `--exclude-fasta`. Use the frozen templates and original ESM-2 calibration if the resulting examples will be evaluated or used for further training with an existing checkpoint.


## Smoke tests

After installing the dataset and docking extras, run:

```bash
python -m unittest discover -s checks -p test_dataset.py -v
```

The small fixtures exercise BindingDB filtering, generated dataset files, supervised training and evaluation, target registration, receptor parsing and historical route inputs. They use substitute route trees and ESM-2 vectors.
