# ATML PA3

Code, configs, logs and results for EE-5102/CS-6304 Programming Assignment 3 (Fall 2026). The analysis is in the separate PDF report; this repository has everything needed to reproduce its numbers.

## Repository layout

```text
configs/            base.yaml (seed) and one config per task, every reported hyperparameter lives here
common/             small shared helpers (config, json/jsonl io, seeding, timing)
task1_finetune/
    prepare_data.py   the 40,000-example MetaMathQA training set and the fixed activation probe set
    data.py           chat formatting, response-only labels, padding
    adapters.py       LoRA / OPLoRA adapter module, SVD bases, factor export
    train.py          one training script for all five variants
    models.py         checkpoint layout, rebuilding each evaluation condition as a plain HF model
    evaluate.py       merge + merged/unmerged logit check, lm-evaluation-harness runs, per-example records
    qualitative.py    168 greedy responses and the 42-response annotation sheet
    diagnostics.py    rho_k, eta, activation ratios R(z), full-model displacement
    spectrum.py       supplementary: share of each starting weight's energy in OPLoRA's top-128 singular directions
    task1.ipynb       thin driver: runs the scripts in order, then builds the tables and figures
task2_vlm/
    prepare_data.py   coco / vqav2 split selection (seed 6304), grounding pairs, image download (cpu only)
    cache_features.py the frozen siglip encoder, run once -> fp16 patch features
    model.py          prompt layout around the visual tokens, connector, feature bank, loss, generation
    train.py          the three-stage curriculum as one resumable run, calibration, interface checks
    evaluate.py       development metrics and the final four-condition evaluation with pair bootstrap
    vqa_eval.py       vqa answer normalization and soft consensus accuracy (ported, see Attribution)
    task2.ipynb       thin driver, then the tables and figures
task3_ttc/
    prepare_data.py   joins and checks the supplied rollout tables, splits (seed 6304), answer groups, correctness (cpu only)
    answers.py        answer equivalence -> math-verify for MATH, the official MathVista rules, boxed-answer extraction
    offline.py        Steps 1 to 3: seeded candidate orders, every selection rule, VisConf, signal analysis, bootstrap
    adaptive.py       Step 5: agreement and quality-aware stopping, development selection, held-out evaluation
    live.py           Step 4: single / parallel / refinement generation with Qwen2.5-3B-Instruct (the only GPU step)
    task3.ipynb       thin driver, then the tables and figures
data/first_plot_questions.yaml   the 24 qualitative prompts
results/task1_finetune/ results/task2_vlm/ results/task3_ttc/   machine-readable logs, metrics, per-example outputs, tables, figures
run_modal.py        the runner used for every reported run (Modal, fixed GPU type)
deviations.md       every deviation from the manual, the open choices it leaves, and the setup differences between tasks
data/task1/ data/task2/ data/task3/ outputs/   generated data, the supplied Task 3 rollouts, images, cached features, checkpoints, not committed
```

## Setup

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install papermill==2.7.0 ipykernel   # to execute the task notebooks
```

Models and datasets are pulled from Hugging Face at runtime: Qwen/Qwen3-0.6B (revision `c1899de289a04d12100db370d81485cdf75e47ca`), google/siglip-base-patch16-224 (revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`), meta-math/MetaMathQA (revision `aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18`), openai/gsm8k, Rowan/hellaswag, and the lm-evaluation-harness task datasets (gsm8k, hellaswag, super_glue/boolq, ai2_arc). Task 2's COCO 2014 annotations and images and the VQAv2 files are downloaded from their official servers by `task2_vlm/prepare_data.py` (URLs in `configs/task2.yaml`). Task 3 uses the supplied ATML-PA3-Data rollouts (Google Drive folder, downloaded with gdown), the MathVista testmini metadata (`data/testmini.json` at commit `452b0f83e0079029fb48d1d1b606d3954d812db1`), and Qwen/Qwen2.5-3B-Instruct (revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`) for the live step; links in `configs/task3.yaml`. All commands run from the repository root.

## How the reported runs were executed

Every reported result was produced with `run_modal.py` on [Modal](https://modal.com), on a single fixed GPU type (`GPU` in `run_modal.py`, an NVIDIA RTX PRO 6000, 96 GB). The image pins the library versions; generated data, checkpoints, evaluation models and results live on a persistent volume (`atml`, under `/pa3`).

```bash
pip install modal && modal setup                     # once

modal run --detach run_modal.py --task task1 --smoke           # smoke test: real model and recipe, 2 updates, 8 docs per benchmark, under /pa3/smoke
modal run --detach run_modal.py --task task1                   # the whole task at the manual's settings, notebook top to bottom
modal run --detach run_modal.py --task task1 --scale           # the reported run: microbatch 16 x accumulation 2, under /pa3/scaled (see deviations.md)

# or the five training runs in parallel containers first (each script skips finished work, and resumes from its
# latest recovery checkpoint if it was interrupted), then the notebook for the rest
modal run --detach run_modal.py --cmd "python -m task1_finetune.prepare_data"
modal run --detach run_modal.py --cmd "python -m task1_finetune.train --method full_ft"   # ... l2sp, lora4, lora16, oplora16

# task 2 -> the cpu-only download first, then the smoke test and the reported run (both --scale, see deviations.md)
modal run --detach run_modal.py --cpu --cmd "python -m task2_vlm.prepare_data"
modal run --detach run_modal.py --task task2 --smoke --scale
modal run --detach run_modal.py --task task2 --scale

# task 3 -> on its own card (an L4, TASK_GPUS in run_modal.py); the offline steps are cpu work inside the same run.
# the supplied rollouts were put on the volume first (prepare_data falls back to a gdown download if they are missing)
modal volume put atml data/task3/ATML-PA3-Data /pa3/data/task3/ATML-PA3-Data
modal run --detach run_modal.py --task task3 --smoke
modal run --detach run_modal.py --task task3

modal volume get atml /pa3/scaled/results ./         # pull the reported results/ back (/pa3/results for a spec-settings run)
```

The same notebook runs locally on a GPU without Modal:

```bash
papermill --cwd . task1_finetune/task1.ipynb task1_finetune/task1_executed.ipynb -p CONFIG configs/task1.yaml
papermill --cwd . task1_finetune/task1.ipynb task1_finetune/task1_executed.ipynb -p RUN_SCRIPTS False   # rebuild tables/figures only
```

## Task 1: LLM fine-tuning (`configs/task1.yaml`, notebook `task1_finetune/task1.ipynb`)

```bash
python -m task1_finetune.prepare_data                         # training set + probe set (both fixed before any model result)
python -m task1_finetune.train --method full_ft               # Step 1, also l2sp (Step 2), lora4, lora16 (Step 3), oplora16 (Step 4)
python -m task1_finetune.evaluate --condition base            # Step 5, also full_ft_mid, full_ft, l2sp, lora4, lora16, oplora16
python -m task1_finetune.qualitative                          # Step 7, all seven conditions, then the annotation sheet
python -m task1_finetune.diagnostics                          # Step 6 weight-space / activation diagnostics
```

Deviations from the manual's setup (microbatch 16 x accumulation 2, same effective batch of 32) and the open choices it leaves (GPU, evaluation batch sizes, scoring details) are listed with their measurements in [`deviations.md`](deviations.md).

Implementation notes:

- Training data: the first 40,000 MetaMathQA rows in source order that have a non-empty query and response, a formatted length of at most 2,048 tokens, and neither `query` nor `original_question` equal to a GSM8K test question after normalization (lowercase, non-alphanumeric runs collapsed to one space). Kept row indices and drop counts: `results/task1_finetune/train_selection.json`.
- Loss: next-token cross entropy on response tokens and `<|im_end|>` only, summed over the 32 examples of an update and divided by their total number of supervised tokens.
- Precision: FP16 autocast with dynamic gradient scaling, unscaled before clipping. Full FT keeps FP32 master weights; adapter runs a frozen FP16 backbone with FP32 factors. AdamW states are FP32. Skipped updates are listed in `train_summary_*.json`.
- L2-SP: Omega = 1/2 sum (theta - theta0)^2 over every unique parameter (tied embedding once, not averaged), theta0 a frozen FP32 copy on the GPU. Its gradient lambda (theta - theta0) is added to the unscaled gradients before clipping, which equals backpropagating lambda Omega.
- LoRA and OPLoRA share one adapter module (`adapters.py`) and the same seeded initialization; OPLoRA computes its bases once in FP32 (`bases.safetensors`) and applies the projections to the factors at every forward pass. The exported factors are the projected ones, and the projection residuals are recorded in `train_summary_oplora16.json`.
- Evaluation: lm-evaluation-harness `gsm8k_cot` with `num_fewshot=0`, chat template, `enable_thinking=False`, greedy, `max_gen_toks=1024`; `hellaswag`, `boolq`, `arc_easy` zero-shot without chat template; FP16, complete splits. The harness version and task configs used are saved in each `eval_*.json`.
- GSM8K truncation flag: the harness keeps only decoded text, so a response is flagged as truncated when it re-encodes to at least 1,016 tokens (cap minus 8, allowing for re-encoding merges).
- Qualitative annotation: the 42-row sheet is labelled by hand; the rubric (label values and the rule for each aspect) is in `deviations.md` (Task 1, 2.8).

## Task 2: VLM and visual grounding (`configs/task2.yaml`, notebook `task2_vlm/task2.ipynb`)

```bash
python -m task2_vlm.prepare_data       # splits (seed 6304), grounding pairs, mismatched mapping, image download
python -m task2_vlm.cache_features     # siglip patch features once, fp16, then the encoder is gone
python -m task2_vlm.train              # Steps 1 to 4: checks, calibration, caption -> instruct -> vqa, development metrics
python -m task2_vlm.evaluate           # Step 5: both final-test suites under correct / complementary / mismatched / no image
python -m task2_vlm.train --config configs/task2_49.yaml   # optional Step 7 (49 visual tokens), then evaluate the same way
```

Implementation notes:

- Prompt: Qwen's chat template with thinking disabled, the 196 visual tokens inside the user message before the question. Without an image the visual tokens are dropped and the text is unchanged.
- Connector: linear 768 -> 1024, GELU, linear 1024 -> 1024, FP32, times a fixed output scale calibrated on training captions (`calibration.json`).
- Loss: per-example mean NLL over target tokens and `<|im_end|>`, averaged over the 16 examples of an update; visual tokens, prompt and padding masked.
- Curriculum: one continuous run, 2,250 + 1,500 + 750 updates; the mixture draws of stages 2 and 3 are seeded per update, so the run resumes from any checkpoint without sampler state.
- Labelling rules for the annotation-derived questions, every open choice, and the measured probe numbers are in `deviations.md` (Task 2).

## Task 3: Test-time compute (`configs/task3.yaml`, notebook `task3_ttc/task3.ipynb`)

```bash
python -m task3_ttc.prepare_data       # Step 1 setup: join + alignment checks, 200 / 800 split, 50 live questions, answer groups
python -m task3_ttc.offline            # Steps 1 to 3: ten seeded orders, budgets 1 to 32, every rule, signals at N = 32
python -m task3_ttc.adaptive           # Step 5: settings chosen on development, frozen, then held-out evaluation
python -m task3_ttc.live               # Step 4: 50 MATH questions x (single, 3 parallel, 1 revision) on one GPU
```

Implementation notes:

- Answer groups are formed without gold answers: math-verify equivalence for MATH, the official MathVista normalization (option letters, integer / float precision / list rules) for MathVista. Missing extractions form one NO_ANSWER group, always incorrect, kept in voting and cost.
- Every method at every budget sees the same nested prefixes of the same ten orders; ties go to the earliest supporting rollout in the order, RWC uses average ranks.
- VisConf calibration (5th / 95th percentile of the pooled token KL) is recomputed for each candidate prefix from the revealed rollouts only, and so are the RWC ranks and margins the adaptive controllers use.
- The offline steps need no GPU and finish in about a minute after the 6-minute data preparation.
- Every open choice (scorer details, unconvertible MathVista answers, seeds of the live calls, the GPU) is in `deviations.md` (Task 3).

## Where each result lives (Task 3)

| What | File(s) under `results/task3_ttc/` |
|---|---|
| Splits, live question ids | `splits.json` |
| Alignment checks, extraction and scorer stats | `prepare_summary.json` |
| Per-rollout labels and scores, answer groups | `rollouts_{math,mathvista}.csv`, `answer_groups_{math,mathvista}.json` |
| Candidate orders, per-prefix selections (selected answer group of every method, cost, coverage, votes, RWC margins) | `orders_*.json`, `selections_*.parquet` |
| Per-question results | `per_question_*.csv` |
| Steps 1 and 2 tables, bootstrap differences | `offline_summary.csv`, `offline_bootstrap.csv`, `table_step1_sampling.csv`, `table_methods_*.csv`, `table_bootstrap_n32.csv` |
| Step 3 signals | `signals_summary.json`, `signals_per_question_*.csv`, `confidence_vision_rollouts.csv`, `table_signals.csv`, `table_fusion.csv`, `trajectories.json` |
| Step 4 live generation | `live_generations.jsonl` (prompts, responses, seeds, lengths, truncation), `live_questions.csv`, `live_summary.json`, `table_live.csv` |
| Step 5 adaptive | `adaptive_dev_grid.csv`, `adaptive_selected.json`, `adaptive_eval.csv`, `adaptive_per_question.csv`, `adaptive_early_stops.csv`, `adaptive_bootstrap.csv` |
| Step 6 tables and examples | `table_stopping_vs_pool.csv`, `table_sample_growth.csv`, `examples.json` |
| Resources, exact config | `resources.csv`, `offline_run.json`, `adaptive_run.json`, `run_config.yaml` |
| Figures | `figures/` |

## Where each result lives (Task 2)

| What | File(s) under `results/task2_vlm/` |
|---|---|
| Selected ids and split summary | `selection_ids.json`, `selection_summary.json` |
| Interface checks, calibration | `interface_checks.json`, `calibration.json` |
| Learning curves | `train_log.jsonl`, `dev_log.jsonl`, `dev_metrics.csv` |
| Final results (both suites, four image conditions, pair bootstrap) | `eval_final.json`, `final_grounding_test.csv`, `final_vqa_test.csv`, `final_types.csv`, `advantages.csv` |
| Per-question outputs | `eval_final_examples.jsonl` (references, generations, scores, nll) |
| Resources | `feature_cache.json`, `train_summary.json`, `resources.csv` |
| Exact config of the run | `run_config.yaml` |
| Figures | `figures/` (including `complementary_examples.png`, added after the run from saved results; see `../task2_tables_executed.ipynb`) |
| Optional Step 7, 49 visual tokens | the same files under `results/task2_vlm_49/`, executed notebook `results/task2_49_executed.ipynb` (`modal run --detach run_modal.py --task task2_49 --scale`) |

## Where each result lives (Task 1)

| What | File(s) under `results/task1_finetune/` |
|---|---|
| Benchmark table (7 conditions x 4 benchmarks, pp changes) | `benchmark_table.csv`, `benchmark_raw_table.csv`, `eval_*.json` |
| Per-example outputs | `eval_*_gsm8k.jsonl` (response, extracted answers, truncation), `eval_*_{hellaswag,boolq,arc_easy}.jsonl` |
| GSM8K extraction / truncation | `gsm8k_extraction.csv` |
| Learning curves, L2-SP terms | `train_log_*.jsonl`, `displacement.json` |
| Resources | `train_summary_*.json`, `resources.csv` |
| Diagnostics | `diagnostics_weight_space.csv`, `diagnostics_activation_*.csv`, `diagnostics_table.csv` |
| Supplementary: energy in OPLoRA's top-128 directions (base model only) | `oplora_spectrum.csv`, `oplora_spectrum_curves.json`, `oplora_spectrum_summary.json` (`python -m task1_finetune.spectrum`) |
| Merge check | `merge_check_*.json` |
| Qualitative | `qualitative_*.jsonl`, `qualitative_grouped.json`, `qualitative_annotation_sheet.csv` |
| Probe set | `probe_set.json` |
| Figures | `figures/` |
| Exact config of the reported run | `run_config.yaml` (the `--scale` config, paths relative to the Modal volume) |
| Executed notebooks | `../task1_executed.ipynb` (the run itself), `../task1_tables_executed.ipynb` (tables and figures rebuilt from the saved results, no scripts rerun) |

## Reproducibility notes

- Seed 6304 everywhere (`configs/base.yaml`); all hyperparameters are in `configs/`.
- Every training run starts from the original checkpoint and sees the same examples in the same order.
- Re-running is resumable: finished runs, evaluations and generations are skipped; delete the matching `outputs/` and `results/` entries to recompute.
- GPU kernels and FP16 arithmetic are not bit-deterministic, so a re-run can differ slightly from the committed logs.

## Attribution

- `data/first_plot_questions.yaml`: from the Emergent Misalignment repository ([emergent-misalignment/emergent-misalignment](https://github.com/emergent-misalignment/emergent-misalignment), `evaluation/first_plot_questions.yaml`), unchanged.
- Benchmarks are scored with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness); the HellaSwag probe contexts use its `hellaswag/utils.preprocess`.
- `task3_ttc/answers.py`: the MathVista normalization follows the official MathVista evaluation code ([lupantech/MathVista](https://github.com/lupantech/MathVista), `evaluation/calculate_score.py`, `normalize_extracted_answer`); MATH equivalence uses [Math-Verify](https://github.com/huggingface/Math-Verify); the boxed-answer extraction follows the rule in the supplied data README.
- `task2_vlm/vqa_eval.py`: the answer normalization (punctuation, number words, articles, contractions) is ported from the official VQA evaluation code ([GT-Vision-Lab/VQA](https://github.com/GT-Vision-Lab/VQA), `PythonEvaluationTools/vqaEvaluation/vqaEval.py`); the soft accuracy follows the formula in the assignment.
- Public models and datasets: Qwen3-0.6B (Qwen team), SigLIP base patch16-224 (Google), MetaMathQA, GSM8K, HellaSwag, BoolQ, ARC, COCO 2014 captions and instances, VQAv2 with complementary pairs, the supplied ATML-PA3-Data rollouts (Qwen2.5-3B-Instruct on MATH, Qwen2.5-VL-3B-Instruct on MathVista testmini), MathVista testmini metadata, Qwen2.5-3B-Instruct.
- Libraries: PyTorch, Transformers, Accelerate, Datasets, safetensors, lm-evaluation-harness, pandas, NumPy, matplotlib, Pillow, SciPy, PyArrow, Math-Verify, gdown, papermill, Modal (versions in `requirements.txt` and `run_modal.py`).

No other external code was reused.
