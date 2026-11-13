# modal runner for pa3, one runner for every task
#
# the repo code is baked into the image, and the things that must survive between runs (generated data, checkpoints,
# evaluation models, results, the hugging face cache) live on the same 'atml' volume pa0 to pa2 used, under /pa3
#
# smoke test, real model and real recipe, a couple of updates and a few evaluation docs, results under /pa3/smoke:
#     modal run --detach run_modal.py --task task1 --smoke
#
# the real run, detached, which is what pa0 and pa1 used. the container keeps running on modal's side and the
# results land in the volume, so a dropped laptop connection or a closed terminal cannot kill the run:
#     modal run --detach run_modal.py --task task1
#
# the assignment's own settings are the default. --scale swaps in the larger microbatch described in
# SCALED_OVERRIDES below, sized to the card's memory, which is faster but is a deliberate deviation from the values
# the assignment fixes (see deviations.md). it writes under /pa3/scaled and never mixes with the spec results.
# --smoke --scale smoke-tests the scaled settings:
#     modal run --detach run_modal.py --task task1 --scale
#
# one script on its own (every script skips work that is already on the volume), e.g. the five training runs in
# parallel containers, then the task notebook to fill in everything else:
#     modal run --detach run_modal.py --cmd "python -m task1_finetune.train --method lora4"
#
# --cpu runs a --cmd in a container without a gpu, for work that never touches one (task 2's downloads):
#     modal run --detach run_modal.py --cpu --cmd "python -m task2_vlm.prepare_data"
#
# task 3 runs on its own card (TASK_GPUS), smoke first, then the reported run (no --scale, nothing to scale):
#     modal run --detach run_modal.py --task task3 --smoke
#     modal run --detach run_modal.py --task task3
#
# watch a detached run:   modal app logs atml-pa3
# pull the results back:  modal volume get atml /pa3/results ./

import subprocess
from pathlib import Path

import modal

APP_NAME = 'atml-pa3'
VOLUME_NAME = 'atml' #same volume as pa0 to pa2, everything here stays under /pa3 so the assignments do not mix
VOLUME_PATH = '/mnt/atml'
REMOTE_STORE = f'{VOLUME_PATH}/pa3' #persistent side, on the volume
REMOTE_REPO = '/root/pa3' #code side, baked into the image

#a single fixed gpu type for every reported run, so training time, peak memory and evaluation time are comparable
#and reportable. the rtx pro 6000 (blackwell, 96 gb) runs the fp16 recipe natively with the cu128 torch build, and
#its memory is what the --scale microbatch below is sized against. task 2 uses the same card -> measured training
#step probe (4,500 updates, worst-case 320-token sequences, checkpointing on, card + 8 cores + 32 gib per hour) ->
#    l4        8 x 2 -> 66.9 min ($1.60)   16 x 1 -> 76.6 min ($1.83)
#    l40s      8 x 2 -> 37.9 min ($1.63)   16 x 1 -> 21.3 min ($0.92)
#    pro 6000  8 x 2 -> 25.0 min ($1.53)   16 x 1 -> 12.2 min ($0.75)
GPU = 'RTX-PRO-6000'
SMOKE_GPU = 'RTX-PRO-6000' #same card, so the smoke test also checks memory
#task 3's only gpu work is live generation at batch size 1 with a 3b fp16 model (~6 gb of weights, ~150k new tokens
#in total). one-token-at-a-time decoding is bound by memory bandwidth and python overhead, not compute, so the l4
#($0.80/h) gives the most tokens per dollar. its comparison is internal to task 3, so the card differs from tasks 1-2
TASK_GPUS = {'task3': 'L4'}
CPU_CORES = 8.0 #tokenisation, lm-eval's request building and checkpoint io run on the cpu, as in pa1
MEMORY_MB = 32768 #the merge check holds two fp32 copies of the 0.6b model in ram (~5 gb)
TIMEOUT_SECONDS = 24 * 60 * 60 #modal's maximum, five 1,250-update runs plus seven full evaluations on one card

#directories created on the volume and linked into the repo copy, so the scripts' relative paths transparently
#read and write the volume. data/task1 (the generated training file) and data/task2 (coco/vqav2 metadata, the
#selected images and splits, the cached siglip features). data/ itself holds tracked repo files
PERSISTENT_DIRS = ['data/task1', 'data/task2', 'data/task3', 'outputs', 'results']
VARIANT_DIRS = ['smoke', 'scaled'] #where the smoke and --scale runs write, see make_variant_config

LOCAL_REPO = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version='3.11') #the python version pa0 and pa1 ran on
    #cuda 12.8 builds from the pytorch index, kept separate from the pypi packages below
    .pip_install('torch==2.8.0', index_url='https://download.pytorch.org/whl/cu128')
    .pip_install_from_requirements(str(LOCAL_REPO / 'requirements.txt'))
    .pip_install('papermill==2.7.0', 'ipykernel')
    #the model and dataset downloads land on the volume instead of the container disk, so they happen once, not
    #once per run. parallel --cmd containers started before the cache exists would download side by side
    #expandable segments -> the caching allocator grows its blocks instead of keeping one per tensor shape, so
    #microbatches of changing padded length do not fragment the card's memory near the --scale ceiling
    .env({'HF_HOME': f'{REMOTE_STORE}/hf_cache', 'TOKENIZERS_PARALLELISM': 'false',
          'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'})
    #copy=True bakes the code into the image, so a detached run does not depend on the laptop staying up
    .add_local_dir(
        LOCAL_REPO, remote_path=REMOTE_REPO, copy=True,
        ignore=[*PERSISTENT_DIRS, *VARIANT_DIRS, '.git', '**/__pycache__', '**/.ipynb_checkpoints', '*_executed.ipynb'],
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)

#each task is a thin notebook that runs the task's scripts in order and then builds its tables and figures.
#papermill executes it top to bottom, and the executed copy is kept next to the results as the record of the run
#task2_49 -> the optional step 7, the same notebook and recipe with 49 visual tokens (configs/task2_49.yaml)
TASK_NOTEBOOKS = {'task1': 'task1_finetune/task1.ipynb', 'task2': 'task2_vlm/task2.ipynb', 'task2_49': 'task2_vlm/task2.ipynb',
                  'task3': 'task3_ttc/task3.ipynb'}
TASK_CONFIGS = {'task1': 'configs/task1.yaml', 'task2': 'configs/task2.yaml', 'task2_49': 'configs/task2_49.yaml',
                'task3': 'configs/task3.yaml'}
#config keys that hold a location the run writes, moved under /pa3/<variant> for the smoke and --scale runs. task 2's
#data_dir is shared on purpose -> the downloaded images and cached features are the same for every variant
OUTPUT_KEYS = {'task1': ['outputs_dir', 'results_dir', 'train_file'], 'task2': ['outputs_dir', 'results_dir'],
               'task2_49': ['outputs_dir', 'results_dir'], 'task3': ['results_dir']}

#smoke sizes -> real model, real sequence lengths, batch sizes and precision recipe, just a few updates and docs
SMOKE_OVERRIDES = {
    'task1': {'train_examples': 64, 'checkpoint_updates': [1, 2], 'eval_limit': 8, 'qualitative_max_new_tokens': 64},
    'task2': {'checkpoint_every': 4, 'dev_eval_every': 4, 'eval_limit_pairs': 4, 'bootstrap_resamples': 200},
    #task 3 -> 20 questions per split offline, 2 live questions at the real token caps
    'task3': {'question_limit': 20, 'live_limit': 2, 'bootstrap_resamples': 200},
}
SMOKE_OVERRIDES['task2_49'] = SMOKE_OVERRIDES['task2']
SMOKE_CONDITION_UPDATES = {'task1': {'full_ft_mid': 1, 'full_ft': 2, 'l2sp': 2, 'lora4': 2, 'lora16': 2, 'oplora16': 2}}
SMOKE_STAGE_UPDATES = {'task2': [4, 4, 4], 'task2_49': [4, 4, 4]} #updates per curriculum stage in a smoke test

#--scale -> microbatch 2 x 16 accumulation becomes 16 x 2, so the effective batch (32), the update count (1,250) and
#the loss normalisation (every microbatch divided by the token count of the whole effective batch) are unchanged and
#each update is the same up to fp16 rounding order and the lora dropout masks. peak memory and training time then
#describe this setting, not the prescribed one, so a scaled run is reported as a deviation (deviations.md).
#sized on the rtx pro 6000 (95.6 gib usable) with a worst-case probe, l2-sp (the heaviest run: fp32 weights,
#grads, two adam moments, theta0) on one microbatch padded to the longest training example (1,941 tokens) ->
#    microbatch 8 -> 39.5 gib peak, 16 -> 72.1 gib (89.6 reserved), 32 -> out of memory
#the full-vocabulary logits (152k) grow linearly with the microbatch and dominate, so 16 is the ceiling.
#gsm8k generation 64 -> 128 changes speed only (250 + 1,024 tokens at 128 -> 19.0 gib peak). the likelihood batch
#stays 64 -> 128 x 1,024 tokens already peaks at 75.3 gib and the longest boolq prompt is ~1,270 tokens
#task 2 -> microbatch 8 x 2 becomes 16 x 1. the loss is a per-example mean averaged over the 16 examples of the
#effective batch, and every microbatch adds its sum of per-example means / 16, so again each update is the same up
#to fp16 rounding order and the lora dropout masks. measured peak 6.4 gib (3.8 at 8 x 2), training 25.0 -> 12.2 min
SCALED_OVERRIDES = {
    'task1': {'micro_batch': 16, 'grad_accum': 2, 'gsm8k_batch_size': 128},
    'task2': {'micro_batch': 16, 'grad_accum': 1},
    'task2_49': {'micro_batch': 16, 'grad_accum': 1}, #same setting as the 196-token run, so time and memory compare
}


def link_persistent_dirs():
    #repo/outputs -> volume/pa3/outputs and so on
    for name in PERSISTENT_DIRS + VARIANT_DIRS:
        target = Path(REMOTE_STORE) / name
        target.mkdir(parents=True, exist_ok=True)
        link = Path(REMOTE_REPO) / name
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            if link.exists():
                subprocess.run(['rm', '-rf', str(link)], check=True)
            link.symlink_to(target)


def make_variant_config(task, config, smoke, scale):
    #writes a copy of the task config with the overrides and every output location moved under /pa3/smoke or
    #/pa3/scaled, so a smoke or scaled run can never be mistaken for, or skip over, a spec result
    import sys
    import yaml
    sys.path.insert(0, REMOTE_REPO)
    from common.utils import load_yaml

    cfg = load_yaml(config)
    variant = 'smoke' if smoke else 'scaled'
    if scale:
        cfg.update(SCALED_OVERRIDES[task]) #task 3 has no batch setting to scale
    if smoke:
        cfg.update(SMOKE_OVERRIDES[task])
        for condition, update in SMOKE_CONDITION_UPDATES.get(task, {}).items():
            cfg['conditions'][condition]['update'] = update
        for stage, updates in zip(cfg.get('stages', []), SMOKE_STAGE_UPDATES.get(task, [])):
            stage['updates'] = updates
    for key in OUTPUT_KEYS[task]:
        cfg[key] = f'{variant}/{cfg[key]}'
    path = Path(REMOTE_STORE) / variant / Path(config).name
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(path), variant, [Path(REMOTE_REPO) / cfg[key] for key in OUTPUT_KEYS[task]]


@app.function(gpu=GPU, cpu=CPU_CORES, memory=MEMORY_MB, timeout=TIMEOUT_SECONDS, volumes={VOLUME_PATH: volume})
def run(task, smoke, scale):
    import shutil

    volume.reload() #picks up whatever earlier runs left on the volume
    link_persistent_dirs()
    subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', shell=True)

    import logging
    import papermill

    #papermill reports each cell's output through python logging, without this nothing reaches the modal logs
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    config, variant, executed = TASK_CONFIGS[task], 'spec', f'{REMOTE_STORE}/results/{task}_executed.ipynb'
    if smoke or scale:
        config, variant, written = make_variant_config(task, config, smoke, scale)
        executed = f'{REMOTE_STORE}/{variant}/results/{task}_executed.ipynb'
        if smoke:
            #a smoke test always starts clean, otherwise the skip-if-already-done checks would hide the code under test
            for path in written:
                shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)
    Path(executed).parent.mkdir(parents=True, exist_ok=True)
    print(f'running {task} ({variant}{", scaled" if smoke and scale else ""}) with {config}')

    try:
        papermill.execute_notebook(
            f'{REMOTE_REPO}/{TASK_NOTEBOOKS[task]}', executed,
            parameters={'CONFIG': config},
            cwd=REMOTE_REPO, #the scripts resolve configs/, data/, results/ from the repo root
            kernel_name='python3',
            progress_bar=False,
            log_output=True, #every cell's output goes to the modal logs, so a detached run can still be followed
        )
    finally:
        #committed even when a cell raises, so the partially executed notebook and whatever results exist are kept
        volume.commit()
        print(f'results committed under {REMOTE_STORE}')


def run_command(command):
    volume.reload()
    link_persistent_dirs()
    try:
        subprocess.run(command, shell=True, cwd=REMOTE_REPO, check=True)
    finally:
        volume.commit()


@app.function(gpu=GPU, cpu=CPU_CORES, memory=MEMORY_MB, timeout=TIMEOUT_SECONDS, volumes={VOLUME_PATH: volume})
def shell(command):
    #runs one command inside the same environment and on the same card as the task runs
    run_command(command)


@app.function(cpu=CPU_CORES, memory=MEMORY_MB, timeout=TIMEOUT_SECONDS, volumes={VOLUME_PATH: volume})
def cpu_shell(command):
    #the same environment without a gpu, for downloads and other cpu-only work (billed for cores and memory only)
    run_command(command)


def wait(call):
    #spawned calls are waited on -> the logs still stream to the terminal, but a dropped connection between the laptop
    #and modal only ends the wait. with .remote() that connection error was raised inside the entrypoint, and an
    #entrypoint that exits with an error makes modal cancel the running call, even under --detach. errors raised
    #inside the run itself still come through call.get()
    try:
        call.get()
    except (modal.exception.ConnectionError, TimeoutError) as error:
        print(f'lost the connection to modal ({error!r}). with --detach the run carries on, follow it with '
              f'modal app logs {APP_NAME}')


@app.local_entrypoint()
def main(task: str = 'task1', smoke: bool = False, scale: bool = False, cmd: str = '', cpu: bool = False):
    if cmd:
        wait((cpu_shell if cpu else shell).spawn(cmd))
        return
    #the smoke test can swap in its own card, the real run keeps the fixed GPU constant. task 3 uses its own card
    #for both
    function = run.with_options(gpu=TASK_GPUS[task]) if task in TASK_GPUS else run.with_options(gpu=SMOKE_GPU) if smoke else run
    wait(function.spawn(task, smoke, scale))
