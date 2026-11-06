# aggregated-delta

Code for *Efficient Lookahead Encoding and Abstracted Width for Learning General Policies in Classical Planning* (NeurIPS 2026).
Datasets, trained checkpoints and planner outputs: [doi:10.5281/zenodo.24007972](https://doi.org/10.5281/zenodo.24007972).

```
striplan/          training and planning (Python package with a C++/nanobind backend)
wheels/            prebuilt wheels, CPython 3.12, linux x86-64
docker/            Dockerfile, Apptainer definition, pinned requirements, wheel build script
build_docker.sh    builds the container with docker or podman
build_apptainer.sh builds it with Apptainer
```

`striplan` builds on [pymimir](https://github.com/maichmueller/mimir) (a fork of mimir with abstracted IW, commit `26c62e7`, version 0.14.1),
[pymimir-rgnn](https://github.com/simon-stahlberg/mimir-rgnn) 0.2.5 and [pymimir-rl](https://pypi.org/project/pymimir-rl/) 0.2.6. Their wheels are in `wheels/`.

## Run

Download and unpack the data archive, then build the container and plan from the data folder:

```
./build_docker.sh                         # or ./build_apptainer.sh
cd /path/to/data
docker run --rm -it --platform linux/amd64 -v "$PWD:/data" beiw      # GPU: add --gpus all
python -m striplan.plan --cpu \
    --domain datasets/sokoban-ipc/test/domain.pddl --problem datasets/sokoban-ipc/test/p02-easy.pddl \
    --model checkpoints/AIW-AD/sokoban-ipc/seed_2/best-2-20-19.pt \
    --plan_output_dir out --summary_output_dir out --flat_plans  # drop --cpu to plan on the GPU
```

The encoding and IW settings are read from the checkpoint. With Apptainer:
`apptainer exec --nv --bind "$PWD:/data" --pwd /data /path/to/beiw.sif python -m striplan.plan ...`.
With podman, use `--device nvidia.com/gpu=all` for GPUs and `-v "$PWD:/data:Z"`.

The image is x86-64 only (about 8 GB, mostly CUDA libraries); on ARM hosts it runs under emulation.

## Train

Each run trains one domain; the paper used 5 seeds per (column, domain), some columns more. From the data folder, inside the container:

```
python -m striplan.train --train datasets/<domain>/train --validation datasets/<domain>/validation \
    --output runs/<domain>/<seed> --no_wandb \
    --hindsight lifted --reward constant --num_rollouts 4 --num_episodes 1000 \
    --validation_strategy test_curriculum --validation_horizon 100 --validation_num_workers 4 \
    --iw1_incremental_first_applicability --iw1_precheck_add_effect_novelty \
    <column flags>
```

Here `AD` stands for `--iw_depth_mode depth_nodes --iw_depth_precedes_mode mlp --iw_depth_order_aux_loss --iw_depth_order_aux_weight 1`,
and `AIW` for `--projective_iw --typed_projection --keep_goal_nonunary_atoms`.

| column (data folder) | column flags |
|---|---|
| AIW–AD (`AIW-AD`) | `--encoding concurrent_internal_tree` AIW AD |
| IW–AD (`IW-AD`) | `--encoding concurrent_internal_tree` AD |
| AIW–Ext (`AIW-Ext`) | `--encoding sequential_external` AIW |
| IW–Ext (`IW-Ext`) | `--encoding sequential_external` |
| BAIW (`BAIW`) | `--encoding concurrent_internal_tree --projective_iw --keep_goal_nonunary_atoms` AD |
| C-AIW (`C-AIW`) | `--encoding concurrent_internal_tree --iw_beam_width 1000 --iw_layer_ordering_strategy goal_count_more` AIW AD |
| ID (`Encoding-ID`) | `--encoding sequential_delta` AIW |
| Int (`Encoding-Int`) | `--encoding sequential_internal` AIW |
| No Depthnodes (`Ablation-NoDepthnodes`) | `--encoding concurrent_internal_tree` AIW |
| No Tree-Atoms (`Ablation-NoTreeAtoms`) | `--encoding concurrent_internal --projective_iw --keep_goal_nonunary_atoms` AD |
| No Depthnodes + No Tree-Atoms (`Ablation-NoDepthnodes-NoTreeAtoms`) | `--encoding concurrent_internal --iw_depth_mode depth_nodes --iw_depth_precedes_mode mlp` AIW |
| Reduced Goals (`Ablation-ReducedGoals`) | `--encoding concurrent_internal_tree --projective_iw --typed_projection` AD |
| Flat–AA (`Flat-AA`) | `--encoding ground_actions --flat --layers 60 --max_buffer_size 1000 --train_horizon 100 --validation_horizon 200`, without the `--iw1_*` flags |
| Flat AD (`Flat-AD`) | `--encoding concurrent_internal --flat --layers 60 --max_buffer_size 1000 --train_horizon 100 --validation_horizon 200`, without the `--iw1_*` flags |

Drop `--no_wandb` to log to Weights & Biases. `python -m striplan.train --help` lists all options.

## Rebuild the wheels

```
podman run --rm -v "$PWD:/code:Z" -w /code quay.io/pypa/manylinux_2_28_x86_64:2025.11.09-2 bash docker/build_wheels.sh
```

This fetches mimir and mimir-rgnn from GitHub at the pinned commits. To build without them, copy `code/src/` from the data archive (snapshots of both, and git bundles of the mimir dependencies Loki, nauty and tree-compression) to `src/`.
pymimir and striplan exchange objects through nanobind, so both must use the same nanobind internals generation: the script builds pymimir with nanobind 2.7.0, and striplan's conan build uses 2.9.2, both generation 16.

## License

GPL-3.0 (see `LICENSE`), as for mimir and mimir-rgnn, which striplan links against.
