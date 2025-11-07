

# CodeACT-R: A Cognitive Simulation Framework for Human Attention in Code Reading ![CodeACT-R](repo/icon(1).png)

This repository reproduces the CodeACT-R simulation pipeline described in the paper **“CodeACT-R: A Cognitive Simulation Framework
for Human Attention in Code Reading”**.  The Python implementation centers on `simulation.py`, which generates synthetic eye-tracking scanpaths over C++ stimuli while maintaining ACT-R-style cognitive bookkeeping (retrieval requests, buffer updates, and action logs) via the local `pyactr` implementation.

The workflow has two primary modes:

1. **Zero-shot (`--pattern=zero`)** – simulate new target stimuli using only aggregate complexity statistics from the training set.
2. **Augmented (`--pattern=aug`)** – bootstrap the simulator with selected human scanpaths, then perturb them via chunk-level sampling to increase data diversity.

Both modes share the same command-line interface and draw on common data resources.

---

## Repository Structure (C++ Simulation)

| Path | Purpose |
| --- | --- |
| `simulation.py` | Main entry point for generating C++ scanpaths. Implements the two simulation patterns and the pyACT-R logging hooks. |
| `path_seman.py` | Utility that computes Markov-style path transition tables (`generate_path`) from empirical scanpaths. |
| `sti_with_backward.py` | Contains heuristics (`find_variable_and_statements_from_file`) for “quick” strategies that jump between key code elements. |
| `all_scanpath_repeat/` | Human scanpath corpora aggregated per stimulus (`sti_*_all.csv`). Used in the augmented pattern as the experience pool. |
| `stim_info/` | C++ source files (`*.cpp`) and their semantic annotations (`*_sem.csv`). Each target stimulus `t` must appear in this directory. |
| `cross_valid_simu*/` | Output folders populated automatically (one per augmentation level). |

All other Python modules and Java scripts belong to the Java-side experiments and are not required for the C++ workflow.

---

## Simulation Patterns

### 1. Zero-shot Pattern (`--pattern=zero`)
This mode mirrors the “Code Comprehension without augmentation” configuration described in the paper.  It:

1. Computes aggregate complexity for the training set (`train_complexity`) and target test file (`test_complexity`).
2. Samples step counts using `set_step_zero`, which blends the two complexity measures (`comp_measure`) to decide how many fixations to generate.
3. For each simulated fixation step:
   - Uses empirical transition tables (via `generate_path`) when sufficient data exists.
   - Falls back to the `null`, `quick`, or `quick_comp` heuristics to select lines based on code structure, while logging the cognitive context.

The zero-shot pipeline is deterministic for a given `--seed`, and it writes CSV files to `cross_valid_simu/`.

### 2. Augmented Pattern (`--pattern=aug`)
This mode implements the “augmented training” approach from the paper:

1. Selects a subset of participant IDs (`pid`) from `all_scanpath_repeat/sti_<id>_all.csv` using `select_chunks_half`.  This splits the pool into halves (`half_chunk`, `chunk`), emulating knowledge transfer between participant clusters.
2. Uses the sampled real scanpaths (`df_stm`) to build next-step distributions (`generate_path`).
3. Generates synthetic scanpaths by stochastically replaying and perturbing those distributions.  When `step` is small, it alternates between `quick_aug` and `quick_aug_comp` strategies.
4. Outputs CSV files to `cross_valid_simu{augnum[0]}/`.

Because this mode reuses human traces, you must provide augmentation metadata via `--aug`.  If not supplied, the script defaults to `'91'` (use the entire remaining list, label chunk “1”).

---

## Command-Line Arguments

| Argument | Required | Description |
| --- | --- | --- |
| `--number` | ✔ | Number of simulated scanpaths to generate per target stimulus `t`. |
| `--seed` | ✔ | Random seed applied to Python, NumPy, and sampling routines for reproducibility. |
| `--pattern` | ✔ | Either `zero` or `aug`, selecting the pipeline described above. |
| `--s_list` | ✔ | Space-separated list of training stimuli IDs (e.g., `"stim1 stim5"`).  Each ID should correspond to files under `all_scanpath_repeat/` (prefixed with `sti_`). |
| `--t` | ✔ | Target stimulus ID.  The script expects `stim_info/<t>.cpp` and `stim_info/<t>_sem.csv`. |
| `--aug` | (aug only) | Two-character string controlling the augmentation behavior. First char selects the percentage of the remaining participant pool (`3` = 30%, `9` = 100%); second char is an arbitrary label embedded in output filenames. Defaults to `91` if omitted. |

> **Tip:** When passing multiple values to `--s_list`, wrap the list in quotes so the shell does not split the argument: `--s_list "stim1 stim5 stim13"`.

---

## Running the Simulator

The script assumes you are inside the repository root (`/home/yueke/CodeACT-R`).  All commands use Python 3.10+ with `pandas`, `numpy`, and the bundled `pyactr` package on `PYTHONPATH`.

### Zero-shot Example
```bash
python overall_sim_pyactr_strict.py \
  --number 3 \
  --seed 42 \
  --pattern zero \
  --s_list "stim1 stim3 stim5" \
  --t stim2
```
Outputs will appear under `cross_valid_simu/`:
```
cross_valid_simu/stim2_stim1xstim3xstim5_0_aug0_chunk0_seed42_simu.csv
...
```

### Augmented Example
```bash
python overall_sim_pyactr_strict.py \
  --number 5 \
  --seed 7 \
  --pattern aug \
  --s_list "stim1 stim5 stim13" \
  --t stim2 \
  --aug 52
```
This creates `cross_valid_simu5/` if it does not exist and writes CSV files with names such as:
```
cross_valid_simu5/stim2_stim1xstim5xstim13_0_aug52_chunk2_seed7_simu.csv
```

### Output Format
Each CSV row represents a simulated fixation with the following columns:

| Column | Description |
| --- | --- |
| `node` | Token snippet (up to 50 chars) corresponding to the fixated AST node. |
| `Line` | 1-based source line selected by the simulator. |
| `Column` | 1-based column offset within the line. |
| `semanic` | Semantic label derived from the annotated `*_sem.csv` file. |
| `controlnum` | Control-structure identifier used for transition matching. |
| `feature` | Secondary feature label (e.g., `loop`, `condition`). |
| `complexity` | Scalar complexity score copied from the semantic CSV. |
| `corr_section` | Original normalized section value from the target dataset. |
| `section` | Simulated timestamp (step index / total steps). |
| `memtype` | Cognitive strategy invoked (`short_path`, `long_path`, `quick_aug_comp`, etc.). |
| `pidselect` | Participant IDs contributing to this run (comma separated); `NA` in zero-shot mode. |
| `half_pid` | IDs in the “half chunk” used for augmented sampling; `NA` in zero-shot mode. |

Additional ACT-R-inspired state—such as retrieval requests, buffer contents, and action executions—is logged internally but omitted from the CSV for compatibility with downstream tooling. When running the augmented pattern, `pidselect` and `half_pid` trace which empirical scanpaths seeded the simulation; the zero-shot pattern writes `NA` for these fields.

---

## Relation to the ASE NIER Paper

The two simulation patterns correspond to the main experimental configurations reported in **“ASE NIER: ACT-R based gaze-path generation for source code”**:

- The **zero-shot pattern** approximates the baseline condition, where gaze sequences are generated using macro-level complexity statistics.
- The **augmented pattern** matches the paper’s data augmentation strategy, leveraging subsets of real scanpaths to seed further synthetic sequences.

The pyACT-R hooks embedded in `overall_sim_pyactr_strict.py` extend the paper’s conceptual model by logging retrieval requests, buffer contents, and action completions for each fixation decision.  These logs do not change the output but provide richer traces for cognitive analysis or downstream tooling.

For methodological background (e.g., how complexity metrics are computed, why certain heuristics are employed, and how augmentation affects performance), refer to `ASE_nier__ACTR (1).pdf`.

---

## Troubleshooting & Tips

- **Missing directories** – The script auto-creates `cross_valid_simu*/` folders. If you relocate the repository, ensure `stim_info/` and `all_scanpath_repeat/` remain readable.
- **Stimulus naming** – Training stimuli in `--s_list` should match the IDs used in `all_scanpath_repeat/sti_<ID>_all.csv` (typically `sti_####`).  The script strips prefixes internally.
- **Aug argument** – Always supply two characters (e.g., `52`, `91`).  The script reads them separately: `augnum[0]` controls the slice size, `augnum[1]` labels the chunk in filenames.
- **Reproducibility** – Setting `--seed` fixes the random selection of step counts, participant subsets, and stochastic transitions, producing identical CSVs across runs.

---

## Evaluation Script: `cross_valid_eva.py`

After generating synthetic scanpaths you can score them against withheld human data using `cross_valid_eva.py`. This script mirrors the evaluation procedure described in the ASE NIER paper by computing weighted Levenshtein distances between simulated line sequences and empirical gaze paths.

### Inputs

- `cross_valid_simu*/` – Simulation outputs from `overall_sim_pyactr_strict.py`.
- `all_scanpath_repeat/sti_<ID>_all.csv` – Aggregated human scanpaths per stimulus (evaluation pool).
- `stim_info/<t>.cpp` and `stim_info/<t>_sem.csv` – Source and semantic annotations for the target.

### CLI

```
python cross_valid_eva.py \
  --s_list "stim1 stim5" \
  --t stim2 \
  --seed 42 \
  --aug 52      # use 0 for zero-shot evaluation
```

Arguments mirror the simulator:

| Argument | Description |
| --- | --- |
| `--s_list` | Same training set used during simulation (space-separated). |
| `--t` | Target stimulus ID to evaluate. |
| `--seed` | Seed for reproducible random baselines. |
| `--aug` | Augmentation code. Set to `0` for zero-shot evaluation, otherwise use the same code as the generator (e.g., `52`). |

### Outputs

- Augmented runs write to `cross_valid_result{aug[0]}/`.
- Zero-shot runs write to `cross_valid_result/`.
  The directories are created automatically if they do not exist.

#### Augmented Output Columns

| Column | Description |
| --- | --- |
| `stim` | Target stimulus ID. |
| `cross` | Composite label (`tname`) summarizing training stimuli. |
| `num` | Simulation index corresponding to the source CSV suffix. |
| `seed` | Evaluation seed (mirrors the simulator). |
| `aug` | Augmentation code passed on the CLI. |
| `minsimu` | Pair of human scanpaths with minimum distance to the simulation (`[(sublist, simulation)]`). |
| `dist` | Weighted Levenshtein distance between the simulated lines and the closest human path. |
| `pidselect` | IDs of participants whose scanpaths seeded the augmented simulation. |
| `half_pid` | IDs from the complementary “half chunk” used when seeding evaluation paths. |

#### Zero-shot Output Columns

| Column | Description |
| --- | --- |
| `stim` | Target stimulus ID. |
| `cross` | Training-set summary label (same as above). |
| `num` | Simulation index pulled from the filename. |
| `seed` | Evaluation seed. |
| `aug` | Always `0` for zero-shot evaluation. |
| `minsimu` | Closest empirical scanpath to the simulation (line list). |
| `minranline` | Closest empirical scanpath to a random line list of equal length. |
| `minranall` | Closest empirical scanpath to a random line list drawn across the full training distribution. |
| `dist` | Distance between the simulation and its closest empirical path. |
| `rand_dist_line` | Distance between the equal-length random baseline and its closest empirical path. |
| `rand_dist_overall` | Distance for the overall random baseline. |

These columns let you compare real-vs-simulated similarity against two random baselines, reproducing the evaluation in the ASE NIER paper.
