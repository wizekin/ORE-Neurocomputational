# Running ORE tests on Kaggle (free cloud runtime)

This folder provides a Kaggle-ready runner for the repository's normal pytest suite and the randomized theory/algorithm evaluation used by the Streamlit Experiment Workbench.

## Important: GPU versus CPU

The current test runner and most ORE tests are ordinary Python, NumPy, SciPy, SQLite, filesystem, and validation workloads. They run on the CPU. Selecting a Kaggle GPU does not automatically accelerate pytest or the randomized Workbench runner. A GPU helps only when a tested operation explicitly uses a GPU-enabled framework (for example, PyTorch CUDA).

Use Kaggle primarily to move sustained computation away from your PC. Start with a Kaggle CPU session; choose GPU only if the particular tests you are running use CUDA. Free accelerator availability and quotas can change, and sessions may be queued or time-limited.

## Quick start

1. Open cloud_testing/ORE_Kaggle_Test_Runner.ipynb in Kaggle Notebooks: https://www.kaggle.com/code
2. Sign in and create a notebook from the notebook file.
3. In Settings → Accelerator, choose None for ordinary pytest and randomized ORE testing. Choose an available GPU only for tests that explicitly use GPU-enabled code.
4. Run the notebook cell. It clones the public GitHub repository, installs development dependencies, runs pytest, and then runs the randomized Workbench evaluation.
5. Download ORE-cloud-test-results.zip from the notebook output/files panel. It contains the pytest log, cloud summary, and randomized evaluation report archive.

The notebook defaults to 5 random trials per eligible callable as a practical first cloud run. Set TRIALS_PER_CALLABLE to 25 (or another value) for a larger run. Set MAX_CALLABLES_PER_PACKAGE = 0 to include every discovered eligible callable.

## Run the script yourself

```bash
python -m pip install -e ".[dev]"
python cloud_testing/run_cloud_tests.py --trials-per-callable 5
```

Examples:

```bash
python cloud_testing/run_cloud_tests.py --trials-per-callable 25
python cloud_testing/run_cloud_tests.py --trials-per-callable 10 --random-seed 20261127
python cloud_testing/run_cloud_tests.py --skip-randomized
python cloud_testing/run_cloud_tests.py --skip-pytest --trials-per-callable 10
```

## Reports and safety

- Cloud runner output is written under the selected --output-dir, defaulting to ore_reports/cloud_runs/.
- The existing randomized Workbench runner continues to write under ore_reports/randomized_theory_tests/.
- The runner does not disable or modify tests.
- It does not write to GitHub. It only clones and tests the checked-out source in the cloud runtime.
- Test failures are preserved in pytest.log; the runner still attempts randomized evaluation and packages the results.
- Randomized tests can execute code paths in ORE packages. Use a clean Kaggle session and do not attach private credentials or unrelated datasets.

## Kaggle resource notes

Kaggle notebook sessions have finite runtimes and limited free accelerator availability. Check current notebook settings and account quota before a long run. A GPU is a scarce resource; do not consume GPU quota for CPU-only tests.