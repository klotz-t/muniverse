### Loading a Dataset

```python
from muniverse.datasets import load_dataset

# Load a dataset from Harvard Dataverse
dataset = load_dataset("neuromotion-test", output_dir="./data")
```

### Running Decomposition

```python
from muniverse.algorithms import decompose_recording

# Run decomposition using CBSS algorithm
results, metadata = decompose_recording(
    data="path/to/emg_data.edf",
    method="cbss"
)

# Or use SCD algorithm (requires container)
results, metadata = decompose_recording(
    data="path/to/emg_data.edf",
    method="scd",
    container="path/to/muniverse_scd.sif",
    engine="singularity"
)
```