# Installation

### Install the latest stable version

`MUniverse` is available on [`PyPI`](https://pypi.org/project/muniverse-emg/). We recomment to install the latest stable version via:

```sh
pip install muniverse-emg
```

### Install from source

You can also install `MUniverse` from source:

(1.) **Clone the repository:**

```sh
git clone https://github.com/dfarinagroup/muniverse.git
cd muniverse
```

(2.) **Create and activate a virtual environment:**
```sh
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

(3.) **Install the package:**
```bash
# Install core dependencies
pip install -e .

# Install with external decomposition algorithms (optional)
pip install -e ".[algs]"

# Install with development dependencies (optional)
pip install -e ".[dev]"

# Install with documentation dependencies (optional)
pip install -e ".[docs]"

# Install all optional dependencies
pip install -e ".[algs,dev,docs]"
```