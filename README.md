# ml_framework

Reusable ML/DL framework library.

## Why is the package under `src/`?

This repo uses the common Python packaging pattern called **src layout**:

* Source code lives in [`src/ml_framework/`](src/ml_framework:1)
* Packaging config points setuptools to `src/` via [`pyproject.toml:12`](pyproject.toml:12)

This prevents accidentally importing from the repo root during development and matches how the
package behaves after installation.

## Install / Use

### Option A) Use as a Git submodule + editable install (recommended)

In your *other* project:

```bash
git submodule add https://github.com/NamhoKimKR/ml_framework.git libs/ml_framework
python -m pip install -e ./libs/ml_framework
```

Then:

```python
import ml_framework
```

### Option B) Install directly from GitHub

```bash
python -m pip install "git+https://github.com/NamhoKimKR/ml_framework.git@main"
```
