# Misc Python tools

A Python package with various utilities.

## Installation

```bash
pip install ptetools
```

For development use the -e flag when installing from source code.

```bash
pip install -e . --upgrade
```


## Development

Build new package
```
python -m build . --sdist
twine upload ...
````
Testing
```
pytest --cov=ptetools
```
