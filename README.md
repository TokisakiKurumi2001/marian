# What is this repo

One encoder and two decoders training on a parallel corpura of English and France from HuggingFace dataset.
- One decoder for translating
- One decoder for reconstructing the original sentence

**Credit**: This project will not be success if not for the API of HuggingFace team.

# How to install

1. Install the environment using environment.yml from this repo [marian](https://github.com/TokisakiKurumi2001/marian). We use miniconda3

```bash
conda create -f environment.yml
```

2. Install this package

```bash
pip install compromise-marian
```