import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='compromise_marian',
    packages=['compromise_marian'],
    version='0.1.2',
    description='Marian model but with two decoders',
    author='Le Minh Khoi',
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    py_modules=["compromise_marian"],
    install_requires=['torch', 'pytorch-lightning', 'wandb', 'transformers', 'datasets', 'tokenizers', 'evaluate', 'sacrebleu', 'sentencepiece', 'sacremoses']
)