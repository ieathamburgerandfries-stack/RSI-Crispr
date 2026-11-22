"""RSI09 -- sgRNA cutting-efficiency feature engineering for E. coli.

Normally driven from a notebook -- see notebooks/RSI09_pipeline.ipynb:

    import sgrna.harness.notebook as nb
    nb.setup(); nb.build_index(); nb.build_features("all"); nb.ablate(["c_folding"])

Layout
------
notebook.py           the notebook front end; wraps every step below
config.py             every path and constant the project uses
genome.py             NC_000913.2 loading, circular coordinates, oriC/ter
qct.py                decode protospacers from QCT columns; extend the tensors
io_utils.py           feature-block conventions
build_guide_index.py  step 0: locate all 13,880 guides on the chromosome
build_features.py     step 1: build one or more feature families
build_matrix.py       step 2: join chosen families onto the published matrix
run_ablation.py       step 3: 5-seed leakage-safe CV, one family at a time

features/             one module per family, A through H
"""

__version__ = "0.1.0"
