"""Guide design: from a genome and a cut site to the SpCas9 guide SLICER predicts will cut best there.

    python -m sgrna.design.design_guide --genome genome.fasta --cut-site 1234567

`train_model` fits the shipped models, `design_guide` is the tool, `check_design` verifies it against the stored feature matrices,
`design_benchmark` measures how much picking by SLICER's score beats picking at random among the guides near a site.
"""
