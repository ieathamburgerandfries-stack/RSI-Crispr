# Notice for the files in this folder

**`resources/qct_lookup_tables.json`** holds the quantum-chemical values of mono-, di-, tri- and tetranucleotides and base pairs published by Noshay et al. in
https://github.com/nosha003/sgRNA_iRF (files `HL.Bond.Basepair.txt`, `HL.Bond.Monomer.txt`, `HL.Bond.Dimer.txt`, `HL.Bond.Trimer.txt`, `HL.Bond.Tetramer.txt`, commit 696c376), re-formatted
as JSON and restricted to DNA k-mers; every value equals theirs to within a relative 1e-6 (ours are rounded). That repository is distributed under the MIT licence:

```
MIT License

Copyright (c) 2022 Jaclyn Noshay

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**`resources/v_column_mapping.json`** says which (position, k-mer) indicator each position-by-k-mer column of the feature matrix of Noshay et al. (2023, *Nucleic Acids Research* 51:10147,
CC BY 4.0, https://doi.org/10.1093/nar/gkad736) is. It was decoded from that matrix; credit Noshay et al.

**`slicer_tuned.*` and `slicer_default.*`** are SLICER models trained on the 33,567 curated guides of the *E. coli* SpCas9 screen: the curated release of Browne et al. (2026, CC BY,
https://doi.org/10.7717/peerj.20706) of the screen of Guo et al. (2018, *Nucleic Acids Research* 46:7052, CC BY-NC, https://doi.org/10.1093/nar/gky572). Because the training guides and scores
come from a CC BY-NC article, use the models for non-commercial research, and cite all three papers.

This is a reading of the source licences, not legal advice.
