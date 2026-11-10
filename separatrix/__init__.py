"""separatrix: a measurement toolkit for generative separatrix reconstruction.

The package holds the reusable pieces; the runnable experiments live in
``experiments/``.  Module map:

``dynamics``   Newton map, vectorised basin labelling
``dataset``    labelled uniform grids (stage 1 of the paper's pipeline)
``classifier`` MLP + MC-dropout entropy (stage 2), torch/GPU or scikit-learn
``refine``     the entropy-driven iterative-refinement loop (stage 3)
``evaluate``   geometric fractal-band oracle and honest metrics
``islands``    buried-island catalogue and per-island recovery statistics
``peninsula``  corridor-thickness / root-to-tip coordinates on the fractal
``stats``      seed-averaged curves, freeze boundary, bootstrap intervals
``plotting``   the paper's Fig. 1 reproduction and the metric-bias figure
``paths``      where figures, tables and caches are written

Reference: E. R. Crabtree, D. G. Giovanis, A. Georgiou, G. Datseris,
I. G. Kevrekidis (2026), *Generative Learning of Separatrices*,
arXiv:2608.14743.
"""

__version__ = "0.2.0"
