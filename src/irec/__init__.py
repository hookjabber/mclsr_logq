"""IRec, reduced to the MCLSR + SASRec study.

Every component is registered under a ``config_name`` and built from JSON by ``create_from_config``:

* ``dataset``   -- ``base`` (registry root, sequence readers), ``sequences`` (``mclsr``, ``sasrec_comparison``),
                   ``graph`` (the three graphs of the paper), ``samplers`` and ``negative_samplers``;
* ``models``    -- ``mclsr`` (paper eq. 1-7, 9-10, 13, 16), ``sasrec`` (the comparison encoder), ``base``;
* ``loss``      -- ``retrieval`` (L_P: in-batch softmax with the logQ forms, exact softmax, sampled negatives),
                   ``contrastive`` (L_IL / L_UC / L_IC forms), ``base`` (composite objectives), ``counts`` (q tables);
* ``metric``, ``callbacks``, ``dataloader``, ``optimizer``, ``utils``;
* ``train``     -- the exploratory entry point (``python -m irec.train --params <config>``); the confirmatory
                   runner lives in ``scripts/train_confirmatory.py``.
"""
