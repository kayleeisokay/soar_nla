"""Google's IFEval instruction checkers (Zhou et al., 2023), vendored.

Source: https://github.com/google-research/google-research, commit
81820f0136b22de81a66abce091ebbb2012e068b, directory
``instruction_following_eval/`` (``instructions.py``,
``instructions_registry.py``, ``instructions_util.py``). Apache License 2.0;
see ``NOTICE`` in this directory. Only the package-absolute imports were
changed to relative imports.

Third-party requirements: ``absl-py``, ``langdetect``, ``nltk`` (with the
``punkt`` and ``punkt_tab`` tokenizer data), ``immutabledict``.
"""
