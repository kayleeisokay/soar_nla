"""Faithfulness experiments: does a more faithful explanation reconstruct better?

Three parts, one per paper subsection:
  fve.py, paper_numbers.py  AV vs Sonnet explanations through the custom L32 NLA's AR, at the
                            warm-start SFT and RL checkpoints, plus the random-pairing control.
  sentence_judge.py         sentence-level faithfulness of the same Sonnet and AV explanations.
  surgical_edits/           factual corrections of official-NLA AV explanations.
"""
