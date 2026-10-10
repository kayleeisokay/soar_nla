"""GCG on AV explanations: does an explanation that reconstructs better read as more faithful?

Each beta is optimized separately on the objective
`MSE(normalize(AR(x)), normalize(a)) / Var(v) + beta * mean_token_NLL_AV(x | a)`.
"""

BETAS = (0.1, 0.01, 0.0)
