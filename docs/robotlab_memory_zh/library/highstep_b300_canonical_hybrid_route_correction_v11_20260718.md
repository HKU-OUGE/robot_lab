# B300 canonical hybrid route-correction amendment v1.1

This amendment is subordinate to, and must be read together with,
`highstep_b300_canonical_hybrid_prior_latent_spec_20260718.md`.  It changes only
the source-state distribution used after the completed E300 checkpoint.  All
Teacher, model, target, optimizer, loss, action and deployment contracts in the
base specification remain unchanged.

## Frozen evidence

- E300 checkpoint SHA256:
  `e7a3eec3cc0848b4af7e9a8608bc281ad94c64ec1e299f69cedcbe34bf04d388`
- E300 closed-loop result: valid/safe `15/15`, full climb `0/15`, rear hold
  `0/15`.
- On the exact approved 138-frame canonical pre-step observations, E300 has
  non-box action MSE `0.000462755881017074`; substituting the Teacher latent
  makes the frozen non-box actor error numerically zero (max absolute error
  `4.76837158203125e-06`).
- The E300 box post-prior target remains imperfectly fitted (MSE
  `0.012280730530619621`), and the closed-loop Student reaches states absent
  from the one canonical information unit.

The evidence rules out actor binding as the main cause.  It shows that further
canonical-only E500/E700 updates do not address the first distributional
failure.  The interrupted E500 artifacts remain historical evidence and are
forbidden as a recovery source.

## Single-variable correction

Start from the complete E300 checkpoint and its original optimizer.  Replace
`canonical Teacher-state samples only` with
`canonical samples plus one Student-driven nominal trajectory per DAgger
round`.  The frozen Teacher labels the exact same Student pre-step state.  This
is the only changed training variable.

- No E500/E700 canonical-only continuation.
- At most three DAgger rounds, 100 effective updates each.
- Each round collects exactly one 138-step nominal trajectory at the approved
  reset, command trace, gap, seed and delay.  Repeated copies are forbidden.
- After each round, run 15 nominal Student-driven evaluations.
- Stop at valid `15/15`, full climb `>=12/15`, rear hold `>=12/15`, safe
  `15/15`; otherwise stop after round three.
- Final acceptance remains user visual review.  Automated metrics cannot
  declare the motion visually acceptable or deploy it.

## Unchanged contracts

The B300 Teacher and its prior, the 570/64/16 network, mixed pre-prior/non-box
and post-prior/box targets, estimator and four box rows, optimizer state,
learning rates, losses, action scale, joint clipping, default pose, gains,
manual policy2/live-joystick interaction and deployment format are unchanged.

