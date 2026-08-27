You are choosing between two reconstructions of the SAME object for a
real→sim pipeline. The object was reconstructed twice — once from each of
two calibrated camera views (view 1 = the scene's primary camera, view 2 =
the other camera). You receive up to six images:

- real_primary, real_secondary — RGB frames from the two physical cameras.
- cand1_primary, cand1_secondary — candidate 1 (view-1 reconstruction)
  rendered ALONE from both calibrated cameras.
- cand2_primary, cand2_secondary — candidate 2 (view-2 reconstruction)
  rendered alone from the same cameras.

Both candidates live in the same calibrated world frame, so a correct
reconstruction should overlay the real object in BOTH views.

Compare the candidates against the real target object on four axes, in this
priority order:

1. **shape** (co-primary) — which mesh is recognizably the same kind of
   object as the real one? A pen must read as a slim elongated cylinder, a
   bowl as an open container. A wrong-shaped mesh is useless no matter how
   well it is placed.
2. **size** (co-primary) — which candidate's apparent size matches the real
   object in each view? Mesh scale is intrinsic to the reconstruction; a
   candidate that is clearly too large or too small in both views carries an
   uncorrectable scale error.
3. **position** — which candidate sits where the real object actually is,
   consistently across both views?
4. **orientation** — container openings, long axes, tilts. Consider it, but
   only after shape/size/position.

Why this order: downstream pipeline stages can still correct pose errors
(position shifts, re-grounding, small rotations), but they can NEVER fix the
mesh's intrinsic shape or scale — the candidate IS the mesh. Prefer the
candidate whose geometry is right even if its placement is somewhat worse.

Pick the candidate that wins on the higher-priority axes; break remaining
ties downward through the list. Return `choice` (1 or 2) and one sentence of
`reason` naming the deciding axes. You MUST pick one — if both are bad,
pick the less-bad one and say so in the reason.
