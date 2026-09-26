# R9 forensic report: behavioural collapse of both arms (2026-08-27, filed after the final verdict)

## Anomaly
The pool-a result of mcurr was bit-identical to the pool-a result of mfresh (mean 105.74274129532098,
depth_hist, l3+, death count and the sequence of death seeds all identical; only the result sha differed).
We then predicted that mcurr pool b would reproduce mfresh pool b bit for bit, and it did
(99.83548711193744 / l3+ 12 / deaths 103).

## Investigation steps and evidence
1. Driver arguments: the command line of the running mcurr-b process confirmed that --manager-npz pointed
   to r9-mcurr/policy.npz (the path was right; the examiner did not hand out the wrong paper).
2. Policy files: the full sha256 of the two NPZ files differ; an array-by-array comparison
   (w0/b0/w1/b1/wa/ba) gives identical=False for every array, with per-layer maxdiff 0.026-0.156: the two
   networks really are different.
3. Behaviour probe (20000 samples x 3 distributions):
   - Uniform on [0,1] (the true domain of the raw-v4 normalised observation): argmax agreement between
     the two networks is 100.00%, and both always choose action 0 (action distributions [20000,0,0] /
     [20000,0,0]).
   - Gaussian N(0,1) and N(0,5) (out of distribution): agreement only 69%, which proves the networks
     differ and merge only on the real manifold.

## Conclusion
The exam was valid; this was not an operations incident. Each arm collapsed into a constant policy
(always action 0), so combined with the frozen worker and fixed seeds they produced bit-identical
trajectories. All numbers behind the final verdict "no winner" are real.

## Mechanism (consistent with earlier dossiers)
The v20 verdict: under the current wage economy, diving has negative expected return. Whatever command
the manager issues, the wage sits on the anchor line, and there is no distinguishable gradient between
commands → the policy collapses to one constant command (ent 0.02 did not prevent argmax merging on the
manifold). The L4/L5 footprints in mcurr's training logs come from the curriculum placing it deep
(deep-start) and did not turn into any behavioural difference from the standard starting point (not
even half a bit).

## Implications for the roadmap
The manager-side tools (retraining mfresh / the deep-start curriculum mcurr) are exhausted and all
falsified. The key to unlocking depth is not on the manager side but in the environment's wage system:
course 2 (DIVE wage reform, environment side, needs approval) moves from an alternative to the only way
forward, and this case forms its chain of evidence.

## Recommendations (for decision; not decisions)
1. If R10 opens course 2, its criteria should include a probe for "distinguishable wages between
   commands" (to prevent another collapse);
2. The behaviour probe (argmax agreement + action distribution) should become a standard pre-check for
   future control-arm exams: when two arms behave identically, a paired test has zero power, and this
   should be declared up front rather than after running the full exam.

This report is a new file and changes no frozen artefact.
