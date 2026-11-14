# Paper 5 patch to the ntn-rrc module (Nov 2026)

`ns-3-dev` is shared with running campaigns, so this revision did not touch it or its build. The paper's
ns-3 data come from one standalone binary, `../bin/p5-ul-timing`, built by `build.sh` from the files here against
the existing optimized libraries in `ns-3-dev/build/lib`.

- `orig/`: the source files as copied from `contrib/ntn-rrc` on 3 Nov 2026 (the precompensator on 4 Nov).
- `ntn-sib19.{h,cc}`, `ntn-ul-timing-precompensator.{h,cc}`, `ntn-rrc-ul-timing.cc`: patched copies. The 4 Nov revision adds `--leadS` and `--leadAware` (content announced ahead of its epoch, fit and rule over [-L, V]) and lets the UE propagate backward before the epoch; with no lead it reproduces the 3 Nov binary run for run (`../regress_lead0.py`). `ntn-rrc_p5.patch` is the diff against `orig/`.
- `apply_edits.py`, `ex_helpers.inc`: how the patched copies were produced from `orig/`.

The patched broadcaster is compiled under another class name (`-DNtnSib19Broadcaster=NtnSib19BroadcasterP5`, with
`P5_STANDALONE` renaming its log component and TypeId). Library code built with `-fno-semantic-interposition` would
not reliably be overridden, and registering a second TypeId or log component under the same name is fatal, so there is
deliberately no symbol override: the library keeps its own class and the executable uses its own.
Checked: with equivalent flags the binary reproduces the shared-build example run (1626 samples, 10 reads,
max 367.869 ns), and `--implNs` changes the chosen validity, so the patched rule is the one running.

**To merge at the shared build:** apply `ntn-rrc_p5.patch` under `contrib/ntn-rrc` (model and example paths), drop the
`P5_STANDALONE` blocks and change the include in the example back to `"ns3/ntn-sib19.h"`, then rebuild. Nothing else depends on the rename.
