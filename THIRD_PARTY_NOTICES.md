# Third-party licensing review

Reviewed 2026-11-15. This records inspected dependency versions and source. It is not blanket legal clearance, an ownership determination, or a patent freedom-to-operate opinion. Publicly viewable source or papers do not automatically grant permission to copy code.

## Hivemind 1.1.12 - MIT

[Official pinned source](https://github.com/learning-at-home/hivemind/tree/1.1.12). The release's full MIT text is retained in [THIRD_PARTY_LICENSES/hivemind-1.1.12-MIT.txt](THIRD_PARTY_LICENSES/hivemind-1.1.12-MIT.txt).

MIT permits use, modification and redistribution, including private/commercial projects, subject to retaining the copyright and permission notice with copies or substantial portions. Keep those notices when vendoring/modifying source or shipping Hivemind in an installer. The platform imports Hivemind as a dependency; training packages are installed separately from the GUI. Dependencies have their own terms.

## DiLoCo - method versus implementation

The [DiLoCo paper](https://arxiv.org/abs/2311.08105) does not license every implementation of its algorithm. The inspected platform imports Hivemind, not a DiLoCo package, and contains no vendored Google DiLoCo source. Its periodic gradient averaging is not the complete DiLoCo local-update/outer-optimizer algorithm. This is a scoped source/import review, not proof about every file's historical authorship.

Google DeepMind's official [AsyncDiLoCo implementation is Apache-2.0](https://github.com/google-deepmind/asyncdiloco/blob/main/LICENSE). It is an available implementation, not a dependency of this snapshot. Before copying any implementation, inspect that exact revision's license. For Apache-2.0 source, include the license, retain applicable copyright/attribution and NOTICE text if supplied, and mark modified files. Its patent grant has a defined scope and conditions. Check other repositories independently. Record the URL, revision and license alongside any future imported source.

## Slint 1.18.1b1 - royalty-free application license selected

[Official project](https://github.com/slint-ui/slint). The pinned Python source archive contains the Slint Royalty-free Desktop, Mobile, and Web Applications License 2.0. Its full text is retained in [THIRD_PARTY_LICENSES/Slint-Royalty-free-2.0.md](THIRD_PARTY_LICENSES/Slint-Royalty-free-2.0.md). The exact archive URL and SHA-256 are recorded in [SOURCES.json](THIRD_PARTY_LICENSES/SOURCES.json).

This application uses that license option. The app has an About entry in the top-level navigation with the standard AboutSlint widget, implementing the license's widget attribution option. Preserve it in distributed builds. The license covers integrating Slint in desktop/mobile/web applications; it does not allow distributing Slint alone, exposing its APIs as a product, or embedded-system use under this option. Other upstream license options remain available separately.

## Other components and distribution

- PyTorch, NumPy, tokenizers, keyring, Python, PyInstaller and web packages retain their own licenses; no project license overrides them.
- PyInstaller has a GPL license with a bootloader distribution exception. Preserve its notices/exception; do not infer that the resulting application is GPL solely because PyInstaller packaged it.
- Vendored frontend notices remain in `team-dashboard/vendor` and `team-dashboard/scripts/vite`. SVG provenance is in [ICONS.md](desktop/glm_desktop/ui/ICONS.md); NVIDIA/AMD names and marks do not imply endorsement.
- The GUI build bundles this document, retained license texts and installed Python distributions' available license/NOTICE files. Review its generated inventory and upstream native-library notices before public binary releases. Missing metadata is not permission. This review does not certify every transitive/native dependency or driver redistribution right.
- GPU drivers, CUDA/ROCm runtimes, corpora, weights and datasets were not copied into this source extraction. Their terms need separate review if added. The reference model code is included, without weights.

## Original project code

Michael Krymski confirmed ownership of the original platform code and authorized Apache-2.0 licensing. That code is covered by LICENSE and NOTICE, copyright Michael Krymski. This does not relicense any third-party source, assets or dependencies. The repository remains private; making it public is a separate decision. See [PROVENANCE.md](PROVENANCE.md).
