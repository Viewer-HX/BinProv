# Hugging Face release

The public model repository is **`XuViewer/binprov`**.

Published on 2026-09-06: <https://huggingface.co/XuViewer/binprov>. The public
snapshot contains 32 release files. A clean Hub download reproduced the local
`model.pt` SHA-256
`93efdc354956bd8e3f5abc254b97538947b856e185c737f891f00ddb95065247` and passed
the bundled MPS inference smoke test.

- `XuViewer` keeps the model under the author's personal account and independent
  of organization membership. It can be transferred to an organization later.
- `binprov` matches the project and paper name. The current `opt4` task,
  architecture, context width, and seed are recorded in the model card and
  release metadata rather than the repository name.
- `XuViewer/binprov` was unoccupied when checked on 2026-09-06.

The upload source is the gitignored
`results/gpu_release/opt4_wide_seed29/export/` directory. Its root
`README.md` is the Hub model card and includes valid metadata, direct-use
instructions, metrics, training provenance, limitations, and citation. The
directory bundles the minimal BinProv runtime and uses no `trust_remote_code`.

Publishing uses a write-capable user token. Saving it as a Git credential is not
required for the `hf` CLI upload path used here.

```bash
hf auth login
hf auth whoami --format json
hf repos create XuViewer/binprov --type model --public
hf upload-large-folder XuViewer/binprov \
  results/gpu_release/opt4_wide_seed29/export \
  --repo-type model
```

The published page renders the MIT license and `binprov` library metadata. The
clean-download smoke check above completed before the release was recorded here.
