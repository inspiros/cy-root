# Documentation Website

The mdBook site reuses the [manual](../README.md) through include files; edit
those pages rather than duplicating content. Assets come from the repository's
`assets/` directory.

## Build And Preview

Install [mdBook 0.5](https://rust-lang.github.io/mdBook/guide/installation.html)
and Python 3, then run from the repository root:

```sh
mdbook build docs/site
mdbook serve docs/site --hostname 127.0.0.1 --port 3000
```

Output defaults to `/tmp/cyroot-book-preview`; use `--dest-dir PATH` elsewhere.
The site targets GitHub Pages at `/cy-root/`.
MathJax needs network access to render equations.

## Deploy To GitHub Pages

Set the repository's Pages source to **GitHub Actions**, then open
[Actions → Documentation](https://github.com/inspiros/cy-root/actions/workflows/docs.yml)
and choose **Run workflow** on `master`. It checks documentation links, builds
the site and deploys to <https://inspiros.github.io/cy-root/>. Deployment runs
only when manually requested; pushes and tags do not trigger it.

## Check And Extend

```sh
.venv/bin/python docs/site/check_docs.py
```

Add chapters to `src/SUMMARY.md` with includes of the canonical manual pages.
The preprocessor preserves equations and pins source and example links to the
release tag of the version in `cyroot/_version.py`.
Colors and layout live in `theme/cyroot.css`; the existing logos are reused.

The checker also supports `--snippets --compiled-example-dir PATH` after building
the native examples. The package tests remain `python tests/run_tests.py`.
