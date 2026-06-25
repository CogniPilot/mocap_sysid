# Benchmark Site

This directory is a static GitHub Pages application for browsing benchmark results.
Browser dependencies that are not vendored are pinned in `package.json` and
`package-lock.json`; do not commit `node_modules/`.

Run locally from the repository root:

```bash
./results.py serve-site
```

Then open the printed local URL.

Refresh the data bundle with:

```bash
./results.py web-data
```

Run a fast end-to-end setup check with:

```bash
./results.py check-setup
```

The app reads JSON from `site/public/data/`. It does not run benchmark methods in the browser; benchmark execution remains Python-native and the site is only an interactive results viewer.
