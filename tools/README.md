# Site tools

- `build_blog.py`: builds `blog/`, `sitemap.xml` and the post list on the home page from `posts/*.md` (`pip install markdown`, then `python3 tools/build_blog.py`). Post images go in `blog/img/<slug>/`.
- `extract_bench.py`: regenerates `data/depth.json` from the ds4 benchmark runs; paste the JSON into the `#depth-data` script tag in `index.html`.
