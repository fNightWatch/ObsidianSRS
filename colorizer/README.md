# Obsidian Graph Colorizer (Python base project)

This small Python tool does two things:

1) **Auto-writes Properties (YAML frontmatter) in your notes**, based on rules that look at:
   - path (folder)
   - tags
   - regex matches in note content

2) **Optionally syncs Obsidian Graph View "Groups"** (stored in `.obsidian/graph.json`) so that:
   - each distinct `graph_color` value gets a Graph group
   - the group color is set to that same hex value

This gets you **automatic coloring in Graph View**, driven by a note property, without writing an Obsidian plugin.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp config.example.yaml config.yaml

python -m obsidian_graph_colorizer --vault "/path/to/Vault" --config config.yaml sync
```

## Watch mode

```bash
python -m obsidian_graph_colorizer --vault "/path/to/Vault" --config config.yaml watch
```

## Notes / caveats

- The tool updates YAML frontmatter **line-by-line**, to avoid reformatting the whole block.
- It only writes **scalar string properties** in this starter project.
- If you enable Graph syncing, it's best to run the tool while Obsidian is closed,
  because Obsidian may overwrite `.obsidian/graph.json` while it is running.

## Extend

- Add more properties in `auto_properties`.
- Add more rules under each property.
