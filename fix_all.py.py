"""Run from the root of the sc-elt repo:  python fix_all.py   (keep this file OUTSIDE the repo)."""
import pathlib, re

def edit(path, fn):
    p = pathlib.Path(path)
    raw = p.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    s = raw.replace("\r\n", "\n")
    new = fn(s)
    if new == s:
        print("no change:", path); return
    if crlf:
        new = new.replace("\n", "\r\n")
    p.write_bytes(new.encode("utf-8"))
    print("fixed:", path)

def sub(old, new):
    def f(s):
        if new in s: return s
        assert old in s, f"pattern not found: {old!r}"
        return s.replace(old, new, 1)
    return f

# 1. register looped_dit (the 4 one-line edits)
edit("src/eval.py",   sub('if model_name == "dit":', 'if model_name in ("dit", "looped_dit"):'))
edit("src/losses.py", sub('if cfg["model"]["name"] == "dit":', 'if cfg["model"]["name"] in ("dit", "looped_dit"):'))
edit("src/sample.py", sub('if name == "dit":', 'if name in ("dit", "looped_dit"):'))
edit("src/train.py",  sub('    "dit": build_dit_trainer,\n',
                          '    "dit": build_dit_trainer,\n    "looped_dit": build_dit_trainer,   # Looped-DiT-B / ELT share the DiT trainer\n'))

# 2. pos_embed fix from branch zanteni-repository-check (register_buffer clashed with self.pos_embed)
edit("src/model.py", sub(
'''            self.pos_embed = build_2d_sincos_pos_embed(d_model=cfg.hidden_size,grid_h=cfg.grid_h,grid_w=cfg.grid_w)
            self.register_buffer("pos_embed",self.pos_embed.unsqueeze(0),persistent=False)
''',
'''            pos_embed = build_2d_sincos_pos_embed(
                d_model=cfg.hidden_size,
                grid_h=cfg.grid_h,
                grid_w=cfg.grid_w,
            )
            self.register_buffer(
                "pos_embed",
                pos_embed.unsqueeze(0),
                persistent=False,
            )
'''))

# 3. README
def readme(s):
    s = re.sub(r"<!-- VERIFY: batch \(32 x 4.*?-->\n\n", "", s, flags=re.S)
    s = s.replace("**Wajdi Zenteni**", "**Wajdi Zanteni**")
    if "├── LICENSE" not in s:
        s = s.replace("├── requirements.txt\n", "├── requirements.txt\n├── LICENSE\n", 1)
    if "## License" not in s:
        s = s.rstrip("\n") + "\n\n## License\n\nReleased under the [MIT License](LICENSE).\n"
    return s
edit("README.md", readme)

# 4. repair .gitignore
def gitignore(s):
    lines = [l for l in s.split("\n") if l.strip() not in ("assets/raw/", "samples_grid.pngassets/raw/", "# Raw W&B screenshots (kept locally)")]
    while lines and lines[-1] == "": lines.pop()
    if "samples_grid.png" not in lines: lines.append("samples_grid.png")
    lines += ["", "# Raw W&B screenshots (kept locally)", "assets/raw/", ""]
    return "\n".join(lines)
edit(".gitignore", gitignore)
print("done")