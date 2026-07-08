




def make_fig_name(sweep: str, run_name: str, kind: str, **tags) -> str:
    tag_str = "__".join(f"{k}{v}" for k, v in tags.items())
    return f"{sweep}__{run_name}__{kind}" + (f"__{tag_str}" if tag_str else "") + ".png"

def make_meta_text(**kv) -> str:
    return " | ".join(f"{k}={v}" for k, v in kv.items() if v is not None)

def add_meta_footer(fig, text: str, y: float = 0.01):
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=8, color="gray")