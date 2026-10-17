import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from oncora.config import DATA_INTERIM, FIGURES  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.hashsalt": "oncora", "font.size": 9})


def trainable_model_ids() -> list[str]:
    """Cell lines with both CCLE expression and DSS3 labels (the 985 from the bridge)."""
    b = pl.read_parquet(DATA_INTERIM / "cellline_bridge.parquet")
    return b.filter("in_ccle", "in_dss3")["model_id"].to_list()


def save_fig(fig, name: str) -> str:
    """Save under reports/figures and return the path relative to reports/ for markdown links."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return f"figures/{name}"
