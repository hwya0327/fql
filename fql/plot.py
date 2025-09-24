import os
import re
import numpy as np
import pandas as pd
from tbparse import SummaryReader
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

env_configs = {
    "Hopper-v4":      {"steps": 1_000_000, "yticks": np.arange(0,  4500,  500), "xticks": np.arange(0, 1_000_001, 2_00_000)},
    "HalfCheetah-v4": {"steps": 5_000_000, "yticks": np.arange(0, 21000, 3000), "xticks": np.arange(0, 5_000_001, 1_000_000)},
    "Walker2d-v4":    {"steps": 5_000_000, "yticks": np.arange(0,  7000, 1000), "xticks": np.arange(0, 5_000_001, 1_000_000)},
    "Ant-v4":         {"steps": 5_000_000, "yticks": np.arange(-1500,  9000, 1500), "xticks": np.arange(0, 5_000_001, 1_000_000)},
    "Humanoid-v4":    {"steps": 5_000_000, "yticks": np.arange(0,  8000, 1000), "xticks": np.arange(0, 5_000_001, 1_000_000)},
}

palette = {
    "DDPG": "#2ca02c",
    "MEOW": "#1f77b4",
    "SAC":  "#ff7f0e",
    "TD3":  "#8c564b",
    "FQL":  "#17becf",
}

order = ["DDPG", "TD3", "MEOW", "SAC", "FQL"]

plt.rcParams["font.size"] = 40
plt.rcParams["axes.labelsize"] = 44
plt.rcParams["axes.titlesize"] = 44
plt.rcParams["xtick.labelsize"] = 36
plt.rcParams["ytick.labelsize"] = 36
plt.rcParams["legend.fontsize"] = 28
plt.rcParams["figure.titlesize"] = 48

def mean_std(a):
    a = np.asarray(a, float)
    n = a.size
    if n == 0:  return 0.0, 0.0
    if n == 1:  return float(a[0]), 0.0
    return float(a.mean()), float(a.std(ddof=1))

def _parse_dirname(dir_name: str):
    toks = [t for t in str(dir_name).split("/") if t]
    if not toks:
        return "UNKNOWN", None, None, dir_name
    algo = toks[0].upper()
    beta = method = seed = None
    if algo == "FQL":
        rest = toks[1:]
        if len(rest) >= 1:
            beta = rest[0]
        if len(rest) >= 2:
            method = rest[1]
        if len(rest) >= 3:
            m = re.fullmatch(r"\d+", rest[2]) or re.fullmatch(r"seed[_\-]?(\d+)", rest[2], flags=re.IGNORECASE)
            if m:
                seed = m.group(1) if hasattr(m, "groups") and m.groups() else rest[2]
        if seed is None:
            for t in reversed(rest):
                m = re.search(r"(?:^|[_\-])(\d+)$", t)
                if m:
                    seed = m.group(1)
                    break
    else:
        if len(toks) >= 2:
            m = re.fullmatch(r"\d+", toks[1]) or re.fullmatch(r"seed[_\-]?(\d+)", toks[1], flags=re.IGNORECASE)
            if m:
                seed = m.group(1) if hasattr(m, "groups") and m.groups() else toks[1]
        if seed is None:
            for t in reversed(toks[1:]):
                m = re.search(r"(?:^|[_\-])(\d+)$", t)
                if m:
                    seed = m.group(1)
                    break
    if seed is None:
        seed = dir_name
    return algo, beta, method, str(seed)

def _resolve_per_env(env, spec, default=None):
    if spec is None:
        return None
    if isinstance(spec, dict):
        return spec.get(env, default)
    return spec

def load_df(env, max_steps, beta_spec=None, method_spec=None, default_beta=None, default_method=None):
    df = SummaryReader(os.path.join("runs", env), pivot=True, extra_columns={"dir_name"}).scalars
    if df.empty:
        return df
    df = df.loc[df["Test/return"].notna(), ["step", "Test/return", "dir_name"]].copy()
    parsed = df["dir_name"].apply(_parse_dirname)
    df["algo"]   = parsed.apply(lambda x: x[0]).astype("string")
    df["beta"]   = parsed.apply(lambda x: x[1]).astype("string")
    df["method"] = parsed.apply(lambda x: x[2]).astype("string")
    df["seed"]   = parsed.apply(lambda x: x[3]).astype("string")
    df = df[df["seed"].str.fullmatch(r"[1-5]")]
    target_beta   = _resolve_per_env(env, beta_spec, default_beta)
    target_method = _resolve_per_env(env, method_spec, default_method)
    if target_beta is not None:
        is_fql = df["algo"].str.upper().eq("FQL")
        df = pd.concat([
            df.loc[~is_fql],
            df.loc[is_fql & df["beta"].notna() & (df["beta"].str.lower() == str(target_beta).lower())]
        ], ignore_index=True)
    if target_method is not None:
        is_fql = df["algo"].str.upper().eq("FQL")
        df = pd.concat([
            df.loc[~is_fql],
            df.loc[is_fql & df["method"].notna() & (df["method"].str.lower() == str(target_method).lower())]
        ], ignore_index=True)
    if df.empty:
        return df
    df["step"]        = pd.to_numeric(df["step"], downcast="integer")
    df["Test/return"] = pd.to_numeric(df["Test/return"], downcast="float")
    df["algo"]        = df["algo"].astype("category")
    df["seed"]        = df["seed"].astype("category")
    max_by_algo = df.groupby("algo", observed=True)["step"].transform("max")
    df["step"]  = df["step"] * np.where(max_by_algo < 10000, 10000, 1)
    return df.loc[df["step"] <= max_steps].reset_index(drop=True)

def stats(df, env, order):
    if df.empty:
        return
    present_algos = [a for a in order if a in df["algo"].cat.categories]
    for algo in present_algos:
        adf = df.loc[df["algo"] == algo]
        if adf.empty:
            continue
        cutoff = int(adf.groupby("seed", observed=True)["step"].max().min())
        valid  = adf.loc[adf["step"] <= cutoff, ["seed", "step", "Test/return"]]
        r      = valid["Test/return"].fillna(0.0)
        mc     = r.groupby(valid["step"], observed=True).mean()
        if mc.empty:
            continue
        best_step    = int(mc.idxmax())
        grouped      = adf.sort_values(["seed", "step"]).groupby("seed", observed=True)["Test/return"]
        max_vals_s   = grouped.max()
        final_vals_s = grouped.last()
        best_vals_s  = valid.loc[valid["step"] == best_step].groupby("seed", observed=True)["Test/return"].mean().fillna(0.0)
        seeds        = adf["seed"].unique().tolist()
        best_vals    = [float(best_vals_s.get(s, 0.0))  for s in seeds]
        max_vals     = [float(max_vals_s.get(s, 0.0))   for s in seeds]
        final_vals   = [float(final_vals_s.get(s, 0.0)) for s in seeds]
        m1, s1 = mean_std(best_vals)
        m2, s2 = mean_std(max_vals)
        m3, s3 = mean_std(final_vals)
        print(f"[{env}] {algo:<5}: step: {m1:.2f} ± {s1:.2f} | seed: {m2:.2f} ± {s2:.2f} | final: {m3:.2f} ± {s3:.2f}")

def draw_curves(env, df, cfg, order, palette, out_png, out_pdf):
    if df.empty:
        return
    present_algos = [a for a in order if a in df["algo"].cat.categories]
    if not present_algos:
        return
    agg = (
        df.groupby(["algo", "step"], observed=True)["Test/return"]
          .agg(mean="mean", sd=lambda x: x.std(ddof=1))
          .reset_index()
    )
    plt.figure(figsize=(16, 16))
    ax = plt.gca()
    for algo in present_algos:
        sub = agg.loc[agg["algo"] == algo].sort_values("step")
        x   = sub["step"].to_numpy()
        y   = sub["mean"].to_numpy(float)
        sd  = np.nan_to_num(sub["sd"].to_numpy(float), nan=0.0)
        c   = palette.get(algo)
        ax.plot(x, y, linewidth=2.0, color=c)
        ax.fill_between(x, y - sd, y + sd, alpha=0.15, linewidth=0, color=c)
    ax.set_xlim(0, cfg["steps"])
    ax.set_xticks(cfg["xticks"])
    ax.set_yticks(cfg["yticks"])
    ax.set_xlabel("Timesteps", labelpad=15)
    if env in ("Hopper-v4", "Walker2d-v4"):
        ax.set_ylabel("Average Return", labelpad=20)
    else:
        ax.set_ylabel("")
    ax.grid(True, alpha=0.3, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0)
    plt.close()

def plot(env, df, cfg, order, palette):
    if df.empty:
        return
    draw_curves(env, df, cfg, order, palette,
                out_png=f"figures/{env}.png",
                out_pdf=f"figures/{env}.pdf")

def save_legend_bar(labels, colors, out="figures/legend_bar.png"):
    handles = [Patch(facecolor=colors[l], edgecolor=colors[l], alpha=0.9) for l in labels]
    fig = plt.figure(figsize=(8, 0.7))
    ax = fig.add_subplot(111)
    ax.axis("off")
    fig.legend(handles, labels, loc="center", ncol=len(labels),
               frameon=True, fancybox=True, framealpha=0.35, edgecolor="#d9d9d9",
               handlelength=1.2, handletextpad=0.6, columnspacing=1.8)
    plt.tight_layout(pad=0.3)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    beta_by_env = {
        "Hopper-v4": "5.0",
        "HalfCheetah-v4": "2.0",
        "Walker2d-v4": "2.0",
        "Ant-v4": "0.0",
        "Humanoid-v4": "0.0",
    }

    method_by_env = {
        "Hopper-v4": "Critic",
        "HalfCheetah-v4": "Critic",
        "Walker2d-v4": "Critic",
        "Ant-v4": "Critic",
        "Humanoid-v4": "Critic",
    }
    
    for env, cfg in env_configs.items():
        df = load_df(
            env, cfg["steps"],
            beta_spec=beta_by_env,
            method_spec=method_by_env,
            default_beta=None,
            default_method=None,
        )
        stats(df, env, order)
        plot(env, df, cfg, order, palette)
    save_legend_bar(["DDPG", "TD3", "MEOW", "SAC", "FQL"], {**palette})