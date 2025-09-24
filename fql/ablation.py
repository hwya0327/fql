import os
import re
import math
import numpy as np
import pandas as pd
from tbparse import SummaryReader
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import torch
import torch.nn as nn

env_configs = {
    "Hopper-v4":      {"steps": 1_000_000, "yticks": np.arange(0,  4500,  500), "xticks": np.arange(0, 1_000_001, 2_00_000)},
    "HalfCheetah-v4": {"steps": 5_000_000, "yticks": np.arange(0, 21000, 3000), "xticks": np.arange(0, 5_000_001, 1_000_000)},
    "Walker2d-v4":    {"steps": 5_000_000, "yticks": np.arange(0,  7000, 1000), "xticks": np.arange(0, 5_000_001, 1_000_000)},
    "Ant-v4":         {"steps": 5_000_000, "yticks": np.arange(-1500,  9000, 1500), "xticks": np.arange(0, 5_000_001, 1_000_000)},
    "Humanoid-v4":    {"steps": 5_000_000, "yticks": np.arange(0,  8000, 1000), "xticks": np.arange(0, 5_000_001, 1_000_000)},
}

plt.rcParams["font.size"] = 40
plt.rcParams["axes.labelsize"] = 44
plt.rcParams["axes.titlesize"] = 44
plt.rcParams["xtick.labelsize"] = 36
plt.rcParams["ytick.labelsize"] = 36
plt.rcParams["legend.fontsize"] = 28
plt.rcParams["figure.titlesize"] = 48

BETA_PALETTE = {
    "β=0.0":  "#2ca02c", "β=1.0":   "#9467bd",
    "β=2.0":   "#1f77b4", "β=5.0":   "#ff7f0e", 
}
METHOD_PALETTE = {
    "Critic":  "#2ca02c", "Augmentation": "#8c564b",
}

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1_net(x), self.q2_net(x)

    def q1(self, state, action):
        return self.q1_net(torch.cat([state, action], dim=1))

def _grid(n, max_cols=3):
    cols = min(max_cols, max(1, n))
    rows = int(math.ceil(n / cols)) if n > 0 else 1
    return rows, cols

def _parse_dirname(dir_name: str):
    toks = [t for t in str(dir_name).split("/") if t]
    if not toks:
        return "UNKNOWN", None, None, dir_name
    algo = toks[0].upper()
    beta = method = seed = None
    if algo == "FQL":
        rest = toks[1:]
        if len(rest) >= 1: beta = rest[0]
        if len(rest) >= 2: method = rest[1]
        if len(rest) >= 3:
            m = re.fullmatch(r"\d+", rest[2]) or re.fullmatch(r"seed[_\-]?(\d+)", rest[2], flags=re.IGNORECASE)
            if m: seed = m.group(1) if hasattr(m, "groups") and m.groups() else rest[2]
        if seed is None:
            for t in reversed(rest):
                m = re.search(r"(?:^|[_\-])(\d+)$", t)
                if m:
                    seed = m.group(1)
                    break
    else:
        if len(toks) >= 2:
            m = re.fullmatch(r"\d+", toks[1]) or re.fullmatch(r"seed[_\-]?(\d+)", toks[1], flags=re.IGNORECASE)
            if m: seed = m.group(1) if hasattr(m, "groups") and m.groups() else toks[1]
        if seed is None:
            for t in reversed(toks[1:]):
                m = re.search(r"(?:^|[_\-])(\d+)$", t)
                if m:
                    seed = m.group(1)
                    break
    if seed is None: seed = dir_name
    return algo, beta, method, str(seed)

def draw_curves(df, cfg, order, palette, out_png, out_pdf, ylabel_text="", show_legend=False, show_labels=False):
    if df.empty: return
    present = [a for a in order if a in df["algo"].cat.categories]
    if not present: return
    agg = (
        df.groupby(["algo", "step"], observed=True)["Test/return"]
          .agg(mean="mean", sd=lambda x: x.std(ddof=1))
          .reset_index()
    )
    plt.figure(figsize=(16, 16))
    ax = plt.gca()
    for algo in present:
        sub = agg.loc[agg["algo"] == algo].sort_values("step")
        x, y = sub["step"].to_numpy(), sub["mean"].to_numpy(float)
        sd = np.nan_to_num(sub["sd"].to_numpy(float), nan=0.0)
        c = palette.get(algo, None)
        ax.plot(x, y, linewidth=2.0, color=c, label=algo if show_labels else None)
        ax.fill_between(x, y - sd, y + sd, alpha=0.15, linewidth=0, color=c)
    
    ax.set_xlim(0, cfg["steps"])
    ax.set_xticks(cfg["xticks"])
    ax.set_yticks(cfg["yticks"])
    ax.set_xlabel("Timesteps", labelpad=15)
    ax.set_ylabel(ylabel_text)
    ax.grid(True, alpha=0.3, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if show_legend:
        ax.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0)
    plt.close()

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

def load_fql_data(env: str, max_steps: int):
    log_dir = os.path.join("runs", env)
    if not os.path.isdir(log_dir):
        print(f"Warning: Directory not found for env '{env}' at '{log_dir}'")
        return pd.DataFrame()

    df = SummaryReader(log_dir, pivot=True, extra_columns={"dir_name"}).scalars
    if df.empty or "Test/return" not in df.columns:
        return pd.DataFrame()

    df = df.loc[df["Test/return"].notna(), ["step", "Test/return", "dir_name"]].copy()
    
    parsed = df["dir_name"].apply(_parse_dirname)
    df["algo"]   = parsed.apply(lambda x: x[0]).astype("string")
    df["beta"]   = parsed.apply(lambda x: x[1]).astype("string")
    df["method"] = parsed.apply(lambda x: x[2]).astype("string")
    df["seed"]   = parsed.apply(lambda x: x[3]).astype("string")

    df = df[
        (df["algo"].str.upper() == "FQL") &
        df["seed"].str.fullmatch(r"[1-5]") &
        df["beta"].notna() &
        df["method"].notna()
    ].copy()

    if df.empty: return df

    df["step"]        = pd.to_numeric(df["step"], downcast="integer")
    df["Test/return"] = pd.to_numeric(df["Test/return"], downcast="float")
    
    max_by_run = df.groupby(["dir_name"], observed=True)["step"].transform("max")
    df["step"] = df["step"] * np.where(max_by_run < 10000, 10000, 1)
    
    return df.loc[df["step"] <= max_steps].reset_index(drop=True)

def generate_beta_ablation_plots(df: pd.DataFrame, env: str, cfg: dict, outdir: str):
    for method in ["Critic", "Augmentation"]:
        method_df = df[df["method"].str.capitalize() == method].copy()
        if method_df.empty:
            continue

        method_df["algo"] = "β=" + method_df["beta"].astype(str)
        method_df["algo"] = method_df["algo"].astype("category")

        labels = sorted(method_df["algo"].cat.categories, key=lambda s: float(s.split("=")[-1]))
        palette = {lab: BETA_PALETTE.get(lab, "#333333") for lab in labels}

        print(f"[{env}] Generating beta ablation plot for method: {method}")
        draw_curves(
            method_df, cfg, labels, palette,
            os.path.join(outdir, f"{env}_beta_{method}.png"),
            os.path.join(outdir, f"{env}_beta_{method}.pdf"),
            ylabel_text="Average Return"
        )

def generate_method_comparison_plot(df: pd.DataFrame, env: str, cfg: dict, beta: str, outdir: str):
    beta_df = df[df["beta"] == beta].copy()
    
    beta_df = beta_df[beta_df["method"].str.capitalize().isin(["Critic", "Augmentation"])].copy()
    if beta_df.empty:
        return
    
    beta_df["algo"] = beta_df["method"].str.capitalize().astype("category")
    labels = [lbl for lbl in ["Critic", "Augmentation"] if lbl in beta_df["algo"].cat.categories]

    if labels:
        print(f"[{env}] Generating method comparison plot for beta: {beta}")
        draw_curves(
            beta_df, cfg, labels, METHOD_PALETTE,
            os.path.join(outdir, f"{env}_method_{beta}.png"),
            os.path.join(outdir, f"{env}_method_{beta}.pdf"),
        )

def _action_range(env):
    return (-0.4, 0.4) if env == "Humanoid-v4" else (-1.0, 1.0)

def load_distribution_data(env: str, seed):
    s_path = os.path.join("buffers", f"buffer_{env}_{seed}_state.npy")
    a_path = os.path.join("buffers", f"buffer_{env}_{seed}_action.npy")
    ah_path = os.path.join("buffers", f"buffer_{env}_{seed}_action_h.npy")
    
    if not (os.path.isfile(a_path) and os.path.isfile(ah_path)):
        print(f"Warning: Distribution data not found for {env} with seed {seed}")
        return None, None
        
    state = np.load(s_path, allow_pickle=True)
    action = np.load(a_path, allow_pickle=True)
    action_h = np.load(ah_path, allow_pickle=True)
    return state, action, action_h

def generate_distribution_plots(state, action, action_h, env, seed, outdir="figures", bins=100, cols=3):
    if action is None or action_h is None:
        return
        
    assert state.ndim == 2
    assert action.ndim == 2
    assert action_h.ndim == 3
    
    xmin, xmax = _action_range(env)
    xlim = (xmin, xmax)
    
    plot_params = {
        'axes.labelsize': 14, 'axes.titlesize': 14,
        'xtick.labelsize': 10, 'ytick.labelsize': 12,
    }

    state_dim = state.shape[1]
    T, action_dim = action.shape
    rows, cols_grid = _grid(action_dim, max_cols=cols)

    filename = os.path.join("models", f"policy_{env}_{seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic = Critic(state_dim, action_dim).to(device)
    critic.load_state_dict(torch.load(filename + "_critic"))
    critic.eval()

    def select_action_h(state_h, action_h):
        with torch.no_grad():
            B, A, D = action_h.shape
            
            idx = critic.q1(
                state_h, action_h.reshape(B * A, D)
            ).view(B, A).argmin(dim=1)
            
            return action_h[torch.arange(B, device=device), idx]
        
    B, A, D = action_h.shape
    state_h = torch.from_numpy(state).float().to(device).unsqueeze(1).expand(-1, A, -1).reshape(B * A, -1)
    action_h = torch.from_numpy(action_h).float().to(device)
    action_h = select_action_h(state_h, action_h).cpu().numpy()

    print(f"[{env}] Generating combined distribution plot for seed {seed}...")
    with plt.rc_context(plot_params):
        fig, axes = plt.subplots(rows, cols_grid, figsize=(5*cols_grid, 3.2*rows), squeeze=False)
        axes = axes.flatten()

        for d in range(action_dim):
            ax = axes[d]
            
            ax.hist(action[:, d], bins=bins, range=xlim, density=True, alpha=0.55, label='action')
            ax.hist(action_h[:, d], bins=bins, range=xlim, density=True, alpha=0.55, histtype="stepfilled", color='C1', label='action_h')

            ax.set_title(f"Dim #{d}", pad=6)
            ax.set_xlim(xlim)
            ax.set_ylim(0, 25)
            ax.set_xticks(np.arange(xmin, xmax + 0.1, 0.2))
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("Value")
            
            if d % cols_grid == 0:
                ax.set_ylabel("Density")
            else:
                ax.set_yticklabels([])
                ax.set_ylabel("")

        for t in range(action_dim, len(axes)):
            axes[t].axis("off")

        fig.subplots_adjust(wspace=0.15, hspace=0.4)
        fig.savefig(os.path.join(outdir, f"{env}_distribution_{seed}.png"), dpi=300, bbox_inches="tight", pad_inches=0)
        fig.savefig(os.path.join(outdir, f"{env}_distribution_{seed}.pdf"), bbox_inches="tight", pad_inches=0)
        plt.close(fig)

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    
    beta_labels = sorted(BETA_PALETTE.keys(), key=lambda s: float(s.split("=")[-1]))
    save_legend_bar(beta_labels, BETA_PALETTE, out="figures/legend_beta.png")
    
    method_labels = list(METHOD_PALETTE.keys())
    save_legend_bar(method_labels, METHOD_PALETTE, out="figures/legend_method.png")

    beta_by_env = {
        "Hopper-v4":      "5.0",
        "HalfCheetah-v4": "2.0",
        "Walker2d-v4":    "2.0",
        "Ant-v4":         "0.0",
        "Humanoid-v4":    "0.0",
    }
    
    for env, cfg in env_configs.items():
        df = load_fql_data(env, cfg["steps"])
        if not df.empty:
            generate_beta_ablation_plots(df, env, cfg, outdir="figures")
            
            beta = beta_by_env.get(env)
            if beta:
                generate_method_comparison_plot(df, env, cfg, beta=beta, outdir="figures")
            else:
                print(f"Warning: No specific beta found for {env} in beta_by_env.")
        else:
            print(f"No FQL data found for {env}. Skipping ablation plots.")
    
    for env, cfg in env_configs.items():
        for seed in range(0, 5):
            state, action, action_h = load_distribution_data(env, seed)
            generate_distribution_plots(state, action, action_h, env, seed, outdir="figures")