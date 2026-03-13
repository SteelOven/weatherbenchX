"""
Weather Data Explorer
=====================
Run:  streamlit run weather_dashboard.py

Edit load_datasets() below to point to your data.
"""

import numpy as np
import xarray as xr
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ══════════════════════════════════════════════════════════════════════════════
#  ① EDIT THIS — register your datasets
# ══════════════════════════════════════════════════════════════════════════════

def load_datasets() -> dict:
    """
    Return {display_name: xr.Dataset}.
    All entries appear in every dropdown throughout the app.

    Example:
        t = xr.open_dataset("./tmp_target.nc")
        p = xr.open_dataset("./tmp_pred.nc")
        return {
            "ERA5 Target":     t,
            "HRES Prediction": p,
            "Error":           p - t,
        }
    """
    t = xr.open_dataset("./tmp_target.nc")
    p = xr.open_dataset("./tmp_pred.nc")
    return {
        "ERA5 Target":     t,
        "HRES Prediction": p,
        "Error":           p - t,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Demo data  (remove once you plug in real datasets)
# ══════════════════════════════════════════════════════════════════════════════

def _make_demo_data():
    rng  = np.random.default_rng(0)
    lats = np.linspace(-90, 90, 32)
    lons = np.linspace(0, 355, 64)
    init_times = pd.date_range("2020-01-01", periods=4, freq="12h").values.astype("datetime64[ns]")
    lead_times = np.array([6, 12, 24], dtype="timedelta64[h]").astype("timedelta64[ns]")

    def synth(bias=0, noise=1.5):
        base = (np.cos(np.deg2rad(lats))[:, None]
                * np.sin(np.linspace(0, 4*np.pi, len(lons)))[None, :])
        arr  = (bias + 20 * base[None, None]
                + noise * rng.standard_normal((len(init_times), len(lead_times), len(lats), len(lons))))
        return xr.DataArray(
            arr + 273.15,
            dims=["init_time","lead_time","latitude","longitude"],
            coords=dict(init_time=init_times, lead_time=lead_times,
                        latitude=lats, longitude=lons),
        )

    t2m_t = synth(0,   1.0);  z500_t = synth(55000, 200)
    t2m_p = synth(0.5, 1.8);  z500_p = synth(55000, 350)
    target = xr.Dataset({"2m_temperature": t2m_t, "geopotential": z500_t})
    pred   = xr.Dataset({"2m_temperature": t2m_p, "geopotential": z500_p})
    return {
        "ERA5 (target)":         target,
        "HRES (prediction)":     pred,
        "Error (pred−target)":   pred - target,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════════════════════

LAT_DIMS = {"latitude","lat","y"}
LON_DIMS = {"longitude","lon","x"}

def _lat_lon(da):
    lat = next((d for d in da.dims if d in LAT_DIMS), None)
    lon = next((d for d in da.dims if d in LON_DIMS), None)
    return lat, lon

def _extra_dims(da):
    lat, lon = _lat_lon(da)
    return [d for d in da.dims if d not in (lat, lon)]

def _fmt(val):
    if isinstance(val, np.datetime64):  return str(pd.Timestamp(val))[:19]
    if isinstance(val, np.timedelta64): return f"{int(pd.Timedelta(val).total_seconds()//3600)}h"
    return str(val)

def _slice2d(da, sel: dict):
    return da.isel({d: i for d, i in sel.items()}).squeeze()

def _stats(da2d):
    v = da2d.values.ravel()
    v = v[~np.isnan(v)]
    return dict(min=float(v.min()), max=float(v.max()),
                mean=float(v.mean()), std=float(v.std()))

def _heatmap(da2d, title, cscale, zmin=None, zmax=None):
    lat, lon = _lat_lon(da2d)
    # Ensure (lat, lon) dim order → latitude on y-axis, longitude on x-axis
    if lat and lon and da2d.dims != (lat, lon):
        da2d = da2d.transpose(lat, lon)
    lat_vals = da2d[lat].values if lat else None
    lon_vals = da2d[lon].values if lon else None
    # If latitude runs high→low (90→−90), flip so north is always up
    y_autorange = "reversed" if (lat_vals is not None and lat_vals[0] > lat_vals[-1]) else True
    fig = go.Figure(go.Heatmap(
        z=da2d.values,
        x=lon_vals,
        y=lat_vals,
        colorscale=cscale,
        zmin=zmin, zmax=zmax,
        colorbar=dict(thickness=12, len=0.9, tickfont=dict(size=10)),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color="#ddd"), x=0.02),
        margin=dict(l=5, r=5, t=36, b=5),
        height=340,
        paper_bgcolor="#111318",
        plot_bgcolor="#111318",
        xaxis=dict(title="longitude", color="#777", gridcolor="#222", zeroline=False),
        yaxis=dict(title="latitude",  color="#777", gridcolor="#222", zeroline=False,
                   autorange=y_autorange, scaleanchor="x", scaleratio=0.5),
        font=dict(color="#ccc", size=11),
    )
    return fig

def _stat_line(s):
    return (f"min **{s['min']:.4g}**  ·  max **{s['max']:.4g}**  ·  "
            f"mean **{s['mean']:.4g}**  ·  σ **{s['std']:.4g}**")


# ══════════════════════════════════════════════════════════════════════════════
#  Inline dimension selectors  (key collisions were the bug — now uid-scoped)
# ══════════════════════════════════════════════════════════════════════════════

def dim_selectors(da: xr.DataArray, uid: str) -> dict:
    """
    Renders one selectbox per non-spatial dimension.
    uid must be unique per call site.
    Returns {dim_name: int_index}.
    """
    dims = _extra_dims(da)
    if not dims:
        return {}
    cols = st.columns(len(dims))
    sel  = {}
    for col, dim in zip(cols, dims):
        with col:
            vals   = da[dim].values
            labels = [_fmt(v) for v in vals]
            idx = st.selectbox(
                label=f"{dim}  ({len(vals)} steps)",
                options=range(len(labels)),
                format_func=lambda i, lb=labels: lb[i],
                key=f"sel_{uid}_{dim}",
            )
            sel[dim] = idx
    return sel


# ══════════════════════════════════════════════════════════════════════════════
#  Pages
# ══════════════════════════════════════════════════════════════════════════════

def _get_da(ds, var):
    return ds[var] if isinstance(ds, xr.Dataset) else ds

def _vars(ds):
    return list(ds.data_vars) if isinstance(ds, xr.Dataset) else [ds.name or "value"]


def page_explore(datasets):
    st.subheader("🔍 Explore a single dataset")

    c1, c2, c3 = st.columns([3, 3, 2])
    with c1: ds_name = st.selectbox("Dataset", list(datasets.keys()), key="ex_ds")
    ds = datasets[ds_name]
    with c2: var    = st.selectbox("Variable", _vars(ds), key="ex_var")
    with c3: cscale = st.selectbox("Colorscale", ["Viridis","Plasma","RdBu_r","Turbo","Cividis"], key="ex_cs")

    da  = _get_da(ds, var)
    sel = dim_selectors(da, uid="ex")

    da2d = _slice2d(da, sel)
    v    = da2d.values
    st.plotly_chart(
        _heatmap(da2d, f"{ds_name}  ·  {var}", cscale,
                 float(np.nanmin(v)), float(np.nanmax(v))),
        width='stretch', key="ex_plot",
    )
    st.caption(_stat_line(_stats(da2d)))

    with st.expander("Dataset structure (dims, coords, shape)"):
        st.code(str(ds if isinstance(ds, xr.Dataset) else da))


def page_compare(datasets):
    st.subheader("⚖️ Compare two datasets")
    st.caption("Difference map is Right − Left.")

    names = list(datasets.keys())
    c1, c2, c3, c4 = st.columns([3, 3, 3, 2])
    with c1: left_name  = st.selectbox("Left",     names, index=0,                   key="cmp_l")
    with c2: right_name = st.selectbox("Right",    names, index=min(1,len(names)-1), key="cmp_r")
    ds_l, ds_r = datasets[left_name], datasets[right_name]
    shared = [v for v in _vars(ds_l) if v in _vars(ds_r)] or _vars(ds_l)
    with c3: var    = st.selectbox("Variable", shared, key="cmp_var")
    with c4: cscale = st.selectbox("Colorscale", ["Viridis","Plasma","RdBu_r","Turbo"], key="cmp_cs")

    da_l = _get_da(ds_l, var)
    da_r = _get_da(ds_r, var)
    sel  = dim_selectors(da_l, uid="cmp")

    da2d_l = _slice2d(da_l, sel)
    da2d_r = _slice2d(da_r, {d: sel.get(d,0) for d in _extra_dims(da_r)})

    lock = st.checkbox("Lock colorscale across both panels", value=True, key="cmp_lock")
    vmin = min(float(np.nanmin(da2d_l.values)), float(np.nanmin(da2d_r.values))) if lock else None
    vmax = max(float(np.nanmax(da2d_l.values)), float(np.nanmax(da2d_r.values))) if lock else None

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Left — {left_name}**")
        st.plotly_chart(_heatmap(da2d_l, "", cscale, vmin, vmax), width='stretch', key="cmp_plot_l")
        st.caption(_stat_line(_stats(da2d_l)))
    with col2:
        st.markdown(f"**Right — {right_name}**")
        st.plotly_chart(_heatmap(da2d_r, "", cscale, vmin, vmax), width='stretch', key="cmp_plot_r")
        st.caption(_stat_line(_stats(da2d_r)))

    st.markdown("**Difference — Right minus Left:**")
    try:
        diff = da2d_r - da2d_l
        dabs = float(np.nanmax(np.abs(diff.values)))
        st.plotly_chart(_heatmap(diff, "", "RdBu_r", -dabs, dabs), width='stretch', key="cmp_plot_diff")
        st.caption(_stat_line(_stats(diff)))
    except Exception as e:
        st.warning(f"Cannot compute difference: {e}")


def page_tpe(datasets):
    st.subheader("🎯 Target · Prediction · Error")

    names = list(datasets.keys())
    c1, c2, c3, c4 = st.columns([3, 3, 3, 2])
    with c1: tgt_name = st.selectbox("Target",     names, index=0,                   key="tpe_t")
    with c2: prd_name = st.selectbox("Prediction", names, index=min(1,len(names)-1), key="tpe_p")
    ds_t, ds_p = datasets[tgt_name], datasets[prd_name]
    shared = [v for v in _vars(ds_t) if v in _vars(ds_p)] or _vars(ds_t)
    with c3: var    = st.selectbox("Variable", shared, key="tpe_var")
    with c4: cscale = st.selectbox("Colorscale", ["Viridis","Plasma","RdBu_r","Turbo"], key="tpe_cs")

    da_t = _get_da(ds_t, var)
    da_p = _get_da(ds_p, var)
    sel  = dim_selectors(da_t, uid="tpe")

    da2d_t = _slice2d(da_t, sel)
    da2d_p = _slice2d(da_p, {d: sel.get(d,0) for d in _extra_dims(da_p)})

    lock = st.checkbox("Lock colorscale for Target & Prediction", value=True, key="tpe_lock")
    vmin = min(float(np.nanmin(da2d_t.values)), float(np.nanmin(da2d_p.values))) if lock else None
    vmax = max(float(np.nanmax(da2d_t.values)), float(np.nanmax(da2d_p.values))) if lock else None

    try:
        err  = da2d_p - da2d_t
        eabs = float(np.nanmax(np.abs(err.values)))
        has_err = True
    except Exception:
        has_err = False

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**🎯 Target — {tgt_name}**")
        st.plotly_chart(_heatmap(da2d_t, "", cscale, vmin, vmax), width='stretch', key="tpe_plot_t")
        st.caption(_stat_line(_stats(da2d_t)))
    with col2:
        st.markdown(f"**🔮 Prediction — {prd_name}**")
        st.plotly_chart(_heatmap(da2d_p, "", cscale, vmin, vmax), width='stretch', key="tpe_plot_p")
        st.caption(_stat_line(_stats(da2d_p)))
    with col3:
        st.markdown("**⚠️ Error — Pred minus Target**")
        if has_err:
            st.plotly_chart(_heatmap(err, "", "RdBu_r", -eabs, eabs), width='stretch', key="tpe_plot_e")
            st.caption(_stat_line(_stats(err)))
        else:
            st.info("Could not compute error (shape mismatch?)")

    with st.expander("Stats table"):
        rows = [{"panel":"Target",    **{k:f"{v:.4g}" for k,v in _stats(da2d_t).items()}},
                {"panel":"Prediction",**{k:f"{v:.4g}" for k,v in _stats(da2d_p).items()}}]
        if has_err:
            rows.append({"panel":"Error",**{k:f"{v:.4g}" for k,v in _stats(err).items()}})
        st.dataframe(pd.DataFrame(rows), hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Weather Explorer", page_icon="🌦", layout="wide")
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500&family=IBM+Plex+Sans:wght@300;400&display=swap');
      html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
      h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
      .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
      div[data-testid="stCaption"] { color: #888; font-size: 0.78rem; margin-top: -8px; }
      div[data-testid="stRadio"] label { font-size: 0.88rem; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🌦 Weather Data Explorer")

    @st.cache_resource
    def _load():
        return load_datasets()

    with st.spinner("Loading datasets…"):
        datasets = _load()

    if not datasets:
        st.error("No datasets found. Edit `load_datasets()` at the top of this file.")
        return

    # Show what's loaded
    with st.expander(f"{len(datasets)} dataset(s) loaded — click to see shapes", expanded=False):
        for name, ds in datasets.items():
            if isinstance(ds, xr.Dataset):
                info = ", ".join(f"{v}: {dict(ds[v].sizes)}" for v in ds.data_vars)
            else:
                info = str(dict(ds.sizes))
            st.markdown(f"**{name}** — {info}")

    st.divider()

    # ── Radio, not tabs — tabs cause widget key collisions across renders ──
    mode = st.radio(
        "Mode",
        ["🔍 Explore", "⚖️ Compare two datasets", "🎯 Target · Prediction · Error"],
        horizontal=True,
        key="mode",
    )
    st.divider()

    if   mode == "🔍 Explore":                         page_explore(datasets)
    elif mode == "⚖️ Compare two datasets":            page_compare(datasets)
    else:                                              page_tpe(datasets)


if __name__ == "__main__":
    main()