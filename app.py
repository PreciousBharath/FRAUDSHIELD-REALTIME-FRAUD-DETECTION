"""FraudShield — real-time credit card fraud detection (Streamlit)."""
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import auth, ui
from core.engine import CATEGORIES, FEATURES, decision, explain, generate, score, stream_batch, train

st.set_page_config(page_title="FraudShield · Real-Time Fraud Detection", page_icon="🛡️", layout="wide")
ui.inject_css()

APP, ICON = "FraudShield", "🛡️"
if not auth.auth_page(APP, "Real-time credit card fraud detection powered by machine learning.",
                      ["⚡ Live transaction stream with instant risk scoring", "🧠 Gradient Boosting · Random Forest · Isolation Forest",
                       "🔍 Explainable decisions for every transaction", "📁 Batch CSV scoring & downloadable reports"], ICON,
                      demo={"name": "Risk Analyst", "email": "demo@fraudshield.ai", "password": "Demo@1234"}):
    st.stop()


@st.cache_data(show_spinner="Generating 20,000 transactions…")
def load_data():
    return generate(20000)


@st.cache_resource(show_spinner="Training fraud models…")
def load_models():
    return train(load_data())


df, bundle = load_data(), load_models()
page = ui.sidebar(APP, ICON, ["📡 Live Monitor", "🔎 Check Transaction", "📁 Batch Scoring", "📊 Model Lab", "📈 Fraud Analytics", "👤 Account"])
model_names = list(bundle["results"])

# ------------------------------------------------------------------ LIVE MONITOR
if page == "📡 Live Monitor":
    ui.hero("Live Transaction Monitor", "Every incoming card transaction is scored by the model in real time", "📡")
    ss = st.session_state
    ss.setdefault("feed", pd.DataFrame())
    ss.setdefault("running", False)
    ss.setdefault("rng", np.random.default_rng())
    c1, c2, c3, c4 = st.columns([1, 1, 2, 2])
    def _toggle():
        ss.running = not ss.running
        auth.log("stream", "started" if ss.running else "paused")
    c1.button("⏸ Pause" if ss.running else "▶ Start", type="primary", width="stretch", on_click=_toggle)
    if c2.button("🧹 Reset", width="stretch"):
        ss.feed = pd.DataFrame()
    thr = c3.slider("Block threshold", 0.1, 0.95, 0.6, 0.05, help="Probability above which a transaction is blocked")
    model_name = c4.selectbox("Scoring model", model_names, index=1)

    @st.fragment(run_every=1.5 if ss.running else None)
    def live():
        if ss.running:
            new = stream_batch(ss.rng, int(ss.rng.integers(2, 6)))
            p, an = score(bundle, new, model_name)
            new["risk"] = np.round(p, 3)
            new["anomaly"] = np.round(an, 3)
            new["time"] = pd.Timestamp.now().strftime("%H:%M:%S")
            ss.feed = pd.concat([new, ss.feed]).head(400)
        feed = ss.feed
        status = '<span class="pulse"></span><b>STREAMING</b>' if ss.running else "⏸ <b>Paused</b> — press Start"
        st.markdown(f"<div class='card' style='padding:10px 16px'>{status} · {len(feed)} transactions in window</div>", unsafe_allow_html=True)
        k = st.columns(4)
        flagged = feed[feed["risk"] >= thr] if len(feed) else feed
        review = feed[(feed["risk"] >= thr * .5) & (feed["risk"] < thr)] if len(feed) else feed
        ui.kpi(k[0], "Processed", f"{len(feed):,}", "💳")
        ui.kpi(k[1], "Blocked", f"{len(flagged):,}", "⛔")
        ui.kpi(k[2], "Under review", f"{len(review):,}", "🕵️")
        ui.kpi(k[3], "Amount protected", f"₹{flagged['amount'].sum():,.0f}" if len(feed) else "₹0", "💰")
        if not len(feed):
            st.info("Press **Start** to begin streaming live transactions.")
            return
        left, right = st.columns([1.6, 1])
        with left:
            f = feed.head(80).iloc[::-1]
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=f["risk"], mode="lines+markers", line=dict(width=2.5, shape="spline"), marker=dict(size=7, color=f["risk"], colorscale="RdYlGn_r", cmin=0, cmax=1), name="risk"))
            fig.add_hline(y=thr, line_dash="dash", line_color="#ef4444", annotation_text="block")
            fig.update_layout(title="Risk score stream", height=300, yaxis_range=[0, 1], showlegend=False)
            st.plotly_chart(fig, width="stretch")
        with right:
            cr = feed.groupby("category").agg(risk=("risk", "mean"), n=("risk", "size")).reset_index().sort_values("risk")
            m = px.bar(cr, x="risk", y="category", orientation="h", color="risk", color_continuous_scale="RdYlGn_r", range_color=[0, 1], text="n", height=300)
            m.update_layout(title="Average live risk by merchant category", coloraxis_showscale=False)
            st.plotly_chart(m, width="stretch")
        t = feed.head(25).copy()
        t["decision"] = [decision(x, thr)[0] for x in t["risk"]]
        st.dataframe(t[["time", "txn_id", "merchant", "city", "amount", "risk", "anomaly", "decision"]], width="stretch", hide_index=True,
                     column_config={"risk": st.column_config.ProgressColumn("Fraud risk", min_value=0, max_value=1, format="%.2f"), "amount": st.column_config.NumberColumn("Amount (₹)", format="₹%.0f")})
        if len(flagged):
            st.markdown("#### 🚨 Latest alerts")
            for _, r in flagged.head(4).iterrows():
                st.markdown(f"<div class='card' style='border-left:4px solid #ef4444;padding:10px 14px'><b>{r.txn_id}</b> · {r.merchant} · {r.city} · <b>₹{r.amount:,.0f}</b> <span class='badge red'>risk {r.risk:.2f}</span> {'<span class=badge>new device</span>' if r.new_device else ''}{'<span class=badge>foreign</span>' if r.foreign else ''}</div>", unsafe_allow_html=True)

    live()

# ------------------------------------------------------------------ CHECK TRANSACTION
elif page == "🔎 Check Transaction":
    ui.hero("Check a Transaction", "Score a single transaction and see exactly why the model decided", "🔎")
    with st.form("tx"):
        c = st.columns(4)
        amount = c[0].number_input("Amount (₹)", 1.0, 1_000_000.0, 4500.0, step=100.0)
        avg = c[1].number_input("Customer's average spend (₹)", 100.0, 200_000.0, 1500.0, step=100.0)
        hour = c[2].slider("Hour of day", 0, 23, 2)
        cat = c[3].selectbox("Merchant category", CATEGORIES, index=1)
        c = st.columns(4)
        dist = c[0].number_input("Distance from home (km)", 0.0, 5000.0, 120.0)
        vel = c[1].number_input("Transactions in last hour", 0, 30, 4)
        age = c[2].number_input("Account age (days)", 1, 7000, 90)
        model_name = c[3].selectbox("Model", model_names, index=1)
        c = st.columns(3)
        foreign = c[0].toggle("Foreign transaction", True)
        present = c[1].toggle("Card present", False)
        newdev = c[2].toggle("New device", True)
        thr = st.slider("Block threshold", 0.1, 0.95, 0.6, 0.05)
        go_ = st.form_submit_button("⚡ Analyse transaction", type="primary", width="stretch")
    if go_:
        from core.engine import CAT_RISK
        row = pd.DataFrame([{"amount": amount, "hour": hour, "distance_km": dist, "txn_last_hour": vel, "account_age_days": age, "foreign": int(foreign),
                             "card_present": int(present), "new_device": int(newdev), "category_risk": CAT_RISK[cat], "amount_vs_avg": amount / avg}])
        with st.spinner("Scoring…"):
            time.sleep(0.4)
            p, contrib = explain(bundle, row, model_name)
            _, an = score(bundle, row, model_name)
        label, color = decision(p, thr)
        auth.log("check", f"₹{amount:,.0f} {cat} → {label} ({p:.2f})")
        l, r = st.columns([1, 1.3])
        with l:
            g = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number={"suffix": "%"}, title={"text": "Fraud probability"},
                                       gauge={"axis": {"range": [0, 100]}, "bar": {"color": {"red": "#ef4444", "amber": "#f59e0b", "green": "#10b981"}[color]},
                                              "steps": [{"range": [0, thr * 50], "color": "rgba(16,185,129,.15)"}, {"range": [thr * 50, thr * 100], "color": "rgba(245,158,11,.15)"}, {"range": [thr * 100, 100], "color": "rgba(239,68,68,.15)"}],
                                              "threshold": {"line": {"color": "white", "width": 3}, "value": thr * 100}}))
            g.update_layout(height=300)
            st.plotly_chart(g, width="stretch")
            st.markdown(f"<div class='card' style='text-align:center'><span class='badge {color}' style='font-size:1.1rem;padding:8px 18px'>{label}</span><p style='margin-top:10px'>Anomaly score: <b>{an[0]:.3f}</b> (Isolation Forest)</p></div>", unsafe_allow_html=True)
        with r:
            top = contrib.reindex(contrib.abs().sort_values().index).tail(8)
            fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=["#ef4444" if v > 0 else "#10b981" for v in top.values]))
            fig.update_layout(title="Why? Feature impact on risk (red = increases risk)", height=360)
            st.plotly_chart(fig, width="stretch")
            reasons = [f.replace("_", " ") for f, v in contrib.sort_values(ascending=False).items() if v > 0.01][:3]
            st.markdown(f"**Top risk drivers:** {', '.join(reasons) if reasons else 'none — transaction looks normal'}")

# ------------------------------------------------------------------ BATCH
elif page == "📁 Batch Scoring":
    ui.hero("Batch Scoring", "Upload a CSV of transactions and score them all at once", "📁")
    sample = generate(200, seed=99).drop(columns=["is_fraud"])
    st.download_button("⬇️ Download sample CSV template", sample.to_csv(index=False), "fraudshield_sample.csv", "text/csv")
    up = st.file_uploader("Upload transactions CSV", type="csv")
    data = pd.read_csv(up) if up else (sample if st.toggle("Use the sample file", True) else None)
    if data is not None:
        missing = [f for f in FEATURES if f not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            thr = st.slider("Block threshold", 0.1, 0.95, 0.6, 0.05)
            p, an = score(bundle, data)
            out = data.copy()
            out["fraud_probability"], out["anomaly_score"] = np.round(p, 4), np.round(an, 4)
            out["decision"] = [decision(x, thr)[0] for x in p]
            k = st.columns(4)
            ui.kpi(k[0], "Rows scored", f"{len(out):,}", "🧾")
            ui.kpi(k[1], "Blocked", f"{(out.decision == 'BLOCK').sum():,}", "⛔")
            ui.kpi(k[2], "Review", f"{(out.decision == 'REVIEW').sum():,}", "🕵️")
            ui.kpi(k[3], "Approved", f"{(out.decision == 'APPROVE').sum():,}", "✅")
            st.plotly_chart(px.histogram(out, x="fraud_probability", color="decision", nbins=40, title="Risk distribution", color_discrete_map={"BLOCK": "#ef4444", "REVIEW": "#f59e0b", "APPROVE": "#10b981"}), width="stretch")
            st.dataframe(out.sort_values("fraud_probability", ascending=False), width="stretch", hide_index=True)
            st.download_button("⬇️ Download scored results", out.to_csv(index=False), "fraudshield_scored.csv", "text/csv", type="primary")
            auth.log("batch", f"{len(out)} rows scored")

# ------------------------------------------------------------------ MODEL LAB
elif page == "📊 Model Lab":
    ui.hero("Model Lab", "Compare models, inspect curves and tune the decision threshold by business cost", "📊")
    res = bundle["results"]
    table = pd.DataFrame({n: {k: r[k] for k in ["roc_auc", "pr_auc", "precision", "recall", "f1"]} for n, r in res.items()}).T
    st.dataframe(table.style.format("{:.3f}").background_gradient(cmap="Purples"), width="stretch")
    c1, c2 = st.columns(2)
    roc = go.Figure()
    pr = go.Figure()
    for n, r in res.items():
        roc.add_trace(go.Scatter(x=r["roc"][0], y=r["roc"][1], name=f"{n} (AUC {r['roc_auc']:.3f})", line=dict(width=3)))
        pr.add_trace(go.Scatter(x=r["pr"][1], y=r["pr"][0], name=f"{n} (AP {r['pr_auc']:.3f})", line=dict(width=3)))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), showlegend=False))
    roc.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate", height=380, legend=dict(orientation="h", y=-0.25))
    pr.update_layout(title="Precision–Recall curve", xaxis_title="Recall", yaxis_title="Precision", height=380, legend=dict(orientation="h", y=-0.25))
    c1.plotly_chart(roc, width="stretch")
    c2.plotly_chart(pr, width="stretch")
    c1, c2 = st.columns(2)
    name = c1.selectbox("Model", model_names)
    imp = bundle["importance"]
    c2.plotly_chart(px.bar(x=imp.values, y=imp.index, orientation="h", title="Feature importance (Gradient Boosting)", height=360), width="stretch")
    st.markdown("### 💸 Threshold tuning by business cost")
    c = st.columns(3)
    t = c[0].slider("Threshold", 0.05, 0.95, 0.5, 0.05)
    fn_cost = c[1].number_input("Cost of a missed fraud (₹)", 100, 1_000_000, 25000)
    fp_cost = c[2].number_input("Cost of blocking a good customer (₹)", 10, 100000, 500)
    y, p = bundle["y_test"], res[name]["proba"]
    ts = np.linspace(0.05, 0.95, 37)
    costs = [((p < x) & (y == 1)).sum() * fn_cost + ((p >= x) & (y == 0)).sum() * fp_cost for x in ts]
    best = ts[int(np.argmin(costs))]
    fig = go.Figure(go.Scatter(x=ts, y=costs, fill="tozeroy", line=dict(width=3)))
    fig.add_vline(x=best, line_dash="dash", line_color="#10b981", annotation_text=f"optimal {best:.2f}")
    fig.add_vline(x=t, line_dash="dot", line_color="#f59e0b", annotation_text="current")
    fig.update_layout(title="Total business cost vs threshold", height=330)
    st.plotly_chart(fig, width="stretch")
    cm = np.array([[((p < t) & (y == 0)).sum(), ((p >= t) & (y == 0)).sum()], [((p < t) & (y == 1)).sum(), ((p >= t) & (y == 1)).sum()]])
    with c1:
        st.plotly_chart(px.imshow(cm, text_auto=True, x=["Pred legit", "Pred fraud"], y=["Actual legit", "Actual fraud"], color_continuous_scale="Purples", title=f"Confusion matrix @ {t:.2f}", height=360), width="stretch")

# ------------------------------------------------------------------ ANALYTICS
elif page == "📈 Fraud Analytics":
    ui.hero("Fraud Analytics", f"Patterns across {len(df):,} historical transactions", "📈")
    k = st.columns(4)
    ui.kpi(k[0], "Transactions", f"{len(df):,}", "🧾")
    ui.kpi(k[1], "Fraud cases", f"{df.is_fraud.sum():,}", "🚨")
    ui.kpi(k[2], "Fraud rate", f"{df.is_fraud.mean() * 100:.2f}%", "📉")
    ui.kpi(k[3], "Fraud value", f"₹{df.loc[df.is_fraud == 1, 'amount'].sum() / 1e5:,.1f}L", "💸")
    c1, c2 = st.columns(2)
    h = df.groupby("hour")["is_fraud"].mean().reset_index()
    c1.plotly_chart(px.area(h, x="hour", y="is_fraud", title="Fraud rate by hour of day", markers=True), width="stretch")
    cg = df.groupby("category")["is_fraud"].mean().sort_values().reset_index()
    c2.plotly_chart(px.bar(cg, x="is_fraud", y="category", orientation="h", title="Fraud rate by merchant category", color="is_fraud", color_continuous_scale="Reds"), width="stretch")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.box(df.assign(label=df.is_fraud.map({0: "Legit", 1: "Fraud"})), x="label", y="amount_vs_avg", color="label", log_y=True, title="Amount vs customer average"), width="stretch")
    ct = df.groupby("city")["is_fraud"].agg(["mean", "count"]).reset_index().sort_values("mean")
    c2.plotly_chart(px.bar(ct, x="mean", y="city", orientation="h", color="count", title="Fraud rate by city (colour = volume)", labels={"mean": "fraud rate"}), width="stretch")

else:
    ui.account_page()
