import streamlit as st
import numpy as np
import plotly.graph_objects as go
import tempfile
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MINT · Brain Content Analyser",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; }

.stMetric { background: #111; border: 1px solid #222; border-radius: 8px; padding: 12px; }
.stProgress > div > div { background: #00ff88; }
section[data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 MINT Brain Content Analyser")
st.markdown("*Powered by Meta's TRIBE v2 — predicts second-by-second neural engagement from your video*")
st.divider()

# ── Upload ────────────────────────────────────────────────────────────────────
col_up, col_info = st.columns([1, 1])

with col_up:
    uploaded_file = st.file_uploader(
        "Upload your video",
        type=["mp4", "mov"],
        help="Keep it under 3 minutes for best results"
    )
    if uploaded_file:
        st.video(uploaded_file)

with col_info:
    st.markdown("### How it works")
    st.markdown("""
    1. **Extracts** audio, visuals, and speech from your video
    2. **Runs** Meta's TRIBE v2 brain encoding model
    3. **Predicts** neural activation across 20,000+ cortical points
    4. **Translates** the output into plain-English content insights
    """)
    st.markdown("**Estimated time:** 2–5 minutes per video")

    if uploaded_file:
        analyse = st.button("🔬 Run Brain Analysis", type="primary", use_container_width=True)
    else:
        st.info("Upload a video to get started")
        analyse = False

# ── Analysis ──────────────────────────────────────────────────────────────────
if uploaded_file and analyse:

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        video_path = tmp.name

    try:
        with st.spinner("Loading TRIBE v2 model (first run downloads ~4GB, cached after)..."):
            from tribev2 import TribeModel

            @st.cache_resource(show_spinner=False)
            def load_model():
                return TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")

            model = load_model()

        with st.spinner("Extracting audio, video & speech features..."):
            df = model.get_events_dataframe(video_path=video_path)

        with st.spinner("Predicting brain responses across cortical surface..."):
            preds, segments = model.predict(events=df)
            # preds: (n_timesteps, n_vertices ~20k)

        # ── Compute metrics ───────────────────────────────────────────────────

        activation = preds.mean(axis=1)           # mean across all brain vertices per second
        n_timesteps = len(activation)
        timestamps  = list(range(n_timesteps))

        # Normalise to 0–100
        a_min, a_max = activation.min(), activation.max()
        act_norm = ((activation - a_min) / (a_max - a_min + 1e-8)) * 100

        overall_score = float(act_norm.mean())
        peak_score    = float(act_norm.max())
        peak_second   = int(act_norm.argmax())
        sustained_pct = float((act_norm > 60).mean() * 100)   # % of video above 60/100
        variance      = float(act_norm.std())

        # Retention: second-half vs first-half engagement
        mid         = n_timesteps // 2
        first_half  = act_norm[:mid].mean()
        second_half = act_norm[mid:].mean()
        retention   = float((second_half / (first_half + 1e-8)) * 100)

        # Hook: first 3 seconds
        hook_score = float(act_norm[:min(3, n_timesteps)].mean())

        # Rough region scores using vertex position on fsaverage5
        # Not anatomically precise — indicative directional signal only
        n_verts  = preds.shape[1]
        quarter  = n_verts // 4

        def region_norm(raw_scores):
            r_min = min(raw_scores)
            r_max = max(raw_scores)
            return [((x - r_min) / (r_max - r_min + 1e-8)) * 100 for x in raw_scores]

        visual_raw    = float(preds[:, -quarter:].mean())
        auditory_raw  = float(preds[:, quarter:quarter*2].mean())
        attention_raw = float(preds[:, :quarter].mean())
        language_raw  = float(preds[:, quarter*2:quarter*3].mean())

        visual_n, auditory_n, attention_n, language_n = region_norm(
            [visual_raw, auditory_raw, attention_raw, language_raw]
        )

        # ── Results UI ────────────────────────────────────────────────────────

        st.divider()
        st.markdown("## Results")

        # Top metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Brain Engagement Score",  f"{overall_score:.0f} / 100")
        m2.metric("Peak Activation",         f"{peak_score:.0f} / 100",  f"at {peak_second}s")
        m3.metric("Sustained Attention",     f"{sustained_pct:.0f}%",    "of video above 60/100")
        delta_str = f"+{retention-100:.0f}% vs opener" if retention >= 100 else f"{retention-100:.0f}% vs opener"
        m4.metric("Second-Half Retention",   f"{retention:.0f}%",        delta_str)

        # Time series chart
        st.markdown("### Engagement Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps, y=act_norm.tolist(),
            mode="lines",
            name="Neural Engagement",
            line=dict(color="#00ff88", width=2),
            fill="tozeroy", fillcolor="rgba(0,255,136,0.08)"
        ))
        fig.add_hline(y=60, line_dash="dot", line_color="#888",
                      annotation_text="Attention threshold (60)", annotation_position="bottom right")
        fig.add_vline(x=peak_second, line_dash="dash", line_color="#ff6b6b",
                      annotation_text=f"Peak @ {peak_second}s", annotation_position="top left")
        fig.update_layout(
            xaxis_title="Time (seconds)",
            yaxis_title="Neural Engagement (0–100)",
            yaxis_range=[0, 108],
            template="plotly_dark",
            height=340,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Region breakdown
        st.markdown("### Brain Region Breakdown")
        st.caption("Indicative signal — directional, not anatomically precise.")
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("**👁️ Visual Processing**")
            st.progress(int(visual_n),    text=f"{visual_n:.0f} / 100")
            st.markdown("**👂 Auditory Processing**")
            st.progress(int(auditory_n),  text=f"{auditory_n:.0f} / 100")
        with rc2:
            st.markdown("**🎯 Attention & Memory**")
            st.progress(int(attention_n), text=f"{attention_n:.0f} / 100")
            st.markdown("**💬 Language & Speech**")
            st.progress(int(language_n),  text=f"{language_n:.0f} / 100")

        # Plain-English Summary
        st.divider()
        st.markdown("### 🧠 What This Means for Your Content")

        insights = []

        # Overall
        if overall_score >= 70:
            insights.append("✅ **Strong overall engagement** — this content consistently activates neural attention and processing.")
        elif overall_score >= 45:
            insights.append("⚠️ **Moderate engagement** — the brain responds in places, but there's room to be more consistently stimulating.")
        else:
            insights.append("❌ **Low engagement** — the brain isn't strongly activated. This content may feel passive or forgettable.")

        # Hook
        if hook_score >= 65:
            insights.append(f"✅ **Strong hook** — opening 3 seconds score {hook_score:.0f}/100. You're capturing attention immediately.")
        elif hook_score >= 40:
            insights.append(f"⚠️ **Average hook** — opening 3 seconds score {hook_score:.0f}/100. Consider leading with a face, bold visual, or direct question.")
        else:
            insights.append(f"❌ **Weak hook** — opening only scores {hook_score:.0f}/100. In short-form, this likely causes scroll-through. Lead with your most engaging moment.")

        # Retention
        if retention >= 95:
            insights.append("✅ **Excellent retention** — the second half matches the first. Viewers who start are likely to finish.")
        elif retention >= 75:
            insights.append(f"⚠️ **Moderate retention** — engagement drops to {retention:.0f}% of the opening by the second half. Tighten the ending or build toward a stronger payoff.")
        else:
            insights.append(f"❌ **Drop-off detected** — engagement falls significantly in the second half ({retention:.0f}% of opener). Cut length or restructure around the peak moment.")

        # Consistency
        if variance < 12:
            insights.append("💡 **Consistent but flat** — engagement is stable but lacks a spike. Add a high-energy moment mid-video to create a memorable peak.")
        elif variance > 28:
            insights.append("💡 **Highly variable** — big spikes and drops. Works for dynamic content but risks losing viewers during low-activation segments.")

        # Regions
        if visual_n >= 65:
            insights.append("👁️ **Strong visual processing** — motion, contrast, or faces are working. Keep using dynamic visuals.")
        elif visual_n < 35:
            insights.append("👁️ **Weak visual processing** — add more close-up shots, motion, or face-forward framing to drive visual engagement.")

        if language_n >= 65:
            insights.append("💬 **High language activation** — speech and text are landing well. Captions or voiceover are contributing.")

        if auditory_n >= 65:
            insights.append("👂 **Strong auditory engagement** — your audio (music, voice, SFX) is doing real work. Don't underinvest here.")

        for insight in insights:
            st.markdown(insight)

        st.info(f"🔥 **Peak moment at {peak_second}s** — this is your most neurally engaging moment. If repurposing this content, clip around this timestamp.")

        # Raw data expander for curious members
        with st.expander("📊 Raw activation data (for the curious)"):
            st.markdown("Mean neural activation score per second:")
            raw_df_data = {"Second": timestamps, "Activation Score": [f"{v:.2f}" for v in act_norm.tolist()]}
            import pandas as pd
            st.dataframe(pd.DataFrame(raw_df_data), use_container_width=True, height=200)

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.markdown("**Common fixes:**")
        st.markdown("- Make sure `HF_TOKEN` is set in your Space secrets (needed to download LLaMA weights)")
        st.markdown("- Try a shorter video (under 2 minutes)")
        st.markdown("- Check the Space logs for the full traceback")

    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)
