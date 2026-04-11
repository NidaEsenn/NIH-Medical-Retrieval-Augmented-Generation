from __future__ import annotations

from typing import Dict, List

import streamlit as st

from medrag_ui.retrieval_pipeline import run_retrieval


st.set_page_config(
    page_title="MedRAG",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f5f1e8;
            --panel: rgba(255, 252, 246, 0.88);
            --panel-strong: #fffaf0;
            --text: #182126;
            --muted: #5c6b73;
            --line: rgba(24, 33, 38, 0.12);
            --accent: #0f766e;
            --accent-2: #c2410c;
            --shadow: 0 20px 60px rgba(30, 41, 59, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(194, 65, 12, 0.14), transparent 26%),
                linear-gradient(180deg, #faf6ee 0%, var(--bg) 100%);
            color: var(--text);
            font-family: "Avenir Next", "Segoe UI", sans-serif;
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        h1, h2, h3 {
            font-family: "Iowan Old Style", "Palatino Linotype", serif;
            color: var(--text);
            letter-spacing: -0.02em;
        }

        .hero {
            background: linear-gradient(135deg, rgba(255,255,255,0.82), rgba(255,247,237,0.92));
            border: 1px solid rgba(255,255,255,0.65);
            border-radius: 28px;
            padding: 2rem 2rem 1.5rem 2rem;
            box-shadow: var(--shadow);
            overflow: hidden;
            position: relative;
        }

        .hero:before {
            content: "";
            position: absolute;
            inset: auto -40px -50px auto;
            width: 220px;
            height: 220px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(15,118,110,0.18), rgba(15,118,110,0));
        }

        .hero-kicker {
            color: var(--accent-2);
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            margin-bottom: 0.6rem;
        }

        .hero-text {
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.7;
            max-width: 55rem;
        }

        .stat-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1.3rem;
        }

        .stat-card {
            background: rgba(255,255,255,0.72);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            backdrop-filter: blur(8px);
        }

        .stat-label {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .stat-value {
            color: var(--text);
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 0.2rem;
        }

        .section-card {
            background: var(--panel);
            border: 1px solid rgba(255,255,255,0.6);
            border-radius: 24px;
            padding: 1.25rem;
            box-shadow: var(--shadow);
            backdrop-filter: blur(12px);
        }

        .answer-card {
            background: linear-gradient(180deg, rgba(15,118,110,0.08), rgba(255,255,255,0.86));
            border: 1px solid rgba(15, 118, 110, 0.15);
            border-radius: 24px;
            padding: 1.4rem;
            box-shadow: var(--shadow);
        }

        .caption {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.6;
        }

        .citation-card {
            background: var(--panel-strong);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem;
            margin-bottom: 0.85rem;
        }

        .retrieval-tag {
            display: inline-block;
            background: rgba(15, 118, 110, 0.10);
            color: var(--accent);
            border: 1px solid rgba(15, 118, 110, 0.15);
            border-radius: 999px;
            padding: 0.25rem 0.65rem;
            font-size: 0.8rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }

        @media (max-width: 900px) {
            .stat-row {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Verifiable Medical QA</div>
            <h1>MedRAG</h1>
            <p class="hero-text">
                A retrieval-augmented medical question answering interface built for traceability.
                Users can compare sparse, dense, and hybrid retrieval while inspecting the evidence
                used to ground each answer.
            </p>
            <div class="stat-row">
                <div class="stat-card">
                    <div class="stat-label">Dataset</div>
                    <div class="stat-value">MedQuAD</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">LLM Runtime</div>
                    <div class="stat-value">Local Only</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Output Style</div>
                    <div class="stat-value">Answer + Citations</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> Dict[str, int | str]:
    st.sidebar.markdown("## Retrieval Controls")
    strategy = st.sidebar.radio(
        "Retrieval strategy",
        options=["Hybrid", "Sparse (BM25)", "Dense"],
        help="Switch between your retrieval baselines and comparison modes.",
    )
    top_k = st.sidebar.slider("Top-k passages", min_value=2, max_value=10, value=5)
    rerank_depth = st.sidebar.slider("Dense re-rank depth", min_value=5, max_value=25, value=10)
    temperature = st.sidebar.slider("Generation temperature", min_value=0.0, max_value=1.0, value=0.2)

    st.sidebar.markdown("## Evaluation Lens")
    st.sidebar.checkbox("Show retrieval diagnostics", value=True, key="show_diagnostics")
    st.sidebar.checkbox("Show evidence panel", value=True, key="show_evidence")
    st.sidebar.checkbox("Show answer confidence", value=True, key="show_confidence")

    st.sidebar.markdown("## Notes")
    st.sidebar.caption(
        "This frontend now runs BM25, dense retrieval, and hybrid retrieval directly. "
        "The final step later is swapping the extractive answer preview with your local LLM."
    )

    return {
        "strategy": strategy,
        "top_k": top_k,
        "rerank_depth": rerank_depth,
        "temperature": temperature,
    }


def render_answer(result: Dict[str, object]) -> None:
    st.markdown('<div class="retrieval-tag">Grounded generation</div>', unsafe_allow_html=True)
    st.markdown(f"### Answer")
    st.markdown('<div class="answer-card">', unsafe_allow_html=True)
    st.write(result["answer"])
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("show_confidence", True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Retriever", result["retrieval_strategy"])
        col2.metric("Grounding score", result["grounding_score"])
        col3.metric("Citations", len(result["citations"]))


def render_citations(citations: List[Dict[str, str]]) -> None:
    st.markdown("### Supporting Evidence")
    for idx, citation in enumerate(citations, start=1):
        st.markdown('<div class="citation-card">', unsafe_allow_html=True)
        st.markdown(f"**[{idx}] {citation['title']}**")
        st.markdown(
            f"<div class='caption'><strong>Source:</strong> {citation['source']} | "
            f"<strong>Focus:</strong> {citation['focus_area']}</div>",
            unsafe_allow_html=True,
        )
        st.write(citation["snippet"])
        st.markdown(f"[Open source]({citation['url']})")
        st.markdown("</div>", unsafe_allow_html=True)


def render_diagnostics(result: Dict[str, object]) -> None:
    st.markdown("### Retrieval Diagnostics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Metrics")
        for label, value in result["metrics"].items():
            st.metric(label, value)
    with col2:
        st.markdown("#### Interpretation")
        st.markdown(
            f"""
            <div class="section-card">
                <p class="caption">
                    <strong>{result['retrieval_strategy']}</strong> was selected for this run.
                    Retrieval is running against <code>{result['chunks_path']}</code>.
                    The diagnostics panel is ready to expand into Recall@k, Precision@k,
                    and evidence ranking evaluation once your benchmark set is connected.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    load_styles()
    render_hero()
    controls = render_sidebar()

    st.markdown("")
    input_col, info_col = st.columns([1.55, 1])

    with input_col:
        st.markdown("### Ask a Medical Question")
        query = st.text_area(
            "Question",
            value="What are the common symptoms of iron deficiency anemia?",
            height=110,
            label_visibility="collapsed",
            placeholder="Enter a patient-friendly medical question...",
        )
        run = st.button("Run MedRAG", type="primary", use_container_width=True)

    with info_col:
        st.markdown(
            """
            <div class="section-card">
                <h3>Interface Goals</h3>
                <p class="caption">
                    This first frontend version emphasizes grounded answers, visible evidence,
                    and easy comparison across retrieval modes. It is designed for demos now and
                    for backend integration next.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if run:
        with st.spinner("Retrieving evidence and drafting a grounded answer..."):
            result = run_retrieval(
                query=query,
                strategy=controls["strategy"],
                top_k=controls["top_k"],
                candidate_k=controls["rerank_depth"],
            )

        left, right = st.columns([1.25, 1])
        with left:
            render_answer(result)
            if st.session_state.get("show_diagnostics", True):
                st.markdown("")
                render_diagnostics(result)
        with right:
            if st.session_state.get("show_evidence", True):
                render_citations(result["citations"])
    else:
        st.markdown(
            """
            <div class="section-card">
                <h3>Ready for Backend Integration</h3>
                <p class="caption">
                    The current app already supports strategy selection, parameter tuning,
                    evidence display, and answer rendering. Once your retriever and local model
                    are ready, wire them into the response function and the UI can stay intact.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
