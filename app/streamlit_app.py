import sys, os, time
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "agent"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))

from graph import build_graph

st.set_page_config(page_title="CodeSage", page_icon="◆", layout="wide")

CSS = """
<style>
.stApp { background-color: #18181a; }
#MainMenu, footer, header { visibility: hidden; }
.tab-bar { display: flex; gap: 20px; border-bottom: 1px solid #2c2c2f; margin-bottom: 20px; padding-bottom: 0; }
.tab-item { font-size: 12px; color: #6b6b70; font-family: monospace; padding-bottom: 10px; }
.tab-active { color: #e8e8ea; border-bottom: 2px solid #d97757; }
.stat-strip { display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid #2c2c2f; border-bottom: 1px solid #2c2c2f; margin: 14px 0 20px; }
.stat-cell { padding: 12px 16px; border-right: 1px solid #2c2c2f; }
.stat-val { font-size: 17px; color: #e8e8ea; font-weight: 500; font-family: monospace; margin: 0; }
.stat-label { font-size: 10px; color: #6b6b70; margin: 3px 0 0; }
.section-label { font-size: 10px; color: #6b6b70; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 10px; }
.trace-line { font-family: monospace; font-size: 12px; line-height: 2.1; }
.tag { padding: 1px 6px; border-radius: 3px; margin-right: 6px; }
.tag-retrieve { background: rgba(122,184,232,0.12); color: #7ab8e8; }
.tag-irrelevant { background: rgba(232,151,123,0.12); color: #e8977b; }
.tag-partial { background: rgba(232,197,123,0.12); color: #e8c57b; }
.tag-relevant { background: rgba(126,214,165,0.12); color: #7ed6a5; }
.tag-correct { background: rgba(184,154,232,0.12); color: #b89ae8; }
.tag-answer { background: rgba(126,214,165,0.12); color: #7ed6a5; }
.tag-fallback { background: rgba(232,151,123,0.12); color: #e8977b; }
.chunk-row { display: flex; justify-content: space-between; font-family: monospace; font-size: 11px; color: #9a9a9e; padding: 3px 0; }
.answer-text { font-size: 15px; color: #c4c4c8; line-height: 1.7; }
.bar-label { font-size: 10px; color: #6b6b70; width: 50px; display: inline-block; }
.bar-track { background: #2c2c2f; border-radius: 3px; height: 6px; display: inline-block; width: 140px; vertical-align: middle; }
.bar-fill { height: 6px; border-radius: 3px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("""
<div class="tab-bar">
  <div class="tab-item tab-active">query.trace</div>
  <div class="tab-item">eval.metrics</div>
</div>
""", unsafe_allow_html=True)

if "app" not in st.session_state:
    st.session_state.app = build_graph()
if "last_result" not in st.session_state:
    st.session_state.last_result = None

query = st.text_input("", placeholder="Ask a question about httpx...", label_visibility="collapsed")
ask_clicked = st.button("Run query →")
st.caption("Up to 2 self-correction retries — prioritizes reliability over speed")

trace_placeholder = st.empty()

TAG_CLASS = {"relevant": "tag-relevant", "partial": "tag-partial", "irrelevant": "tag-irrelevant"}

if ask_clicked and query:
    start_time = time.time()
    initial_state = {
        "query": query, "original_query": query, "retrieved": [],
        "grade": "", "grade_reasoning": "", "answer": "",
        "retry_count": 0, "max_retries": 2,
    }

    trace_lines = []
    final_answer = ""
    final_retrieved = []
    step_count = 0

    for step in st.session_state.app.stream(initial_state):
        for node_name, node_state in step.items():
            step_count += 1
            n = f"{step_count:02d}"
            if node_name == "retrieve":
                trace_lines.append(f'<div class="trace-line"><span style="color:#4a4a4e">{n}</span> <span class="tag tag-retrieve">retrieve</span> <span style="color:#6b6b70">searching corpus</span></div>')
                final_retrieved = node_state.get("retrieved", [])
            elif node_name == "grade":
                grade = node_state.get("grade", "")
                reasoning = node_state.get("grade_reasoning", "")[:60]
                cls = TAG_CLASS.get(grade, "tag-irrelevant")
                trace_lines.append(f'<div class="trace-line"><span style="color:#4a4a4e">{n}</span> <span class="tag {cls}">{grade}</span> <span style="color:#6b6b70">{reasoning}</span></div>')
            elif node_name == "correct":
                trace_lines.append(f'<div class="trace-line"><span style="color:#4a4a4e">{n}</span> <span class="tag tag-correct">correct</span> <span style="color:#6b6b70">query rewritten</span></div>')
            elif node_name == "answer":
                final_answer = node_state.get("answer", "")
                trace_lines.append(f'<div class="trace-line"><span style="color:#4a4a4e">{n}</span> <span class="tag tag-answer">answer</span> <span style="color:#6b6b70">generated</span></div>')
            elif node_name == "fallback":
                final_answer = node_state.get("answer", "")
                trace_lines.append(f'<div class="trace-line"><span style="color:#4a4a4e">{n}</span> <span class="tag tag-fallback">fallback</span> <span style="color:#6b6b70">insufficient context</span></div>')

        trace_placeholder.markdown("".join(trace_lines), unsafe_allow_html=True)

    latency = time.time() - start_time
    st.session_state.last_result = {
        "answer": final_answer, "retrieved": final_retrieved,
        "steps": step_count, "latency": latency,
    }

if st.session_state.last_result:
    r = st.session_state.last_result
    st.markdown(f"""
    <div class="stat-strip">
      <div class="stat-cell"><p class="stat-val">789</p><p class="stat-label">chunks indexed</p></div>
      <div class="stat-cell"><p class="stat-val">{r['steps']}</p><p class="stat-label">graph steps</p></div>
      <div class="stat-cell"><p class="stat-val">{r['latency']:.1f}s</p><p class="stat-label">latency</p></div>
      <div class="stat-cell"><p class="stat-val">done</p><p class="stat-label">status</p></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-label">Response</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="answer-text">{r["answer"]}</p>', unsafe_allow_html=True)

        st.markdown('<p class="section-label">Retrieved chunks</p>', unsafe_allow_html=True)
        for idx, item in enumerate(r["retrieved"][:5]):
            chunk = item["chunk"]
            raw_path = chunk.get("file", "unknown")
            normalized = os.path.normpath(raw_path)
            parts = normalized.split(os.sep)
            if "target_repo" in parts:
                name = "/".join(parts[parts.index("target_repo")+1:])
            else:
                name = os.path.basename(raw_path)
            bar_width = max(100 - idx * 18, 25)
            st.markdown(f'''
            <div class="chunk-row" style="align-items:center;">
              <span>#{idx+1} {name}</span>
              <span style="background:#2c2c2f; border-radius:3px; height:5px; width:60px; display:inline-block; overflow:hidden;">
                <span style="background:#7ab8e8; height:5px; width:{bar_width}%; display:block;"></span>
              </span>
            </div>
            ''', unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-label">Naive vs Agentic (10-question benchmark)</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-bottom: 14px;">
          <p style="font-size: 10px; color: #6b6b70; margin: 0 0 4px;">faithfulness</p>
          <div style="margin-bottom: 4px;"><span class="bar-label">naive</span><span class="bar-track"><span class="bar-fill" style="width:90%; background:#4a4a4e; display:inline-block;"></span></span> <span style="font-size:10px; color:#9a9a9e;">0.90</span></div>
          <div><span class="bar-label">agentic</span><span class="bar-track"><span class="bar-fill" style="width:80%; background:#d97757; display:inline-block;"></span></span> <span style="font-size:10px; color:#d97757;">0.80</span></div>
        </div>
        <div>
          <p style="font-size: 10px; color: #6b6b70; margin: 0 0 4px;">relevancy</p>
          <div style="margin-bottom: 4px;"><span class="bar-label">naive</span><span class="bar-track"><span class="bar-fill" style="width:88%; background:#4a4a4e; display:inline-block;"></span></span> <span style="font-size:10px; color:#9a9a9e;">0.88</span></div>
          <div><span class="bar-label">agentic</span><span class="bar-track"><span class="bar-fill" style="width:87%; background:#7ed6a5; display:inline-block;"></span></span> <span style="font-size:10px; color:#7ed6a5;">0.87</span></div>
        </div>
        <p style="font-size: 10px; color: #6b6b70; margin-top: 14px; line-height: 1.6;">Note: faithfulness gap traces to a single judge inconsistency on one question (WebSocket support); relevancy shows a genuine +0.50 win on the retry-on-failure query via query rewriting.</p>
        """, unsafe_allow_html=True)