"""
baseball-mlops Streamlit ダッシュボード

Marcel 予測 vs ML 予測 vs 実績 の 3 列比較
W&B モデルバージョン・最終更新日時を表示
"""

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="baseball-mlops",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@media (max-width: 768px) {
    [data-testid="stMetric"] { padding: 0.3rem 0.4rem; }
    [data-testid="stMetricLabel"] { font-size: 0.75rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    [data-testid="stHorizontalBlock"] { gap: 0.3rem !important; }
    .stDataFrame td, .stDataFrame th { font-size: 0.8rem !important; }
}
</style>
""", unsafe_allow_html=True)

API_URL = os.environ.get("API_URL", "http://localhost:8002")
# predictions/ を優先（git管理・Streamlit Cloud用）、なければ data/projections/
_BASE = Path(__file__).parent.parent
PRED_DIR = _BASE / "predictions"
PROJ_DIR = _BASE / "data" / "projections"


# ---------------------------------------------------------------------------
# データ取得
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_predictions(kind: str) -> pd.DataFrame:
    fname = "batter_predictions.csv" if kind == "batter" else "pitcher_predictions.csv"
    for d in [PRED_DIR, PROJ_DIR]:
        path = d / fname
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def get_model_info() -> dict:
    try:
        r = requests.get(f"{API_URL}/model/info", timeout=3)
        return r.json()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# ページ
# ---------------------------------------------------------------------------

def page_batters():
    st.header("⚾ 打者 wOBA 予測（翌年）")
    df = load_predictions("batter")
    if df.empty:
        st.warning("予測データがありません。train.py を実行してください。")
        return

    # ソート
    sort_col = st.radio("並び替え", ["ML予測 (pred_woba)", "Marcel予測"], horizontal=True)
    col = "pred_woba" if "ML" in sort_col else "marcel_woba"
    df_show = df.sort_values(col, ascending=False).head(50).reset_index(drop=True)

    # 3 列比較テーブル
    display = df_show[["player", "Team", "Age", "wOBA_last", "marcel_woba", "pred_woba"]].copy()
    display.columns = ["選手名", "チーム", "年齢", "昨季wOBA", "Marcel予測", "ML予測"]
    display["差 (ML-Marcel)"] = (display["ML予測"] - display["Marcel予測"]).round(3)
    st.dataframe(display, use_container_width=True, height=600)

    # 散布図: Marcel vs ML
    st.subheader("Marcel vs ML — 乖離が大きい選手をハイライト")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_show["marcel_woba"], y=df_show["pred_woba"],
        mode="markers+text",
        text=df_show["player"],
        textposition="top center",
        marker=dict(
            size=8,
            color=(df_show["pred_woba"] - df_show["marcel_woba"]).abs(),
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="|ML - Marcel|"),
        ),
    ))
    # y=x の対角線
    lo = min(df_show["marcel_woba"].min(), df_show["pred_woba"].min()) - 0.01
    hi = max(df_show["marcel_woba"].max(), df_show["pred_woba"].max()) + 0.01
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                  line=dict(dash="dash", color="gray"))
    fig.update_layout(
        xaxis_title="Marcel 予測 wOBA",
        yaxis_title="ML 予測 wOBA",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})


def page_pitchers():
    st.header("⚾ 投手 xFIP 予測（翌年）")
    df = load_predictions("pitcher")
    if df.empty:
        st.warning("予測データがありません。train.py を実行してください。")
        return

    sort_col = st.radio("並び替え", ["ML予測 (pred_xfip)", "Marcel予測"], horizontal=True)
    col = "pred_xfip" if "ML" in sort_col else "marcel_xfip"
    df_show = df.sort_values(col, ascending=True).head(50).reset_index(drop=True)

    display = df_show[["player", "Team", "Age", "xFIP_last", "marcel_xfip", "pred_xfip"]].copy()
    display.columns = ["選手名", "チーム", "年齢", "昨季xFIP", "Marcel予測", "ML予測"]
    display["差 (ML-Marcel)"] = (display["ML予測"] - display["Marcel予測"]).round(2)
    st.dataframe(display, use_container_width=True, height=600)

    st.subheader("Marcel vs ML")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_show["marcel_xfip"], y=df_show["pred_xfip"],
        mode="markers+text",
        text=df_show["player"],
        textposition="top center",
        marker=dict(
            size=8,
            color=(df_show["pred_xfip"] - df_show["marcel_xfip"]).abs(),
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="|ML - Marcel|"),
        ),
    ))
    lo = min(df_show["marcel_xfip"].min(), df_show["pred_xfip"].min()) - 0.1
    hi = max(df_show["marcel_xfip"].max(), df_show["pred_xfip"].max()) + 0.1
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                  line=dict(dash="dash", color="gray"))
    fig.update_layout(
        xaxis_title="Marcel 予測 xFIP",
        yaxis_title="ML 予測 xFIP",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})


def page_about():
    st.header("このプロジェクトについて")
    st.markdown("""
    ### baseball-mlops

    **MLB Statcast × MLOps** — 選手成績予測パイプライン

    | 項目 | 内容 |
    |---|---|
    | データ | MLB Statcast (pybaseball) — EV / Barrel% / xwOBA / sprint speed 等 |
    | モデル | LightGBM（5-fold CV）|
    | ベースライン | Marcel 法（5/4/3 加重平均 + 平均回帰 + 年齢調整） |
    | 予測ターゲット | 打者: 翌年 wOBA / 投手: 翌年 xFIP |
    | 再学習 | GitHub Actions cron（毎週月曜）|
    | モデル管理 | W&B Model Registry（production タグ）|
    | API | FastAPI — W&B から 6 時間ごとに自動ロード |

    ### NPB Hawk-Eye への移植

    Statcast = NPB Hawk-Eye と同じトラッキングデータ形式。
    `runs_per_game` パラメータを変えるだけで NPB にも対応できる設計。
    """)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

info = get_model_info()

# ヘッダー: モデル情報バッジ
col1, col2, col3 = st.columns([3, 2, 2])
with col1:
    st.title("⚾ baseball-mlops")
with col2:
    if info:
        st.metric("wOBA model", f"v{info.get('woba_model_version', '?')}")
with col3:
    if info:
        updated = info.get("last_updated", "")[:10]
        st.metric("最終更新", updated)

st.divider()

# メインエリアのナビ（サイドバーが閉じているスマホでも操作できる）
page = st.radio(
    "ページを選択",
    ["打者 wOBA 予測", "投手 xFIP 予測", "About"],
    horizontal=True,
    label_visibility="collapsed",
)

if page == "打者 wOBA 予測":
    page_batters()
elif page == "投手 xFIP 予測":
    page_pitchers()
else:
    page_about()
