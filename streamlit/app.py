"""
baseball-mlops Streamlit ダッシュボード

Marcel 予測 vs ML 予測 vs 実績 の 3 列比較
W&B モデルバージョン・最終更新日時を表示
"""

import json
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

def _load_bayes_coef() -> dict:
    coef_path = _BASE / "predictions" / "bayes_coef.json"
    if coef_path.exists():
        try:
            return json.loads(coef_path.read_text())
        except Exception:
            pass
    return {}


def page_batters():
    st.header("⚾ 打者 wOBA 予測（翌年）")
    df = load_predictions("batter")
    if df.empty:
        st.warning("予測データがありません。train.py を実行してください。")
        return

    has_bayes = "bayes_woba" in df.columns
    has_ensemble = "ensemble_woba" in df.columns

    # ソート
    sort_opts = ["ML予測 (pred_woba)", "Marcel予測"]
    if has_bayes:
        sort_opts.insert(1, "Bayes予測")
    if has_ensemble:
        sort_opts.insert(1, "アンサンブル予測")
    sort_col = st.radio("並び替え", sort_opts, horizontal=True)
    if "アンサンブル" in sort_col:
        col = "ensemble_woba"
    elif "Bayes" in sort_col:
        col = "bayes_woba"
    elif "ML" in sort_col:
        col = "pred_woba"
    else:
        col = "marcel_woba"
    df_show = df.sort_values(col, ascending=False).head(50).reset_index(drop=True)

    # テーブル列構成
    base_cols = ["player", "Team", "Age", "wOBA_last", "marcel_woba", "pred_woba"]
    base_names = ["選手名", "チーム", "年齢", "昨季wOBA", "Marcel予測", "ML予測"]
    if has_bayes:
        base_cols += ["bayes_woba", "ci_lo80", "ci_hi80"]
        base_names += ["Bayes予測", "CI下限(80%)", "CI上限(80%)"]
    if has_ensemble:
        base_cols += ["ensemble_woba"]
        base_names += ["アンサンブル予測"]
    display = df_show[base_cols].copy()
    display.columns = base_names
    display["差 (ML-Marcel)"] = (display["ML予測"] - display["Marcel予測"]).round(3)
    st.dataframe(display, use_container_width=True, height=600)

    # CI バーチャート（Bayes あり時のみ）
    if has_bayes:
        st.subheader("Bayes 予測 + 80% 信頼区間（上位30名）")
        df_ci = df_show.dropna(subset=["bayes_woba", "ci_lo80", "ci_hi80"]).head(30)
        if not df_ci.empty:
            err_lo = (df_ci["bayes_woba"] - df_ci["ci_lo80"]).clip(lower=0)
            err_hi = (df_ci["ci_hi80"] - df_ci["bayes_woba"]).clip(lower=0)
            fig_ci = go.Figure()
            fig_ci.add_trace(go.Bar(
                x=df_ci["player"],
                y=df_ci["bayes_woba"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=err_hi.tolist(),
                    arrayminus=err_lo.tolist(),
                ),
                name="Bayes wOBA",
                hovertemplate="<b>%{x}</b><br>Bayes: %{y:.3f}<extra></extra>",
            ))
            fig_ci.update_layout(
                xaxis_tickangle=-45,
                yaxis_title="予測 wOBA",
                height=450,
            )
            st.plotly_chart(fig_ci, use_container_width=True)

    # β係数エクスパンダー
    coef_data = _load_bayes_coef()
    if "batter" in coef_data:
        with st.expander("Bayes Ridge β係数（Statcast特徴量の寄与）"):
            coef_items = sorted(
                coef_data["batter"].items(),
                key=lambda x: abs(x[1]["coef"]), reverse=True
            )
            names = [k for k, _ in coef_items]
            values = [v["coef"] for _, v in coef_items]
            colors = ["#1a73e8" if v >= 0 else "#d93025" for v in values]
            fig_coef = go.Figure(go.Bar(
                x=values, y=names,
                orientation="h",
                marker_color=colors,
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            fig_coef.update_layout(
                xaxis_title="β係数（z-score スケール）",
                height=max(250, len(names) * 35),
            )
            st.plotly_chart(fig_coef, use_container_width=True)

    # 散布図: Marcel vs ML（乖離Top10のみラベル表示、他はホバーで確認）
    st.subheader("Marcel vs ML — 乖離が大きい選手をハイライト")
    diff = (df_show["pred_woba"] - df_show["marcel_woba"]).abs()
    top10_idx = diff.nlargest(10).index
    labels = df_show["player"].where(df_show.index.isin(top10_idx), "")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_show["marcel_woba"], y=df_show["pred_woba"],
        mode="markers+text",
        text=labels,
        textposition="top center",
        hovertemplate="<b>%{customdata}</b><br>Marcel: %{x:.3f}<br>ML: %{y:.3f}<extra></extra>",
        customdata=df_show["player"],
        marker=dict(
            size=9,
            color=diff,
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="|ML - Marcel|"),
        ),
    ))
    lo = min(df_show["marcel_woba"].min(), df_show["pred_woba"].min()) - 0.01
    hi = max(df_show["marcel_woba"].max(), df_show["pred_woba"].max()) + 0.01
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                  line=dict(dash="dash", color="gray"))
    fig.update_layout(
        xaxis_title="Marcel 予測 wOBA",
        yaxis_title="ML 予測 wOBA",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


def page_pitchers():
    st.header("⚾ 投手 xFIP 予測（翌年）")
    df = load_predictions("pitcher")
    if df.empty:
        st.warning("予測データがありません。train.py を実行してください。")
        return

    has_bayes = "bayes_xfip" in df.columns
    has_ensemble = "ensemble_xfip" in df.columns

    sort_opts = ["ML予測 (pred_xfip)", "Marcel予測"]
    if has_bayes:
        sort_opts.insert(1, "Bayes予測")
    if has_ensemble:
        sort_opts.insert(1, "アンサンブル予測")
    sort_col = st.radio("並び替え", sort_opts, horizontal=True)
    if "アンサンブル" in sort_col:
        col = "ensemble_xfip"
    elif "Bayes" in sort_col:
        col = "bayes_xfip"
    elif "ML" in sort_col:
        col = "pred_xfip"
    else:
        col = "marcel_xfip"
    df_show = df.sort_values(col, ascending=True).head(50).reset_index(drop=True)

    base_cols = ["player", "Team", "Age", "xFIP_last", "marcel_xfip", "pred_xfip"]
    base_names = ["選手名", "チーム", "年齢", "昨季xFIP", "Marcel予測", "ML予測"]
    if has_bayes:
        base_cols += ["bayes_xfip", "ci_lo80", "ci_hi80"]
        base_names += ["Bayes予測", "CI下限(80%)", "CI上限(80%)"]
    if has_ensemble:
        base_cols += ["ensemble_xfip"]
        base_names += ["アンサンブル予測"]
    display = df_show[base_cols].copy()
    display.columns = base_names
    display["差 (ML-Marcel)"] = (display["ML予測"] - display["Marcel予測"]).round(2)
    st.dataframe(display, use_container_width=True, height=600)

    # CI バーチャート（Bayes あり時のみ）
    if has_bayes:
        st.subheader("Bayes 予測 + 80% 信頼区間（上位30名、xFIP低い順）")
        df_ci = df_show.dropna(subset=["bayes_xfip", "ci_lo80", "ci_hi80"]).head(30)
        if not df_ci.empty:
            err_lo = (df_ci["bayes_xfip"] - df_ci["ci_lo80"]).clip(lower=0)
            err_hi = (df_ci["ci_hi80"] - df_ci["bayes_xfip"]).clip(lower=0)
            fig_ci = go.Figure()
            fig_ci.add_trace(go.Bar(
                x=df_ci["player"],
                y=df_ci["bayes_xfip"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=err_hi.tolist(),
                    arrayminus=err_lo.tolist(),
                ),
                name="Bayes xFIP",
                hovertemplate="<b>%{x}</b><br>Bayes: %{y:.2f}<extra></extra>",
            ))
            fig_ci.update_layout(
                xaxis_tickangle=-45,
                yaxis_title="予測 xFIP",
                height=450,
            )
            st.plotly_chart(fig_ci, use_container_width=True)

    # β係数エクスパンダー
    coef_data = _load_bayes_coef()
    if "pitcher" in coef_data:
        with st.expander("Bayes Ridge β係数（Statcast特徴量の寄与）"):
            coef_items = sorted(
                coef_data["pitcher"].items(),
                key=lambda x: abs(x[1]["coef"]), reverse=True
            )
            names = [k for k, _ in coef_items]
            values = [v["coef"] for _, v in coef_items]
            colors = ["#d93025" if v >= 0 else "#1a73e8" for v in values]  # xFIPは高いと悪い
            fig_coef = go.Figure(go.Bar(
                x=values, y=names,
                orientation="h",
                marker_color=colors,
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            fig_coef.update_layout(
                xaxis_title="β係数（z-score スケール）",
                height=max(250, len(names) * 35),
            )
            st.plotly_chart(fig_coef, use_container_width=True)

    st.subheader("Marcel vs ML — 乖離が大きい選手をハイライト")
    diff = (df_show["pred_xfip"] - df_show["marcel_xfip"]).abs()
    top10_idx = diff.nlargest(10).index
    labels = df_show["player"].where(df_show.index.isin(top10_idx), "")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_show["marcel_xfip"], y=df_show["pred_xfip"],
        mode="markers+text",
        text=labels,
        textposition="top center",
        hovertemplate="<b>%{customdata}</b><br>Marcel: %{x:.2f}<br>ML: %{y:.2f}<extra></extra>",
        customdata=df_show["player"],
        marker=dict(
            size=9,
            color=diff,
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
    st.plotly_chart(fig, use_container_width=True)


def page_spring():
    st.header("🌸 Spring Training 2026 検証")
    st.caption("オープン戦実績 vs モデル予測の比較（develop 環境のみ）")

    bat_spring_path = PRED_DIR / "spring_batting_2026.csv"
    pit_spring_path = PRED_DIR / "spring_pitching_2026.csv"
    bat_pred = load_predictions("batter")
    pit_pred = load_predictions("pitcher")

    tab1, tab2 = st.tabs(["打者", "投手"])

    with tab1:
        if not bat_spring_path.exists():
            st.info("オープン戦データ取得中（毎日 JST 23:00 更新）")
        elif bat_pred.empty:
            st.warning("予測データがありません。")
        else:
            spring = pd.read_csv(bat_spring_path)
            bat_merge_cols = ["player", "pred_woba", "marcel_woba"]
            if "ensemble_woba" in bat_pred.columns:
                bat_merge_cols.append("ensemble_woba")
            merged = spring.merge(bat_pred[bat_merge_cols], on="player", how="inner")
            if merged.empty:
                st.info(f"予測データと一致する選手が見つかりません（オープン戦出場: {len(spring)} 人）")
            else:
                disp_cols = ["player", "Team", "PA", "wOBA", "pred_woba", "marcel_woba"]
                disp_names = ["選手名", "チーム", "打席数", "実績wOBA", "ML予測", "Marcel予測"]
                if "ensemble_woba" in merged.columns:
                    disp_cols.append("ensemble_woba")
                    disp_names.append("アンサンブル予測")
                display = merged[disp_cols].copy()
                display.columns = disp_names
                display["ML誤差"] = (display["実績wOBA"] - display["ML予測"]).round(3)
                display["Marcel誤差"] = (display["実績wOBA"] - display["Marcel予測"]).round(3)
                st.dataframe(display.sort_values("打席数", ascending=False),
                             use_container_width=True)
                ml_mae = display["ML誤差"].abs().mean()
                mar_mae = display["Marcel誤差"].abs().mean()
                c1, c2 = st.columns(2)
                c1.metric("ML MAE (暫定)", f"{ml_mae:.4f}")
                c2.metric("Marcel MAE (暫定)", f"{mar_mae:.4f}",
                          delta=f"{mar_mae - ml_mae:+.4f}", delta_color="inverse")
                st.caption(f"※ オープン戦は参考値。サンプル数: {len(display)} 選手")

    with tab2:
        if not pit_spring_path.exists():
            st.info("オープン戦データ取得中（毎日 JST 23:00 更新）")
        elif pit_pred.empty:
            st.warning("予測データがありません。")
        else:
            spring = pd.read_csv(pit_spring_path)
            pit_merge_cols = ["player", "pred_xfip", "marcel_xfip"]
            if "ensemble_xfip" in pit_pred.columns:
                pit_merge_cols.append("ensemble_xfip")
            merged = spring.merge(pit_pred[pit_merge_cols], on="player", how="inner")
            if merged.empty:
                st.info(f"予測データと一致する選手が見つかりません（オープン戦出場: {len(spring)} 人）")
            else:
                disp_cols = ["player", "Team", "IP", "xFIP", "pred_xfip", "marcel_xfip"]
                disp_names = ["選手名", "チーム", "投球回", "実績xFIP", "ML予測", "Marcel予測"]
                if "ensemble_xfip" in merged.columns:
                    disp_cols.append("ensemble_xfip")
                    disp_names.append("アンサンブル予測")
                display = merged[disp_cols].copy()
                display.columns = disp_names
                display["ML誤差"] = (display["実績xFIP"] - display["ML予測"]).round(3)
                display["Marcel誤差"] = (display["実績xFIP"] - display["Marcel予測"]).round(3)
                st.dataframe(display.sort_values("投球回", ascending=False),
                             use_container_width=True)
                ml_mae = display["ML誤差"].abs().mean()
                mar_mae = display["Marcel誤差"].abs().mean()
                c1, c2 = st.columns(2)
                c1.metric("ML MAE (暫定)", f"{ml_mae:.4f}")
                c2.metric("Marcel MAE (暫定)", f"{mar_mae:.4f}",
                          delta=f"{mar_mae - ml_mae:+.4f}", delta_color="inverse")
                st.caption(f"※ オープン戦は参考値。サンプル数: {len(display)} 選手")


def page_backtest():
    st.header("Backtest: Year-by-Year Stability")

    summary_path = _BASE / "predictions" / "backtest" / "backtest_summary.json"
    if not summary_path.exists():
        st.info("バックテスト未実行。GitHub Actions の Backtest Analysis を実行してください。")
        return

    summary = json.loads(summary_path.read_text())

    tab_bat, tab_pit = st.tabs(["打者 wOBA", "投手 xFIP"])

    for tab, kind, metric in [(tab_bat, "batter", "wOBA"), (tab_pit, "pitcher", "xFIP")]:
        with tab:
            data = summary[kind]
            yearly = pd.DataFrame(data["yearly"])

            # 全年度勝利かどうか
            if data["ml_wins_all_years"]:
                st.success(f"ML は全 {data['total_years']} 年度で Marcel に勝利")
            else:
                st.warning(f"ML は {data['ml_win_years']}/{data['total_years']} 年度で Marcel に勝利")

            # 年度別 MAE 棒グラフ
            st.subheader(f"年度別 MAE 比較 ({metric})")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=yearly["year"], y=yearly["marcel_mae"],
                name="Marcel", marker_color="#9e9e9e",
            ))
            fig.add_trace(go.Bar(
                x=yearly["year"], y=yearly["ml_mae"],
                name="ML (LightGBM)", marker_color="#1a73e8",
            ))
            fig.update_layout(
                barmode="group",
                xaxis_title="Year", yaxis_title="MAE",
                xaxis=dict(dtick=1),
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # 年度別テーブル
            display = yearly.copy()
            display.columns = ["年度", "選手数", "ML MAE", "Marcel MAE",
                               "ML勝利", "改善率(%)"]
            st.dataframe(display, use_container_width=True, hide_index=True)

            # Pre/Post 2023
            era = data.get("era_split", {})
            if era:
                st.subheader("2023年ルール変更 前後比較")
                cols = st.columns(len(era))
                for col, (label, info) in zip(cols, era.items()):
                    period = "~2022" if "pre" in label else "2023~"
                    with col:
                        st.metric(f"{period} ML MAE", f"{info['ml_mae_weighted']:.4f}")
                        st.metric(f"{period} Marcel MAE", f"{info['marcel_mae_weighted']:.4f}",
                                  delta=f"{info['improvement_pct']:+.1f}%",
                                  delta_color="normal")
                        st.caption(f"{info['n_players_total']}選手, {info['years']}")

            # 2025 True Holdout
            holdout = data.get("holdout", {})
            if holdout:
                st.subheader(f"{holdout['year']} True Holdout")
                st.caption("Optuna + CV では一切使用していない完全未知データでの評価")
                h1, h2, h3 = st.columns(3)
                h1.metric(f"ML MAE ({metric})", f"{holdout['ml_mae']:.4f}")
                h2.metric(f"Marcel MAE ({metric})", f"{holdout['marcel_mae']:.4f}",
                          delta=f"{holdout['improvement_pct']:+.1f}%",
                          delta_color="normal")
                h3.metric("選手数", holdout["n_players"])
                if holdout.get("ml_wins"):
                    st.success(f"ML は 2025 holdout でも Marcel に勝利（{holdout['improvement_pct']:.1f}% 改善）")
                else:
                    st.warning("Marcel が 2025 holdout で ML に勝利")

            # 大外れ選手
            outlier_path = _BASE / "predictions" / "backtest" / f"outliers_{kind}.csv"
            if outlier_path.exists():
                st.subheader("大外れ選手 TOP 20")
                outliers = pd.read_csv(outlier_path).head(20)
                disp_cols = ["player", "season", "actual", "ml_pred",
                             "marcel_pred", "ml_error", "ml_worse_than_marcel"]
                disp_names = ["選手名", "年度", "実績", "ML予測",
                              "Marcel予測", "ML誤差", "MLがMarcelより悪い"]
                st.dataframe(
                    outliers[disp_cols].rename(columns=dict(zip(disp_cols, disp_names))),
                    use_container_width=True, hide_index=True,
                )
                st.caption("ケガ・急激な覚醒・移籍など、データに含まれない要因で予測が外れた可能性あり")


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
    ["打者 wOBA 予測", "投手 xFIP 予測", "🌸 Spring Training 検証", "Backtest", "About"],
    horizontal=True,
    label_visibility="collapsed",
)

if page == "打者 wOBA 予測":
    page_batters()
elif page == "投手 xFIP 予測":
    page_pitchers()
elif page == "🌸 Spring Training 検証":
    page_spring()
elif page == "Backtest":
    page_backtest()
else:
    page_about()
