# npb-mlops

**NPB player/team performance prediction with MLOps pipeline**

MLB・NPBの過去データから選手OPSとチーム勝率を予測し、シーズン中も自動で再学習・更新するMLOpsパイプライン。

[Marcel法による統計予測（npb-prediction）](https://github.com/yasumorishima/npb-prediction) とは異なるアプローチで、機械学習モデルをCI/CDで継続運用する。

---

## 特徴

| 項目 | 内容 |
|------|------|
| 予測ターゲット | 選手OPS（打者）+ チーム勝率 |
| モデル | LightGBM（ローリングウィンドウ学習） |
| 自動再学習 | GitHub Actions cron（週次） |
| モデル管理 | Weights & Biases（バージョン履歴・学習ログ） |
| API | FastAPI（`/predict/player` / `/predict/team`） |
| ダッシュボード | Streamlit（Marcel予測 vs ML予測 vs 実績の3列比較） |
| データソース | baseball-data.com / npb.jp |

---

## アーキテクチャ

```
[GitHub Actions cron 毎週月曜]
  ↓ データ取得（baseball-data.com）
  ↓ 特徴量生成
  ↓ LightGBM 再学習（2015〜今季途中）
  ↓ W&B にモデルバージョン・メトリクス記録
  ↓ モデルファイル更新（旧 vs 新の性能比較）
  ↓ FastAPI 経由で推論提供
  ↓ Streamlit 自動反映

[Streamlit ダッシュボード]
  Marcel予測 | ML予測 | 実績（途中） | 差分
```

---

## ローリングウィンドウ学習

シーズン終了を待たずに再学習できる設計：

```
Training: 2015〜2024（完全シーズン）+ 2025〜今季途中
                                        ↑ 週次でここが増える
Validation: 直近1シーズン（walk-forward CV）
```

---

## ローカル実行

```bash
pip install -r requirements.txt
python src/train.py        # モデル学習
uvicorn api.main:app --reload --port 8002
streamlit run streamlit/app.py
```

---

## データクレジット

- [baseball-data.com](https://baseball-data.com)
- [npb.jp](https://npb.jp)

---

*Built with Claude Code / Powered by LightGBM + W&B + FastAPI + Streamlit*
