from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "UNH", "XOM", "SPY"]
ASSETS = [ticker for ticker in TICKERS if ticker != "SPY"]
BENCHMARK = "SPY"
START_DATE = "2021-01-01"
END_DATE = "2026-06-05"
RISK_FREE_RATE = 0.04
TRADING_DAYS = 252
RANDOM_STATE = 42


def ensure_dirs() -> None:
    for directory in [DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
        directory.mkdir(exist_ok=True)


def unix_timestamp(date_text: str, end_of_day: bool = False) -> int:
    dt = datetime.fromisoformat(date_text).replace(tzinfo=timezone.utc)
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59)
    return int(dt.timestamp())


def fetch_yahoo_history(ticker: str) -> pd.DataFrame:
    cache_path = DATA_DIR / f"{ticker.lower()}_prices.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["date"])

    params = urlencode(
        {
            "period1": unix_timestamp(START_DATE),
            "period2": unix_timestamp(END_DATE, end_of_day=True),
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?{params}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 finance-education-project"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    result = payload["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adjclose = result["indicators"].get("adjclose", [{}])[0].get("adjclose", quote["close"])
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s").date,
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "adj_close": adjclose,
            "volume": quote["volume"],
        }
    )
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna(subset=["adj_close"]).sort_values("date").reset_index(drop=True)
    frame["ticker"] = ticker
    frame.to_csv(cache_path, index=False)
    return frame


def load_market_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_frames = [fetch_yahoo_history(ticker) for ticker in TICKERS]
    raw = pd.concat(raw_frames, ignore_index=True)
    raw.to_csv(DATA_DIR / "market_prices_raw.csv", index=False)

    prices = raw.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    volume = raw.pivot(index="date", columns="ticker", values="volume").sort_index()
    prices = prices.ffill().dropna()
    volume = volume.reindex(prices.index).ffill()
    prices.to_csv(DATA_DIR / "aligned_adjusted_close.csv")
    volume.to_csv(DATA_DIR / "aligned_volume.csv")

    quality_rows = []
    for ticker in TICKERS:
        subset = raw[raw["ticker"] == ticker]
        quality_rows.append(
            {
                "ticker": ticker,
                "rows": len(subset),
                "start": subset["date"].min().date().isoformat(),
                "end": subset["date"].max().date().isoformat(),
                "missing_adj_close": int(subset["adj_close"].isna().sum()),
                "duplicate_dates": int(subset.duplicated(["ticker", "date"]).sum()),
            }
        )
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(OUTPUT_DIR / "data_quality_report.csv", index=False)
    return raw, prices, quality


def drawdown(series: pd.Series) -> pd.Series:
    wealth = (1 + series).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1


def max_drawdown(series: pd.Series) -> float:
    return float(drawdown(series).min())


def sortino_ratio(returns: pd.Series) -> float:
    downside = returns[returns < 0].std() * math.sqrt(TRADING_DAYS)
    if downside == 0 or np.isnan(downside):
        return np.nan
    return float((returns.mean() * TRADING_DAYS - RISK_FREE_RATE) / downside)


def asset_metrics(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    benchmark = returns[BENCHMARK]
    for ticker in returns.columns:
        r = returns[ticker]
        beta = np.cov(r, benchmark)[0, 1] / np.var(benchmark) if ticker != BENCHMARK else 1.0
        alpha = (r.mean() * TRADING_DAYS - RISK_FREE_RATE) - beta * (benchmark.mean() * TRADING_DAYS - RISK_FREE_RATE)
        rows.append(
            {
                "ticker": ticker,
                "annual_return": r.mean() * TRADING_DAYS,
                "annual_volatility": r.std() * math.sqrt(TRADING_DAYS),
                "sharpe": (r.mean() * TRADING_DAYS - RISK_FREE_RATE) / (r.std() * math.sqrt(TRADING_DAYS)),
                "sortino": sortino_ratio(r),
                "max_drawdown": max_drawdown(r),
                "var_95_daily": r.quantile(0.05),
                "cvar_95_daily": r[r <= r.quantile(0.05)].mean(),
                "beta_to_spy": beta,
                "alpha_to_spy": alpha,
                "positive_day_rate": (r > 0).mean(),
            }
        )
    metrics = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    metrics.to_csv(OUTPUT_DIR / "asset_risk_metrics.csv", index=False)
    return metrics


def portfolio_stats(weights: np.ndarray, mean_returns: pd.Series, covariance: pd.DataFrame) -> tuple[float, float, float]:
    annual_return = float(np.dot(weights, mean_returns) * TRADING_DAYS)
    annual_vol = float(np.sqrt(weights.T @ (covariance * TRADING_DAYS).values @ weights))
    sharpe = (annual_return - RISK_FREE_RATE) / annual_vol if annual_vol else np.nan
    return annual_return, annual_vol, sharpe


def optimize_portfolios(asset_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    mean_returns = asset_returns.mean()
    covariance = asset_returns.cov()
    rows = []
    for _ in range(6500):
        weights = rng.dirichlet(np.ones(len(ASSETS)))
        annual_return, annual_vol, sharpe = portfolio_stats(weights, mean_returns, covariance)
        rows.append(
            {
                "annual_return": annual_return,
                "annual_volatility": annual_vol,
                "sharpe": sharpe,
                **{f"weight_{asset}": weight for asset, weight in zip(ASSETS, weights, strict=False)},
            }
        )
    frontier = pd.DataFrame(rows)
    frontier.to_csv(OUTPUT_DIR / "efficient_frontier_simulation.csv", index=False)

    bounds = tuple((0, 1) for _ in ASSETS)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    start = np.repeat(1 / len(ASSETS), len(ASSETS))

    def neg_sharpe(weights: np.ndarray) -> float:
        return -portfolio_stats(weights, mean_returns, covariance)[2]

    def volatility(weights: np.ndarray) -> float:
        return portfolio_stats(weights, mean_returns, covariance)[1]

    max_sharpe = minimize(neg_sharpe, start, method="SLSQP", bounds=bounds, constraints=constraints)
    min_vol = minimize(volatility, start, method="SLSQP", bounds=bounds, constraints=constraints)

    portfolios = []
    for label, result in [("Max Sharpe", max_sharpe), ("Min Volatility", min_vol), ("Equal Weight", None)]:
        weights = start if result is None else result.x
        annual_return, annual_vol, sharpe = portfolio_stats(weights, mean_returns, covariance)
        portfolios.append(
            {
                "portfolio": label,
                "annual_return": annual_return,
                "annual_volatility": annual_vol,
                "sharpe": sharpe,
                **{asset: weight for asset, weight in zip(ASSETS, weights, strict=False)},
            }
        )
    optimized = pd.DataFrame(portfolios)
    optimized.to_csv(OUTPUT_DIR / "optimized_portfolios.csv", index=False)
    return frontier, optimized


def build_market_regimes(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    spy = returns[BENCHMARK]
    features = pd.DataFrame(index=returns.index)
    features["spy_21d_return"] = (1 + spy).rolling(21).apply(np.prod, raw=True) - 1
    features["spy_21d_volatility"] = spy.rolling(21).std() * math.sqrt(TRADING_DAYS)
    features["spy_drawdown"] = drawdown(spy)
    features["market_breadth_21d"] = ((1 + returns[ASSETS]).rolling(21).apply(np.prod, raw=True) - 1 > 0).mean(axis=1)
    features = features.dropna()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=20)
    labels = kmeans.fit_predict(scaled)
    regimes = features.copy()
    regimes["regime_id"] = labels + 1
    profile = regimes.groupby("regime_id").mean().round(4).reset_index()
    profile["label"] = profile.apply(
        lambda row: "Risk-Off Drawdown"
        if row["spy_21d_return"] < 0 and row["spy_21d_volatility"] > profile["spy_21d_volatility"].median()
        else "Bullish Expansion"
        if row["spy_21d_return"] > 0 and row["market_breadth_21d"] > 0.55
        else "Choppy Recovery"
        if row["spy_drawdown"] < -0.08
        else "Quiet Range",
        axis=1,
    )
    regimes = regimes.merge(profile[["regime_id", "label"]], on="regime_id", how="left").set_index(features.index)
    regimes.to_csv(OUTPUT_DIR / "market_regimes.csv")
    profile.to_csv(OUTPUT_DIR / "regime_profiles.csv", index=False)
    return regimes, profile


def detect_anomalies(returns: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=returns.index)
    features["spy_return"] = returns[BENCHMARK]
    features["spy_abs_return"] = returns[BENCHMARK].abs()
    features["cross_asset_volatility"] = returns[ASSETS].std(axis=1)
    features["cross_asset_mean"] = returns[ASSETS].mean(axis=1)
    features["market_breadth"] = (returns[ASSETS] > 0).mean(axis=1)
    features = features.dropna()
    model = IsolationForest(contamination=0.025, random_state=RANDOM_STATE)
    labels = model.fit_predict(features)
    scores = -model.score_samples(features)
    anomalies = features.copy()
    anomalies["anomaly_score"] = scores
    anomalies["is_anomaly"] = (labels == -1).astype(int)
    watchlist = anomalies[anomalies["is_anomaly"] == 1].sort_values("anomaly_score", ascending=False).copy()
    watchlist["spy_close"] = prices.loc[watchlist.index, BENCHMARK]
    watchlist.to_csv(OUTPUT_DIR / "market_anomaly_watchlist.csv")
    return anomalies


def make_prediction_dataset(returns: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    spy = returns[BENCHMARK]
    data = pd.DataFrame(index=returns.index)
    forward_5d_return = (1 + spy).shift(-5).rolling(5).apply(np.prod, raw=True) - 1
    data["target_up_next_5d"] = (forward_5d_return > 0).astype(int)
    data["spy_return_1d"] = spy
    data["spy_return_5d"] = spy.rolling(5).sum()
    data["spy_return_21d"] = spy.rolling(21).sum()
    data["spy_vol_10d"] = spy.rolling(10).std() * math.sqrt(TRADING_DAYS)
    data["spy_vol_21d"] = spy.rolling(21).std() * math.sqrt(TRADING_DAYS)
    data["drawdown"] = drawdown(spy)
    data["market_breadth"] = (returns[ASSETS] > 0).mean(axis=1)
    data["cross_asset_volatility"] = returns[ASSETS].std(axis=1)
    data["nvda_momentum_10d"] = returns["NVDA"].rolling(10).sum()
    data["xom_momentum_10d"] = returns["XOM"].rolling(10).sum()
    data["regime_id"] = regimes["regime_id"].reindex(data.index).ffill()
    return data.dropna().iloc[:-5]


def train_predictive_models(model_data: pd.DataFrame) -> tuple[pd.DataFrame, Pipeline, pd.DataFrame]:
    x = model_data.drop(columns=["target_up_next_5d"])
    y = model_data["target_up_next_5d"].astype(int)
    split = int(len(model_data) * 0.75)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    candidates = {
        "Logistic Regression": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1200))]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=350, min_samples_leaf=8, random_state=RANDOM_STATE, class_weight="balanced")),
            ]
        ),
    }
    tscv = TimeSeriesSplit(n_splits=5)
    rows = []
    fitted = {}
    for name, pipe in candidates.items():
        cv_auc = cross_val_score(pipe, x_train, y_train, cv=tscv, scoring="roc_auc").mean()
        pipe.fit(x_train, y_train)
        probabilities = pipe.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.52).astype(int)
        rows.append(
            {
                "model": name,
                "cv_roc_auc": cv_auc,
                "test_roc_auc": roc_auc_score(y_test, probabilities),
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
            }
        )
        fitted[name] = pipe

    leaderboard = pd.DataFrame(rows).sort_values("test_roc_auc", ascending=False).reset_index(drop=True)
    champion_name = str(leaderboard.loc[0, "model"])
    champion = fitted[champion_name]
    joblib.dump(champion, MODEL_DIR / "spy_direction_champion.joblib")
    leaderboard.to_csv(OUTPUT_DIR / "model_leaderboard.csv", index=False)

    probabilities = champion.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.52).astype(int)
    prediction_frame = x_test.copy()
    prediction_frame["actual_up_next_5d"] = y_test
    prediction_frame["predicted_probability"] = probabilities
    prediction_frame["predicted_up_next_5d"] = predictions
    prediction_frame.to_csv(OUTPUT_DIR / "spy_direction_predictions.csv")

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, linewidth=2.5, label=f"{champion_name} AUC={roc_auc_score(y_test, probabilities):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777")
    ax.set_title("SPY Next-5-Day Direction ROC")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_fig(fig, "model_roc_curve.png")

    matrix = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(matrix, display_labels=["Down/Flat", "Up"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Champion Model Confusion Matrix")
    save_fig(fig, "model_confusion_matrix.png")

    if champion_name == "Random Forest":
        importances = champion.named_steps["model"].feature_importances_
    else:
        importances = np.abs(champion.named_steps["model"].coef_[0])
    importance = pd.DataFrame({"feature": x.columns, "importance": importances}).sort_values("importance", ascending=False)
    importance.to_csv(OUTPUT_DIR / "model_feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    top = importance.head(10).sort_values("importance")
    ax.barh(top["feature"], top["importance"], color="#0f766e", edgecolor="#222")
    ax.set_title("Prediction Model Feature Importance")
    ax.grid(axis="x", alpha=0.25)
    save_fig(fig, "model_feature_importance.png")
    return leaderboard, champion, prediction_frame


def backtest_strategy(returns: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    spy = returns[BENCHMARK].reindex(predictions.index)
    next_day_return = spy.shift(-1).reindex(predictions.index)
    signal = (predictions["predicted_probability"] >= 0.52).astype(float)
    strategy_return = (signal * next_day_return).dropna()
    buy_hold_return = next_day_return.reindex(strategy_return.index)
    backtest = pd.DataFrame({"strategy": strategy_return, "buy_hold": buy_hold_return})
    backtest["strategy_growth"] = (1 + backtest["strategy"]).cumprod()
    backtest["buy_hold_growth"] = (1 + backtest["buy_hold"]).cumprod()
    backtest.to_csv(OUTPUT_DIR / "strategy_backtest.csv")
    return backtest


def performance_summary(return_series: pd.Series) -> dict[str, float]:
    return {
        "annual_return": float(return_series.mean() * TRADING_DAYS),
        "annual_volatility": float(return_series.std() * math.sqrt(TRADING_DAYS)),
        "sharpe": float((return_series.mean() * TRADING_DAYS - RISK_FREE_RATE) / (return_series.std() * math.sqrt(TRADING_DAYS))),
        "max_drawdown": max_drawdown(return_series),
        "var_95_daily": float(return_series.quantile(0.05)),
        "cvar_95_daily": float(return_series[return_series <= return_series.quantile(0.05)].mean()),
    }


def save_fig(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close(fig)


def plot_outputs(prices: pd.DataFrame, returns: pd.DataFrame, metrics: pd.DataFrame, frontier: pd.DataFrame, optimized: pd.DataFrame, regimes: pd.DataFrame, anomalies: pd.DataFrame, backtest: pd.DataFrame) -> None:
    normalized = prices / prices.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    normalized.plot(ax=ax, linewidth=1.8)
    ax.set_title("Normalized Price Performance")
    ax.set_ylabel("Growth of $100")
    ax.grid(alpha=0.25)
    save_fig(fig, "normalized_prices.png")

    corr = returns.corr().round(2)
    corr.to_csv(OUTPUT_DIR / "return_correlation_matrix.csv")
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), labels=corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), labels=corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Daily Return Correlation")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, "correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(metrics["annual_volatility"], metrics["annual_return"], s=85, color="#2563eb")
    for _, row in metrics.iterrows():
        ax.text(row["annual_volatility"], row["annual_return"], row["ticker"], fontsize=9, ha="left", va="bottom")
    ax.set_title("Risk vs Return by Asset")
    ax.set_xlabel("Annual Volatility")
    ax.set_ylabel("Annual Return")
    ax.grid(alpha=0.25)
    save_fig(fig, "risk_return_scatter.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(frontier["annual_volatility"], frontier["annual_return"], c=frontier["sharpe"], cmap="viridis", s=12, alpha=0.7)
    for _, row in optimized.iterrows():
        ax.scatter(row["annual_volatility"], row["annual_return"], s=130, edgecolor="#111", label=row["portfolio"])
    ax.set_title("Monte Carlo Efficient Frontier")
    ax.set_xlabel("Annual Volatility")
    ax.set_ylabel("Annual Return")
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, label="Sharpe")
    ax.legend(frameon=False)
    save_fig(fig, "efficient_frontier.png")

    alloc = optimized.set_index("portfolio")[ASSETS].T
    fig, ax = plt.subplots(figsize=(9, 5))
    alloc.plot(kind="bar", ax=ax, edgecolor="#222")
    ax.set_title("Optimized Portfolio Weights")
    ax.set_ylabel("Weight")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, "portfolio_allocations.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    drawdown(returns[BENCHMARK]).plot(ax=ax, color="#dc2626", linewidth=2)
    ax.set_title("SPY Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.25)
    save_fig(fig, "spy_drawdown.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    (returns[BENCHMARK].rolling(21).std() * math.sqrt(TRADING_DAYS)).plot(ax=ax, color="#7c3aed", linewidth=2)
    ax.set_title("SPY Rolling 21-Day Volatility")
    ax.set_ylabel("Annualized Volatility")
    ax.grid(alpha=0.25)
    save_fig(fig, "rolling_volatility.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    regime_plot = regimes["regime_id"].reindex(returns.index).ffill()
    ax.scatter(regime_plot.index, returns[BENCHMARK].reindex(regime_plot.index).cumsum(), c=regime_plot, cmap="tab10", s=12)
    ax.set_title("Market Regime Timeline")
    ax.set_ylabel("Cumulative SPY Log Proxy")
    ax.grid(alpha=0.25)
    save_fig(fig, "market_regime_timeline.png")

    anomaly_points = anomalies[anomalies["is_anomaly"] == 1]
    fig, ax = plt.subplots(figsize=(10, 5))
    returns[BENCHMARK].plot(ax=ax, color="#334155", linewidth=1)
    ax.scatter(anomaly_points.index, anomaly_points["spy_return"], color="#dc2626", s=35, label="Anomaly")
    ax.set_title("Detected Market Anomaly Days")
    ax.set_ylabel("SPY Daily Return")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_fig(fig, "market_anomalies.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    backtest[["strategy_growth", "buy_hold_growth"]].plot(ax=ax, linewidth=2.5)
    ax.set_title("Signal Strategy vs Buy-and-Hold")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.25)
    save_fig(fig, "strategy_backtest.png")


def table(frame: pd.DataFrame, rows: int | None = None) -> str:
    if rows:
        frame = frame.head(rows)
    return frame.to_html(index=False, border=0, classes="data-table")


def pct(value: float) -> str:
    return f"{value:.2%}"


def money(value: float) -> str:
    return f"${value:,.2f}"


def build_report(quality: pd.DataFrame, metrics: pd.DataFrame, optimized: pd.DataFrame, regime_profile: pd.DataFrame, leaderboard: pd.DataFrame, backtest: pd.DataFrame) -> None:
    best_asset = metrics.iloc[0]
    max_sharpe = optimized[optimized["portfolio"] == "Max Sharpe"].iloc[0]
    strategy_perf = performance_summary(backtest["strategy"])
    buy_hold_perf = performance_summary(backtest["buy_hold"])
    summary = {
        "analysis_start": START_DATE,
        "analysis_end": END_DATE,
        "best_asset_by_sharpe": str(best_asset["ticker"]),
        "best_asset_sharpe": float(best_asset["sharpe"]),
        "max_sharpe_portfolio_sharpe": float(max_sharpe["sharpe"]),
        "strategy_sharpe": strategy_perf["sharpe"],
        "buy_hold_sharpe": buy_hold_perf["sharpe"],
        "champion_model": str(leaderboard.loc[0, "model"]),
        "champion_auc": float(leaderboard.loc[0, "test_roc_auc"]),
    }
    (OUTPUT_DIR / "executive_insights.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Real-World Finance Data Science Report</title>
  <style>
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; color: #172026; background: #f5f7fa; }}
    header {{ background: #0b1324; color: white; padding: 34px 42px; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 28px 20px 48px; }}
    h1, h2 {{ margin: 0 0 12px; }}
    p, li {{ line-height: 1.55; }}
    section {{ margin-top: 32px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 22px 0; }}
    .kpi {{ background: white; border: 1px solid #d7dee8; border-radius: 8px; padding: 18px; }}
    .kpi span {{ display: block; color: #52616f; font-size: 13px; margin-bottom: 8px; }}
    .kpi strong {{ font-size: 22px; }}
    .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 18px; }}
    figure {{ margin: 0; background: white; border: 1px solid #d7dee8; border-radius: 8px; padding: 12px; }}
    img {{ width: 100%; height: auto; display: block; }}
    .data-table {{ border-collapse: collapse; width: 100%; background: white; margin-top: 12px; }}
    .data-table th, .data-table td {{ border: 1px solid #d7dee8; padding: 9px 10px; text-align: left; font-size: 14px; }}
    .data-table th {{ background: #edf2f7; }}
    code {{ background: #e8edf4; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>Real-World Finance Data Science Project</h1>
    <p>End-to-end market analysis, portfolio optimization, anomaly detection, prediction, and backtesting using historical stock data.</p>
  </header>
  <main>
    <div class="kpis">
      <div class="kpi"><span>Analysis Window</span><strong>{html.escape(START_DATE)} to {html.escape(END_DATE)}</strong></div>
      <div class="kpi"><span>Best Asset Sharpe</span><strong>{html.escape(str(best_asset["ticker"]))}: {best_asset["sharpe"]:.2f}</strong></div>
      <div class="kpi"><span>Max-Sharpe Portfolio</span><strong>{max_sharpe["sharpe"]:.2f}</strong></div>
      <div class="kpi"><span>Champion Model</span><strong>{html.escape(str(leaderboard.loc[0, "model"]))}</strong></div>
      <div class="kpi"><span>Model ROC AUC</span><strong>{leaderboard.loc[0, "test_roc_auc"]:.3f}</strong></div>
      <div class="kpi"><span>Strategy Sharpe</span><strong>{strategy_perf["sharpe"]:.2f}</strong></div>
    </div>

    <section>
      <h2>Conclusions</h2>
      <ul>
        <li><strong>{html.escape(str(best_asset["ticker"]))}</strong> ranked highest by standalone Sharpe ratio during this window.</li>
        <li>The optimized max-Sharpe portfolio improved diversification by combining return, volatility, and covariance information.</li>
        <li>Regime clustering separated calm, bullish, choppy, and drawdown-like environments without hand labels.</li>
        <li>The predictive model is evaluated with time-aware splits and should be treated as a research signal, not investment advice.</li>
        <li>The backtest compares a simple probability-threshold strategy against SPY buy-and-hold on held-out dates.</li>
      </ul>
    </section>

    <section>
      <h2>Data Quality</h2>
      {table(quality)}
    </section>

    <section>
      <h2>Visual Evidence</h2>
      <div class="charts">
        <figure><img src="normalized_prices.png" alt="Normalized prices"></figure>
        <figure><img src="risk_return_scatter.png" alt="Risk return scatter"></figure>
        <figure><img src="correlation_heatmap.png" alt="Correlation heatmap"></figure>
        <figure><img src="efficient_frontier.png" alt="Efficient frontier"></figure>
        <figure><img src="portfolio_allocations.png" alt="Portfolio allocations"></figure>
        <figure><img src="spy_drawdown.png" alt="SPY drawdown"></figure>
        <figure><img src="rolling_volatility.png" alt="Rolling volatility"></figure>
        <figure><img src="market_regime_timeline.png" alt="Market regime timeline"></figure>
        <figure><img src="market_anomalies.png" alt="Market anomalies"></figure>
        <figure><img src="model_roc_curve.png" alt="Model ROC"></figure>
        <figure><img src="model_confusion_matrix.png" alt="Model confusion matrix"></figure>
        <figure><img src="model_feature_importance.png" alt="Model feature importance"></figure>
        <figure><img src="strategy_backtest.png" alt="Strategy backtest"></figure>
      </div>
    </section>

    <section>
      <h2>Asset Risk Metrics</h2>
      {table(metrics.round(4))}
    </section>

    <section>
      <h2>Optimized Portfolios</h2>
      {table(optimized.round(4))}
    </section>

    <section>
      <h2>Market Regimes</h2>
      {table(regime_profile.round(4))}
    </section>

    <section>
      <h2>Predictive Model Leaderboard</h2>
      {table(leaderboard.round(4))}
    </section>

    <section>
      <h2>Backtest Summary</h2>
      {table(pd.DataFrame([{"strategy": "Signal Strategy", **strategy_perf}, {"strategy": "SPY Buy and Hold", **buy_hold_perf}]).round(4))}
    </section>
  </main>
</body>
</html>
"""
    (OUTPUT_DIR / "finance_report.html").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    raw, prices, quality = load_market_data()
    returns = prices.pct_change().dropna()
    returns.to_csv(DATA_DIR / "daily_returns.csv")
    metrics = asset_metrics(returns)
    frontier, optimized = optimize_portfolios(returns[ASSETS])
    regimes, regime_profile = build_market_regimes(returns)
    anomalies = detect_anomalies(returns, prices)
    model_data = make_prediction_dataset(returns, regimes)
    model_data.to_csv(DATA_DIR / "spy_prediction_features.csv")
    leaderboard, champion, predictions = train_predictive_models(model_data)
    backtest = backtest_strategy(returns, predictions)
    plot_outputs(prices, returns, metrics, frontier, optimized, regimes, anomalies, backtest)
    build_report(quality, metrics, optimized, regime_profile, leaderboard, backtest)
    print(f"Raw market data: {DATA_DIR / 'market_prices_raw.csv'}")
    print(f"Finance report: {OUTPUT_DIR / 'finance_report.html'}")
    print(f"Executive insights: {OUTPUT_DIR / 'executive_insights.json'}")
    print(f"Champion model: {MODEL_DIR / 'spy_direction_champion.joblib'}")


if __name__ == "__main__":
    main()
