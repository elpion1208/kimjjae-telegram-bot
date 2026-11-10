"""
변형티큐법 원천 데이터 수집기 (Data Fetcher)
- Yahoo Finance (yfinance): TQQQ, QQQ(배당조정/분할조정), SPY, ^IXIC, ^VIX(일봉+5분봉), DX-Y.NYB(일봉+1시간봉), KRW=X
- FRED 공식 CSV 엔드포인트: DGS6MO, DGS2, DGS10, DGS20 (별도 API 키 불필요)
"""

from __future__ import annotations
import concurrent.futures
import io
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
import pytz
import requests
import yfinance as yf

from config import StrategyConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)
ET_TZ = pytz.timezone("America/New_York")


@dataclass
class MarketDataBundle:
    # 일봉 종가/시가 시리즈 (배당조정, 인덱스는 YYYY-MM-DD 문자열 또는 Timestamp)
    tqqq_close: pd.Series
    tqqq_high: pd.Series
    tqqq_low: pd.Series
    qqq_close: pd.Series
    qqq_rsi_close: pd.Series     # 분할조정(splits only) 또는 배당조정 QQQ 종가
    spy_close: pd.Series
    ixic_volume: pd.Series
    vix_close: pd.Series
    dxy_daily_close: pd.Series
    usdkrw_close: pd.Series

    # FRED 금리 일봉 시리즈 (결측 제외한 유효 관측치 시계열)
    dgs6m: pd.Series
    dgs2: pd.Series
    dgs10: pd.Series
    dgs20: pd.Series

    # 1차(PRIMARY) 전용 장마감(16:00/13:00 ET) 스냅샷 데이터
    vix_primary_close_by_date: Dict[str, float]
    dxy_cohort_a_current: Optional[float]
    dxy_cohort_a_ref: Optional[float]
    dxy_cohort_a_date: Optional[str]

    # V2 원금보호 손절 기준가(익일 시가 체결 기준) 산출용 TQQQ 수정시가
    tqqq_open: Optional[pd.Series] = None


def _normalize_date_index(series: pd.Series) -> pd.Series:
    """시계열 인덱스를 미국 동부시간(또는 날짜) 기준 'YYYY-MM-DD' 문자열로 통일합니다."""
    if series.empty:
        return series
    s = series.copy()
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert(ET_TZ)
    s.index = idx.strftime("%Y-%m-%d")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.dropna()


def fetch_yf_daily(
    symbol: str,
    period: str = "5y",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """yfinance에서 일봉 데이터를 가져옵니다."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval="1d", auto_adjust=auto_adjust, actions=False)
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo Finance 일봉 데이터 수집 실패: {symbol}")
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(ET_TZ)
    df = df.copy()
    df.index = idx.strftime("%Y-%m-%d")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


_TREASURY_CSV_CACHE: Dict[str, pd.Series] = {}


def _fetch_us_treasury_official_all(start_year: int = 2021) -> Dict[str, pd.Series]:
    """
    미국 재무부(home.treasury.gov) 공식 일별 국채금리 CSV에서
    DGS6MO('6 Mo'), DGS2('2 Yr'), DGS10('10 Yr'), DGS20('20 Yr') 원천 데이터를 병렬 수집합니다.
    (FRED의 DGS 시리즈와 수치 100% 동일하며, 클라우드/GitHub Actions/Colab IP 차단 영향을 받지 않습니다.)
    """
    global _TREASURY_CSV_CACHE
    if _TREASURY_CSV_CACHE:
        return _TREASURY_CSV_CACHE

    current_year = datetime.now().year
    col_map = {
        "DGS6MO": "6 Mo",
        "DGS2": "2 Yr",
        "DGS10": "10 Yr",
        "DGS20": "20 Yr",
    }
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    def _fetch_one_year(yr: int):
        url = (
            f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
            f"daily-treasury-rates.csv/{yr}/all?type=daily_treasury_yield_curve"
            f"&field_tdr_date_value={yr}&page&_format=csv"
        )
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200 and "Date" in resp.text:
                return pd.read_csv(io.StringIO(resp.text))
        except Exception as e:
            logger.warning(f"미 재무부 {yr}년 금리 CSV 조회 경고: {e}")
        return None

    frames = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for df_yr in ex.map(_fetch_one_year, range(start_year, current_year + 1)):
            if df_yr is not None:
                frames.append(df_yr)

    if not frames:
        return {}

    df_all = pd.concat(frames, ignore_index=True)
    df_all["date_str"] = pd.to_datetime(df_all["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df_all = df_all.dropna(subset=["date_str"]).sort_values("date_str")

    result: Dict[str, pd.Series] = {}
    for fred_id, t_col in col_map.items():
        if t_col in df_all.columns:
            vals = pd.to_numeric(df_all[t_col], errors="coerce")
            s = pd.Series(vals.values, index=df_all["date_str"].values, name=fred_id).dropna()
            s = s[~s.index.duplicated(keep="last")].sort_index()
            result[fred_id] = s

    _TREASURY_CSV_CACHE = result
    return result


def fetch_fred_series(series_id: str, start: str = "2021-01-01") -> pd.Series:
    """
    미국 국채 금리 시계열 수집 (3단계 자동 폴백):
    1순위: 미국 재무부(home.treasury.gov) 공식 Daily Treasury Yield Curve CSV 병렬 수집 (FRED 원천 데이터, 초고속)
    2순위: FRED 공식 CSV (cosd 파라미터로 용량 최적화, 타임아웃 6초)
    3순위: Yahoo Finance 국채 금리 지수 (^IRX, ^FVX, ^TNX, ^TYX)
    """
    # 1순위: 미국 재무부 공식 CSV (병렬 캐시)
    treasury_map = _fetch_us_treasury_official_all(start_year=int(start[:4]))
    if series_id in treasury_map and not treasury_map[series_id].empty:
        s = treasury_map[series_id]
        return s[s.index >= start]

    # 2순위: FRED 공식 CSV
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = requests.get(url, headers=headers, timeout=6)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        date_col, val_col = df.columns[0], df.columns[1]
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=[val_col])
        s = pd.Series(df[val_col].values, index=df[date_col].astype(str).values, name=series_id)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        if not s.empty:
            return s[s.index >= start]
    except Exception as e:
        logger.warning(f"FRED({series_id}) 응답 지연 -> Yahoo 폴백 전환: {e}")

    # 3순위: Yahoo Finance 국채금리 티커 폴백
    yf_fallback_map = {
        "DGS6MO": "^IRX",
        "DGS2": "^FVX",
        "DGS10": "^TNX",
        "DGS20": "^TYX",
    }
    yf_sym = yf_fallback_map.get(series_id, "^TNX")
    logger.info(f"Yahoo Finance 국채금리 대체 티커({yf_sym})로 {series_id} 수집 중...")
    df_yf = fetch_yf_daily(yf_sym, period="5y", auto_adjust=False)
    s = _normalize_date_index(df_yf["Close"])
    s.name = series_id
    return s[s.index >= start]


def fetch_vix_primary_close_map(vix_symbol: str = "^VIX") -> Dict[str, float]:
    """
    주석 14절 & requestVixPrimaryClose():
    VIX 5분봉에서 미국 정규장 마감 시각(16:00 ET 또는 단축장 13:00 ET)에 닫힌 봉의 종가를 추출합니다.
    (yfinance 5분봉의 타임스탬프는 봉 시작 시각이므로 15:55 ET 봉의 종가가 16:00 ET 마감값입니다.)
    """
    out: Dict[str, float] = {}
    try:
        ticker = yf.Ticker(vix_symbol)
        df_5m = ticker.history(period="60d", interval="5m", auto_adjust=False)
        if df_5m is None or df_5m.empty:
            return out
        idx = pd.to_datetime(df_5m.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC").tz_convert(ET_TZ)
        else:
            idx = idx.tz_convert(ET_TZ)
        df_5m = df_5m.copy()
        df_5m["et_dt"] = idx
        df_5m["date_str"] = df_5m["et_dt"].dt.strftime("%Y-%m-%d")
        df_5m["hm"] = df_5m["et_dt"].dt.strftime("%H:%M")

        # 15:55~16:00 ET 봉 (시작 15:55) 또는 조기마감일 12:55~13:00 ET 봉 (시작 12:55)
        for date_str, group in df_5m.groupby("date_str"):
            bar_16 = group[group["hm"] == "15:55"]
            if not bar_16.empty:
                out[date_str] = float(bar_16["Close"].iloc[-1])
            else:
                bar_13 = group[group["hm"] == "12:55"]
                if not bar_13.empty:
                    out[date_str] = float(bar_13["Close"].iloc[-1])
    except Exception as e:
        logger.warning(f"VIX 5분봉 스냅샷 조회 중 경고 (일봉 대체 사용): {e}")
    return out


def fetch_dxy_cohort_a_snapshot(
    dxy_symbol: str = "DX-Y.NYB",
    roc_len: int = 120,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    주석 12절 & c086DxyRing(16):
    DXY 1시간봉(60m)에서 주식 마감 시각(16:00 ET,即 15:00~16:00 ET 봉)의 당일 값과
    동일 시각 N번째(120관측) 이전 세션 값을 반환합니다.
    """
    try:
        ticker = yf.Ticker(dxy_symbol)
        df_1h = ticker.history(period="2y", interval="1h", auto_adjust=False)
        if df_1h is None or df_1h.empty:
            return None, None, None
        idx = pd.to_datetime(df_1h.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC").tz_convert(ET_TZ)
        else:
            idx = idx.tz_convert(ET_TZ)
        df_1h = df_1h.copy()
        df_1h["et_dt"] = idx
        df_1h["date_str"] = df_1h["et_dt"].dt.strftime("%Y-%m-%d")
        df_1h["hour"] = df_1h["et_dt"].dt.hour

        # 15:00 ET 시작 1시간봉의 종가 = 16:00 ET 스냅샷
        cohort_16 = df_1h[df_1h["hour"] == 15].dropna(subset=["Close"])
        cohort_16 = cohort_16[~cohort_16["date_str"].duplicated(keep="last")].sort_values("date_str")
        if len(cohort_16) > roc_len:
            c_val = float(cohort_16["Close"].iloc[-1])
            ref_val = float(cohort_16["Close"].iloc[-1 - roc_len])
            c_date = str(cohort_16["date_str"].iloc[-1])
            return c_val, ref_val, c_date
    except Exception as e:
        logger.warning(f"DXY 1시간봉(Cohort A) 조회 중 경고 (일봉 Cohort B 사용): {e}")
    return None, None, None


def fetch_all_market_data(cfg: StrategyConfig = DEFAULT_CONFIG) -> MarketDataBundle:
    """전략 판정에 필요한 모든 원천 데이터를 한 번에 수집합니다."""
    logger.info("원천 시장 데이터(Yahoo Finance & FRED) 수집 시작...")

    # 1. 주식/ETF (배당+분할 조정)
    use_adj = cfg.use_adjusted_price and (cfg.adjusted_mode == "dividends")
    df_tqqq = fetch_yf_daily(cfg.target_symbol, period="5y", auto_adjust=use_adj)
    df_qqq = fetch_yf_daily(cfg.reference_symbol, period="5y", auto_adjust=use_adj)
    df_spy = fetch_yf_daily(cfg.spy_symbol, period="5y", auto_adjust=use_adj)

    # 2. QQQ RSI용 분할조정(splits-only) 종가
    if cfg.use_tv_default_rsi:
        df_qqq_split = fetch_yf_daily(cfg.reference_symbol, period="5y", auto_adjust=False)
        qqq_rsi_close = _normalize_date_index(df_qqq_split["Close"])
    else:
        qqq_rsi_close = _normalize_date_index(df_qqq["Close"])

    # 3. 나스닥 종합 거래량, VIX, DXY, USD/KRW
    df_ixic = fetch_yf_daily(cfg.nasdaq_vol_symbol, period="2y", auto_adjust=False)
    df_vix = fetch_yf_daily(cfg.vix_symbol, period="5y", auto_adjust=False)
    df_dxy = fetch_yf_daily(cfg.dxy_symbol, period="5y", auto_adjust=False)
    df_krw = fetch_yf_daily(cfg.usdkrw_symbol, period="1y", auto_adjust=False)

    # 4. FRED 국채 금리 4종
    dgs6m = fetch_fred_series(cfg.fred_dgs6m)
    dgs2 = fetch_fred_series(cfg.fred_dgs2)
    dgs10 = fetch_fred_series(cfg.fred_dgs10)
    dgs20 = fetch_fred_series(cfg.fred_dgs20)

    # 5. 장마감 시각(16:00 ET) 스냅샷 (VIX 5분봉 & DXY 1시간봉 Cohort A)
    vix_primary_map = fetch_vix_primary_close_map(cfg.vix_symbol)
    dxy_a_c, dxy_a_ref, dxy_a_date = fetch_dxy_cohort_a_snapshot(cfg.dxy_symbol, cfg.t3_dxy_roc_len)

    logger.info(
        f"데이터 수집 완료: TQQQ 최신일={df_tqqq.index[-1]}, "
        f"DGS2 최신일={dgs2.index[-1] if not dgs2.empty else 'N/A'}"
    )

    return MarketDataBundle(
        tqqq_close=_normalize_date_index(df_tqqq["Close"]),
        tqqq_high=_normalize_date_index(df_tqqq["High"]),
        tqqq_low=_normalize_date_index(df_tqqq["Low"]),
        qqq_close=_normalize_date_index(df_qqq["Close"]),
        qqq_rsi_close=qqq_rsi_close,
        spy_close=_normalize_date_index(df_spy["Close"]),
        ixic_volume=_normalize_date_index(df_ixic["Volume"]),
        vix_close=_normalize_date_index(df_vix["Close"]),
        dxy_daily_close=_normalize_date_index(df_dxy["Close"]),
        usdkrw_close=_normalize_date_index(df_krw["Close"]),
        dgs6m=dgs6m,
        dgs2=dgs2,
        dgs10=dgs10,
        dgs20=dgs20,
        vix_primary_close_by_date=vix_primary_map,
        dxy_cohort_a_current=dxy_a_c,
        dxy_cohort_a_ref=dxy_a_ref,
        dxy_cohort_a_date=dxy_a_date,
        tqqq_open=_normalize_date_index(df_tqqq["Open"]),
    )
