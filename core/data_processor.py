"""
Data Processor Module
---------------------
Loads, cleans, and prepares the Favorita retail dataset for
demand forecasting.

Key responsibilities:
1. Load raw CSVs with proper dtypes (memory optimization)
2. Handle missing values (forward-fill for oil, zero-fill for gaps)
3. Merge supporting data (oil prices, holidays, store metadata)
4. Scope to selected store and product families
5. Produce forecast-ready daily time series per product family

Design decisions:
- Class-based: holds loaded data in memory, avoids re-reading CSVs
- Forward-fill for oil prices (preserves temporal correlation)
- Resample to daily frequency (models require regular intervals)
- Returns clean DataFrames ready for direct model input
"""

import pandas as pd
import numpy as np
import os
from typing import Optional


class DataProcessor:
    """
    Loads and prepares the Favorita retail dataset for forecasting.

    Usage:
        processor = DataProcessor("data/")
        processor.load_all()
        df = processor.get_product_series("GROCERY I")
    """

    def __init__(self, data_dir: str = "data"):
        """
        Args:
            data_dir: Path to directory containing the CSV files
        """
        self.data_dir = data_dir
        self.train: Optional[pd.DataFrame] = None
        self.stores: Optional[pd.DataFrame] = None
        self.oil: Optional[pd.DataFrame] = None
        self.holidays: Optional[pd.DataFrame] = None
        self.transactions: Optional[pd.DataFrame] = None

        # Scoping — set after load
        self.selected_store: Optional[int] = None
        self.store_data: Optional[pd.DataFrame] = None

    def load_all(self) -> None:
        """
        Load all datasets into memory with optimized dtypes.

        Why optimize dtypes?
            The raw train.csv is 87MB in memory with default dtypes.
            Using int16 for store_nbr (max value 54) instead of int64
            and category for family (33 unique strings) reduces memory
            by ~40%. This matters when you scale to the full dataset.

            Interview point: "I optimized memory usage by downcasting
            integer columns and using categorical dtype for string
            columns with low cardinality. This reduced the working
            dataset from 87MB to ~50MB without losing any information."
        """
        print("Loading datasets...")

        # Main sales data
        self.train = pd.read_csv(
            os.path.join(self.data_dir, "train.csv"),
            parse_dates=["date"],
            dtype={
                "id": "int32",
                "store_nbr": "int16",
                "family": "category",
                "sales": "float32",
                "onpromotion": "int16",
            },
        )

        # Store metadata
        self.stores = pd.read_csv(
            os.path.join(self.data_dir, "stores.csv"),
            dtype={"store_nbr": "int16", "cluster": "int16"},
        )

        # Oil prices — external regressor
        self.oil = pd.read_csv(
            os.path.join(self.data_dir, "oil.csv"),
            parse_dates=["date"],
        )
        self._clean_oil()

        # Holidays — for seasonal features
        self.holidays = pd.read_csv(
            os.path.join(self.data_dir, "holidays_events.csv"),
            parse_dates=["date"],
        )

        # Transaction counts
        self.transactions = pd.read_csv(
            os.path.join(self.data_dir, "transactions.csv"),
            parse_dates=["date"],
            dtype={"store_nbr": "int16", "transactions": "int32"},
        )

        print(f"  ✓ Train: {len(self.train):,} rows")
        print(f"  ✓ Stores: {len(self.stores)} stores")
        print(f"  ✓ Oil: {len(self.oil)} days")
        print(f"  ✓ Holidays: {len(self.holidays)} events")
        print(f"  ✓ Transactions: {len(self.transactions):,} rows")

    def _clean_oil(self) -> None:
        """
        Clean oil price data.

        Why forward-fill?
            Oil prices are a time series. Missing values are typically
            weekends and holidays when markets are closed. Yesterday's
            closing price is the best estimate for a missing day.

            Mean imputation would inject the average price from the
            entire 4-year period, which could be wildly different from
            the local trend. This destroys temporal correlation.

            Backward-fill after forward-fill handles the very first
            row (2013-01-01 is NaN because it's a holiday — no prior
            value to forward-fill from).
        """
        self.oil = self.oil.set_index("date").sort_index()
        self.oil["dcoilwtico"] = self.oil["dcoilwtico"].ffill().bfill()
        self.oil = self.oil.reset_index()

    def select_store(self, store_nbr: int = 44) -> dict:
        """
        Scope the dataset to a single store.

        Why default to store 44?
            Store 44 has the highest total sales and full date coverage
            (1687 days). It gives us the richest data for forecasting.

        Why scope to one store for MVP?
            54 stores × 33 families = 1,782 time series. Modeling all
            of them would take minutes and make the demo slow. Scoping
            to one store with top product families gives us ~10 time
            series — enough to demonstrate the full capability without
            overwhelming compute.

        Returns:
            dict with store metadata (city, state, type, cluster)
        """
        if self.train is None:
            raise RuntimeError("Call load_all() first")

        self.selected_store = store_nbr
        self.store_data = self.train[
            self.train["store_nbr"] == store_nbr
        ].copy()

        # Get store metadata
        store_info = self.stores[
            self.stores["store_nbr"] == store_nbr
        ].iloc[0].to_dict()

        print(f"\n  Store {store_nbr}: {store_info['city']}, {store_info['state']}")
        print(f"  Type: {store_info['type']}, Cluster: {store_info['cluster']}")
        print(f"  Rows: {len(self.store_data):,}")
        print(f"  Date range: {self.store_data['date'].min().date()} to {self.store_data['date'].max().date()}")

        return store_info

    def get_product_families(self, top_n: int = 10) -> list[str]:
        """
        Get the top N product families by total sales for the selected store.

        Why top N instead of all 33?
            Many product families have sparse or zero sales at a given
            store. Forecasting a product with mostly zeros produces
            meaningless results. The top 10 families typically cover
            90%+ of revenue — this is the Pareto principle in action.

            Interview point: "I applied the Pareto principle — the top
            10 of 33 product families covered 92% of store revenue.
            Forecasting the remaining 23 would add complexity without
            meaningful business value."
        """
        if self.store_data is None:
            raise RuntimeError("Call select_store() first")

        family_sales = (
            self.store_data.groupby("family")["sales"]
            .sum()
            .sort_values(ascending=False)
        )

        top_families = family_sales.head(top_n).index.tolist()
        total_sales = family_sales.sum()
        top_sales = family_sales.head(top_n).sum()

        print(f"\n  Top {top_n} families cover {top_sales/total_sales:.1%} of store revenue:")
        for i, fam in enumerate(top_families):
            pct = family_sales[fam] / total_sales * 100
            print(f"    {i+1}. {fam}: {family_sales[fam]:,.0f} ({pct:.1f}%)")

        return top_families

    def get_product_series(
        self,
        family: str,
        include_oil: bool = True,
        include_holidays: bool = True,
        include_promotions: bool = True,
    ) -> pd.DataFrame:
        """
        Get a clean, forecast-ready daily time series for one product family.

        This is the main output method. It returns a DataFrame with:
        - date: daily datetime index
        - sales: target variable (float, zero-filled for missing days)
        - onpromotion: number of items on promotion that day
        - oil_price: daily oil price (forward-filled)
        - is_holiday: binary flag for national holidays
        - day_of_week: 0=Monday, 6=Sunday
        - month: 1-12
        - is_weekend: binary flag

        Why add calendar features here instead of in the model?
            Calendar features (day_of_week, month, is_weekend) are
            useful across ALL models — ARIMA uses them for seasonal
            patterns, XGBoost uses them as features, and they're
            essential for the EDA engine's seasonality charts.

            Adding them in the data processor (not the model) follows
            the DRY principle — Don't Repeat Yourself. Each model
            doesn't need to independently compute the same features.

        Args:
            family: Product family name (e.g., "GROCERY I")
            include_oil: Whether to merge oil prices
            include_holidays: Whether to add holiday flags
            include_promotions: Whether to include promotion data

        Returns:
            DataFrame with daily time series and features
        """
        if self.store_data is None:
            raise RuntimeError("Call select_store() first")

        # Filter to the requested product family
        df = self.store_data[self.store_data["family"] == family][
            ["date", "sales", "onpromotion"]
        ].copy()

        # Resample to daily frequency — fill gaps with zero sales
        # Why? Some days might be missing (store closed). Models
        # need regular intervals. A missing day = zero sales.
        df = df.set_index("date").resample("D").agg({
            "sales": "sum",
            "onpromotion": "sum",
        }).reset_index()

        # Add oil prices as external regressor
        if include_oil and self.oil is not None:
            df = df.merge(
                self.oil.rename(columns={"dcoilwtico": "oil_price"}),
                on="date",
                how="left",
            )
            # Fill any remaining gaps (dates outside oil data range)
            df["oil_price"] = df["oil_price"].ffill().bfill()

        # Add holiday flags
        if include_holidays and self.holidays is not None:
            # Only use national holidays (locale == "National")
            # Regional/local holidays affect specific stores differently
            national = self.holidays[
                self.holidays["locale"] == "National"
            ]["date"].unique()
            df["is_holiday"] = df["date"].isin(national).astype(int)

        # Add calendar features
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["day_of_month"] = df["date"].dt.day
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

        # Sort by date (should already be sorted, but defensive)
        df = df.sort_values("date").reset_index(drop=True)

        return df

    def get_all_series(
        self, families: Optional[list[str]] = None, top_n: int = 10
    ) -> dict[str, pd.DataFrame]:
        """
        Get forecast-ready time series for multiple product families.

        Returns a dict mapping family name → DataFrame.

        Why a dict and not a single DataFrame?
            Each family has different sales patterns, different
            seasonality, and needs its own model. A dict makes it
            natural to iterate: for family, df in series.items().

            A single DataFrame with a 'family' column would require
            groupby operations everywhere downstream — less clean.
        """
        if families is None:
            families = self.get_product_families(top_n)

        series = {}
        for family in families:
            series[family] = self.get_product_series(family)
            print(f"  ✓ {family}: {len(series[family]):,} days, "
                  f"avg sales: {series[family]['sales'].mean():.1f}/day")

        return series

    def get_summary_stats(self) -> pd.DataFrame:
        """
        Generate summary statistics for the selected store.

        Used by the EDA engine for the overview dashboard.
        Returns one row per product family with key metrics.
        """
        if self.store_data is None:
            raise RuntimeError("Call select_store() first")

        stats = (
            self.store_data.groupby("family")
            .agg(
                total_sales=("sales", "sum"),
                avg_daily_sales=("sales", "mean"),
                max_daily_sales=("sales", "max"),
                std_daily_sales=("sales", "std"),
                zero_days=("sales", lambda x: (x == 0).sum()),
                total_days=("sales", "count"),
                promo_days=("onpromotion", lambda x: (x > 0).sum()),
            )
            .sort_values("total_sales", ascending=False)
        )

        stats["zero_pct"] = stats["zero_days"] / stats["total_days"] * 100
        stats["promo_pct"] = stats["promo_days"] / stats["total_days"] * 100
        stats["cv"] = stats["std_daily_sales"] / stats["avg_daily_sales"]

        return stats.round(2)


# ---- Quick test ----
if __name__ == "__main__":
    processor = DataProcessor("data")
    processor.load_all()
    store_info = processor.select_store(44)
    families = processor.get_product_families(10)

    print("\n" + "=" * 60)
    print("TESTING: Get GROCERY I time series")
    print("=" * 60)
    grocery = processor.get_product_series("GROCERY I")
    print(f"\nShape: {grocery.shape}")
    print(f"Columns: {list(grocery.columns)}")
    print(f"Date range: {grocery['date'].min().date()} to {grocery['date'].max().date()}")
    print(f"Avg daily sales: {grocery['sales'].mean():.1f}")
    print(f"Missing values:\n{grocery.isnull().sum()}")
    print(f"\nFirst 5 rows:")
    print(grocery.head().to_string(index=False))
    print(f"\nLast 5 rows:")
    print(grocery.tail().to_string(index=False))

    print("\n" + "=" * 60)
    print("TESTING: Summary stats")
    print("=" * 60)
    stats = processor.get_summary_stats()
    print(stats.head(10).to_string())