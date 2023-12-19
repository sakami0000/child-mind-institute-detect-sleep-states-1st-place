from itertools import chain

import polars as pl


class FeatureCreator:
    def __init__(self, signal_names: list[str], timeseries_cols: list[str]) -> None:
        self.signal_names = signal_names
        self.timeseries_cols = timeseries_cols

    def create(self, df: pl.DataFrame) -> pl.DataFrame:
        return df


class LIDFeatureCreator(FeatureCreator):
    def __init__(
        self, signal_names: list[str], timeseries_cols: list[str], window_size: int = 120
    ) -> None:
        self.signal_names = signal_names
        self.timeseries_cols = timeseries_cols
        self.window_size = window_size

    def create(self, df: pl.DataFrame) -> pl.DataFrame:
        # sum over 10minutes
        df = df.with_columns(
            pl.max_horizontal(pl.col("enmo") - 0.02, 0)
            .rolling_sum(self.window_size)
            .over(self.timeseries_cols)
            .alias("activity_count")
            .backward_fill()
        ).with_columns((100 / (pl.col("activity_count") + 1)).alias("lid"))
        return df


class TimeSeriesStatsFeatureCreator:
    def __init__(
        self, signal_names: list[str], window_sizes: list[int], timeseries_cols: list[str]
    ):
        self.signal_names = signal_names
        self.window_sizes = window_sizes
        self.timeseries_cols = timeseries_cols

    def create(self, df: pl.DataFrame) -> pl.DataFrame:
        agg_cols = list(
            chain.from_iterable(
                [
                    [
                        pl.col(column)
                        .rolling_mean(window_size, center=True, min_periods=1)
                        .over(self.timeseries_cols)
                        .name.suffix(f"_{window_size}_rolling_mean"),
                        pl.col(column)
                        .rolling_std(window_size, center=True, min_periods=1)
                        .over(self.timeseries_cols)
                        .name.suffix(f"_{window_size}_rolling_std"),
                        pl.col(column)
                        .rolling_max(window_size, center=True, min_periods=1)
                        .over(self.timeseries_cols)
                        .name.suffix(f"_{window_size}_rolling_max"),
                    ]
                    for window_size in self.window_sizes
                    for column in self.signal_names
                ]
            )
        )
        df = df.with_columns(agg_cols)
        return df


class TimeSeriesStatsDiffFeatureCreator:
    def __init__(
        self, signal_names: list[str], window_sizes: list[int], timeseries_cols: list[str]
    ):
        self.signal_names = signal_names
        self.window_sizes = window_sizes
        self.timeseries_cols = timeseries_cols

    def create(self, df: pl.DataFrame) -> pl.DataFrame:
        agg_cols = list(
            chain.from_iterable(
                [
                    [
                        (
                            pl.col(column)
                            .rolling_mean(window_size, min_periods=1)
                            .over(self.timeseries_cols)
                            - pl.col(column)
                        ).name.suffix(f"_diff_prev{window_size}_rolling_mean"),
                    ]
                    for window_size in self.window_sizes
                    for column in self.signal_names
                ]
            )
        )
        df = df.with_columns(agg_cols)
        agg_cols = list(
            chain.from_iterable(
                [
                    [
                        (
                            pl.col(column)
                            .rolling_mean(window_size, closed="right", min_periods=1)
                            .over(self.timeseries_cols)
                            - pl.col(column)
                        ).name.suffix(f"_diff_lead{window_size}_rolling_mean"),
                    ]
                    for window_size in self.window_sizes
                    for column in self.signal_names
                ]
            )
        )
        df = df.with_columns(agg_cols)
        return df


class TimeSeriesDiffFeatureCreator:
    def __init__(
        self, signal_names: list[str], window_sizes: list[int], timeseries_cols: list[str]
    ):
        self.signal_names = signal_names
        self.window_sizes = window_sizes
        self.timeseries_cols = timeseries_cols

    def create(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            [
                (pl.col(column) - pl.col(column).shift(1).over(self.timeseries_cols))
                .backward_fill()
                .name.suffix("_diff_lag")
                for column in self.signal_names
            ]
        )
        return df


class TimeFeatureCreator:
    def __init__(
        self, signal_names: list[str], window_sizes: list[int], timeseries_cols: list[str]
    ):
        self.signal_names = signal_names
        self.window_sizes = window_sizes
        self.timeseries_cols = timeseries_cols

    def create(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            (pl.col("timestamp").dt.hour() + pl.col("timestamp").dt.minute()).alias("time_idx"),
            pl.col("timestamp").dt.hour().alias("hour"),
            pl.col("timestamp").dt.minute().alias("minute"),
            (pl.col("timestamp").dt.minute() % 15).alias("minute_mod15"),
            pl.col("timestamp").dt.weekday().alias("weekday"),
        )
        return df
