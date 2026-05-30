from feast import FeatureView, Entity, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

# Data source
aqi_source = FileSource(
    path="data/karachi_features.parquet",
    timestamp_field="time"
)

# Entity
aqi_entity = Entity(
    name="aqi_id",
    join_keys=["aqi_id"],
    value_type=None
)

# Feature View
aqi_fv = FeatureView(
    name="aqi_features",
    entities=[aqi_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="temperature_2m", dtype=Float32),
        Field(name="relative_humidity_2m", dtype=Float32),
        Field(name="wind_speed_10m", dtype=Float32),
        Field(name="aqi", dtype=Float32),
        Field(name="aqi_change", dtype=Float32),
        Field(name="aqi_lag_1", dtype=Float32),
        Field(name="aqi_lag_2", dtype=Float32),
        Field(name="aqi_lag_3", dtype=Float32),
        Field(name="aqi_rolling_3h", dtype=Float32),
        Field(name="aqi_rolling_6h", dtype=Float32),
        Field(name="future_aqi", dtype=Float32),
        Field(name="hour", dtype=Int64),
        Field(name="day", dtype=Int64),
        Field(name="month", dtype=Int64),
        Field(name="weekday", dtype=Int64),
    ],
    source=aqi_source
)