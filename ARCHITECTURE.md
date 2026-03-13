# WeatherBench X Architecture Guide

This document provides a deep dive into the modular design of WeatherBench X, a framework built for flexible, scalable, and physically consistent evaluation of weather models.

---

## 1. Architectural Philosophy
WeatherBench X is designed around the principle of **functional decoupling**. Instead of a monolithic "compute RMSE" function, the framework breaks the evaluation pipeline into independent, pluggable stages:
1.  **Loading & Alignment:** Fetching and regridding data.
2.  **Statistic Computation:** Calculating raw, additive values from prediction/target pairs.
3.  **Aggregation:** Reducing dimensions (time/space) using weights and bins.
4.  **Metric Finalization:** Deriving final scores from aggregated statistics.

This separation allows for massive scalability (using Apache Beam/Dask) and extreme research flexibility.

---

## 2. Core Modules & Abstractions

### A. Data Loading (`weatherbenchX.data_loaders`)
The `DataLoader` is the entry point for data. Its primary role is to provide chunks of data that are broadcastable against each other.
*   **Data Sources & Storage:**
    *   **Google Cloud Storage (GCS):** Natively supports GCS buckets (e.g., `gs://weatherbench2/datasets`) via `xarray`'s GCS backends.
    *   **Local Filesystem:** Standard local paths are supported for all file types.
    *   **In-Memory:** Can pass an already-opened `xarray.Dataset` directly via the `ds` parameter.
*   **Supported File Types:**
    *   **Zarr (`.zarr`):** Primary format for gridded datasets (e.g., ERA5, ML model outputs).
    *   **NetCDF/GRIB:** Any format supported by `xarray.open_dataset` (e.g., `.nc`, `.grib`).
    *   **Parquet (`.parquet`):** Used for tabular/sparse observation data (e.g., METAR station data).
*   **Internal Representation:** Regardless of the source or format, all data is eventually loaded into **`xarray`** objects. Sparse data is read into `pandas.DataFrame` for filtering before conversion to an `xarray.Dataset` (usually with an `index` dimension).
*   **Key Classes:**
    *   `XarrayDataLoader`: Standard loader for gridded data.
    *   `SparseParquetDataLoader`: Optimized for point observations (METAR).
*   **Flexibility:** Loaders handle on-the-fly tasks like dimension renaming (ECMWF standard: `init_time`, `lead_time`, `latitude`, `longitude`), adding NaN masks, computing derived variables, and calling interpolation logic before the data reaches the metrics engine.

### B. Metrics & Statistics (`weatherbenchX.metrics`)
WeatherBench X makes a critical distinction between a **Metric** and a **Statistic**.
*   **`Statistic`:** A function of a prediction/target pair that produces an additive value (e.g., Squared Error, Absolute Error). It preserves dimensions that will be reduced later (like time or space).
*   **`Metric`:** A higher-level object that specifies which `Statistics` it needs and how to compute the final value from their **means**.
    *   *Example:* The `RMSE` metric requests the `SquaredError` statistic. Once the mean squared error is calculated across all data, the `RMSE` metric simply takes the square root.
*   **`PerVariableMetric/Statistic`:** Specialized bases that automatically map computations across all variables in a dataset.

### C. Aggregation (`weatherbenchX.aggregation`)
Aggregation is the process of reducing high-dimensional statistics into a final score.
*   **`Aggregator`:** Defines *how* to reduce data. It takes a list of dimensions to reduce (`reduce_dims`), a list of `Weighting` objects, and a list of `Binning` objects.
*   **`AggregationState`:** The "engine" of the framework. It stores two things: `sum_weighted_statistics` and `sum_weights`.
    *   Because it stores sums rather than means, `AggregationState` objects can be added together (`state_A + state_B`). This is what allows WeatherBench X to run on distributed systems like Apache Beam—each worker computes a local sum, and they are all combined at the end.

### D. Binning (`weatherbenchX.binning`)
Binning allows for "disaggregated" evaluation—seeing how a model performs in specific sub-groups.
*   **Spatial Bins:** `LandSea` (land vs. ocean), `Regions` (e.g., Tropics, Extratropics), `LatitudeBins`.
*   **Temporal Bins:** `ByTimeUnit` (e.g., performance by hour of day or month of year).
*   **Coordinate Bins:** `BySets` (e.g., performance for a specific set of weather stations).

### E. Weighting (`weatherbenchX.weighting`)
Ensures that data points contribute correctly to the final average.
*   **`GridAreaWeighting`:** The most common weighting, which adjusts for the decreasing area of grid boxes towards the poles in a lat/lon grid (using a `cos(latitude)` factor).

### F. Interpolation (`weatherbenchX.interpolations`)
Handles the spatial mapping between models and targets.
*   **`InterpolateToReferenceCoords`:** Maps a prediction grid to the target grid.
*   **`GridToSparseWithAltitudeAdjustment`:** A sophisticated interpolator for station data that adjusts temperature and wind based on the elevation difference between the model's orography and the actual station height.

---

## 3. The Evaluation Lifecycle

1.  **Instantiate Components:** Define your `DataLoader`, your list of `Metrics`, and your `Aggregator` (with desired bins/weights).
2.  **Chunking:** The system (via `beam_pipeline` or a simple loop) iterates through time chunks.
3.  **Compute:** For each chunk:
    *   `Statistic.compute(pred, target)` is called.
    *   `Aggregator.aggregate_statistics(stats)` applies weights/bins and returns an `AggregationState`.
4.  **Merge:** All `AggregationState` objects are summed into a single global state.
5.  **Finalize:** `AggregationState.metric_values(metrics)` is called. This:
    *   Divides `sum_weighted_statistics` by `sum_weights` to get the mean.
    *   Calls `Metric.values_from_mean_statistics` to get the final scores.

---

## 4. Summary of Flexibility
*   **Scalability:** The `AggregationState` design means the memory footprint remains constant regardless of the total time period being evaluated.
*   **Extensibility:** Adding a new metric only requires defining a new `Statistic` class and a simple `values_from_mean_statistics` formula.
*   **Consistency:** By centralizing weights and bins in the `Aggregator`, the framework ensures that different metrics (RMSE, Bias, ACC) are all calculated over the exact same data points and regions.
