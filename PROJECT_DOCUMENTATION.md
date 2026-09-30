# Chemistry project — end-to-end documentation

## 1) What this project does (high level)

This project builds a small ML pipeline + web app that predicts:

- **Absorbance** of a chemistry sample (a regression problem), and
- **Solution color** as an **RGB triplet** (another regression problem),

from two user inputs:

- **Volume** (mL)
- **Concentration**

The workflow is:

1. Start from an Excel dataset (`chem_excel_final.xlsx`) that contains experimental rows with volume labels (like `1ml`, `2ml`, …), concentration, absorbance, and color values (HEX + R/G/B).
2. Clean/prepare the data and train:
   - Model A: \(f(volume, conc) \rightarrow absorbance\)
   - Model B: \(g(volume, conc, predicted\_absorbance) \rightarrow (R,G,B)\)
3. Save trained artifacts to `artifacts/` (joblib files).
4. Serve predictions through a **Streamlit** app (`web_app.py`) that:
   - predicts absorbance, then
   - predicts RGB color and renders a HEX color swatch.

Separately, there is a notebook that recreates “comparison” line charts from an Excel workbook (`comparisons.xlsx`).

## 2) Project contents

### 2.1 Data files

- `chem_excel_final.xlsx`
  - Main dataset used for EDA + training.
  - Columns (as used in the ML notebook) are normalized to:
    - `name` (volume label like `1ml`, `2ml`, `3ml`, `4ml`, `5ml`)
    - `conc` (concentration)
    - `wl_nm` (wavelength nm; appears constant \(640\) in EDA output)
    - `abs` (absorbance)
    - `hex` (HEX color like `#0093C9`)
    - `r`, `g`, `b` (RGB channels)
    - other columns like `photos`, `hsl`
- `comparisons.xlsx`
  - A workbook containing “Actual vs Predicted absorbance” values already filled in.
  - Used by `comparisons_plots_replicate.ipynb` to recreate the charts programmatically.

### 2.2 Notebooks

- `chem_excel_eda.ipynb`
  - Loads `chem_excel_final.xlsx`
  - Cleans duplicates, inspects schema, and performs EDA plots (seaborn/plotly).
- `chemistry_ml_project.ipynb`
  - The core ML notebook.
  - Cleans the dataset, trains **two models**, evaluates them, and saves `joblib` artifacts.
- `comparisons_plots_replicate.ipynb`
  - Recreates line charts from `comparisons.xlsx` (and generates some visual color comparisons).

### 2.3 App

- `web_app.py`
  - A Streamlit app titled **“Chemistry Color Predictor”**
  - Loads trained artifacts from `artifacts/`
  - Takes `volume_ml` + `concentration`
  - Predicts absorbance, then predicts RGB and displays a HEX swatch.

### 2.4 Dependencies

- `requirements.txt` contains:
  - `joblib`, `numpy`, `pandas`, `scikit-learn`, `streamlit`, `openpyxl`
  - Note: the EDA / plotting notebooks also use `matplotlib`, `seaborn`, and `plotly` (these are imported in notebooks even if not listed in `requirements.txt`).

## 3) End-to-end data flow (detailed)

### Step A — EDA (`chem_excel_eda.ipynb`)

1. Loads `chem_excel_final.xlsx` (sheet `Sheet1`).
2. Prints shape and column types; example columns seen:
   - `name`, `conc`, `wl(nm)`, `abs`, `HEX`, `R`, `G`, `B`, `HSL` (and an `Unnamed: 0` column).
3. Removes duplicates and explores distributions/relationships with plots.

Output: exploratory plots and basic dataset understanding (no model artifacts).

### Step B — Model training (`chemistry_ml_project.ipynb`)

#### B1) Data loading

- Reads `chem_excel_final.xlsx` and renames columns into a consistent set:
  - `blank`, `name`, `conc`, `wl_nm`, `abs`, `photos`, `hex`, `r`, `g`, `b`, `hsl`

#### B2) Feature engineering and cleaning rules

The notebook builds the following derived columns:

- **`volume_label`**: cleaned string version of `name` (e.g., `5ml`)
- **`volume_ml`**: numeric value extracted from `volume_label` via regex (e.g., `5.0`)

Then it converts `conc`, `abs`, and `r/g/b` to numeric and validates `hex` with:

- HEX must match `^#[0-9A-F]{6}$` (upper-cased)

Rows are kept only if all of these are present and valid:

- `volume_ml`, `conc`, `abs`, `hex`, and all of `r/g/b`

Finally, it removes an **obvious absorbance outlier/typo**:

- filters `abs <= 10` (the dataset contains a value `52` that would dominate training)

The notebook prints a quick summary like:

- raw rows: **101**
- usable rows after cleaning: **90**
- volumes present: `['1ml', '2ml', '3ml', '4ml', '5ml']`
- absorbance range after cleaning: ~`0.09` to `2.0`

#### B3) Train/test split

It defines base feature columns:

- `feature_cols = ["volume_ml", "conc"]`

Then it splits:

- inputs: `X = clean[["volume_ml", "conc"]]`
- targets:
  - `y_abs = clean["abs"]`
  - `y_rgb = clean[["r", "g", "b"]]`

And runs:

- `train_test_split(..., test_size=0.2, random_state=42)`

#### B4) Model A — Absorbance regressor

- Model: `RandomForestRegressor(n_estimators=400, random_state=42, ...)`
- Predicts absorbance for the test set: `abs_pred = abs_model.predict(X_test)`
- Evaluates with:
  - `MAE`
  - `RMSE`
  - `R²`

#### B5) Model B — RGB regressor (colour)

This model uses a *two-stage* approach:

1. Use Model A to generate absorbance predictions for train/test:
   - `train_abs_pred = abs_model.predict(X_train)`
   - `test_abs_pred = abs_model.predict(X_test)`
2. Build color model features:
   - `color_feature_cols = ["volume_ml", "conc", "abs_pred"]`
   - `X_rgb_train = X_train.assign(abs_pred=train_abs_pred)`
   - `X_rgb_test = X_test.assign(abs_pred=test_abs_pred)`
3. Train:
   - `rgb_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=500, random_state=42, ...))`
4. Predict RGB and evaluate (overall + per-channel):
   - `MAE`, `RMSE`, `R²`

#### B6) Saved artifacts

The notebook creates (if you run the “save” cell) an `artifacts/` directory and saves:

- `artifacts/absorbance_model.joblib`
- `artifacts/colour_model_rgb.joblib`
- `artifacts/model_metadata.joblib`

Metadata written includes keys like:

- `abs_feature_cols`
- `color_feature_cols`
- `clean_rows`
- `volume_labels` (unique volume labels seen in training, e.g. `1ml..5ml`)

**Important:** In the current workspace snapshot, `artifacts/` is not present yet, which implies the training notebook’s “save artifacts” cell likely hasn’t been run (or outputs were deleted). The Streamlit app requires these files.

### Step C — Streamlit app (`web_app.py`)

#### C1) What it loads

The app expects:

- `artifacts/absorbance_model.joblib`
- `artifacts/colour_model_rgb.joblib`
- `artifacts/model_metadata.joblib` (optional but used to show “known volumes” and suggested volume range)

It caches model loading with `@st.cache_resource`.

#### C2) Prediction pipeline in the app

Given user inputs:

- `volume_ml`
- `concentration`

It runs:

1. `predicted_absorbance = abs_model.predict([[volume_ml, conc]])`
2. Builds a second-row feature frame including `abs_pred = predicted_absorbance`
3. Predicts `predicted_rgb = rgb_model.predict([volume_ml, conc, abs_pred])`
4. Converts RGB → HEX and displays a color swatch.

The app warns if the user’s volume is outside the training range inferred from metadata.

## 4) Comparison charts workflow (`comparisons_plots_replicate.ipynb`)

This notebook does **not** train models. Instead it:

1. Loads `comparisons.xlsx` via `openpyxl.load_workbook(..., data_only=True)`.
2. Iterates over the rows and extracts a dataframe with columns like:
   - `volume` (e.g. `1ml`)
   - `conc`
   - `wl_nm`
   - `actual_abs`
   - `predicted_abs` (already present in the Excel sheet)
   - `actual_colour` (HEX)
3. Plots, per volume, a line chart of:
   - actual absorbance vs concentration
   - predicted absorbance vs concentration
   - and reports average absolute error per volume.
4. For a visual color comparison, it creates a continuous mapping:
   - builds an “absorbance → color” colormap using `actual_abs` + `actual_colour` anchor points,
   - then maps `predicted_abs` through that colormap to produce a `predicted_colour` HEX.

So, in this notebook, **color is derived from predicted absorbance via interpolation**, not from the RGB model used in the Streamlit app.

## 5) How to run (end-to-end)

### 5.1 Install dependencies

From the project folder:

```bash
pip install -r requirements.txt
```

If the EDA/comparison notebooks fail due to missing plotting libraries, install:

```bash
pip install matplotlib seaborn plotly
```

### 5.2 Train models and generate artifacts

Open and run:

- `chemistry_ml_project.ipynb`

Run all cells through the “save artifacts” section. After completion you should have:

- `artifacts/absorbance_model.joblib`
- `artifacts/colour_model_rgb.joblib`
- `artifacts/model_metadata.joblib`

### 5.3 Run the web app

From the project folder:

```bash
streamlit run web_app.py
```

If you see an error like “Model files are missing…”, go back to **5.2** and run the training notebook save cell.

### 5.4 Recreate comparison plots

Open and run:

- `comparisons_plots_replicate.ipynb`

This reads `comparisons.xlsx` and generates the charts in-notebook.

## 6) Outputs you should expect

- **EDA notebook**: tables + plots describing distributions, duplicates, missingness, and relationships.
- **ML notebook**:
  - printed metrics for absorbance and RGB models (MAE/RMSE/R²),
  - saved `joblib` artifacts under `artifacts/`.
- **Streamlit app**:
  - predicted absorbance (float)
  - predicted RGB (integers 0–255)
  - predicted HEX and a visual swatch
- **Comparison plots notebook**:
  - per-volume “actual vs predicted absorbance” line plots,
  - a sample table including `predicted_colour` derived from absorbance interpolation.

## 7) Key assumptions / constraints in the current implementation

- **Volume parsing** assumes volume strings contain a numeric value (e.g. `5ml` → `5.0`).
- The dataset currently appears to contain volumes in the set `1ml..5ml` (from training summary).
- Color model training uses **predicted absorbance** (not measured absorbance) as an input feature to simulate the app’s two-stage prediction.
- The notebook explicitly removes absorbance values \(> 10\) to avoid a single extreme outlier (`52`) dominating the regression.
- The Streamlit app is only as good as the artifacts present in `artifacts/`.

## 8) Troubleshooting

- **Streamlit error: model files missing**
  - Run the artifact-saving cell in `chemistry_ml_project.ipynb` to create `artifacts/*.joblib`.
- **Volume out of range warning**
  - Use volumes within the training metadata range; predictions outside it may extrapolate poorly.
- **Plots/notebooks error: missing seaborn/plotly/matplotlib**
  - Install extra plotting dependencies (see 5.1).

