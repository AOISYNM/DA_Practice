import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.special import boxcox1p

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="House Value Estimator", page_icon="🏡", layout="wide")

FONT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --paper: #EBE5D6;
    --paper-raised: #F4F0E5;
    --ink: #262319;
    --ink-soft: #5C5747;
    --brass: #A0702F;
    --brass-dark: #7C5423;
    --teal: #3E5C51;
    --line: #D7CDB4;
}

html, body, [class*="css"]  {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--ink);
}

.stApp {
    background-color: var(--paper);
}

h1, h2, h3, .estimate-figure {
    font-family: 'Fraunces', serif;
}

/* header */
.masthead {
    border-bottom: 2px solid var(--ink);
    padding-bottom: 1.1rem;
    margin-bottom: 1.6rem;
}
.masthead .eyebrow {
    font-size: 0.8rem;
    color: var(--brass-dark);
    letter-spacing: 0.02em;
    margin-bottom: 0.2rem;
}
.masthead h1 {
    font-size: 2.4rem;
    font-weight: 500;
    margin: 0;
    line-height: 1.1;
}
.masthead p {
    color: var(--ink-soft);
    margin-top: 0.4rem;
    max-width: 46ch;
}

/* section labels inside the form */
.section-label {
    font-family: 'Fraunces', serif;
    font-size: 1.05rem;
    font-weight: 500;
    color: var(--ink);
    border-bottom: 1px solid var(--line);
    padding-bottom: 0.35rem;
    margin: 1.6rem 0 0.9rem 0;
}
.section-label:first-child { margin-top: 0; }

/* result card */
.estimate-card {
    background: var(--paper-raised);
    border: 1px solid var(--line);
    border-radius: 2px;
    padding: 1.8rem 1.8rem 1.5rem 1.8rem;
    position: sticky;
    top: 1rem;
}
.estimate-card .tag {
    font-size: 0.78rem;
    color: var(--ink-soft);
    text-transform: none;
    border-left: 3px solid var(--brass);
    padding-left: 0.5rem;
}
.estimate-figure {
    font-size: 2.9rem;
    font-weight: 600;
    color: var(--teal);
    margin: 0.5rem 0 0.1rem 0;
    line-height: 1;
}
.estimate-range {
    color: var(--ink-soft);
    font-size: 0.9rem;
    margin-bottom: 1.1rem;
}
.estimate-divider {
    border: none;
    border-top: 1px solid var(--line);
    margin: 1.1rem 0;
}
.estimate-note {
    font-size: 0.82rem;
    color: var(--ink-soft);
    line-height: 1.5;
}
.estimate-placeholder {
    color: var(--ink-soft);
    font-size: 0.92rem;
    line-height: 1.6;
}

/* form widget tweaks */
div[data-testid="stForm"] {
    border: 1px solid var(--line);
    border-radius: 2px;
    background: var(--paper-raised);
    padding: 1.6rem 1.8rem;
}
.stButton>button, div[data-testid="stFormSubmitButton"] button {
    background-color: var(--ink);
    color: var(--paper);
    border: none;
    border-radius: 2px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
    padding: 0.55rem 1.4rem;
}
.stButton>button:hover, div[data-testid="stFormSubmitButton"] button:hover {
    background-color: var(--brass-dark);
    color: var(--paper);
}
</style>
"""
st.markdown(FONT_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Load artifacts
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    preprocessor = joblib.load("preprocessor.joblib")
    return model, preprocessor

model, pp = load_artifacts()


# ----------------------------------------------------------------------
# Preprocessing: raw single-row dict -> scaled feature row, using the
# exact fitted state captured at training time (preprocessor.joblib)
# ----------------------------------------------------------------------
def build_feature_row(raw: dict, pp: dict) -> pd.DataFrame:
    row = {}
    for col in pp["raw_numeric_cols"]:
        row[col] = raw.get(col, pp["raw_numeric_defaults"].get(col, 0))
    for col in pp["raw_categorical_cols"]:
        row[col] = raw.get(col, pp["raw_categorical_defaults"].get(col, ""))
    df = pd.DataFrame([row])

    # feature engineering (mirrors Main.ipynb's feature_engineering)
    df["TotalSF"] = df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"].fillna(0) + df["2ndFlrSF"].fillna(0)
    df["TotalPorchSF"] = (
        df["OpenPorchSF"].fillna(0) + df["EnclosedPorch"].fillna(0)
        + df["3SsnPorch"].fillna(0) + df["ScreenPorch"].fillna(0)
    )
    df["TotalBath"] = (
        df["FullBath"].fillna(0) + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0) + 0.5 * df["BsmtHalfBath"].fillna(0)
    )
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["IsRemod"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["IsNew"] = (df["YrSold"] == df["YearBuilt"]).astype(int)
    df["LotFrontage"] = df["LotFrontage"].fillna(pp["lot_frontage_median"])
    if "GarageYrBlt" in df.columns:
        df = df.drop("GarageYrBlt", axis=1)
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

    num = df.select_dtypes(include=["float64", "int64"]).copy()
    cat = df.select_dtypes(include=["object"]).copy()

    # skew handling (Box-Cox on the columns fitted from train)
    for col in pp["skewed_cols"]:
        if col in num.columns:
            num[col] = boxcox1p(num[col].astype(float), pp["boxcox_lambda"])

    # categorical fill + one-hot, aligned to training columns
    for col in cat.columns:
        if col in pp["none_cols"]:
            cat[col] = cat[col].fillna("0")
        if cat[col].isnull().any():
            cat[col] = cat[col].fillna(pp["cat_modes"].get(col, ""))
    cat_enc = pd.get_dummies(cat, dtype=int)

    final = pd.concat([num.reset_index(drop=True), cat_enc.reset_index(drop=True)], axis=1)
    final = final.reindex(columns=pp["final_columns"], fill_value=0)
    for c, med in pp["numeric_medians"].items():
        if c in final.columns:
            final[c] = final[c].fillna(med)

    return final


def predict_price(raw: dict) -> float:
    row = build_feature_row(raw, pp)
    scaled = pp["scaler"].transform(row)
    pred_log = model.predict(scaled)[0]
    return float(np.expm1(pred_log))


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="masthead">
        <div class="eyebrow">Ames, Iowa &nbsp;·&nbsp; residential sales model</div>
        <h1>House Value Estimator</h1>
        <p>Fill in the details of a property below. The estimate comes from a Lasso
        regression trained on ~1,460 historical Ames home sales.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

opts = pp["categorical_options"]

def choices(col, fallback):
    return opts.get(col, fallback)


# ----------------------------------------------------------------------
# Layout: form (left) + result (right)
# ----------------------------------------------------------------------
left, right = st.columns([1.5, 1], gap="large")

with left:
    with st.form("house_form"):
        st.markdown('<div class="section-label">Location &amp; lot</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            neighborhood = st.selectbox("Neighborhood", choices("Neighborhood", ["NAmes"]))
            ms_zoning = st.selectbox("Zoning", choices("MSZoning", ["RL"]))
            lot_area = st.number_input("Lot area (sq ft)", min_value=0, value=9000, step=100)
        with c2:
            lot_frontage = st.number_input("Lot frontage (ft)", min_value=0, value=70, step=1)
            bldg_type = st.selectbox("Building type", choices("BldgType", ["1Fam"]))
            house_style = st.selectbox("House style", choices("HouseStyle", ["2Story"]))

        st.markdown('<div class="section-label">Structure &amp; size</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            overall_qual = st.slider("Overall material & finish quality", 1, 10, 6)
            year_built = st.number_input("Year built", min_value=1870, max_value=2026, value=1995, step=1)
            year_remod = st.number_input("Year remodeled", min_value=1870, max_value=2026, value=1995, step=1)
            gr_liv_area = st.number_input("Above-ground living area (sq ft)", min_value=0, value=1500, step=50)
        with c4:
            overall_cond = st.slider("Overall condition", 1, 10, 5)
            total_bsmt_sf = st.number_input("Total basement area (sq ft)", min_value=0, value=800, step=50)
            first_flr_sf = st.number_input("1st floor area (sq ft)", min_value=0, value=1000, step=50)
            second_flr_sf = st.number_input("2nd floor area (sq ft)", min_value=0, value=500, step=50)

        st.markdown('<div class="section-label">Rooms &amp; amenities</div>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            full_bath = st.number_input("Full bathrooms", min_value=0, max_value=6, value=2)
            bsmt_full_bath = st.number_input("Basement full bathrooms", min_value=0, max_value=4, value=0)
            bedroom = st.number_input("Bedrooms above grade", min_value=0, max_value=10, value=3)
            tot_rms = st.number_input("Total rooms above grade", min_value=0, max_value=20, value=6)
        with c6:
            half_bath = st.number_input("Half bathrooms", min_value=0, max_value=4, value=1)
            bsmt_half_bath = st.number_input("Basement half bathrooms", min_value=0, max_value=4, value=0)
            fireplaces = st.number_input("Fireplaces", min_value=0, max_value=5, value=1)
            kitchen_qual = st.selectbox("Kitchen quality", choices("KitchenQual", ["TA"]))

        c7, c8 = st.columns(2)
        with c7:
            garage_cars = st.number_input("Garage capacity (cars)", min_value=0, max_value=6, value=2)
        with c8:
            garage_area = st.number_input("Garage area (sq ft)", min_value=0, value=480, step=20)

        st.markdown('<div class="section-label">Condition &amp; sale</div>', unsafe_allow_html=True)
        c9, c10 = st.columns(2)
        with c9:
            exter_qual = st.selectbox("Exterior quality", choices("ExterQual", ["TA"]))
            central_air = st.selectbox("Central air", choices("CentralAir", ["Y"]))
        with c10:
            sale_condition = st.selectbox("Sale condition", choices("SaleCondition", ["Normal"]))
            mo_sold = st.slider("Month sold", 1, 12, 6)

        yr_sold = st.number_input("Year sold", min_value=1990, max_value=2026, value=2010, step=1)

        submitted = st.form_submit_button("Estimate value")

with right:
    st.markdown('<div class="estimate-card">', unsafe_allow_html=True)
    if submitted:
        raw = {
            "Neighborhood": neighborhood,
            "MSZoning": ms_zoning,
            "LotArea": lot_area,
            "LotFrontage": lot_frontage,
            "BldgType": bldg_type,
            "HouseStyle": house_style,
            "OverallQual": overall_qual,
            "OverallCond": overall_cond,
            "YearBuilt": year_built,
            "YearRemodAdd": year_remod,
            "GrLivArea": gr_liv_area,
            "TotalBsmtSF": total_bsmt_sf,
            "1stFlrSF": first_flr_sf,
            "2ndFlrSF": second_flr_sf,
            "FullBath": full_bath,
            "HalfBath": half_bath,
            "BsmtFullBath": bsmt_full_bath,
            "BsmtHalfBath": bsmt_half_bath,
            "BedroomAbvGr": bedroom,
            "TotRmsAbvGrd": tot_rms,
            "Fireplaces": fireplaces,
            "KitchenQual": kitchen_qual,
            "GarageCars": garage_cars,
            "GarageArea": garage_area,
            "ExterQual": exter_qual,
            "CentralAir": central_air,
            "SaleCondition": sale_condition,
            "MoSold": mo_sold,
            "YrSold": yr_sold,
        }
        try:
            price = predict_price(raw)
            low, high = price * 0.92, price * 1.08
            st.markdown('<div class="tag">Estimated sale price</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="estimate-figure">${price:,.0f}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="estimate-range">Likely range ${low:,.0f} – ${high:,.0f}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<hr class="estimate-divider">', unsafe_allow_html=True)
            st.markdown(
                '<div class="estimate-note">Based on a Lasso regression fit on Ames, Iowa '
                'home sales. Typical error on held-out data is about $24,000, so treat this '
                'as a starting point rather than an appraisal.</div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Couldn't produce an estimate: {e}")
    else:
        st.markdown('<div class="tag">Estimated sale price</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="estimate-placeholder">Fill in the property details and press '
            '"Estimate value" to see a price here.</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)