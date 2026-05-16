import streamlit as st
import cv2
import joblib
import numpy as np

from PIL import Image
from skimage.feature import local_binary_pattern
from skimage.feature import hog


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Indian Coin Denomination Prediction",
    page_icon="🪙",
    layout="centered"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* Main */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Title */

.title {

    font-size: 54px;

    font-weight: 800;

    margin-bottom: 10px;
}

/* Subtitle */

.subtitle {

    font-size: 22px;

    color: gray;

    margin-bottom: 30px;
}

/* Result Card */

.result-card {

    background-color: rgba(0, 128, 0, 0.15);

    padding: 22px;

    border-radius: 14px;

    margin-top: 25px;

    border: 1px solid rgba(0,255,0,0.2);
}

/* Result Text */

.result-text {

    font-size: 28px;

    font-weight: 700;

    color: #22c55e;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_models():

    model = joblib.load("models/rf_model.pkl")

    scaler = joblib.load("models/scaler.pkl")

    return model, scaler


try:

    model, scaler = load_models()

except Exception as e:

    st.error(f"Error loading model files: {e}")

    st.stop()


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(image):

    image = np.array(image)

    image = cv2.resize(image, (256, 256))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # =====================================================
    # CIRCLE DETECTION
    # =====================================================

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=50,
        maxRadius=120
    )

    radius = 0

    segmented = gray

    if circles is not None:

        circles = np.round(circles[0, :]).astype("int")

        x, y, r = circles[0]

        radius = r

        mask = np.zeros(gray.shape, dtype="uint8")

        cv2.circle(mask, (x, y), r, 255, -1)

        segmented = cv2.bitwise_and(gray, gray, mask=mask)

    # =====================================================
    # DIAMETER RATIO
    # =====================================================

    diameter_ratio = radius / 128.0

    # =====================================================
    # EDGE DENSITY
    # =====================================================

    edges = cv2.Canny(segmented, 100, 200)

    edge_density = np.sum(edges > 0) / (256 * 256)

    # =====================================================
    # LBP FEATURES
    # =====================================================

    radius_lbp = 3

    points = 8 * radius_lbp

    lbp = local_binary_pattern(
        segmented,
        points,
        radius_lbp,
        method="uniform"
    )

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, points + 3),
        range=(0, points + 2)
    )

    lbp_hist = lbp_hist.astype("float")

    lbp_hist /= (lbp_hist.sum() + 1e-6)

    # =====================================================
    # HOG FEATURES
    # =====================================================

    hog_features = hog(
        segmented,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )

    hog_features = hog_features[:200]

    # =====================================================
    # HSV HISTOGRAM
    # =====================================================

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [8, 8],
        [0, 180, 0, 256]
    )

    hist = cv2.normalize(hist, hist).flatten()

    # =====================================================
    # FINAL FEATURE VECTOR
    # =====================================================

    features = np.hstack([
        diameter_ratio,
        edge_density,
        lbp_hist,
        hog_features,
        hist
    ])

    return features


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">🪙 Indian Coin Denomination Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload an Indian coin image to predict the denomination.</div>',
    unsafe_allow_html=True
)


# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Coin Image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Coin",
            use_container_width=True
        )

        with st.spinner("Predicting denomination..."):

            features = extract_features(image)

            features = scaler.transform([features])

            prediction = model.predict(features)[0]

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-text">
                    Predicted Coin Denomination: ₹{prediction}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:

        st.error(f"Prediction failed: {e}")
