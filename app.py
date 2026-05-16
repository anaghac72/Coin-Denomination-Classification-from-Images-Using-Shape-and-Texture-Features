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
    page_title="Indian Coin Prediction",
    page_icon="🪙",
    layout="centered"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #111111;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}

.result-box {
    background-color: #f5f7fa;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
}

.result {
    font-size: 42px;
    font-weight: bold;
    color: #00875a;
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

    # PIL -> NumPy
    image = np.array(image)

    # Resize image
    image = cv2.resize(image, (256, 256))

    # RGB -> BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Gray image
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

    # MUST MATCH TRAINING
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
# UI
# =========================================================

st.markdown(
    '<p class="title">🪙 Indian Coin Prediction</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Upload an Indian coin image to predict denomination</p>',
    unsafe_allow_html=True
)

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

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Uploaded Coin",
                use_container_width=True
            )

        with st.spinner("Predicting..."):

            # Extract features
            features = extract_features(image)

            # Scale
            features = scaler.transform([features])

            # Predict
            prediction = model.predict(features)[0]

            # Confidence
            if hasattr(model, "predict_proba"):

                confidence = np.max(
                    model.predict_proba(features)
                ) * 100

            else:

                confidence = 0

        with col2:

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result">₹{prediction}</div>
                    <br>
                    <b>Confidence:</b> {confidence:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

    except Exception as e:

        st.error(f"Prediction failed: {e}")
