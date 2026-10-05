import streamlit as st
import pandas as pd
import pickle

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AutoPrice AI",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS DESIGN
# =====================================================

st.markdown(
    """
    <style>

    /* Main Application Background */
    .stApp {
        background: linear-gradient(
            135deg,
            #0f2027 0%,
            #203a43 45%,
            #2c5364 100%
        );
        color: white;
    }

    /* Main Content Width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #101820 0%,
            #1c2b36 100%
        );
        border-right: 1px solid #426778;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Main Heading */
    .main-title {
        text-align: center;
        font-size: 46px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 5px;
        letter-spacing: 1px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #c8e6f0;
        margin-bottom: 30px;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.20);
        border-radius: 20px;
        padding: 25px;
        margin-top: 15px;
        margin-bottom: 20px;
        box-shadow: 0px 8px 25px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
    }

    .card-heading {
        color: #7ee8fa;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .card-text {
        color: #e6f7fb;
        font-size: 16px;
        line-height: 1.6;
    }

    /* Prediction Result */
    .prediction-card {
        background: linear-gradient(
            135deg,
            #00b09b,
            #96c93d
        );
        border-radius: 22px;
        padding: 35px;
        text-align: center;
        margin-top: 25px;
        margin-bottom: 25px;
        box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.35);
    }

    .prediction-label {
        font-size: 21px;
        color: white;
        font-weight: 600;
    }

    .prediction-price {
        font-size: 48px;
        font-weight: 900;
        color: white;
        margin-top: 10px;
    }

    .prediction-note {
        font-size: 14px;
        color: white;
        margin-top: 10px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3.2em;
        background: linear-gradient(
            90deg,
            #00c6ff,
            #0072ff
        );
        color: white;
        font-size: 18px;
        font-weight: 700;
        border: none;
        transition: 0.3s;
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            #0072ff,
            #00c6ff
        );
        transform: scale(1.02);
    }

    /* Input Labels */
    label {
        font-weight: 600 !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.10);
        border-radius: 10px;
        color: white !important;
        font-weight: 600;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #b8d8e3;
        font-size: 14px;
        margin-top: 35px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# LOAD MODEL AND DATASET
# =====================================================

@st.cache_resource
def load_model():
    with open("car_price_model.pkl", "rb") as file:
        saved_data = pickle.load(file)

    return saved_data["model"], saved_data["features"]


@st.cache_data
def load_dataset():
    return pd.read_csv("autos_dataset.csv")


try:
    model, features = load_model()
    dataset = load_dataset()

except FileNotFoundError:
    st.error(
        "Required file not found. Please keep "
        "app.py, car_price_model.pkl and autos_dataset.csv "
        "in the same folder."
    )
    st.stop()

except Exception as error:
    st.error("Error while loading application files.")
    st.write(error)
    st.stop()

# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="main-title">🚘 AutoPrice AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Smart Car Price Prediction Using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

# =====================================================
# INTRODUCTION CARD
# =====================================================

st.markdown(
    """
    <div class="glass-card">
        <div class="card-heading">✨ Welcome to AutoPrice AI</div>
        <div class="card-text">
            Estimate the price of a car by entering its dimensions,
            engine specifications, weight and mileage details.
            Our intelligent prediction system provides an estimated
            car price within seconds.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.markdown("## 🚗 Car Information")
st.sidebar.write(
    "Enter the specifications of your car below."
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 📏 Vehicle Dimensions")

symboling = st.sidebar.number_input(
    "Symboling",
    min_value=-3,
    max_value=3,
    value=0,
    help="Insurance risk rating of the vehicle."
)

wheel_base = st.sidebar.number_input(
    "Wheel Base",
    min_value=80.0,
    max_value=130.0,
    value=100.0,
    help="Distance between the front and rear wheels."
)

length = st.sidebar.number_input(
    "Length",
    min_value=130.0,
    max_value=210.0,
    value=170.0,
    help="Overall length of the car."
)

width = st.sidebar.number_input(
    "Width",
    min_value=60.0,
    max_value=75.0,
    value=65.0,
    help="Overall width of the car."
)

height = st.sidebar.number_input(
    "Height",
    min_value=45.0,
    max_value=75.0,
    value=55.0,
    help="Overall height of the car."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Engine Specifications")

curb_weight = st.sidebar.number_input(
    "Curb Weight",
    min_value=1000,
    max_value=4500,
    value=2500,
    help="Weight of the vehicle without passengers."
)

num_of_cylinders = st.sidebar.selectbox(
    "Number of Cylinders",
    options=[2, 3, 4, 5, 6, 8, 12],
    index=2
)

engine_size = st.sidebar.number_input(
    "Engine Size",
    min_value=50,
    max_value=500,
    value=120
)

compression_ratio = st.sidebar.number_input(
    "Compression Ratio",
    min_value=5.0,
    max_value=25.0,
    value=9.0
)

horsepower = st.sidebar.number_input(
    "Horsepower",
    min_value=30.0,
    max_value=400.0,
    value=100.0
)

peak_rpm = st.sidebar.number_input(
    "Peak RPM",
    min_value=3000.0,
    max_value=7000.0,
    value=5000.0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⛽ Mileage Information")

city_mpg = st.sidebar.number_input(
    "City Mileage (MPG)",
    min_value=10,
    max_value=60,
    value=25
)

highway_mpg = st.sidebar.number_input(
    "Highway Mileage (MPG)",
    min_value=10,
    max_value=60,
    value=30
)

# =====================================================
# SELECTED DETAILS DISPLAY
# =====================================================

st.markdown(
    """
    <div class="glass-card">
        <div class="card-heading">📝 Your Car Specifications</div>
        <div class="card-text">
            Review the selected values below before generating
            the estimated price.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

car_details = pd.DataFrame({
    "Specification": [
        "Symboling",
        "Wheel Base",
        "Length",
        "Width",
        "Height",
        "Curb Weight",
        "Number of Cylinders",
        "Engine Size",
        "Compression Ratio",
        "Horsepower",
        "Peak RPM",
        "City Mileage",
        "Highway Mileage"
    ],
    "Value": [
        symboling,
        wheel_base,
        length,
        width,
        height,
        curb_weight,
        num_of_cylinders,
        engine_size,
        compression_ratio,
        horsepower,
        peak_rpm,
        city_mpg,
        highway_mpg
    ]
})

st.dataframe(
    car_details,
    use_container_width=True,
    hide_index=True
)

# =====================================================
# PREDICTION BUTTON
# =====================================================

st.markdown("### 🔮 Generate Your Price Estimate")

if st.button("🚀 Predict Car Price"):

    input_data = pd.DataFrame([{
        "symboling": symboling,
        "wheel-base": wheel_base,
        "length": length,
        "width": width,
        "height": height,
        "curb-weight": curb_weight,
        "num-of-cylinders": num_of_cylinders,
        "engine-size": engine_size,
        "compression-ratio": compression_ratio,
        "horsepower": horsepower,
        "peak-rpm": peak_rpm,
        "city-mpg": city_mpg,
        "highway-mpg": highway_mpg
    }])

    # Keep the same column order used during model training
    input_data = input_data[features]

    try:
        predicted_price = model.predict(input_data)[0]

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">
                    🎉 Estimated Car Price
                </div>
                <div class="prediction-price">
                    ${predicted_price:,.2f}
                </div>
                <div class="prediction-note">
                    This amount is an estimated value generated
                    by the Machine Learning model.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        result = input_data.copy()
        result["Estimated Price"] = predicted_price

        csv_data = result.to_csv(index=False)

        st.download_button(
            label="⬇️ Download Prediction Report",
            data=csv_data,
            file_name="car_price_prediction.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error("Prediction could not be generated.")
        st.write(error)

# =====================================================
# DATASET PREVIEW
# =====================================================

with st.expander("📊 View Sample Dataset"):

    st.write(
        "This section displays a sample of the automobile dataset."
    )

    st.dataframe(
        dataset.head(10),
        use_container_width=True
    )

# =====================================================
# FOOTER
# =====================================================

st.markdown(
    """
    <div class="footer">
        🚘 AutoPrice AI | Smart Prediction for Smarter Decisions
    </div>
    """,
    unsafe_allow_html=True
)