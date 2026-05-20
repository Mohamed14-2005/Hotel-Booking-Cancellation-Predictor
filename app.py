import streamlit as st
import pandas as pd
import joblib
import warnings
from collections import UserList
from datetime import date

try:
    import pycountry
except ImportError:
    pycountry = None

st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("hotel_bookings.csv")

meal_display = {
    "BB": "BB - Bed & Breakfast",
    "HB": "HB - Half Board",
    "FB": "FB - Full Board",
    "SC": "SC - Self Catering",
    "Undefined": "Undefined"
}

market_segment_display = {
    "Aviation": "Aviation",
    "Complementary": "Complementary",
    "Corporate": "Corporate",
    "Direct": "Direct",
    "Groups": "Groups",
    "Offline TA/TO": "Offline TA/TO - Travel Agent / Tour Operator",
    "Online TA": "Online TA - Online Travel Agent",
    "Undefined": "Undefined"
}

distribution_channel_display = {
    "Corporate": "Corporate",
    "Direct": "Direct",
    "GDS": "GDS - Global Distribution System",
    "TA/TO": "TA/TO - Travel Agent / Tour Operator",
    "Undefined": "Undefined"
}

country_cache = {"Unknown": "Unknown"}

def get_country_label(code):
    if code in country_cache:
        return country_cache[code]

    label = code
    if code is None or code == "Unknown":
        country_cache[code] = "Unknown"
        return "Unknown"

    if pycountry is not None:
        country = pycountry.countries.get(alpha_2=code)
        if country is None:
            country = pycountry.countries.get(alpha_3=code)
        if country is not None:
            label = f"{code} - {country.name}"

    country_cache[code] = label
    return label

deposit_type_display = {
    "No Deposit": "No Deposit",
    "Non Refund": "Non Refund",
    "Refundable": "Refundable"
}

customer_type_display = {
    "Contract": "Contract",
    "Group": "Group",
    "Transient": "Transient",
    "Transient-Party": "Transient-Party"
}

@st.cache_resource
def load_model():
    try:
        import sklearn.compose._column_transformer as ct

        if not hasattr(ct, "_RemainderColsList"):
            class _RemainderColsList(UserList):
                def __init__(self, columns, future_dtype=None, warning_was_emitted=False, warning_enabled=True):
                    super().__init__(columns)
                    self.future_dtype = future_dtype
                    self.warning_was_emitted = warning_was_emitted
                    self.warning_enabled = warning_enabled

                def _warn(self):
                    if self.warning_enabled and not self.warning_was_emitted:
                        warnings.warn(
                            "Accessing _RemainderColsList contents.",
                            FutureWarning,
                            stacklevel=2,
                        )
                        self.warning_was_emitted = True

                def __getitem__(self, idx):
                    self._warn()
                    return super().__getitem__(idx)

                def __iter__(self):
                    self._warn()
                    return super().__iter__()

                def __repr__(self):
                    return list.__repr__(self.data)

            ct._RemainderColsList = _RemainderColsList
    except Exception:
        pass

    return joblib.load("hotel_model.pkl")

st.title("🏨 Hotel Booking Cancellation Predictor")
st.markdown("Use the form below to enter booking details and predict whether the booking will be canceled.")

try:
    df = load_data()
    model = load_model()
    model_loaded = True
except Exception as exc:
    st.error("Unable to load model or data. Please ensure `hotel_bookings.csv` and `hotel_model.pkl` exist.")
    st.exception(exc)
    model_loaded = False

if model_loaded:
    st.sidebar.header("Options")
    show_data = st.sidebar.checkbox("Show sample data", value=False)

    if show_data:
        st.subheader("Dataset sample")
        st.dataframe(df.head())

    st.header("🔮 Predict Booking Cancellation")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            hotel = st.selectbox("Hotel", sorted(df["hotel"].dropna().unique()))
            lead_time = st.number_input("Lead Time", min_value=0, value=35)
            year_options = sorted(df["arrival_date_year"].dropna().astype(int).unique().tolist())
            if year_options[-1] < 2030:
                year_options.extend(range(year_options[-1] + 1, 2031))
            arrival_date_year = st.selectbox("Arrival Year", year_options)
            arrival_date_month = st.selectbox("Arrival Month", sorted(df["arrival_date_month"].dropna().unique()))
            arrival_date_week_number = st.number_input("Arrival Week Number", min_value=1, max_value=53, value=30)
            arrival_date_day_of_month = st.number_input("Arrival Day of Month", min_value=1, max_value=31, value=15)
            stays_in_weekend_nights = st.number_input("Stays in Weekend Nights", min_value=0, value=0)
            stays_in_week_nights = st.number_input("Stays in Week Nights", min_value=0, value=3)
            adults = st.number_input("Adults", min_value=1, value=2)
            children = st.number_input("Children", min_value=0, value=0)
            babies = st.number_input("Babies", min_value=0, value=0)
            meal_values = sorted(df["meal"].fillna("Undefined").unique())
            meal = st.selectbox("Meal", meal_values, format_func=lambda x: meal_display.get(x, x))
            country_values = sorted(df["country"].fillna("Unknown").unique())
            country = st.selectbox(
                "Country",
                country_values,
                format_func=get_country_label
            )

        with col2:
            market_segment_values = sorted(df["market_segment"].dropna().unique())
            market_segment = st.selectbox(
                "Market Segment",
                market_segment_values,
                format_func=lambda x: market_segment_display.get(x, x)
            )
            distribution_channel_values = sorted(df["distribution_channel"].dropna().unique())
            distribution_channel = st.selectbox(
                "Distribution Channel",
                distribution_channel_values,
                format_func=lambda x: distribution_channel_display.get(x, x)
            )
            is_repeated_guest = st.checkbox("Repeated Guest")
            previous_cancellations = st.number_input("Previous Cancellations", min_value=0, value=0)
            previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", min_value=0, value=0)
            reserved_room_type = st.selectbox("Reserved Room Type", sorted(df["reserved_room_type"].dropna().unique()))
            assigned_room_type = st.selectbox("Assigned Room Type", sorted(df["assigned_room_type"].dropna().unique()))
            booking_changes = st.number_input("Booking Changes", min_value=0, value=0)
            deposit_type_values = sorted(df["deposit_type"].dropna().unique())
            deposit_type = st.selectbox(
                "Deposit Type",
                deposit_type_values,
                format_func=lambda x: deposit_type_display.get(x, x)
            )
            agent = st.number_input("Agent ID", min_value=0.0, value=0.0, step=1.0, format="%.0f")
            company = st.number_input("Company ID", min_value=0.0, value=0.0, step=1.0, format="%.0f")
            days_in_waiting_list = st.number_input("Days in Waiting List", min_value=0, value=0)
            customer_type_values = sorted(df["customer_type"].dropna().unique())
            customer_type = st.selectbox(
                "Customer Type",
                customer_type_values,
                format_func=lambda x: customer_type_display.get(x, x)
            )
            adr = st.number_input("Average Daily Rate", min_value=0.0, value=100.0)
            required_car_parking_spaces = st.number_input("Required Car Parking Spaces", min_value=0, value=0)
            total_of_special_requests = st.number_input("Total Special Requests", min_value=0, value=0)
            reservation_status = st.selectbox("Reservation Status", sorted(df["reservation_status"].dropna().unique()))
            reservation_status_date = st.text_input("Reservation Status Date", value=date.today().strftime("%Y-%m-%d"))

            submitted = st.form_submit_button("Predict Cancellation")

        if submitted:
            input_data = pd.DataFrame([{
                "hotel": hotel,
                "lead_time": lead_time,
                "arrival_date_year": arrival_date_year,
                "arrival_date_month": arrival_date_month,
                "arrival_date_week_number": arrival_date_week_number,
                "arrival_date_day_of_month": arrival_date_day_of_month,
                "stays_in_weekend_nights": stays_in_weekend_nights,
                "stays_in_week_nights": stays_in_week_nights,
                "adults": adults,
                "children": children,
                "babies": babies,
                "meal": meal,
                "country": country,
                "market_segment": market_segment,
                "distribution_channel": distribution_channel,
                "is_repeated_guest": int(is_repeated_guest),
                "previous_cancellations": previous_cancellations,
                "previous_bookings_not_canceled": previous_bookings_not_canceled,
                "reserved_room_type": reserved_room_type,
                "assigned_room_type": assigned_room_type,
                "booking_changes": booking_changes,
                "deposit_type": deposit_type,
                "agent": agent,
                "company": company,
                "days_in_waiting_list": days_in_waiting_list,
                "customer_type": customer_type,
                "adr": adr,
                "required_car_parking_spaces": required_car_parking_spaces,
                "total_of_special_requests": total_of_special_requests,
                "reservation_status": reservation_status,
                "reservation_status_date": reservation_status_date
            }])

            prediction = model.predict(input_data)[0]
            probability = None
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(input_data)[0].max()

            if prediction == 1:
                st.error("🔴 Prediction: This booking is likely to be canceled.")
            else:
                st.success("🟢 Prediction: This booking is likely NOT to be canceled.")

            if probability is not None:
                st.write(f"Prediction confidence: {probability:.2%}")

    st.write("---")
    st.markdown(
        "**Run this command in the same Python environment that has Streamlit installed:** `python -m streamlit run app.py`."
    )
