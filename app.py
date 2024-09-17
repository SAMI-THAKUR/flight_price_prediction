import streamlit as st
from datetime import datetime, timedelta
import pickle
import numpy as np
import requests



# Open the file in binary mode
with open('flight_rf.pkl', 'rb') as file:
    file_content = file.read()  # Read the file content as bytes

# Load the pickle file
modelP = pickle.loads(file_content)  # Use pickle.loads with bytes-like object



# Initialize input arrays based on the feature sequence
xP = np.zeros(28)

def main():
    # Add custom CSS to ensure pointer cursor for dropdowns
    st.markdown("""
        <style>
        .css-1okebmr {
            cursor: pointer;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title of the app
    st.title("Flight Price Prediction")

    # Create a sidebar for inputs
    with st.sidebar:
        st.header("Flight Search Parameters")

        # Input for departure date
        current_date = datetime.now().date()
        max_date = current_date + timedelta(days=49)
        departure_date = st.date_input("Departure Date", min_value=current_date, max_value=max_date)
        date_difference = departure_date - current_date

        # Input for airline
        airline_options = ['Vistara', 'Air India', 'Indigo', 'GO FIRST']
        airline = st.selectbox("Airline", options=airline_options)

        # Input for 'from' location
        from_options = ['Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']
        from_location = st.selectbox("From", options=from_options)

        # Input for 'to' location
        to_options = ['Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']
        to_location = st.selectbox("To", options=to_options)

        # Input for number of stops
        stops_options = ['0', '1', '2+']
        stops = st.selectbox("Number of Stops", options=stops_options)

        # Input for class
        class_options = ['economy', 'business']
        flight_class = st.selectbox("Class", options=class_options)

        # Input for departure time
        departure_time_options = ['Evening', 'Night', 'Afternoon', 'Morning', 'Early_Morning', 'Late_Night']
        departure_time = st.selectbox("Departure Time", options=departure_time_options)

    # Main content area
    st.header("Selected Parameters")
    xP[0] = date_difference.days+1

    # Map airline
    airline_map = {
        'Air India': 1,
        'GO FIRST': 2,
        'Indigo': 3,
        'Vistara': 4
    }
    xP[airline_map.get(airline, 0)-1] = 1

    # Map from location
    from_map = {
        'Bangalore': 6,
        'Chennai': 7,
        'Delhi': 8,
        'Hyderabad': 9,
        'Kolkata': 10,
        'Mumbai': 11
    }
    xP[from_map.get(from_location, 0)-1] = 1

    # Map stops
    stops_map = {
        '0': 12,
        '1': 13,
        '2+': 14
    }
    xP[stops_map.get(stops, 0)-1] = 1

    # Map to location
    to_map = {
        'Bangalore': 15,
        'Chennai': 16,
        'Delhi': 17,
        'Hyderabad': 18,
        'Kolkata': 19,
        'Mumbai': 20
    }
    xP[to_map.get(to_location, 0)-1] = 1

    # Map flight class
    class_map = {
        'business': 21,
        'economy': 22
    }
    xP[class_map.get(flight_class, 0)-1] = 1

    # Map departure time
    departure_time_map = {
        'Afternoon': 23,
        'Early_Morning': 24,
        'Evening': 25,
        'Late_Night': 26,
        'Morning': 27,
        'Night': 28
    }
    xP[departure_time_map.get(departure_time, 0)-1] = 1

    # Display inputs
    st.markdown(f"""
        **Departure Date:** {departure_date}  
        **Airline:** {airline}  
        **From:** {from_location}  
        **To:** {to_location}  
        **Number of Stops:** {stops}  
        **Class:** {flight_class}  
        **Departure Time:** {departure_time}
    """, unsafe_allow_html=True)

    st.write(f"Difference in days: {date_difference.days}")

    # Prediction logic
    if from_location != to_location:
        if st.button("Predict Price"):
            # Predict flight price using modelP
            flight_price_prediction = modelP.predict(np.expand_dims(xP, axis=0))
            st.write(f"Predicted Flight Price: {flight_price_prediction[0].round(0)}")
    else:
        st.write("The 'From' and 'To' locations cannot be the same. Please select different locations.")

if __name__ == "__main__":
    main()
