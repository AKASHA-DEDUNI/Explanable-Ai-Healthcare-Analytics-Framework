import streamlit as st
import requests

# Initialize session state
if "role" not in st.session_state:
    st.session_state.role = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "step" not in st.session_state:
    st.session_state.step = "choose_role"
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

st.title("💬 AI Healthcare Chatbox")

# Step 1 – Choose Role
if st.session_state.step == "choose_role":
    st.subheader("Select Your Role")
    role = st.selectbox("I am a...", ["-- Select --", "Nurse", "Doctor", "Patient"])

    if role != "-- Select --":
        st.session_state.role = role
        st.session_state.chat_history.append(
            {"sender": "System", "message": f"Welcome {role}! Thank you for connecting."}
        )
        st.session_state.step = "choose_action"
        st.rerun()

# Step 2 – Choose Action
elif st.session_state.step == "choose_action":
    st.subheader(f"Hello {st.session_state.role}, what would you like to do?")
    action = st.selectbox(
        "Select an option",
        [
            "-- Select --",
            "Prediction Explanation",
            "Grad-CAM Visualization",
            "Search Medical Info",
            "Add/View Medical Recommendations"
        ]
    )

    if action != "-- Select --":
        st.session_state.chat_history.append({"sender": "System", "message": f"You selected: {action}"})
        st.session_state.step = action.lower().replace(" ", "_")
        st.rerun()

# Step 3 – Prediction Explanation
elif st.session_state.step == "prediction_explanation":
    st.subheader("Prediction Explanation")
    st.write("Example: *The model detected pneumonia with 85% confidence based on image features.*")
    if st.session_state.role == "Patient" and st.session_state.recommendations:
        st.info("Doctor's Recommendations:\n" + "\n".join(f"- {rec}" for rec in st.session_state.recommendations))

# Step 4 – Grad-CAM Visualization
elif st.session_state.step == "grad-cam_visualization":
    st.subheader("Grad-CAM Visualization")
    st.write("This is where you would display the Grad-CAM heatmap.")
    st.image("https://via.placeholder.com/400x300.png?text=Grad-CAM+Heatmap", caption="Grad-CAM Example")
    if st.session_state.role == "Patient" and st.session_state.recommendations:
        st.info("Doctor's Recommendations:\n" + "\n".join(f"- {rec}" for rec in st.session_state.recommendations))

# Step 5 – Web Search
elif st.session_state.step == "search_medical_info":
    st.subheader("Search Medical Information")
    query = st.text_input("Enter your medical query:")

    if query:
        search_url = f"https://api.duckduckgo.com/?q={query}&format=json"
        response = requests.get(search_url).json()
        answer = response.get("AbstractText", "No detailed answer found.")
        st.session_state.chat_history.append({"sender": "System", "message": answer})
        st.write(answer)

    if st.session_state.role == "Patient" and st.session_state.recommendations:
        st.info("Doctor's Recommendations:\n" + "\n".join(f"- {rec}" for rec in st.session_state.recommendations))

# Step 6 – Medical Recommendations
elif st.session_state.step == "add/view_medical_recommendations":
    st.subheader("Medical Recommendations")

    if st.session_state.role == "Doctor":
        new_rec = st.text_area("Enter your recommendation for patients:")
        if st.button("Save Recommendation"):
            if new_rec.strip():
                st.session_state.recommendations.append(new_rec.strip())
                st.success("Recommendation saved!")
    elif st.session_state.role == "Patient":
        if st.session_state.recommendations:
            st.info("Doctor's Recommendations:\n" + "\n".join(f"- {rec}" for rec in st.session_state.recommendations))
        else:
            st.warning("No recommendations from doctors yet.")
    else:
        st.write("Only doctors can add recommendations. Patients can view them here.")

# Display Chat History
st.subheader("💬 Chat History")
for chat in st.session_state.chat_history:
    if chat["sender"] == "System":
        st.markdown(f"**🩺 {chat['sender']}:** {chat['message']}")
    else:
        st.markdown(f"**🙂 You:** {chat['message']}")

# Back button
if st.session_state.step != "choose_role":
    if st.button("⬅ Back"):
        st.session_state.step = "choose_action"
        st.rerun()

