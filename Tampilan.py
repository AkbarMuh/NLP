import streamlit as st
import requests

# Function to predict sentiment
def predict_sentiment(text):
    response = requests.post("http://127.0.0.1:8000/predict_sentiment", json={"text": text})
    result_nb = response.json()["predicted_sentiment_nb"]
    result_svm = response.json()["predicted_sentiment_svm"]
    result_lr = response.json()["predicted_sentiment_lr"]

    acuracy_nb = response.json()["accuracy_nb"]
    acuracy_svm = response.json()["accuracy_svm"]
    acuracy_lr = response.json()["accuracy_lr"]
    return result_nb, result_svm, result_lr, acuracy_nb, acuracy_svm, acuracy_lr

# Main Streamlit app
st.title("Twitter Sentiment Analysis")

# Create tabs
tabs = ["Prediction", "Chat"]
selected_tab = st.sidebar.selectbox("Select Tab", tabs)

# Prediction tab
if selected_tab == "Prediction":
    # Input text for sentiment prediction
    text_to_predict = st.text_area("Input Text for Sentiment Prediction", "I love Nasi Goreng")
    
    # Button to predict sentiment
    if st.button("Predict Sentiment"):
        result_nb, result_svm, result_lr, acuracy_nb, acuracy_svm, acuracy_lr = predict_sentiment(text_to_predict)
         # Display the predictions
        st.write(f"Predicted Sentiment (Naive Bayes) - {acuracy_nb} : {result_nb}")
        st.write(f"Predicted Sentiment (Linear SVM) - {acuracy_svm} : {result_svm}")
        st.write(f"Predicted Sentiment (Logistic Regression) - {acuracy_lr}: {result_lr}")

# Chat tab
elif selected_tab == "Chat":
    # Chat-like interface
    st.title("Chat Messages")
    messages = []
    # Input and send messages
    message = st.chat_input("Type a message:")
    if message:
        result_nb, result_svm, result_lr, acuracy_nb, acuracy_svm, acuracy_lr = predict_sentiment(message)
        if "akurasi" in message.lower():
            messages.append({"user": message, "response_nb": f"Server (Naive Bayes): {acuracy_nb}", 
                        "response_svm": f"Server (Linear SVM): {acuracy_svm}", 
                        "response_lr": f"Server (Logistic Regression): {acuracy_lr}"})
        else:
            messages.append({"user": message, "response_nb": f"Server (Naive Bayes): {result_nb}", 
                            "response_svm": f"Server (Linear SVM): {result_svm}", 
                            "response_lr": f"Server (Logistic Regression): {result_lr}"})

    # Display all messages
    for msg in messages:
        with st.chat_message("user"):
            st.write(f"{msg['user']}")

        with st.chat_message("assistant"):
            st.write(f"{msg['response_nb']}")
            st.write(f"{msg['response_svm']}")
            st.write(f"{msg['response_lr']}")
