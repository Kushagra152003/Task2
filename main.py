import os
import autogen
import pandas as pd
import io
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression  # Example classification model
from flask import Flask, request, jsonify
import requests  # For testing
from dotenv import load_dotenv
from autogen import UserProxyAgent, AssistantAgent
load_dotenv()

# --- 1. API Key and Configuration ---
api_key = os.environ.get("Groq_api_key")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")

config_list = [
    {
        "model": "llama-3.3-70b-versatile",  # Or another suitable Groq model
        "api_type": "groq",
        "api_key": api_key,
    }
]

llama3_8b_config = {
    "cache_seed": 42,
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,
    "code_execution_config": {"use_docker": False}
}

# --- 2. Agent Functions (Tools) ---

def process_data(csv_content: str):
    try:
        df = pd.read_csv(io.StringIO(csv_content))
    except pd.errors.ParserError:
        return "Error: Invalid CSV file format."

    # Data Cleaning and Preprocessing (Adapt to your data)
    df.dropna(inplace=True)  # Handle missing values (example)
    # ... (Add more preprocessing steps as needed - feature engineering, etc.)

    # Separate features and target (adapt to your dataset)
    X = df.drop('genre', axis=1)  # Example: 'genre' is the target
    y = df['genre']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling (Important for many models)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to JSON strings for passing between agents
    train_data = {'X_train': X_train.tolist(), 'y_train': y_train.tolist()}
    test_data = {'X_test': X_test.tolist(), 'y_test': y_test.tolist()}
    scaler_data = {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()}

    return json.dumps({'train': train_data, 'test': test_data, 'scaler': scaler_data})


def train_model(data_json: str):
    try:
        data = json.loads(data_json)
        X_train = pd.DataFrame(data['train']['X_train'])  # Convert back to DataFrame
        y_train = pd.Series(data['train']['y_train'])
        X_test = pd.DataFrame(data['test']['X_test'])
        y_test = pd.Series(data['test']['y_test'])
        scaler_mean = data['scaler']['mean']
        scaler_scale = data['scaler']['scale']

        # Model Training (Example: Logistic Regression)
        model = LogisticRegression()  # Or any other classification model
        model.fit(X_train, y_train)

        # Evaluate the model (example)
        accuracy = model.score(X_test, y_test)
        print(f"Test Accuracy: {accuracy}")

        model_path = "trained_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({'model': model, 'mean': scaler_mean, 'scale': scaler_scale}, f) # Save model and scaler

        return model_path  # Return path to the saved model
    except Exception as e:
        return f"Error training model: {e}"

# --- 3. Deployment (Flask App) ---
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get input data (JSON)
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Load model and scaler
        with open("trained_model.pkl", "rb") as f:
            model_data = pickle.load(f)
            model = model_data['model']
            scaler_mean = model_data['mean']
            scaler_scale = model_data['scale']

        # Scale input data using loaded scaler
        input_data = pd.DataFrame([data]) # Convert to DataFrame
        scaled_data = (input_data - scaler_mean) / scaler_scale # Scale using mean and scale

        prediction = model.predict(scaled_data)[0]  # Make prediction
        return jsonify({'prediction': str(prediction)}), 200 # Convert prediction to string

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- 4. Agents ---
data_preprocessor = autogen.AssistantAgent(
    name="data_preprocessor",
    system_message="You are a data preprocessing expert. You will receive CSV data as input. Your job is to clean, transform, and prepare this data for machine learning. Return the processed data as a JSON string.",
    function_map={"process_data": process_data},
)

model_developer = autogen.AssistantAgent(
    name="model_developer",
    system_message="You are a machine learning expert. You will receive processed data as a JSON string. Your job is to train a suitable classification model and return the path to the saved model.",
    function_map={"train_model": train_model},
)

deployer = autogen.AssistantAgent(
    name="deployer",
    system_message="You are a model deployment expert. You will receive the path to a trained model. Your job is to start the Flask app.",
    function_map={},  # Deployment is handled separately
)

tester = autogen.UserProxyAgent(
    name="tester",
    system_message="Your task is to test the model. You should send a request to the /predict endpoint and check the prediction."
)

# --- 5. Workflow ---
def state_transition(last_speaker, groupchat):
    if last_speaker is data_preprocessor:
        return model_developer
    elif last_speaker is model_developer:
        return deployer
    elif last_speaker is deployer:
        return tester
    elif last_speaker is tester:
        return None

groupchat = autogen.GroupChat(
    agents=[data_preprocessor, model_developer, deployer, tester],
    messages=[],
    max_round=6,
    speaker_selection_method=state_transition,
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llama3_8b_config)

# --- 6. Load CSV Data ---
csv_file_path = r"C:\Users\Hp\Desktop\TASK2\versions\1\imdb_top_1000.csv"  # Replace with your CSV file path
try:
    with open(csv_file_path, "r", encoding="utf-8") as file:  # Handle encoding
        csv_content = file.read()
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_file_path}")
    exit()

# --- 7. Initiate Chat ---
data_preprocessor.initiate_chat(
    manager, message=f"Here is the CSV data:\n```csv\n{csv_content}\n```"
)

# --- 8. Process Agent Messages ---
for message in groupchat.messages:
    if message["role"] == "assistant" and message["name"] == "data_preprocessor":
        processed_data_json = message["content"]
        model_developer.send(message=processed_data_json, recipient=model_developer)

    elif message["role"] == "assistant" and message["name"] == "model_developer":
        model_path = message["content"]
        deployer.send(message="Deploy the model", recipient=deployer)  # Trigger deployment

        # Start Flask app in a separate process (or use a proper deployment method)
        import multiprocessing
        p = multiprocessing.Process(target=app.run)
        p.start()

        # Test the model (after a short delay to allow the app to start)
        import time
        time.sleep(5)  # Adjust delay as needed

        test_data = {  # Example test data - adapt to your features
            'feature1': 0.5,  # Replace with your actual feature values
            'feature2': 1,
            # ... other features
        }
        
        response = requests.post