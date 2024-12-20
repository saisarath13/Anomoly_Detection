# IoT Anomaly Detection Project

## Overview
This project demonstrates an end-to-end pipeline for anomaly detection in IoT temperature data. It includes data simulation, real-time monitoring using MQTT, and anomaly detection using a trained autoencoder model. The system identifies temperature anomalies by leveraging machine learning techniques and visualization tools.

---

## Features
1. **Temperature Data Simulation**: Simulates IoT temperature sensor data and publishes it to an MQTT broker.  
2. **Real-Time Monitoring**: Subscribes to the MQTT topic, receives data, and visualizes it in real time.  
3. **Anomaly Detection**: Detects anomalies using an autoencoder neural network based on reconstruction error thresholds.  
4. **Visualization**: Provides historical and real-time temperature data analysis.  
5. **Evaluation**: Calculates precision and recall for the anomaly detection model.

---

## Installation
To run the project, ensure you have Python installed and the required libraries. Install the dependencies using the following command:

```bash
pip install paho-mqtt tensorflow pandas numpy matplotlib scikit-learn
```

---

## Project Structure
1. **Data Simulation**: Simulates temperature data using a Python script and publishes it to the MQTT broker.  
2. **Real-Time Monitoring**: Subscribes to the MQTT topic to visualize real-time temperature data. Implements a threshold to flag potential anomalies.  
3. **Historical Analysis**: Simulates historical temperature data with injected anomalies. Normalizes the data and splits it into training and testing sets.  
4. **Autoencoder-Based Anomaly Detection**: Trains an autoencoder model to reconstruct normal data. Uses reconstruction error to identify anomalies.  
5. **Evaluation**: Calculates precision and recall to evaluate the anomaly detection performance.

---

## Usage

### Simulate Temperature Data
Run the script to simulate temperature data and publish it to the MQTT broker:

```python
simulate_temperature_data(duration=30)
```

### Real-Time Monitoring
Start the MQTT subscriber to receive and visualize temperature data in real-time:

```python
# MQTT Client setup
client = mqtt.Client()
client.connect(broker, 1883, 60)
client.subscribe(topic)
client.on_message = on_message

print("Listening for temperature data...")
client.loop_start()
```

### Train the Autoencoder
Train the autoencoder model on historical temperature data:

```python
# Train the Autoencoder
history = model.fit(train_data, train_data, epochs=10, batch_size=32, validation_data=(test_data, test_data))
```

### Anomaly Detection
Identify anomalies in the test data based on the reconstruction error threshold:

```python
# Calculate reconstruction error
reconstruction_error = np.mean(np.square(test_data - reconstructed_data), axis=1)
anomalies = reconstruction_error > threshold
```

### Evaluation
Calculate precision and recall to evaluate the anomaly detection system:

```python
precision = precision_score(ground_truth, predicted_labels)
recall = recall_score(ground_truth, predicted_labels)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
```

---

## Results
1. **Real-Time Visualization**: Shows temperature trends and highlights anomalies in real-time.  
2. **Reconstruction Error Plot**: Visualizes reconstruction errors and detected anomalies.  
3. **Evaluation Metrics**:  
   - Precision: Measures the percentage of true anomalies among detected anomalies.  
   - Recall: Measures the percentage of true anomalies correctly identified.

---

## Requirements
- **Python 3.6 or above**  
- **Libraries**:  
   - paho-mqtt  
   - tensorflow  
   - pandas  
   - numpy  
   - matplotlib  
   - scikit-learn  

---

## Project Flow
1. Simulate IoT temperature data and publish it to the MQTT broker.  
2. Visualize real-time data using MQTT subscriber.  
3. Train an autoencoder model to reconstruct normal temperature data.  
4. Detect anomalies based on reconstruction error threshold.  
5. Evaluate the model using precision and recall.

---

## Future Enhancements
1. Integrate other types of sensor data (e.g., humidity, pressure).  
2. Use more complex deep learning models for anomaly detection.  
3. Deploy the system as a real-time web application.

---

## Author
**Sarath Kumar Kathiravan**  
AI & ML Engineer | Chennai, India  
Email: saisarath1307@gmail.com
