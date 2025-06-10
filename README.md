# Smart Grid Load Forecasting Dashboard

A web application for predicting and visualizing smart grid load forecasts using machine learning.

## Features

- 24-hour load forecasting using XGBoost
- Real-time visualization of predicted loads
- Peak demand prediction and analysis
- RESTful API for predictions
- Interactive dashboard using Streamlit

## Project Structure

```
smart-grid-app/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI backend
│   └── models/
│       └── __init__.py
├── data/
│   └── smart_grid_data.csv  # Your dataset here
├── scripts/
│   └── train_model.py       # Model training script
├── dashboard.py             # Streamlit frontend
├── requirements.txt         # Python dependencies
└── README.md
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .\.venv\Scripts\activate  # On Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your dataset in the `data/` directory as `smart_grid_data.csv`

4. Train the model:
   ```bash
   python scripts/train_model.py
   ```

## Running the Application

1. Start the FastAPI backend:
   ```bash
   uvicorn app.main:app --reload
   ```
   The API will be available at http://localhost:8000

2. In a separate terminal, start the Streamlit dashboard:
   ```bash
   streamlit run dashboard.py
   ```
   The dashboard will be available at http://localhost:8501

## API Endpoints

- `GET /`: API status
- `POST /predict`: Get a single prediction
- `GET /forecast/24h`: Get 24-hour forecast with peak demand

## Data Format

The input dataset (`smart_grid_data.csv`) should contain the following columns:
- Timestamp
- Temperature
- Humidity
- Wind Power (kW)
- Solar Power (kW)
- Predicted Load (kW)

## Contributing

Feel free to submit issues and enhancement requests! 