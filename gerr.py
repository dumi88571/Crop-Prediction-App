# Initialize models and scalers globally
try:
    # Load data
    data = load_data()
    logging.info("Successfully loaded crop data")
except FileNotFoundError:
    logging.critical("crop_data.csv not found. Please ensure the file exists in the correct directory.")
    raise
except Exception as e:
    logging.critical(f"Error loading data: {e}")
    raise

try:
    # Check if pre-trained models exist
    if os.path.exists('crop_prediction_model.joblib') and os.path.exists('crop_feature_scaler.joblib'):
        crop_model = joblib.load('crop_prediction_model.joblib')
        crop_scaler = joblib.load('crop_feature_scaler.joblib')
        logging.info("Loaded pre-trained crop prediction model and scaler")
    else:
        crop_model, crop_scaler = train_crop_model(data)
        logging.info("Trained new crop prediction model")

    if os.path.exists('yield_prediction_model.joblib') and os.path.exists('yield_feature_scaler.joblib'):
        yield_model = joblib.load('yield_prediction_model.joblib')
        yield_scaler = joblib.load('yield_feature_scaler.joblib')
        logging.info("Loaded pre-trained yield prediction model and scaler")
    else:
        yield_model, yield_scaler = train_yield_model(data)
        logging.info("Trained new yield prediction model")
except Exception as e:
    logging.critical(f"Error initializing models: {e}")
    raise

# Run the app
if __name__ == '__main__':
    import socket

    def get_local_ip():
        try:
            # Create a temporary socket to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            logging.warning("Could not determine local IP, defaulting to 127.0.0.1")
            return '127.0.0.1'

    local_ip = get_local_ip()

    print("\n[NETWORK] Crop Prediction App Access URLs:")
    print(f"- Local:    http://127.0.0.1:5000")
    print(f"- Network:  http://{local_ip}:5000")
    print("\n[SERVER] Starting... Press Ctrl+C to stop.\n")

    # Run the app (disable debug in production)
    app.run(host='0.0.0.0', port=5000, debug=False)