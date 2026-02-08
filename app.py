from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Helper function to convert numpy types to Python native types
def convert_to_native_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    return obj

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)
print("Template folder:", app.template_folder)
print("Exists:", os.path.exists(app.template_folder))



# Load the trained model and other files
print("Loading model...")
try:
    model = joblib.load('xgboost_model.pkl')
    feature_names = joblib.load('feature_columns.pkl')
    model_info_raw = joblib.load('model_stats.pkl')
    
    # Convert all numpy types to Python native types
    model_info = {
        'r2_score': float(model_info_raw['r2_score']),
        'mae': float(model_info_raw['mae']),
        'rmse': float(model_info_raw['rmse'])
    }
    
    print("✅ Model loaded successfully!")
    print(f"Model: XGBoost Regressor")
    print(f"R² Score: {model_info['r2_score']:.4f}")
    print(f"MAE: {model_info['mae']:,.2f}")
    print(f"RMSE: {model_info['rmse']:,.2f}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure you have run the training script first to generate the .pkl files")

# Property types and locations
PROPERTY_TYPES = [
    'apartment', 'villa', 'duplex', 'penthouse', 'townhouse', 
    'studio', 'chalet', 'twinhouse', 'standalone villa'
]

LOCATIONS = [
    'New Cairo', 'Sheikh Zayed City', 'New Zayed City', '6th of October', 
    'New Capital', 'North Coast', 'Gouna', 'Hurghada', 'Ras Al Hekma',
    'Maadi', 'Heliopolis', 'Nasr City', 'Zamalek', 'Mohandessin'
]

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict real estate price
    Expects JSON data with all feature values
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract base features
        size_sqm = float(data.get('size_sqm', 0))
        bedrooms = float(data.get('bedrooms', 0))
        bathrooms = float(data.get('bathrooms', 0))
        property_type = data.get('property_type', '')
        location = data.get('location', '')
        price_per_sqft = float(data.get('price_per_sqft', 0))
        
        # Create base feature array
        features = []
        missing_features = []
        
        # Build feature array in correct order
        for feature in feature_names:
            if feature == 'Size_sqm':
                features.append(size_sqm)
            elif feature == 'Bedroom_Num':
                features.append(bedrooms)
            elif feature == 'bathrooms_numeric':
                features.append(bathrooms)
            elif feature == 'Price_per_sqft':
                features.append(price_per_sqft)
            elif feature.startswith('type_'):
                type_name = feature.replace('type_', '')
                features.append(1.0 if type_name == property_type else 0.0)
            elif feature.startswith('location_'):
                loc_name = feature.replace('location_', '')
                features.append(1.0 if loc_name == location else 0.0)
            else:
                features.append(0.0)
                missing_features.append(feature)
        
        # Log if any features were missing
        if missing_features:
            print(f"⚠️  Missing features (defaulted to 0): {missing_features}")
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction - FIXED: Convert to Python float to avoid JSON serialization error
        prediction = float(model.predict(features_array)[0])
        
        # Format prediction
        prediction_formatted = f"{prediction:,.0f}"
        prediction_millions = f"{prediction/1000000:.2f}"
        
        # Calculate price per sqm
        price_per_sqm = prediction / size_sqm if size_sqm > 0 else 0
        
        # Prepare response
        result = {
            'success': True,
            'prediction': prediction_formatted,
            'prediction_egp': prediction,
            'prediction_millions': prediction_millions,
            'price_per_sqm': price_per_sqm,
            'inputs': {
                'size_sqm': size_sqm,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'property_type': property_type,
                'location': location
            },
            'model_info': {
                'model_name': 'XGBoost Regressor',
                'r2_score': model_info['r2_score'],
                'mae': model_info['mae'],
                'rmse': model_info['rmse']
            }
        }
        
        # Convert all numpy types to Python native types
        result = convert_to_native_types(result)
        
        # Log prediction
        print(f"\n{'='*50}")
        print(f"Prediction: EGP {prediction_formatted}")
        print(f"Property: {property_type} in {location}")
        print(f"Price per sqm: EGP {price_per_sqm:,.2f}")
        print(f"{'='*50}\n")
        
        return jsonify(result)
    
    except ValueError as ve:
        return jsonify({'error': f'Invalid data format: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Predict multiple properties at once
    Expects JSON array of property objects
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Expected array of properties'}), 400
        
        results = []
        
        for idx, property_data in enumerate(data):
            # Extract features
            size_sqm = float(property_data.get('size_sqm', 0))
            bedrooms = float(property_data.get('bedrooms', 0))
            bathrooms = float(property_data.get('bathrooms', 0))
            property_type = property_data.get('property_type', '')
            location = property_data.get('location', '')
            price_per_sqft = float(property_data.get('price_per_sqft', 0))
            
            # Build feature array
            features = []
            for feature in feature_names:
                if feature == 'Size_sqm':
                    features.append(size_sqm)
                elif feature == 'Bedroom_Num':
                    features.append(bedrooms)
                elif feature == 'bathrooms_numeric':
                    features.append(bathrooms)
                elif feature == 'Price_per_sqft':
                    features.append(price_per_sqft)
                elif feature.startswith('type_'):
                    type_name = feature.replace('type_', '')
                    features.append(1.0 if type_name == property_type else 0.0)
                elif feature.startswith('location_'):
                    loc_name = feature.replace('location_', '')
                    features.append(1.0 if loc_name == location else 0.0)
                else:
                    features.append(0.0)
            
            # Convert to numpy array
            features_array = np.array(features).reshape(1, -1)
            
            # Make prediction - FIXED: Convert to Python float
            prediction = float(model.predict(features_array)[0])
            
            results.append({
                'property_id': idx + 1,
                'property_type': property_type,
                'location': location,
                'predicted_price': f"{prediction:,.0f}",
                'predicted_price_egp': float(prediction),
                'predicted_millions': f"{prediction/1000000:.2f}"
            })
        
        # Calculate summary statistics
        total_value = sum(r['predicted_price_egp'] for r in results)
        avg_price = total_value / len(results) if results else 0
        
        response = {
            'results': results,
            'summary': {
                'total_properties': len(results),
                'total_value_egp': total_value,
                'average_price_egp': avg_price,
                'total_value_formatted': f"{total_value:,.0f}",
                'average_price_formatted': f"{avg_price:,.0f}"
            }
        }
        
        # Convert all numpy types to Python native types
        response = convert_to_native_types(response)
        
        print(f"\n{'='*50}")
        print(f"Batch Prediction Summary:")
        print(f"Total: {response['summary']['total_properties']}")
        print(f"Total Value: EGP {response['summary']['total_value_formatted']}")
        print(f"Average: EGP {response['summary']['average_price_formatted']}")
        print(f"{'='*50}\n")
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': f'Batch prediction error: {str(e)}'}), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model"""
    try:
        info = {
            'model_name': 'XGBoost Regressor',
            'algorithm': 'Gradient Boosting',
            'r2_score': model_info['r2_score'],
            'mae': model_info['mae'],
            'rmse': model_info['rmse'],
            'total_features': len(feature_names),
            'features': feature_names,
            'property_types': PROPERTY_TYPES,
            'locations': LOCATIONS
        }
        return jsonify(convert_to_native_types(info))
    except Exception as e:
        return jsonify({'error': f'Error getting model info: {str(e)}'}), 500

@app.route('/feature-names', methods=['GET'])
def get_features():
    """Get list of required feature names"""
    try:
        return jsonify({
            'features': feature_names,
            'total': len(feature_names)
        })
    except Exception as e:
        return jsonify({'error': f'Error getting features: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Verify model is loaded
        if model is None:
            return jsonify({'status': 'unhealthy', 'message': 'Model not loaded'}), 503
        
        health_data = {
            'status': 'healthy',
            'model_loaded': True,
            'model_name': 'XGBoost Regressor',
            'r2_score': model_info['r2_score']
        }
        return jsonify(convert_to_native_types(health_data))
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🏠 STARTING EGYPT REAL ESTATE PRICE PREDICTION SERVER")
    print("="*70)
    print("\n📊 Available Endpoints:")
    print("  • GET  /                    - Main web interface")
    print("  • POST /predict             - Single property prediction")
    print("  • POST /predict-batch       - Multiple properties prediction")
    print("  • GET  /model-info          - Get model information")
    print("  • GET  /feature-names       - Get required features")
    print("  • GET  /health              - Health check")
    print("\n🌐 Server starting at: http://localhost:5000")
    print("   Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    app.run(debug=True)