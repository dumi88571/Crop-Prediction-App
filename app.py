import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
import requests

import logging
import traceback
from disease_management import disease_predictor
import io
import json
from datetime import datetime
import random

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='crop_prediction.log',
    filemode='a'
)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'crop_prediction_app'

# Load and prepare data
def load_data():
    data = pd.read_csv('crop_data.csv')
    return data

# Train crop prediction model
def train_crop_model(data):
    # Ensure groundnut is included in the crop labels
    if 'groundnut' not in data['crop'].unique():
        logging.warning("Groundnut not found in training data. Adding synthetic data.")
        # Add synthetic groundnut data if not present
        groundnut_samples = data.sample(n=50, random_state=42).copy()
        groundnut_samples['crop'] = 'groundnut'
        data = pd.concat([data, groundnut_samples], ignore_index=True)
    
    # Prepare features for crop prediction
    X = data[['rainfall', 'temperature', 'humidity', 'soil_ph', 'nitrogen', 'phosphorus', 'potassium']]
    y_crop = data['crop']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_crop, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train crop prediction model
    crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
    crop_model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    joblib.dump(crop_model, 'crop_prediction_model.joblib')
    joblib.dump(scaler, 'crop_feature_scaler.joblib')
    
    return crop_model, scaler

# Train yield prediction model
def train_yield_model(data):
    # Prepare features for yield prediction
    X = data[['rainfall', 'temperature', 'humidity', 'soil_ph', 'nitrogen', 'phosphorus', 'potassium']]
    y_yield = data['yield_per_hectare']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_yield, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train yield prediction model
    yield_model = RandomForestRegressor(n_estimators=100, random_state=42)
    yield_model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    joblib.dump(yield_model, 'yield_prediction_model.joblib')
    joblib.dump(scaler, 'yield_feature_scaler.joblib')
    
    return yield_model, scaler

# Prediction functions
def predict_crop(features, model, scaler, top_n=3):
    """
    Predict top N crops based on environmental conditions with robust error handling
    
    Args:
        features (array): Environmental features
        model (RandomForestClassifier): Trained crop prediction model
        scaler (StandardScaler): Feature scaler
        top_n (int): Number of top crop recommendations to return
    
    Returns:
        list: Top N crop recommendations with probabilities
    """
    try:
        # Ensure features are 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict probabilities with error handling
        try:
            crop_probabilities = model.predict_proba(features_scaled)[0]
        except Exception as prob_error:
            logging.warning(f"Probability prediction error: {prob_error}")
            # Fallback to predict method
            crop_predictions = model.predict(features_scaled)
            crop_probabilities = np.ones(len(model.classes_)) / len(model.classes_)
        
        # Get crop labels from the model
        crop_labels = model.classes_
        
        # Create a list of (crop, probability) tuples and sort
        crop_prob_pairs = list(zip(crop_labels, crop_probabilities))
        sorted_crops = sorted(crop_prob_pairs, key=lambda x: x[1], reverse=True)
        
        # Return top N recommendations with their probabilities
        top_recommendations = [
            {
                'crop': crop, 
                'probability': round(prob * 100, 2)
            } 
            for crop, prob in sorted_crops[:top_n]
        ]
        
        # Ensure at least one recommendation
        if not top_recommendations:
            top_recommendations = [{'crop': 'default', 'probability': 100}]
        
        return top_recommendations
    
    except Exception as e:
        logging.error(f"Crop Prediction Error: {e}")
        return [{'crop': 'default', 'probability': 100}]

def predict_yield(features, model, scaler):
    """
    Predict crop yield with robust error handling
    
    Args:
        features (array): Environmental features
        model (RandomForestRegressor): Trained yield prediction model
        scaler (StandardScaler): Feature scaler
    
    Returns:
        float: Predicted yield per hectare
    """
    try:
        # Ensure features are 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict yield
        predicted_yield = model.predict(features_scaled)[0]
        
        # Add sanity checks
        predicted_yield = max(min(predicted_yield, 10), 0.5)  # Clamp between 0.5 and 10
        
        return round(predicted_yield, 2)
    
    except Exception as e:
        logging.error(f"Yield Prediction Error: {e}")
        return 3.0  # Default reasonable yield

# Sustainability Scoring Function
def calculate_sustainability_score(crop, yield_per_hectare, environmental_data, disease_risk):
    """
    Calculate a comprehensive sustainability score based on multiple factors.
    
    Parameters:
    - crop (str): Type of crop
    - yield_per_hectare (float): Crop yield
    - environmental_data (dict): Environmental conditions
    - disease_risk (float): Disease risk probability
    
    Returns:
    - dict: Sustainability score with breakdown and overall score
    """
    try:
        # Sustainability factors
        factors = {
            'water_efficiency': 0,
            'soil_health': 0,
            'climate_resilience': 0,
            'disease_resistance': 0,
            'yield_potential': 0
        }
        
        # Water Efficiency (based on rainfall and irrigation needs)
        rainfall = environmental_data.get('rainfall', 0)
        factors['water_efficiency'] = min(max(rainfall / 1000, 0), 1) * 20
        
        # Soil Health (based on temperature and humidity)
        temperature = environmental_data.get('temperature', 0)
        humidity = environmental_data.get('humidity', 0)
        soil_health_score = 1 - abs(temperature - 25)/50 * 0.5 - abs(humidity - 60)/50 * 0.5
        factors['soil_health'] = max(soil_health_score, 0) * 20
        
        # Climate Resilience
        climate_variation = abs(temperature - 25) / 10
        factors['climate_resilience'] = max(1 - climate_variation, 0) * 20
        
        # Disease Resistance
        factors['disease_resistance'] = max(1 - disease_risk, 0) * 20
        
        # Yield Potential
        # Normalize yield against typical crop yields
        crop_typical_yields = {
            'wheat': 3.5,
            'rice': 4.0,
            'corn': 5.0,
            'soybean': 2.5,
            'cotton': 1.5,
            'groundnut': 2.0  # Add groundnut yield
        }
        typical_yield = crop_typical_yields.get(crop.lower(), 3.0)
        yield_ratio = min(yield_per_hectare / typical_yield, 1)
        factors['yield_potential'] = yield_ratio * 20
        
        # Calculate overall sustainability score
        sustainability_score = sum(factors.values())
        
        return {
            'overall_score': round(sustainability_score, 2),
            'factors': {k: round(v, 2) for k, v in factors.items()},
            'score_breakdown': f"""
            Water Efficiency: {factors['water_efficiency']:.2f}%
            Soil Health: {factors['soil_health']:.2f}%
            Climate Resilience: {factors['climate_resilience']:.2f}%
            Disease Resistance: {factors['disease_resistance']:.2f}%
            Yield Potential: {factors['yield_potential']:.2f}%
            """
        }
    except Exception as e:
        logging.error(f"Sustainability Score Calculation Error: {e}")
        return {
            'overall_score': 0,
            'factors': {},
            'score_breakdown': 'Calculation Error'
        }

# Country-based Crop Recommendation
COUNTRY_CROP_RECOMMENDATIONS = {
    'United States': ['corn', 'soybean', 'wheat', 'cotton', 'groundnut'],
    'India': ['rice', 'wheat', 'sugarcane', 'cotton', 'groundnut', 'millet'],
    'Brazil': ['soybean', 'sugarcane', 'corn', 'coffee', 'groundnut'],
    'China': ['rice', 'wheat', 'corn', 'soybean', 'groundnut'],
    'Argentina': ['soybean', 'corn', 'wheat', 'sunflower', 'groundnut'],
    'Canada': ['wheat', 'canola', 'barley', 'oats', 'groundnut'],
    'Australia': ['wheat', 'barley', 'cotton', 'canola', 'groundnut'],
    'Nigeria': ['cassava', 'yam', 'maize', 'rice', 'groundnut'],
    'Russia': ['wheat', 'barley', 'sunflower', 'potato', 'groundnut'],
    'Mexico': ['corn', 'beans', 'sugarcane', 'coffee', 'groundnut'],
    'France': ['wheat', 'corn', 'barley', 'rapeseed', 'groundnut'],
    'Germany': ['wheat', 'barley', 'potatoes', 'sugar beet', 'groundnut'],
    'Ukraine': ['wheat', 'corn', 'sunflower', 'barley', 'groundnut'],
    'Indonesia': ['rice', 'cassava', 'corn', 'coconut', 'groundnut'],
    'Turkey': ['wheat', 'barley', 'cotton', 'sugar beet', 'groundnut'],
    'Pakistan': ['wheat', 'cotton', 'rice', 'sugarcane', 'groundnut'],
    'Egypt': ['cotton', 'rice', 'wheat', 'maize', 'groundnut'],
    'Vietnam': ['rice', 'coffee', 'rubber', 'pepper', 'groundnut'],
    'Thailand': ['rice', 'cassava', 'sugarcane', 'corn', 'groundnut'],
    'South Africa': ['corn', 'wheat', 'sugarcane', 'sunflower', 'groundnut'],
    'Spain': ['wheat', 'barley', 'corn', 'olive', 'groundnut'],
    'Italy': ['wheat', 'corn', 'tomatoes', 'grapes', 'groundnut'],
    'United Kingdom': ['wheat', 'barley', 'potatoes', 'oats', 'groundnut'],
    'Poland': ['wheat', 'potatoes', 'sugar beet', 'rye', 'groundnut'],
    'Iran': ['wheat', 'rice', 'barley', 'pistachio', 'groundnut']
}

# Currency Conversion (approximate rates)
def convert_to_usd(price, country='India'):
    """
    Convert market price to USD based on country
    Supports multiple currency conversions
    """
    currency_mapping = {
        'United States': ('USD', 1.0),
        'India': ('INR', 0.012),
        'Brazil': ('BRL', 0.20),
        'China': ('CNY', 0.14),
        'Argentina': ('ARS', 0.011),
        'Canada': ('CAD', 0.74),
        'Australia': ('AUD', 0.66),
        'Nigeria': ('NGN', 0.0013),
        'Russia': ('RUB', 0.011),
        'Mexico': ('MXN', 0.059),
        'France': ('EUR', 1.08),
        'Germany': ('EUR', 1.08),
        'Ukraine': ('UAH', 0.027),
        'Indonesia': ('IDR', 0.000066),
        'Turkey': ('TRY', 0.039),
        'Pakistan': ('PKR', 0.0036),
        'Egypt': ('EGP', 0.032),
        'Vietnam': ('VND', 0.000042),
        'Thailand': ('THB', 0.029),
        'South Africa': ('ZAR', 0.054),
        'Spain': ('EUR', 1.08),
        'Italy': ('EUR', 1.08),
        'United Kingdom': ('GBP', 1.26),
        'Poland': ('PLN', 0.24),
        'Iran': ('IRR', 0.000024)
    }
    
    # Get currency and conversion rate, default to Indian Rupee
    currency, rate = currency_mapping.get(country, ('INR', 0.012))
    
    return round(price * rate, 2)

# PDF Export Function
def generate_prediction_report(prediction_data):
    """
    Generate a detailed PDF report of crop prediction
    """
    pdf_path = 'crop_prediction_report.pdf'
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    report_content = []
    
    title = Paragraph("Crop Prediction & Sustainability Report", styles['Title'])
    report_content.append(title)
    
    details = [
        ['Crop', prediction_data['crop']],
        ['Estimated Yield', f"{prediction_data['yield_per_hectare']} tonnes/hectare"],
        ['Sustainability Score', f"{prediction_data['sustainability_score']}%"],
        ['Water Requirement', f"{prediction_data['water_requirement']} mm/season"],
        ['Market Price', f"${prediction_data['market_price']}/kg"]
    ]
    
    table = Table(details)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.green),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    
    report_content.append(table)
    
    doc.build(report_content)
    return pdf_path

# Advanced Machine Learning Features

# Ensemble Model for Improved Predictions
def create_ensemble_model(data):
    """
    Create an ensemble model combining multiple algorithms
    for more robust crop and yield predictions
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    
    # Prepare features and targets
    features = data.drop(['crop', 'yield_per_hectare'], axis=1)
    crop_target = data['crop']
    yield_target = data['yield_per_hectare']
    
    # Crop Prediction Ensemble
    crop_models = [
        RandomForestClassifier(n_estimators=100, random_state=42),
        # Add more classifiers like GradientBoostingClassifier
    ]
    
    # Yield Prediction Ensemble
    yield_models = [
        RandomForestRegressor(n_estimators=100, random_state=42),
        # Add more regressors like GradientBoostingRegressor
    ]
    
    # Cross-validation for model performance
    def evaluate_model(models, X, y):
        scores = []
        for model in models:
            cv_scores = cross_val_score(model, X, y, cv=5)
            scores.append(cv_scores.mean())
        return scores
    
    crop_model_scores = evaluate_model(crop_models, features, crop_target)
    yield_model_scores = evaluate_model(yield_models, features, yield_target)
    
    return {
        'crop_models': crop_models,
        'yield_models': yield_models,
        'crop_model_scores': crop_model_scores,
        'yield_model_scores': yield_model_scores
    }

# Advanced Feature Engineering
def engineer_advanced_features(data):
    """
    Create new features to improve model performance
    """
    # Interaction terms
    data['temp_humidity_interaction'] = data['temperature'] * data['humidity']
    data['rainfall_nitrogen_interaction'] = data['rainfall'] * data['nitrogen']
    
    # Polynomial features
    data['temperature_squared'] = data['temperature'] ** 2
    data['rainfall_squared'] = data['rainfall'] ** 2
    
    # Ratio features
    data['nutrient_balance'] = (data['nitrogen'] + data['phosphorus'] + data['potassium']) / 3
    
    return data

# Risk Assessment Function
def assess_crop_risk(crop_details):
    """
    Evaluate potential risks for crop cultivation
    """
    risk_factors = {
        'climate_volatility': abs(crop_details['temperature'] - 25),  # Deviation from optimal temp
        'water_stress': max(0, 800 - crop_details['rainfall']),  # Water availability
        'nutrient_imbalance': abs(50 - crop_details['nitrogen']),  # Nitrogen deviation
    }
    
    # Calculate overall risk score
    risk_score = sum(risk_factors.values()) / len(risk_factors)
    
    return {
        'risk_score': risk_score,
        'risk_factors': risk_factors,
        'risk_level': (
            'Low' if risk_score < 20 else 
            'Medium' if risk_score < 50 else 
            'High'
        )
    }

# Climate Change Impact Projection
def project_climate_change_impact(crop, years_ahead=10):
    """
    Simulate potential crop performance under climate change scenarios
    """
    climate_scenarios = {
        'conservative': {'temp_increase': 0.5, 'rainfall_change': -5},
        'moderate': {'temp_increase': 1.0, 'rainfall_change': -10},
        'extreme': {'temp_increase': 2.0, 'rainfall_change': -20}
    }
    
    projections = {}
    for scenario, changes in climate_scenarios.items():
        projections[scenario] = {
            'expected_yield_change': f"{changes['temp_increase'] * 5}%",
            'water_requirement_change': f"{changes['rainfall_change']}%",
            'adaptation_strategies': [
                'Drought-resistant varieties',
                'Improved irrigation',
                'Crop diversification'
            ]
        }
    
    return projections

# Agricultural Knowledge Base and Expert System

# Crop Companion Planting Recommendations
COMPANION_PLANTING_GUIDE = {
    'wheat': {
        'good_companions': ['clover', 'alfalfa'],
        'bad_companions': ['beans', 'potatoes']
    },
    'rice': {
        'good_companions': ['azolla', 'fish'],
        'bad_companions': ['weeds']
    },
    'corn': {
        'good_companions': ['beans', 'squash'],
        'bad_companions': ['tomatoes']
    },
    'soybean': {
        'good_companions': ['corn', 'sunflower'],
        'bad_companions': ['wheat']
    },
    'groundnut': {
        'good_companions': ['maize', 'sorghum'],
        'bad_companions': ['cotton']
    }
}

# Pest and Disease Management
PEST_DISEASE_MANAGEMENT = {
    'wheat': {
        'common_pests': ['aphids', 'rust'],
        'prevention_methods': [
            'Crop rotation',
            'Use resistant varieties',
            'Maintain field hygiene'
        ],
        'organic_treatments': [
            'Neem oil spray',
            'Introduce beneficial insects'
        ]
    },
    'rice': {
        'common_pests': ['brown planthopper', 'blast disease'],
        'prevention_methods': [
            'Water management',
            'Balanced fertilization',
            'Timely planting'
        ],
        'organic_treatments': [
            'Botanical pesticides',
            'Biological control agents'
        ]
    },
    'corn': {
        'common_pests': ['stem borers', 'rootworms'],
        'prevention_methods': [
            'Crop rotation',
            'Use resistant varieties',
            'Maintain field hygiene'
        ],
        'organic_treatments': [
            'Neem oil spray',
            'Introduce beneficial insects'
        ]
    },
    'soybean': {
        'common_pests': ['aphids', 'whiteflies'],
        'prevention_methods': [
            'Crop rotation',
            'Use resistant varieties',
            'Maintain field hygiene'
        ],
        'organic_treatments': [
            'Neem oil spray',
            'Introduce beneficial insects'
        ]
    },
    'groundnut': {
        'common_pests': ['aphids', 'thrips'],
        'prevention_methods': [
            'Crop rotation',
            'Use resistant varieties',
            'Maintain field hygiene'
        ],
        'organic_treatments': [
            'Neem oil spray',
            'Introduce beneficial insects'
        ]
    }
}

# Irrigation and Water Management
def calculate_irrigation_schedule(crop, environmental_conditions):
    """
    Generate a smart irrigation schedule based on crop and environmental factors
    """
    water_requirement_map = {
        'wheat': {
            'base_water': 450,  # mm
            'critical_stages': ['tillering', 'grain-filling'],
            'water_use_efficiency': 0.8,
            'climate_factor_range': (0.7, 1.3)
        },
        'rice': {
            'base_water': 750,  # mm
            'critical_stages': ['transplanting', 'flowering'],
            'water_use_efficiency': 0.6,
            'climate_factor_range': (0.5, 1.5)
        },
        'corn': {
            'base_water': 500,  # mm
            'critical_stages': ['tasseling', 'grain-filling'],
            'water_use_efficiency': 0.75,
            'climate_factor_range': (0.6, 1.4)
        },
        'groundnut': {
            'base_water': 400,  # mm
            'critical_stages': ['flowering', 'pod-filling'],
            'water_use_efficiency': 0.85,
            'climate_factor_range': (0.7, 1.2)
        }
    }
    
    crop_water_data = water_requirement_map.get(crop, {})
    
    # Dynamic water adjustment based on environmental conditions
    water_adjustment_factors = {
        'temperature': environmental_conditions['temperature'] * 1.2,
        'humidity': 100 - environmental_conditions['humidity'],
        'rainfall': environmental_conditions['rainfall'] * 0.5
    }
    
    adjusted_water_requirement = crop_water_data.get('base_water', 500) - sum(water_adjustment_factors.values())
    
    irrigation_schedule = {
        'total_water_requirement': max(0, adjusted_water_requirement),
        'critical_stages': crop_water_data.get('critical_stages', []),
        'irrigation_intervals': ['7-10 days', '14-21 days'],
        'water_saving_tips': [
            'Use drip irrigation',
            'Mulch to reduce evaporation',
            'Monitor soil moisture'
        ]
    }
    
    return irrigation_schedule

# Precision Agriculture Insights
def generate_precision_agriculture_recommendations(crop_details):
    """
    Generate site-specific agricultural recommendations
    """
    precision_insights = {
        'soil_management': {
            'optimal_ph_range': (6.0, 7.0),
            'recommended_nutrients': {
                'nitrogen': (40, 60),
                'phosphorus': (30, 50),
                'potassium': (20, 40)
            }
        },
        'nutrient_optimization': {
            'current_balance': {
                'nitrogen': crop_details['nitrogen'],
                'phosphorus': crop_details['phosphorus'],
                'potassium': crop_details['potassium']
            },
            'recommendations': []
        },
        'precision_technologies': [
            'GPS-guided machinery',
            'Drone crop monitoring',
            'Soil sensors',
            'Variable rate fertilization'
        ]
    }
    
    # Nutrient balance recommendations
    if (precision_insights['soil_management']['recommended_nutrients']['nitrogen'][0] > crop_details['nitrogen']):
        precision_insights['nutrient_optimization']['recommendations'].append('Increase Nitrogen')
    
    if (precision_insights['soil_management']['recommended_nutrients']['phosphorus'][0] > crop_details['phosphorus']):
        precision_insights['nutrient_optimization']['recommendations'].append('Increase Phosphorus')
    
    return precision_insights

# Market Price Prediction and Risk Assessment
import numpy as np
from datetime import datetime, timedelta

class GlobalMarketTrendAnalyzer:
    def __init__(self):
        # Global market trend data
        self.global_market_trends = {
            'wheat': {
                'base_volatility': 0.15,
                'global_factors': {
                    'climate_change': 0.05,
                    'geopolitical_tension': 0.03,
                    'trade_policies': 0.02
                }
            },
            'rice': {
                'base_volatility': 0.18,
                'global_factors': {
                    'climate_change': 0.06,
                    'water_scarcity': 0.04,
                    'population_growth': 0.03
                }
            },
            'corn': {
                'base_volatility': 0.16,
                'global_factors': {
                    'biofuel_demand': 0.04,
                    'climate_change': 0.05,
                    'trade_policies': 0.03
                }
            },
            'groundnut': {
                'base_volatility': 0.14,
                'global_factors': {
                    'climate_change': 0.04,
                    'water_scarcity': 0.03,
                    'population_growth': 0.02
                }
            }
        }

    def analyze_global_trends(self, crop, current_price):
        """
        Analyze global market trends and provide risk assessment
        """
        crop_data = self.global_market_trends.get(crop.lower(), 
            self.global_market_trends.get('wheat'))  # Default to wheat if not found
        
        # Calculate global trend impact
        trend_impact = sum(crop_data['global_factors'].values())
        
        # Risk categorization
        risk_levels = {
            (0, 0.05): 'Low',
            (0.05, 0.1): 'Medium',
            (0.1, float('inf')): 'High'
        }
        
        # Determine risk level
        risk_level = next(
            level for (low, high), level in risk_levels.items() 
            if low <= trend_impact < high
        )
        
        # Future price projection
        future_price_projection = current_price * (1 + trend_impact)
        
        return {
            'trend_impact': round(trend_impact * 100, 2),
            'risk_level': risk_level,
            'future_price_projection': round(future_price_projection, 2),
            'key_global_factors': list(crop_data['global_factors'].keys())
        }

# Global market trend analyzer
global_market_trend_analyzer = GlobalMarketTrendAnalyzer()

class MarketPricePredictor:
    def __init__(self):
        # Expanded crop price data with more robust fallback
        self.crop_price_data = {
            'wheat': {
                'base_price': 250,  # USD per ton
                'volatility_factor': 0.15,
                'seasonal_patterns': {
                    'winter': -0.1,
                    'summer': 0.1
                },
                'default_risk_level': 'Medium'
            },
            'rice': {
                'base_price': 400,
                'volatility_factor': 0.2,
                'seasonal_patterns': {
                    'winter': -0.05,
                    'summer': 0.05
                },
                'default_risk_level': 'Low'
            },
            'corn': {
                'base_price': 300,
                'volatility_factor': 0.18,
                'seasonal_patterns': {
                    'winter': -0.08,
                    'summer': 0.08
                },
                'default_risk_level': 'High'
            },
            'groundnut': {
                'base_price': 350,
                'volatility_factor': 0.14,
                'seasonal_patterns': {
                    'winter': -0.06,
                    'summer': 0.06
                },
                'default_risk_level': 'Medium'
            },
            # Fallback for unknown crops
            'default': {
                'base_price': 200,
                'volatility_factor': 0.1,
                'seasonal_patterns': {
                    'winter': -0.05,
                    'summer': 0.05
                },
                'default_risk_level': 'Medium'
            }
        }

    def predict_market_price(self, crop, current_conditions):
        """
        Predict market price with robust risk assessment
        """
        # Normalize crop name and use default if not found
        crop = crop.lower()
        crop_data = self.crop_price_data.get(crop, self.crop_price_data['default'])
        
        # Ensure all required conditions are present
        current_conditions = current_conditions or {}
        temperature = current_conditions.get('temperature', 25)
        rainfall = current_conditions.get('rainfall', 500)
        
        base_price = crop_data['base_price']
        
        # Seasonal adjustment with safe fallback
        try:
            current_month = datetime.now().month
            season = 'summer' if 5 <= current_month <= 10 else 'winter'
            seasonal_adjustment = crop_data['seasonal_patterns'].get(season, 0)
        except Exception:
            season = 'N/A'
            seasonal_adjustment = 0
        
        # Risk factors calculation with safe defaults
        try:
            risk_factors = {
                'temperature_deviation': abs(temperature - 25) * 0.02,
                'rainfall_deviation': abs(rainfall - 500) * 0.01
            }
        except Exception:
            risk_factors = {'temperature_deviation': 0, 'rainfall_deviation': 0}
        
        # Calculate final price and risk
        price_adjustment = sum(risk_factors.values()) * (1 + seasonal_adjustment)
        predicted_price = base_price * (1 + price_adjustment)
        
        risk_score = sum(risk_factors.values()) * 10  # Normalized risk score
        
        # Add global market trend analysis
        global_trend_insights = global_market_trend_analyzer.analyze_global_trends(
            crop, 
            predicted_price
        )
        
        # Merge insights
        return {
            'predicted_price': round(max(predicted_price, 50), 2),  # Ensure minimum price
            'risk_score': round(min(max(risk_score, 0), 10), 2),  # Clamp risk score
            'risk_level': (
                'Low' if risk_score < 3 else 
                'Medium' if risk_score < 7 else 
                'High'
            ),
            'seasonal_trend': season,
            'global_trend_impact': global_trend_insights['trend_impact'],
            'future_price_projection': global_trend_insights['future_price_projection'],
            'key_global_factors': global_trend_insights['key_global_factors']
        }

# Reinitialize market price predictor
market_price_predictor = MarketPricePredictor()

# Comprehensive Crop Price Calculation
def calculate_crop_price(crop, year=2024):
    """
    Calculate comprehensive crop price with multiple market factors
    
    Args:
        crop (str): Name of the crop
        year (int): Reference year for price calculation
    
    Returns:
        dict: Detailed price information
    """
    # Global and regional price variations
    global_price_factors = {
        'wheat': {
            'base_price_usd_per_kg': 0.25,
            'price_volatility': 0.15,
            'market_factors': {
                'global_demand': 0.1,
                'climate_impact': 0.05,
                'geopolitical_events': 0.03
            },
            'regional_variations': {
                'North America': 1.0,
                'Europe': 1.1,
                'Asia': 0.9,
                'Africa': 0.8
            }
        },
        'rice': {
            'base_price_usd_per_kg': 0.45,
            'price_volatility': 0.2,
            'market_factors': {
                'global_demand': 0.15,
                'climate_impact': 0.07,
                'trade_policies': 0.05
            },
            'regional_variations': {
                'North America': 0.9,
                'Europe': 0.8,
                'Asia': 1.2,
                'Africa': 1.1
            }
        },
        'corn': {
            'base_price_usd_per_kg': 0.20,
            'price_volatility': 0.18,
            'market_factors': {
                'biofuel_demand': 0.1,
                'livestock_feed': 0.08,
                'climate_impact': 0.04
            },
            'regional_variations': {
                'North America': 1.2,
                'Europe': 0.9,
                'Asia': 0.8,
                'Africa': 0.7
            }
        },
        'groundnut': {
            'base_price_usd_per_kg': 1.50,
            'price_volatility': 0.25,
            'market_factors': {
                'global_demand': 0.12,
                'oil_production': 0.06,
                'climate_impact': 0.05
            },
            'regional_variations': {
                'North America': 1.1,
                'Europe': 0.9,
                'Asia': 1.2,
                'Africa': 1.3
            }
        },
        'soybean': {
            'base_price_usd_per_kg': 0.60,
            'price_volatility': 0.22,
            'market_factors': {
                'global_demand': 0.13,
                'biofuel_demand': 0.07,
                'trade_policies': 0.04
            },
            'regional_variations': {
                'North America': 1.2,
                'Europe': 0.9,
                'Asia': 1.0,
                'Africa': 0.8
            }
        },
        'default': {
            'base_price_usd_per_kg': 0.35,
            'price_volatility': 0.16,
            'market_factors': {
                'global_demand': 0.08,
                'climate_impact': 0.04
            },
            'regional_variations': {
                'North America': 1.0,
                'Europe': 0.9,
                'Asia': 1.0,
                'Africa': 0.8
            }
        }
    }
    
    # Get crop-specific price data or use default
    crop_price_data = global_price_factors.get(crop.lower(), global_price_factors['default'])
    
    # Calculate price with market factors
    base_price = crop_price_data['base_price_usd_per_kg']
    price_volatility = crop_price_data['price_volatility']
    
    # Simulate market factor impact
    market_factor_impact = sum(crop_price_data['market_factors'].values())
    
    # Add some randomness to simulate real-world price variations
    random_variation = random.uniform(-price_volatility, price_volatility)
    
    # Calculate final price
    final_price = base_price * (1 + market_factor_impact + random_variation)
    
    return {
        'base_price_usd_per_kg': round(base_price, 2),
        'final_price_usd_per_kg': round(final_price, 2),
        'price_volatility': price_volatility * 100,
        'market_factors': {k: round(v * 100, 2) for k, v in crop_price_data['market_factors'].items()},
        'description': f"Market price for {crop} as of {year}"
    }

# Water Requirement Calculation
def calculate_water_requirement(crop, growing_period_days=120):
    """
    Calculate comprehensive water requirement for a crop
    
    Args:
        crop (str): Name of the crop
        growing_period_days (int): Total growing period in days
    
    Returns:
        dict: Water requirement details
    """
    water_requirement_map = {
        'wheat': {
            'base_water': 450,  # mm
            'critical_stages': ['tillering', 'grain-filling'],
            'water_use_efficiency': 0.8,
            'climate_factor_range': (0.7, 1.3)
        },
        'rice': {
            'base_water': 750,  # mm
            'critical_stages': ['transplanting', 'flowering'],
            'water_use_efficiency': 0.6,
            'climate_factor_range': (0.5, 1.5)
        },
        'corn': {
            'base_water': 500,  # mm
            'critical_stages': ['tasseling', 'grain-filling'],
            'water_use_efficiency': 0.75,
            'climate_factor_range': (0.6, 1.4)
        },
        'groundnut': {
            'base_water': 400,  # mm
            'critical_stages': ['flowering', 'pod-filling'],
            'water_use_efficiency': 0.85,
            'climate_factor_range': (0.7, 1.2)
        },
        'soybean': {
            'base_water': 450,  # mm
            'critical_stages': ['flowering', 'pod-filling'],
            'water_use_efficiency': 0.8,
            'climate_factor_range': (0.7, 1.3)
        },
        'default': {
            'base_water': 500,  # mm
            'critical_stages': ['vegetative', 'reproductive'],
            'water_use_efficiency': 0.7,
            'climate_factor_range': (0.6, 1.4)
        }
    }
    
    # Get crop-specific water data or use default
    crop_water_data = water_requirement_map.get(crop.lower(), water_requirement_map['default'])
    
    # Calculate total water requirement
    base_water = crop_water_data['base_water']
    water_use_efficiency = crop_water_data['water_use_efficiency']
    min_factor, max_factor = crop_water_data['climate_factor_range']
    
    # Adjust water requirement based on growing period and efficiency
    total_water_requirement = base_water * (growing_period_days / 120) * water_use_efficiency
    
    return {
        'total_water_requirement': round(total_water_requirement, 2),
        'base_water': base_water,
        'critical_stages': crop_water_data['critical_stages'],
        'water_use_efficiency': water_use_efficiency * 100,
        'description': f"Water requirement for {crop} during a {growing_period_days}-day growing period"
    }

# Comprehensive Crop Features
def get_crop_comprehensive_details(crop):
    """
    Provide comprehensive details for each crop
    
    Args:
        crop (str): Name of the crop
    
    Returns:
        dict: Comprehensive crop details
    """
    crop_details = {
        'wheat': {
            'nutritional_profile': {
                'protein': '13-14%',
                'carbohydrates': '70-75%',
                'fiber': '2-3%'
            },
            'climate_suitability': {
                'temperature_range': '10-25°C',
                'rainfall_requirement': '350-500 mm',
                'growing_period': '90-130 days'
            },
            'economic_value': {
                'global_production': '771 million tons (2021)',
                'top_producers': ['China', 'India', 'Russia', 'USA']
            },
            'sustainability_score': 0.75,
            'carbon_sequestration': 'Moderate',
            'soil_health_impact': 'Positive'
        },
        'rice': {
            'nutritional_profile': {
                'protein': '7-8%',
                'carbohydrates': '80-85%',
                'fiber': '0.5-1%'
            },
            'climate_suitability': {
                'temperature_range': '20-35°C',
                'rainfall_requirement': '1000-1500 mm',
                'growing_period': '100-150 days'
            },
            'economic_value': {
                'global_production': '518 million tons (2021)',
                'top_producers': ['China', 'India', 'Bangladesh', 'Indonesia']
            },
            'sustainability_score': 0.65,
            'carbon_sequestration': 'Low',
            'soil_health_impact': 'Neutral'
        },
        'corn': {
            'nutritional_profile': {
                'protein': '9-10%',
                'carbohydrates': '70-75%',
                'fiber': '2-3%'
            },
            'climate_suitability': {
                'temperature_range': '15-35°C',
                'rainfall_requirement': '500-800 mm',
                'growing_period': '90-120 days'
            },
            'economic_value': {
                'global_production': '1.2 billion tons (2021)',
                'top_producers': ['USA', 'China', 'Brazil', 'Argentina']
            },
            'sustainability_score': 0.7,
            'carbon_sequestration': 'High',
            'soil_health_impact': 'Moderate'
        },
        'groundnut': {
            'nutritional_profile': {
                'protein': '25-30%',
                'carbohydrates': '20-25%',
                'fiber': '3-4%'
            },
            'climate_suitability': {
                'temperature_range': '20-35°C',
                'rainfall_requirement': '500-700 mm',
                'growing_period': '120-150 days'
            },
            'economic_value': {
                'global_production': '48 million tons (2021)',
                'top_producers': ['China', 'India', 'Nigeria', 'USA']
            },
            'sustainability_score': 0.85,
            'carbon_sequestration': 'High',
            'soil_health_impact': 'Positive'
        },
        'soybean': {
            'nutritional_profile': {
                'protein': '35-40%',
                'carbohydrates': '30-35%',
                'fiber': '3-4%'
            },
            'climate_suitability': {
                'temperature_range': '20-30°C',
                'rainfall_requirement': '450-750 mm',
                'growing_period': '90-120 days'
            },
            'economic_value': {
                'global_production': '369 million tons (2021)',
                'top_producers': ['USA', 'Brazil', 'Argentina', 'China']
            },
            'sustainability_score': 0.8,
            'carbon_sequestration': 'High',
            'soil_health_impact': 'Positive'
        },
        'default': {
            'nutritional_profile': {
                'protein': '10-15%',
                'carbohydrates': '60-70%',
                'fiber': '2-3%'
            },
            'climate_suitability': {
                'temperature_range': '15-30°C',
                'rainfall_requirement': '500-700 mm',
                'growing_period': '100-140 days'
            },
            'economic_value': {
                'global_production': 'Varies',
                'top_producers': ['Global Producers']
            },
            'sustainability_score': 0.7,
            'carbon_sequestration': 'Moderate',
            'soil_health_impact': 'Neutral'
        }
    }
    
    return crop_details.get(crop.lower(), crop_details['default'])

# Routes
@app.route('/')
def index():
    countries = sorted(list(COUNTRY_CROP_RECOMMENDATIONS.keys()))
    return render_template('index.html', countries=countries)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extensive logging for debugging
        logging.info("Starting prediction process")
        logging.info(f"Received form data: {dict(request.form)}")

        # Validate and collect input features with more robust error handling
        try:
            features = np.array([[
                float(request.form.get('rainfall', 500)),  # Default values added
                float(request.form.get('temperature', 25)),
                float(request.form.get('humidity', 60)),
                float(request.form.get('soil_ph', 7.0)),
                float(request.form.get('nitrogen', 50)),
                float(request.form.get('phosphorus', 50)),
                float(request.form.get('potassium', 50))
            ]])
        except ValueError as ve:
            logging.error(f"Input conversion error: {ve}")
            return render_template('error.html', 
                error="Invalid numeric input. Using default environmental conditions.")
            
            # Fallback to default features
            features = np.array([[500, 25, 60, 7.0, 50, 50, 50]])

        # Scale features
        try:
            scaled_features = crop_scaler.transform(features)
        except Exception as scale_error:
            logging.warning(f"Feature scaling error: {scale_error}")
            # Attempt to retrain scaler if possible
            from sklearn.preprocessing import StandardScaler
            crop_scaler = StandardScaler()
            scaled_features = crop_scaler.fit_transform(features)

        # Predict crop and yield with robust error handling
        try:
            top_crop_recommendations = predict_crop(
                features, 
                crop_model, 
                crop_scaler, 
                top_n=3  # Return top 3 crop recommendations
            )
            
            # Ensure we have at least one valid crop recommendation
            if not top_crop_recommendations:
                raise ValueError("No crop recommendations found")
            
            predicted_crop = top_crop_recommendations[0]['crop']
            predicted_yield = predict_yield(scaled_features, yield_model, yield_scaler)
        except Exception as prediction_error:
            logging.error(f"Prediction error: {prediction_error}")
            
            # More diverse fallback recommendations
            top_crop_recommendations = [
                {'crop': 'wheat', 'probability': 0.4},
                {'crop': 'rice', 'probability': 0.3},
                {'crop': 'groundnut', 'probability': 0.3}
            ]
            predicted_crop = top_crop_recommendations[0]['crop']
            predicted_yield = 3.0  # Default reasonable yield

        # Ensure top_crop_recommendations is a list of at least 3 crops
        if len(top_crop_recommendations) < 3:
            additional_crops = [
                {'crop': 'wheat', 'probability': 0.3},
                {'crop': 'rice', 'probability': 0.2},
                {'crop': 'corn', 'probability': 0.1}
            ]
            top_crop_recommendations.extend(
                crop for crop in additional_crops 
                if crop['crop'] not in [rec['crop'] for rec in top_crop_recommendations]
            )
            top_crop_recommendations = top_crop_recommendations[:3]
        
        # Prepare crop details for market price prediction
        crop_details = {
            'crop': predicted_crop,
            'yield_per_hectare': predicted_yield,
            'rainfall': features[0][0],
            'temperature': features[0][1],
            'humidity': features[0][2],
            'nitrogen': features[0][4],
            'phosphorus': features[0][5],
            'potassium': features[0][6]
        }

        # Market price prediction with extensive logging and fallback
        try:
            market_price_insights = market_price_predictor.predict_market_price(predicted_crop, {
                'temperature': crop_details['temperature'],
                'rainfall': crop_details['rainfall']
            })
        except Exception as market_price_error:
            logging.error(f"Market price prediction error: {market_price_error}")
            market_price_insights = {
                'predicted_price': 250,  # Default fallback price
                'risk_score': 5,
                'risk_level': 'Medium',
                'seasonal_trend': 'N/A',
                'global_trend_impact': 0,
                'future_price_projection': 250,
                'key_global_factors': []
            }

        # Country and market price handling
        country = request.form.get('country', 'India')  # Default to India
        country_crops = COUNTRY_CROP_RECOMMENDATIONS.get(country, ['groundnut', 'wheat', 'rice'])
        market_price = 500
        market_price_usd = convert_to_usd(market_price, country)

        # Risk assessment with fallback
        risk_assessment = {
            'risk_level': market_price_insights.get('risk_level', 'Medium'),
            'risk_score': market_price_insights.get('risk_score', 5),
            'global_trend_impact': market_price_insights.get('global_trend_impact', 0)
        }

        # Climate projection with more robust handling
        climate_projection = {
            'Optimistic': {
                'yield_impact': '+5%',
                'temperature_change': '+0.5°C',
                'water_availability': 'Stable'
            },
            'Moderate': {
                'yield_impact': '-2%',
                'temperature_change': '+1.2°C',
                'water_availability': 'Slightly Reduced'
            },
            'Pessimistic': {
                'yield_impact': '-10%',
                'temperature_change': '+2.5°C',
                'water_availability': 'Significantly Reduced'
            }
        }

        # Disease risk assessment
        disease_assessment = {
            'total_risk_score': 0.4,  # Low to moderate risk
            'common_diseases': ['Fungal infections', 'Bacterial blight'],
            'prevention_methods': ['Crop rotation', 'Proper irrigation']
        }

        # Sustainability score calculation
        sustainability_score = calculate_sustainability_score(
            predicted_crop, 
            predicted_yield, 
            crop_details, 
            disease_risk=disease_assessment.get('total_risk_score', 0.5)
        )
        
        # Add comprehensive details for top crop recommendations
        for recommendation in top_crop_recommendations:
            recommendation['water_requirement'] = calculate_water_requirement(recommendation['crop'])
            recommendation['market_price'] = calculate_crop_price(recommendation['crop'])
            recommendation['crop_details'] = get_crop_comprehensive_details(recommendation['crop'])
        
        return render_template('result.html', 
                               crop=predicted_crop, 
                               yield_per_hectare=round(predicted_yield, 2),
                               rainfall=features[0][0],
                               temperature=features[0][1],
                               humidity=features[0][2],
                               market_price=market_price_usd,
                               region_crops=country_crops,
                               market_price_insights=market_price_insights,
                               risk_assessment=risk_assessment,
                               climate_projection=climate_projection,
                               disease_assessment=disease_assessment,
                               disease_risks_json=json.dumps(disease_assessment),
                               sustainability_score=sustainability_score,
                               top_crop_recommendations=top_crop_recommendations)
    
    except Exception as e:
        logging.critical(f"Unexpected error: {e}")
        logging.critical(traceback.format_exc())
        return render_template('error.html', 
            error="An unexpected error occurred. Our team has been notified.")

@app.route('/download_report', methods=['POST'])
def download_report():
    try:
        # Retrieve prediction data from form with comprehensive fallback
        crop = request.form.get('crop', 'Unknown Crop')
        yield_per_hectare = request.form.get('yield_per_hectare', 'N/A')
        market_price = request.form.get('market_price', 'N/A')
        
        # Retrieve risk and sustainability data with safe defaults
        try:
            total_risk_score = float(request.form.get('total_risk_score', 5))
        except (ValueError, TypeError):
            total_risk_score = 5
        
        try:
            sustainability_score = float(request.form.get('sustainability_score', 50))
        except (ValueError, TypeError):
            sustainability_score = 50
        
        sustainability_breakdown = request.form.get('sustainability_breakdown', 'No detailed breakdown available')
        
        # Retrieve disease risks with safe defaults
        disease_risks = request.form.get('disease_risks', '{}')
        try:
            disease_risks = json.loads(disease_risks)
        except (json.JSONDecodeError, TypeError):
            disease_risks = {
                'Fungal Infection': 0.3,
                'Bacterial Blight': 0.2
            }
        
        # Retrieve mitigation strategies
        mitigation_strategies = request.form.get('mitigation_strategies', 
            "1. Implement crop rotation\n"
            "2. Use disease-resistant crop varieties\n"
            "3. Maintain proper soil nutrition\n"
            "4. Monitor and control irrigation"
        )
        
        # Create PDF with more comprehensive information
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Comprehensive report elements
        report_elements = []
        
        # Title
        title = Paragraph(f"Comprehensive Crop Prediction Report: {crop}", styles['Title'])
        report_elements.append(title)
        report_elements.append(Spacer(1, 12))
        
        # Prediction Details Section
        details_table_data = [
            ["Predicted Crop", str(crop)],
            ["Estimated Yield", f"{yield_per_hectare} tons/hectare"],
            ["Market Price", f"${market_price}/ton"],
            ["Generated on", datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        details_table = Table(details_table_data, colWidths=[150, 300])
        details_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        report_elements.append(details_table)
        report_elements.append(Spacer(1, 12))
        
        # Disease Risk Section
        disease_title = Paragraph("Disease Risk Management Assessment", styles['Heading2'])
        report_elements.append(disease_title)
        
        # Disease Risk Table
        disease_table_data = [["Disease", "Risk Level", "Risk Percentage"]]
        for disease, risk in disease_risks.items():
            try:
                risk_percentage = f"{float(risk)*100:.2f}%"
                risk_level = (
                    "Low" if risk < 0.2 else 
                    "Medium" if risk < 0.5 else 
                    "High"
                )
                disease_table_data.append([str(disease), risk_level, risk_percentage])
            except (ValueError, TypeError):
                # Skip invalid risk entries
                continue
        
        disease_table = Table(disease_table_data, colWidths=[200, 100, 100])
        disease_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkgreen),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        report_elements.append(disease_table)
        report_elements.append(Spacer(1, 12))
        
        # Sustainability Score Section
        sustainability_title = Paragraph("Sustainability Assessment", styles['Heading2'])
        report_elements.append(sustainability_title)
        
        sustainability_table_data = [
            ["Overall Sustainability Score", f"{sustainability_score}%"],
            ["Score Breakdown", str(sustainability_breakdown)]
        ]
        
        sustainability_table = Table(sustainability_table_data, colWidths=[200, 300])
        sustainability_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige)
        ]))
        report_elements.append(sustainability_table)
        report_elements.append(Spacer(1, 12))
        
        # Recommended Mitigation Strategies
        strategies_title = Paragraph("Recommended Mitigation Strategies", styles['Heading2'])
        report_elements.append(strategies_title)
        
        strategies_paragraph = Paragraph(str(mitigation_strategies), styles['Normal'])
        report_elements.append(strategies_paragraph)
        
        # Build PDF
        try:
            doc.build(report_elements)
        except Exception as build_error:
            logging.error(f"PDF Build Error: {build_error}")
            logging.error(traceback.format_exc())
            raise
        
        # Move buffer position to the beginning
        buffer.seek(0)
        
        # Send file for download
        return send_file(
            buffer, 
            as_attachment=True, 
            download_name=f'{crop}_crop_prediction_report.pdf',
            mimetype='application/pdf'
        )
    
    except Exception as e:
        # Log full traceback for debugging
        logging.error(f"PDF Generation Error: {e}")
        logging.error(f"Full Traceback: {traceback.format_exc()}")
        
        # Log all form data to help diagnose issues
        logging.error("Form Data Received:")
        for key, value in request.form.items():
            logging.error(f"{key}: {value}")
        
        # Flash a more informative error message
        flash(f"Error generating PDF: {e}", 'error')
        
        # Return a detailed error template
        return render_template('error.html', 
            error=f"Unable to generate PDF report. Error details: {e}")

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
            return '127.0.0.1'
    
    local_ip = get_local_ip()
    
    print("\n[NETWORK] Crop Prediction App Access URLs:")
    print(f"- Local:    http://127.0.0.1:5000")
    print(f"- Network:  http://{local_ip}:5000")
    print("\n[SERVER] Starting... Press Ctrl+C to stop.\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
