import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import logging

class CropDiseasePredictionSystem:
    def __init__(self):
        # Comprehensive disease database
        self.disease_database = {
            'wheat': {
                'diseases': [
                    'Leaf Rust', 
                    'Stem Rust', 
                    'Powdery Mildew', 
                    'Fusarium Head Blight'
                ],
                'risk_factors': [
                    'humidity', 
                    'temperature', 
                    'rainfall', 
                    'soil_moisture'
                ]
            },
            'rice': {
                'diseases': [
                    'Blast', 
                    'Bacterial Leaf Blight', 
                    'Brown Spot', 
                    'Sheath Blight'
                ],
                'risk_factors': [
                    'humidity', 
                    'temperature', 
                    'water_stagnation', 
                    'crop_density'
                ]
            },
            'corn': {
                'diseases': [
                    'Gray Leaf Spot', 
                    'Northern Corn Leaf Blight', 
                    'Corn Rust', 
                    'Maize Streak Virus'
                ],
                'risk_factors': [
                    'humidity', 
                    'temperature', 
                    'wind_speed', 
                    'soil_nutrients'
                ]
            }
        }
        
        # Machine learning model for disease risk prediction
        self.disease_risk_model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42
        )
        
        # Scaler for feature normalization
        self.scaler = StandardScaler()
        
    def preprocess_environmental_data(self, features):
        """
        Normalize and prepare environmental data for prediction
        """
        try:
            # Normalize features
            scaled_features = self.scaler.fit_transform(features)
            return scaled_features
        except Exception as e:
            logging.error(f"Feature preprocessing error: {e}")
            return None
    
    def predict_disease_risk(self, crop, environmental_data):
        """
        Predict disease risk for a specific crop
        
        :param crop: Type of crop
        :param environmental_data: Dict of environmental conditions
        :return: Disease risk assessment
        """
        try:
            # Validate crop
            if crop.lower() not in self.disease_database:
                logging.warning(f"No disease data available for {crop}")
                return self._default_risk_assessment(crop)
            
            # Extract relevant features
            crop_diseases = self.disease_database[crop.lower()]['diseases']
            risk_factors = self.disease_database[crop.lower()]['risk_factors']
            
            # Prepare feature vector
            feature_vector = [
                environmental_data.get(factor, 0) 
                for factor in risk_factors
            ]
            
            # Preprocess features
            processed_features = self.preprocess_environmental_data([feature_vector])
            
            # Predict disease probabilities
            disease_probabilities = self._simulate_disease_probabilities(
                processed_features, 
                crop_diseases
            )
            
            # Generate comprehensive risk assessment
            risk_assessment = {
                'crop': crop,
                'total_risk_score': sum(disease_probabilities.values()),
                'disease_risks': disease_probabilities,
                'recommended_actions': self._generate_mitigation_strategies(
                    disease_probabilities
                )
            }
            
            logging.info(f"Disease Risk Assessment for {crop}: {risk_assessment}")
            return risk_assessment
        
        except Exception as e:
            logging.critical(f"Disease risk prediction error: {e}")
            return self._default_risk_assessment(crop)
    
    def _simulate_disease_probabilities(self, features, diseases):
        """
        Simulate probabilistic disease risks
        """
        probabilities = {}
        for disease in diseases:
            # Simulated probabilistic risk calculation
            base_risk = np.random.uniform(0.05, 0.3)
            feature_impact = np.mean(features[0]) * 0.5
            probabilities[disease] = round(min(base_risk + feature_impact, 1), 3)
        return probabilities
    
    def _generate_mitigation_strategies(self, disease_risks):
        """
        Generate targeted mitigation strategies
        """
        strategies = []
        for disease, risk in disease_risks.items():
            if risk > 0.5:
                strategies.append(f"High-risk mitigation for {disease}")
            elif risk > 0.2:
                strategies.append(f"Preventive measures for {disease}")
        
        return strategies or ["No immediate action required"]
    
    def _default_risk_assessment(self, crop):
        """
        Provide a default risk assessment when specific data is unavailable
        """
        return {
            'crop': crop,
            'total_risk_score': 0.3,
            'disease_risks': {},
            'recommended_actions': [
                "Conduct detailed local agricultural survey",
                "Consult local agricultural experts"
            ]
        }

# Initialize the disease prediction system
disease_predictor = CropDiseasePredictionSystem()
