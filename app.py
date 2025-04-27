# app.py
import os
import torch
import json
import pickle
from flask import Flask, render_template, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification

class SkincareReviewPredictor:
    def __init__(self, 
                 model_weights_path='model_artifacts/best_model_weights.pth',
                 config_path='model_artifacts/model_config.json',
                 tokenizer_path='model_artifacts/tokenizer',
                 label_encoder_path='model_artifacts/label_encoder.pkl'):
        """
        Initialize the predictor with saved model artifacts
        
        Args:
            model_weights_path (str): Path to saved model weights
            config_path (str): Path to model configuration file
            tokenizer_path (str): Path to saved tokenizer
            label_encoder_path (str): Path to saved label encoder
        """
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model configuration
        with open(config_path, 'r') as f:
            self.model_config = json.load(f)
        
        # Load label encoder
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        
        # Load model
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels=self.model_config['num_labels']
        )
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def preprocess_input(self, 
                          skin_type='', 
                          product='', 
                          brand='', 
                          ingredients='', 
                          review='', 
                          max_length=128):
        """
        Preprocess input features into a single text input
        
        Args:
            skin_type (str): Skin type description
            product (str): Product name
            brand (str): Brand name
            ingredients (str): Product ingredients
            review (str): Product review text
            max_length (int): Maximum sequence length for tokenization
        
        Returns:
            dict: Tokenized input
        """
        # Combine input features into a single text
        input_text = f"{skin_type} {product} {brand} {ingredients} {review}".strip()
        
        # Tokenize input
        encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
    
    def predict_category(self, **kwargs):
        """
        Predict category for given input
        
        Args:
            **kwargs: Keyword arguments for input features
        
        Returns:
            str: Predicted category
        """
        # Preprocess input
        processed_input = self.preprocess_input(**kwargs)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(
                input_ids=processed_input['input_ids'], 
                attention_mask=processed_input['attention_mask']
            )
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
        
        # Convert prediction to original category
        predicted_category = self.label_encoder.inverse_transform(preds.cpu().numpy())[0]
        
        return predicted_category
    
    def get_available_categories(self):
        """
        Return list of available categories
        
        Returns:
            list: Available categories
        """
        return list(self.label_encoder.classes_)

# Flask Application
app = Flask(__name__)

# Global predictor instance
predictor = SkincareReviewPredictor()

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main route to render the prediction form and handle predictions
    """
    categories = predictor.get_available_categories()
    
    if request.method == 'POST':
        # Collect form data
        skin_type = request.form.get('skin_type', '')
        product = request.form.get('product', '')
        brand = request.form.get('brand', '')
        ingredients = request.form.get('ingredients', '')
        review = request.form.get('review', '')
        
        try:
            # Predict category
            predicted_category = predictor.predict_category(
                skin_type=skin_type,
                product=product,
                brand=brand,
                ingredients=ingredients,
                review=review
            )
            
            return render_template('index.html', 
                                   categories=categories, 
                                   predicted_category=predicted_category,
                                   skin_type=skin_type,
                                   product=product,
                                   brand=brand,
                                   ingredients=ingredients,
                                   review=review)
        
        except Exception as e:
            return render_template('index.html', 
                                   categories=categories, 
                                   error=str(e))
    
    return render_template('index.html', categories=categories)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    """
    API endpoint for category prediction
    """
    data = request.get_json()
    
    try:
        predicted_category = predictor.predict_category(
            skin_type=data.get('skin_type', ''),
            product=data.get('product', ''),
            brand=data.get('brand', ''),
            ingredients=data.get('ingredients', ''),
            review=data.get('review', '')
        )
        
        return jsonify({
            'success': True,
            'predicted_category': predicted_category
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)