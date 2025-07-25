from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import logging
import statistics
from apify_client import ApifyClient
from er import predict_er
import json

# Read from a JSON file
def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return None

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# More permissive CORS setup
CORS(app, 
     resources={r"/*": {"origins": "*"}}, 
     expose_headers=["Content-Type", "X-Requested-With", "Content-Length"],
     supports_credentials=True,
     allow_headers=["Content-Type", "X-Requested-With", "Authorization"])

# Configure Google AI
API_KEY = "GEMINI-API-KEY"
genai.configure(api_key=API_KEY)

APIFY_TOKEN = 'APIFY-API-KEY'

@app.route('/test', methods=['GET'])
def test_connection():
    return jsonify({'message': 'Server is running'}), 200

@app.route('/generate-caption-from-prompt', methods=['POST'])
def generate_caption():
    try:
        if not request.is_json:
            logger.error("Request does not contain JSON")
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.json
        prompt = data.get('prompt')

        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        logger.info(f"Received prompt request: {prompt[:30]}...")
        
        prompt += " It should be for an Instagram photo. Add relevant emojis if needed. Return a single creative caption of less than 15 words."

        # Load Gemini model
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)

        if response and hasattr(response, 'candidates') and response.candidates:
            caption = response.candidates[0].content.parts[0].text

            prompt = "Choose the best category for the given caption. The categories are: social_media, fashion_beauty, travel_adventure, nature_outdoors, food_culinary, pets_animals, motivation_inspiration, career_jobs, technology_engineering, entertainment_media, gaming, health_wellness, business_entrepreneurship, education_learning, sustainability_environment, arts_culture, legal, politics_government, real_estate, events_networking, nonprofit_philanthropy, diversity_inclusion. Return a single category."
            response = model.generate_content([prompt, caption])

            print("Categories response:", response)
            category = response.candidates[0].content.parts[0].text

            logger.info(f"Generated caption: {caption}")
            return jsonify({'caption': caption})
        else:
            logger.error("Content blocked or no response generated")
            return jsonify({'error': 'Content blocked or no response generated'}), 400
    except Exception as e:
        logger.exception("Error in generate_caption endpoint")
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    
from PIL import Image
import io
import torch
from transformers import (
    AutoProcessor, AutoModelForCausalLM, 
    LxmertTokenizer, LxmertModel
)

git_model_name = "path-to-finetuned-microsoft-git-base-model" 
git_processor = AutoProcessor.from_pretrained(git_model_name)
git_model = AutoModelForCausalLM.from_pretrained(git_model_name)

lxmert_model_name = "path-to-lxmert4hashtag"  
lxmert_tokenizer = LxmertTokenizer.from_pretrained(lxmert_model_name)
lxmert_model = LxmertModel.from_pretrained(lxmert_model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
git_model.to(device)
lxmert_model.to(device)
git_model.eval()
lxmert_model.eval()

def generate_caption_with_git(image_pil):
    try:
        # Process image for GIT model
        inputs = git_processor(images=image_pil, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Generate caption
            generated_ids = git_model.generate(
                pixel_values=inputs.pixel_values,
                max_length=50,
                num_beams=4,
                do_sample=False,
                early_stopping=True
            )
            
        # Decode the generated caption
        initial_caption = git_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return initial_caption.strip()
        
    except Exception as e:
        logger.error(f"Error in GIT caption generation: {str(e)}")
        raise

def generate_hashtags_with_lxmert(image_pil, caption):
        
    try:
        # Preprocess image for LXMERT
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image_pil).unsqueeze(0).to(device)
        
        inputs = lxmert_tokenizer(caption, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = lxmert_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                visual_feats=image_tensor,
                visual_pos=None
            )
            
            # Decode the refined output
            predicted_tokens = torch.argmax(outputs.prediction_logits, dim=-1)
            caption_with_hashtags = lxmert_tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)
            
        return caption_with_hashtags
        
    except Exception as e:
        logger.error(f"Error in LXMERT caption refinement: {str(e)}")
        raise

def generate_caption_and_hashtag(image_pil):
    
    try:
        logger.info("Stage 1: Generating caption with GIT model")
        initial_caption = generate_caption_with_git(image_pil)
        logger.info(f"Initial caption: {initial_caption}")
        
        logger.info("Stage 2: generating hashtags with LXMERT")
        final_caption_with_hashtags = generate_hashtags_with_lxmert(
            image_pil, initial_caption
        )
        
        return final_caption_with_hashtags
        
    except Exception as e:
        logger.error(f"Error in two-stage caption generation: {str(e)}")
        raise

@app.route('/generate-caption-from-image', methods=['POST'])
def generate_caption_from_image():
    try:
        if 'image' not in request.files:
            logger.error("No image part in request")
            return jsonify({'error': 'No image file provided'}), 400
        
        image = request.files['image']

        if image.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400

        # Read image data
        image_bytes = image.read()
        logger.info(f"Received image: {image.filename}, Size: {len(image_bytes)} bytes")
        
        # Convert bytes to PIL Image
        image_pil = Image.open(io.BytesIO(image_bytes))
        
        # Generate caption using two-stage approach
        final_caption_with_hashtags = generate_caption_and_hashtag(image_pil)
        
        logger.info(f"Generated final caption: {final_caption_with_hashtags}")
        return jsonify({'caption': final_caption_with_hashtags})

    except Exception as e:
        logger.exception("Error in generate_caption_from_image endpoint")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


def validate_instagram_username(username: str):
    client = ApifyClient(APIFY_TOKEN)
    run_input = { "usernames": [username] }

    try:
        run = client.actor("YOUR-PARTICULAR-ACTOR-ID").call(run_input=run_input)
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            error = item.get("error")
            is_private = item.get("private")
            followers = item.get("followersCount")

            if error == "not_found":
                return {"valid": False, "exists": False, "is_public": None}

            if is_private is True:
                return {"valid": True, "exists": True, "is_public": False}

            if is_private is False:
                return {"valid": True, "exists": True, "is_public": True, "followers": followers}

        return {"valid": False, "error": "Unexpected response format"}

    except Exception as e:
        return {"valid": False, "error": str(e)}

def get_average_likes_and_comments(username: str):
    client = ApifyClient(APIFY_TOKEN)
    run_input = {
        "username": [username],
        "resultsLimit": 10,
        "onlyPostsNewerThan": "1 month",
        "skipPinnedPosts": True
    }

    try:
        run = client.actor("YOUR-PARTICULAR-ACTOR-ID").call(run_input=run_input)
        likes = []
        comments = []

        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            likes_count = item.get("likesCount")
            comments_count = item.get("commentsCount")
            if likes_count is not None:
                likes.append(likes_count)
            if comments_count is not None:
                comments.append(comments_count)

        avg_likes = round(statistics.mean(likes), 0) if likes else 0
        avg_comments = round(statistics.mean(comments), 0) if comments else 0

        return {
            "average_likes": avg_likes,
            "average_comments": avg_comments
        }

    except Exception as e:
        return {"error": f"Failed to calculate averages: {str(e)}"}


@app.route('/validate-instagram-username', methods=['POST'])
def validate_username_endpoint():
    data = request.get_json()
    username = data.get('username', '').strip()

    if not username:
        return jsonify({'valid': False, 'error': 'Username is required'}), 400

    validation_result = validate_instagram_username(username)

    if validation_result.get("valid") and validation_result.get("is_public"):
        # Fetch averages only if user is valid and public
        print("Fetching averages for:", username)
        #averages = get_average_likes_and_comments(username)
        #validation_result.update(averages)

    return jsonify(validation_result)

@app.route('/instagram-average-engagement', methods=['POST'])
def average_engagement():
    data = request.get_json()
    username = data.get('username', '').strip()

    if not username:
        return jsonify({'error': 'Username is required'}), 400

    result = get_average_likes_and_comments(username)

    if 'error' in result:
        return jsonify({'error': result['error']}), 500

    return jsonify(result)

@app.route('/predict-engagement', methods=['POST'])
def predict_engagement_rate():
    data = request.get_json()
    avg_likes = data.get('average_likes', 0)
    avg_comments = data.get('average_comments', 0)
    followers = data.get('followers', 0)
    caption = data.get('caption', '')
    hashtags = data.get('hashtags', '')

    try:
        result = predict_er(avg_likes, avg_comments, followers, caption, hashtags)
        return jsonify(result)
    
    except Exception as e:
        logger.exception("Error in predict_engagement_rate endpoint")
        return jsonify({'error': str(e)}), 500
    




if __name__ == '__main__':
    app.run(debug=True, port=5001)
