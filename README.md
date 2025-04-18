# Malaria Detection API

A FastAPI-based backend for detecting malaria in blood cell images using deep learning.

## Deployment on Render

### Prerequisites
- A Render account
- Git repository with your code
- Trained model file (malaria_model.h5)

### Deployment Steps

1. **Prepare Your Repository**
   - Ensure all necessary files are committed:
     - `main.py`
     - `requirements.txt`
     - `Dockerfile`
     - `render.yaml`
     - Your trained model file

2. **Create a New Web Service on Render**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" and select "Web Service"
   - Connect your Git repository
   - Select the branch to deploy

3. **Configure the Service**
   - Name: `malaria-detection-api`
   - Environment: `Python`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

4. **Set Environment Variables**
   - `ENVIRONMENT`: `production`
   - `MODEL_PATH`: `./models/malaria_model.h5`
   - `UPLOAD_DIR`: `./uploads`
   - `TEMP_DIR`: `./temp`
   - `PORT`: `10000`

5. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your application

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| ENVIRONMENT | Application environment (development/production) | development |
| MODEL_PATH | Path to the trained model file | ./models/malaria_model.h5 |
| UPLOAD_DIR | Directory for uploaded images | ./uploads |
| TEMP_DIR | Directory for temporary files | ./temp |
| PORT | Port to run the application | 10000 |

### API Endpoints

- `GET /`: Health check endpoint
- `POST /predict`: Upload an image for malaria detection

### Local Development

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the development server:
   ```bash
   uvicorn main:app --reload
   ```

### Notes
- The API uses CORS middleware to allow cross-origin requests
- Uploaded images are saved with timestamps in the filename
- The model should be placed in the `models` directory
- Make sure to set appropriate CORS origins in production 