# Railway Complaint Management System (RCMS)

## 🚉 Overview
An AI-powered system for automated railway complaint processing with multi-modal input support and intelligent classification.

## 🏗️ Architecture
- **Modular Design**: Plug-in based sentiment analysis
- **Multi-Modal**: Text, Image, Audio, Video processing
- **AI Pipeline**: Category, Priority, Escalation prediction
- **Computer Vision**: Visual content analysis
- **REST API**: FastAPI backend
- **Real-time**: Instant complaint processing

## 🛠️ Technology Stack
- **Backend**: Python, FastAPI, SQLAlchemy
- **ML/AI**: scikit-learn, TensorFlow, OpenCV, Whisper
- **Database**: PostgreSQL with async support
- **Testing**: pytest, pytest-asyncio
- **Deployment**: Docker, Kubernetes

## 📁 Project Structure
```
RCMS/
├── src/
│   ├── preprocessing/     # OCR, ASR, text cleaning
│   ├── models/           # ML models (category, priority, escalation)
│   ├── api/              # FastAPI routes and endpoints
│   └── utils/            # Helper functions and utilities
├── data/
│   ├── raw/              # Original complaint data
│   ├── processed/        # Cleaned and preprocessed data
│   └── media/            # Images, audio, video files
├── models/               # Trained model artifacts
├── tests/                # Unit and integration tests
├── config/               # Configuration files
├── frontend/             # Web and mobile interfaces
└── docs/                 # Documentation
```

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone <repository-url>
cd RCMS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup database
python scripts/setup_db.py

# Run tests
pytest

# Start API server
uvicorn src.api.main:app --reload
```

### API Usage
```python
import requests

# Submit complaint
response = requests.post("http://localhost:8000/api/v1/complaints", 
    json={"text": "The train is very dirty and smells bad"})

# Get complaint status
complaint_id = response.json()["id"]
status = requests.get(f"http://localhost:8000/api/v1/complaints/{complaint_id}")
```

## 🧪 Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test module
pytest tests/test_models.py
```

## 🔧 Configuration
Edit `config/settings.yaml` for:
- Model parameters
- API settings
- Database configuration
- Feature toggles (sentiment analysis)

## 📊 Features

### Core ML Pipeline
- ✅ **Category Classification**: Infrastructure, Cleanliness, Safety, Staff, etc.
- ✅ **Priority Prediction**: Low, Medium, High, Critical
- ✅ **Escalation Decision**: Auto-route to appropriate department
- 🔌 **Sentiment Analysis**: Optional plug-in module

### Multi-Modal Processing
- 📝 **Text**: Natural language complaint processing
- 📷 **Images**: OCR text extraction + visual analysis
- 🎵 **Audio**: Speech-to-text conversion
- 🎥 **Video**: Frame extraction + action recognition

### Computer Vision
- 🔍 **Object Detection**: Identify railway infrastructure issues
- 🏷️ **Scene Classification**: Train interior, platform, station areas
- 👁️ **Action Recognition**: Detect incidents in video content

## 🔌 Plug-in Architecture
```python
# Enable/disable sentiment analysis
ENABLE_SENTIMENT = True  # config/settings.yaml

# System automatically adapts
if ENABLE_SENTIMENT:
    result.sentiment = sentiment_analyzer.predict(text)
else:
    result.sentiment = "neutral"
```

## 🌟 Advanced Features
- 🚨 **Real-time Processing**: Instant complaint analysis
- 📈 **Analytics Dashboard**: Complaint trends and patterns
- 🌍 **Multi-language**: Hindi, English, regional languages
- 📱 **Mobile Ready**: React Native app support
- 🔐 **Security**: JWT authentication, role-based access

## 📞 API Endpoints
- `POST /api/v1/complaints` - Submit new complaint
- `GET /api/v1/complaints/{id}` - Get complaint details
- `GET /api/v1/analytics` - Dashboard analytics
- `POST /api/v1/media/upload` - Upload media files

## 🤝 Contributing
1. Fork the repository
2. Create feature branch
3. Write tests for new features
4. Ensure all tests pass
5. Submit pull request

