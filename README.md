# AI Resume Screener Complete

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## Project Overview

AI Resume Screener Complete is an intelligent resume screening system that leverages machine learning and natural language processing to automatically evaluate and rank resumes. The system extracts key features from resumes, applies trained ML models for screening, and provides detailed evaluation metrics through a user-friendly API interface.

## Features

### 📄 Resume Parsing & Feature Extraction
- PDF and DOCX resume parsing
- Text extraction and preprocessing
- Contact information extraction
- Skills and experience identification
- Education background parsing
- Work experience analysis

### 🤖 Machine Learning Models
- Multiple ML algorithms for resume screening
- Feature engineering and selection
- Model training and validation
- Ensemble methods for improved accuracy
- Custom scoring algorithms

### 📊 Evaluation & Analytics
- Comprehensive evaluation metrics
- Performance analysis and reporting
- ROC curves and confusion matrices
- Cross-validation results
- Feature importance analysis

### 🚀 REST API
- Easy-to-use HTTP endpoints
- JSON request/response format
- Batch processing capabilities
- Authentication and rate limiting
- Comprehensive error handling

## Tech Stack

- **Backend**: Python 3.8+, Flask
- **Machine Learning**: scikit-learn, pandas, numpy
- **Text Processing**: NLTK, spaCy
- **Document Parsing**: PyPDF2, python-docx
- **API**: Flask-RESTful, Flask-CORS
- **Database**: SQLite (development), PostgreSQL (production)
- **Deployment**: Docker, Docker Compose
- **Testing**: pytest, unittest

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/jayzz999/AI-Resume-Screener-Complete.git
   cd AI-Resume-Screener-Complete
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required NLTK data**
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

5. **Initialize the application**
   ```bash
   python app.py
   ```

## Configuration

Create a `.env` file in the project root:

```env
# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# Database Configuration
DATABASE_URL=sqlite:///resume_screener.db

# ML Model Configuration
MODEL_PATH=models/
FEATURE_THRESHOLD=0.5
CONFIDENCE_THRESHOLD=0.7

# API Configuration
API_RATE_LIMIT=100
MAX_FILE_SIZE=10MB
```

## Usage Examples

### Starting the Server

```bash
python app.py
```

The API will be available at `http://localhost:5000`

### API Endpoints

#### 1. Screen Single Resume

**POST** `/api/screen`

```bash
curl -X POST \
  http://localhost:5000/api/screen \
  -H 'Content-Type: multipart/form-data' \
  -F 'resume=@path/to/resume.pdf' \
  -F 'job_requirements={"skills":["Python","Machine Learning"],"experience":"2+ years"}'
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "candidate_id": "12345",
    "score": 0.85,
    "ranking": "Highly Qualified",
    "extracted_features": {
      "skills": ["Python", "Machine Learning", "Data Science"],
      "experience_years": 3,
      "education": "Bachelor's in Computer Science"
    },
    "match_analysis": {
      "skill_match": 0.9,
      "experience_match": 0.8,
      "overall_fit": 0.85
    }
  }
}
```

#### 2. Batch Screening

**POST** `/api/batch-screen`

```bash
curl -X POST \
  http://localhost:5000/api/batch-screen \
  -H 'Content-Type: multipart/form-data' \
  -F 'resumes=@resume1.pdf' \
  -F 'resumes=@resume2.pdf' \
  -F 'job_requirements={"skills":["Python","Flask"]}'
```

#### 3. Get Model Performance

**GET** `/api/model/performance`

```bash
curl http://localhost:5000/api/model/performance
```

**Response:**
```json
{
  "accuracy": 0.92,
  "precision": 0.89,
  "recall": 0.94,
  "f1_score": 0.91,
  "auc_roc": 0.96
}
```

## Model Training Guide

### Training New Models

1. **Prepare training data**
   ```bash
   python prepare_data.py --input data/resumes/ --output data/processed/
   ```

2. **Train the model**
   ```bash
   python train_model.py --config config/model_config.json
   ```

3. **Evaluate model performance**
   ```bash
   python evaluate.py --model models/latest_model.pkl --test-data data/test/
   ```

### Model Configuration

Edit `config/model_config.json`:

```json
{
  "algorithms": ["random_forest", "svm", "logistic_regression"],
  "feature_selection": true,
  "cross_validation_folds": 5,
  "hyperparameter_tuning": true,
  "ensemble_method": "voting"
}
```

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Categories

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# API tests
pytest tests/api/ -v
```

### Coverage Report

```bash
pytest --cov=. --cov-report=html
```

## Project Structure

```
AI-Resume-Screener-Complete/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── README.md             # Project documentation
├── config/               # Configuration files
│   └── model_config.json
├── models/               # Trained ML models
│   ├── resume_classifier.pkl
│   └── feature_extractor.pkl
├── src/                  # Source code
│   ├── __init__.py
│   ├── parser_utils.py   # Resume parsing utilities
│   ├── feature_extraction.py # Feature extraction
│   ├── train_model.py    # Model training
│   └── evaluate.py       # Model evaluation
├── tests/                # Test suite
│   ├── unit/
│   ├── integration/
│   └── api/
├── data/                 # Data directory
│   ├── raw/
│   ├── processed/
│   └── test/
└── docker/               # Docker configuration
    ├── Dockerfile
    └── docker-compose.yml
```

## Docker Deployment

### Using Docker Compose

1. **Build and start services**
   ```bash
   docker-compose up --build
   ```

2. **Run in production mode**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

### Using Docker directly

```bash
# Build image
docker build -t ai-resume-screener .

# Run container
docker run -p 5000:5000 ai-resume-screener
```

## Evaluation Metrics Explanation

### Core Metrics

- **Accuracy**: Overall correctness of the model
- **Precision**: Ratio of true positives to total predicted positives
- **Recall**: Ratio of true positives to total actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

### Custom Metrics

- **Skill Match Score**: Measures alignment between candidate skills and job requirements
- **Experience Relevance**: Evaluates work experience relevance to the position
- **Overall Fit Score**: Composite score combining multiple factors

## Future Enhancements

- [ ] Deep learning models (BERT, GPT) for text analysis
- [ ] Real-time resume screening dashboard
- [ ] Integration with ATS systems
- [ ] Advanced bias detection and mitigation
- [ ] Multi-language resume support
- [ ] Video resume analysis capabilities
- [ ] Blockchain-based verification system
- [ ] Mobile application development

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests for new functionality**
5. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
6. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests
- Update documentation
- Ensure backward compatibility
- Add type hints where applicable

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Team & Contact Information

### Development Team
- **Lead Developer**: [jayzz999](https://github.com/jayzz999)
- **Contributors**: See [Contributors](https://github.com/jayzz999/AI-Resume-Screener-Complete/contributors)

### Contact
- **Email**: [contact@resumescreener.ai](mailto:contact@resumescreener.ai)
- **Issues**: [GitHub Issues](https://github.com/jayzz999/AI-Resume-Screener-Complete/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jayzz999/AI-Resume-Screener-Complete/discussions)

### Support

If you find this project helpful, please consider:
- ⭐ Starring the repository
- 🐛 Reporting bugs
- 💡 Suggesting enhancements
- 🤝 Contributing to the codebase

---

**Built with ❤️ by the AI Resume Screener Team**
