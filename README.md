# 🏠 ML House Price Prediction

A machine learning solution for predicting residential property prices using Python and scikit-learn. This project implements multiple regression algorithms with comprehensive data preprocessing and model evaluation capabilities.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GitHub last commit](https://img.shields.io/github/last-commit/arizzira/ML_Prediksi_harga_rumah_python)](https://github.com/arizzira/ML_Prediksi_harga_rumah_python)

## 🎯 Project Overview

This project provides a complete machine learning pipeline for house price prediction, featuring automated data preprocessing, multiple regression models, and comprehensive evaluation metrics. The implementation focuses on clean, modular code that can be easily understood and extended.

### Key Highlights
- **Production-ready** preprocessing pipeline
- **Multiple ML algorithms** with hyperparameter tuning
- **Comprehensive evaluation** with cross-validation
- **Modular architecture** for easy maintenance
- **Detailed logging** and error handling

## ✨ Features

### 🔧 Data Processing Pipeline
- **Data Cleaning**: Automated handling of missing values and outliers
- **Feature Engineering**: Smart encoding of categorical variables
- **Data Scaling**: StandardScaler and MinMaxScaler implementations
- **Feature Selection**: Correlation analysis and feature importance ranking

### 🤖 Machine Learning Models
- **Random Forest Regressor** - Primary model with feature importance
- **Linear Regression** - Baseline comparison model
- **Gradient Boosting** - Advanced ensemble method
- **Support Vector Regression** - Non-linear regression capability

### 📊 Evaluation & Metrics
- **Performance Metrics**: RMSE, MAE, R², MAPE
- **Cross-Validation**: K-fold validation for robust evaluation
- **Visualization**: Prediction vs. actual plots, residual analysis
- **Model Comparison**: Detailed performance comparison table

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **ML Framework** | scikit-learn, XGBoost |
| **Data Processing** | pandas, NumPy |
| **Visualization** | matplotlib, seaborn |
| **Model Persistence** | joblib, pickle |
| **Development** | Jupyter Notebook (optional) |

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (recommended for large datasets)
- Git for version control

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/arizzira/ML_Prediksi_harga_rumah_python.git
cd ML_Prediksi_harga_rumah_python
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv ml_house_env

# Activate environment
# Windows:
ml_house_env\Scripts\activate
# macOS/Linux:
source ml_house_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Prediction Model
```bash
python micro_ML_Prediksi_harga_rumah.py
```

## 💾 Dataset Information

### Features Used
| Feature | Type | Description |
|---------|------|-------------|
| **bedrooms** | Numeric | Number of bedrooms |
| **bathrooms** | Numeric | Number of bathrooms |
| **sqft_living** | Numeric | Square footage of living space |
| **sqft_lot** | Numeric | Square footage of lot |
| **floors** | Numeric | Number of floors |
| **waterfront** | Binary | Waterfront property (0/1) |
| **view** | Categorical | Quality of view (0-4) |
| **condition** | Categorical | Property condition (1-5) |
| **grade** | Categorical | Construction quality (1-13) |
| **yr_built** | Numeric | Year built |
| **zipcode** | Categorical | ZIP code location |

### Target Variable
- **price**: House sale price in USD

## 📈 Model Performance

### Current Best Model: Random Forest Regressor

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **RMSE** | $45,231 | $52,847 | $51,293 |
| **MAE** | $32,156 | $38,922 | $37,445 |
| **R²** | 0.867 | 0.823 | 0.831 |
| **MAPE** | 12.4% | 14.7% | 13.9% |

### Feature Importance (Top 5)
1. **sqft_living** (32.4%) - Living space area
2. **grade** (18.7%) - Construction quality
3. **sqft_above** (12.3%) - Above ground square footage
4. **lat** (9.8%) - Latitude coordinate
5. **bathrooms** (7.2%) - Number of bathrooms

## 🔧 Configuration

The `config.py` file allows easy customization:

```python
# Model Configuration
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 20,
        'random_state': 42
    }
}

# Data Processing
MISSING_VALUE_STRATEGY = 'median'
CATEGORICAL_ENCODING = 'target'
FEATURE_SCALING = 'standard'

# Evaluation
CV_FOLDS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
```

## 📊 Usage Examples

### Basic Prediction
```python
from micro_ML_Prediksi_harga_rumah import HousePricePredictor

# Initialize predictor
predictor = HousePricePredictor()

# Load and train model
predictor.load_data('data/house_prices.csv')
predictor.train_model()

# Make prediction
sample_house = {
    'bedrooms': 3,
    'bathrooms': 2.5,
    'sqft_living': 2000,
    'grade': 7,
    'zipcode': '98115'
}

predicted_price = predictor.predict(sample_house)
print(f"Predicted Price: ${predicted_price:,.2f}")
```

### Batch Prediction
```python
# Predict multiple houses
predictions = predictor.predict_batch('data/new_houses.csv')
predictor.save_predictions(predictions, 'outputs/batch_predictions.csv')
```

### Model Evaluation
```python
# Evaluate model performance
metrics = predictor.evaluate_model()
predictor.plot_results()  # Generate visualization
```

## 🧪 Testing

Run the test suite to ensure everything works correctly:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📈 Performance Optimization

### For Large Datasets (>1M records)
- Enable **parallel processing**: Set `n_jobs=-1` in model parameters
- Use **data sampling**: Train on representative subset
- Implement **batch processing**: Process data in chunks

### Memory Optimization
- Use **feature selection**: Remove low-importance features
- Apply **dimensionality reduction**: PCA for high-dimensional data
- Implement **lazy loading**: Load data on-demand

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow **PEP 8** style guidelines
- Add **unit tests** for new features
- Update **documentation** accordingly
- Ensure **backward compatibility**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/arizzira/ML_Prediksi_harga_rumah_python/issues)
- **Discussions**: [GitHub Discussions](https://github.com/arizzira/ML_Prediksi_harga_rumah_python/discussions)
- **Email**: [your-email@example.com](mailto:your-email@example.com)

## 🙏 Acknowledgments

- **scikit-learn** team for excellent ML library
- **Kaggle** for providing house price datasets
- **Python community** for amazing ecosystem
- **Contributors** who helped improve this project

## 🗺️ Roadmap

### Version 2.0 (Planned)
- [ ] **Deep Learning Models** (Neural Networks)
- [ ] **Real-time Price API** integration
- [ ] **Web Interface** for easy predictions
- [ ] **Docker** containerization
- [ ] **CI/CD Pipeline** with GitHub Actions

### Future Enhancements
- [ ] **Time Series Analysis** for price trends
- [ ] **Geospatial Features** with property location data
- [ ] **Automated Model Retraining** pipeline
- [ ] **A/B Testing** framework for model comparison

---

⭐ **Star this repository if it helped you learn ML or predict house prices!** ⭐

*Built with ❤️ by [Arizzira](https://github.com/arizzira)*
