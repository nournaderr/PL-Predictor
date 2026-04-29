# PL-Predictor
Premier League match outcome classifier

# Project Description
A supervised machine learning system that predicts the outcome of English Premier League 
fixtures — Home Win (H), Draw (D), or Away Win (A) — using pre-match historical 
performance statistics. The model targets sports analytics firms providing decision-support 
tools to football clubs and media broadcasters.

**Data Sources:**
- [Football-Data.co.uk](https://www.football-data.co.uk/englandm.php) — Match statistics (2015/16–2024/25), 3,800 rows
- [Transfermarkt via Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores) — Player & transfer data

# Setup & Installation

### Prerequisites
- Python 3.11
- [Poetry](https://python-poetry.org/docs/#installation)

### Steps
```bash
# 1. Clone the repository
git clone https://github.com/nournaderr/PL-Predictor.git
cd PL-Predictor

# 2. Install dependencies
poetry install

# 3. Copy environment variables template
cp .env.example .env
```

## Running the Project

```bash
# Install dependencies
make install

# Run data preprocessing
make pipeline

# Train models
make train

# Run tests with coverage
make test

# Lint code
make lint
```

## Environment Variables
Copy `.env.example` to `.env` and fill in any required values: