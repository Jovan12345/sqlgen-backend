# Recipe Radar Backend

A Flask-powered API that powers the Recipe Radar smart web app:

- **Natural-language → SQL**: Convert user requests (e.g. “Show recipes with chicken”) into safe Supabase SQL.
- **Hybrid Recommendations**: Combine demographic filtering (city, age, gender) and collaborative filtering (user ratings) to suggest recipes.
- **Analytics Endpoints**: Serve ready-to-plot data for:
  - **Demographics vs. Ratings** (bubble chart)
  - **Correlation Matrix** (heatmap of numeric recipe fields)
  - **Ingredient Co-occurrence** (network graph)

## 🚀 Features

- **/generate-sql** – GPT-powered SQL generation from plain English
- **/recommend** – Return top K recipe recommendations for a given profile ID
- **/data/demographics** – Avg rating vs. rating count + servings (for bubble chart)
- **/data/correlation** – Correlation matrix of prep_time, cook_time, servings, calories, protein
- **/data/cooccurrence** – Ingredient co-occurrence pairs for network analysis

## 🔧 Requirements

- Python 3.8+
- A Supabase project (with tables: `profiles`, `recipes`, `ratings`, `shopping_list`)
- pip
- A `.env` file with:
  - SUPABASE_URL=…
  - SUPABASE_API_KEY=…
  - OPENAI_API_KEY=…

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/yourusername/sqlgen-backend.git
    cd sqlgen-backend
    ```

2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

## Usage

1. Start the backend server:
    ```
    python main.py
    ```

2. Access the API at `http://localhost:8000` (or your configured port).

## API Endpoints

- `POST /generate-sql`  
  Generate SQL from natural language input.  gunicorn>=21.0.0

## Project Structure
