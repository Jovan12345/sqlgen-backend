"""Flask refactor of the FastAPI cooking‑app backend.
Run with:  python main.py  (or use a WSGI server in production)
Install deps: pip install -r requirements.txt or pip install flask flask-cors python-dotenv openai supabase_py pandas scikit-learn
"""
from __future__ import annotations

import json
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from supabase import create_client, Client
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from google.cloud import bigquery
from google.cloud.bigquery import Client
import google.auth
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# ▶️  App & service clients
# ---------------------------------------------------------------------------

load_dotenv()

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(
    app,
    origins=[
        "http://localhost:4200",
        "https://recipe-radar-sl89.onrender.com/"
        ],  #  Front‑end dev URL
    supports_credentials=True,
)

# ---------------------------------------------------------------------------
#  📄 Schema sent to GPT (static for now; generate dynamically if schema evolves)
# ---------------------------------------------------------------------------

SCHEMA = """
Table: profiles
- id (int8, PK)
- first_name (text)
- last_name (text)
- email (text)
- password_hash (text)
- role (text)
- created_at (timestampz)
- updated_at (timestampz)
- user_id (uuid, FK to auth.users.id)
- gender (text)
- age (int8)
- country (text)
- city (text)
- likes (jsonb)
- dislikes (jsonb)
- alergies (jsonb)

Table: recipes
- id (int8, PK)
- name (text)
- description (text)
- ingredients (text)
- instructions (text)
- prep_time (int8)
- cook_time (int8)
- servings (int8)
- difficulty (text)
- category (text)
- image_url (text)
- created_at (timestampz)
- updated_at (timestampz)
- user_id (int8, FK to profiles.id)
- calories_per_serving (numeric)
- protein_per_serving (numeric)

Table: ratings
- id (int8, PK)
- created_at (timestampz)
- recipe_id (int8, FK to recipes.id)
- profile_id (int8, FK to profiles.id)
- rating (int8)
- description (text)

Table: shopping_list
- id (int8, PK)
- ingredient_name (text)
- quantity (float8)
- unit (text)
- created_at (timestampz)
"""

# ---------------------------------------------------------------------------
#  🧠 OpenAI helper
# ---------------------------------------------------------------------------

def build_prompt(question: str) -> str:
    """Compose the system prompt that instructs GPT to output JSON‑only."""
    return f"""You are an intelligent SQL assistant for a cooking application powered by Supabase.
Use the schema below to generate a correct SQL query for Supabase for the user's question.

Schema:
{SCHEMA}

Given a user question, return a JSON object in one of the following formats:

1. To search for recipes. Use joins when needed. For example, to get highly rated recipes, join with the ratings table and calculate average rating per recipe:
{{
  "action": "fetch_recipes",
  "sql": "SELECT r.* FROM recipes r JOIN ratings ra ON r.id = ra.recipe_id WHERE ra.created_at >= now() - interval '7 days' GROUP BY r.id ORDER BY AVG(ra.rating) DESC LIMIT 1"
}}

2. To add ingredients to the shopping list (can be multiple):
{{
  "action": "add_items",
  "params": [
    {{
      "ingredient_name": "flour",
      "quantity": 1,
      "unit": "kg"
    }}
  ]
}}

3. To remove ingredients from the shopping list (can be multiple):
{{
  "action": "delete_items",
  "params": [
    {{ "ingredient_name": "milk" }}
  ]
}}

Only respond with a JSON object. Do not explain, do not use markdown.

User Question:
{question}

SQL Query:"""


def generate_structured_json(question: str) -> Dict[str, Any]:
    """Ask GPT and coerce its answer into JSON, returning error info when needed."""
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a backend assistant that returns structured JSON for executing SQL or calling RPCs on a Supabase-powered recipe app. Do not explain. Return only valid JSON.",
            },
            {"role": "user", "content": build_prompt(question)},
        ],
        temperature=0,
        max_tokens=300,
    )
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON returned from GPT", "raw": content}

# ---------------------------------------------------------------------------
#  🔎 /generate-sql  — SQL assistant endpoint
# ---------------------------------------------------------------------------

@app.post("/generate-sql")
def generate_sql() -> Any:
    payload = request.get_json(force=True)
    question = payload.get("question", "")
    result = generate_structured_json(question)
    return jsonify(result)

# ---------------------------------------------------------------------------
#  🥘  Recommendation engine helpers
# ---------------------------------------------------------------------------

def fetch_data():
    """Return ratings + profile metadata from Supabase as plain lists."""
    try:
        ratings = supabase.table("ratings").select("*").execute().data
        profiles = (
            supabase.table("profiles")
            .select("id, city, age, gender")
            .execute()
            .data
        )
        print(f"Fetched {len(ratings)} ratings and {len(profiles)} profiles from Supabase.")
        return ratings, profiles
    except Exception as e:
        print(f"[fetch_data] Error fetching from Supabase: {e}")
        return [], []


def collaborative_filtering(ratings: List[dict], profile_id: int, top_k: int = 5) -> List[int]:
    """Recommend recipes based on collaborative filtering using user ratings."""
    try:
        print(f"Starting collaborative filtering for profile {profile_id} with top_k={top_k}.")
        if not ratings:
            return []

        ratings_df = pd.DataFrame(ratings)
        print(f"Ratings DataFrame shape: {ratings_df.shape}")
                # Use pivot_table to aggregate duplicates by averaging their ratings
        pivot = ratings_df.pivot_table(
            index="profile_id",
            columns="recipe_id",
            values="rating",
            aggfunc="mean",
            fill_value=0
        )
        print(f"Pivot table shape: {pivot.shape}")

        if profile_id not in pivot.index:
            print(f"No ratings row for profile {profile_id}, skipping CF.")
            return []

        model = NearestNeighbors(metric="cosine", algorithm="brute")
        model.fit(pivot)

        distances, indices = model.kneighbors(
            [pivot.loc[profile_id]],
            n_neighbors=min(top_k + 1, len(pivot))
        )
        similar_users = pivot.index[indices.flatten()[1:]]
        print(f"Found {len(similar_users)} similar users for profile {profile_id}.")

        recommended = ratings_df[ratings_df["profile_id"].isin(similar_users)]
        print(f"Found {len(recommended)} ratings from similar users.")
        
        # Score by average rating and return top_k recipe_ids
        top_recipes = (
            recommended
            .groupby("recipe_id")["rating"]
            .mean()
            .sort_values(ascending=False)
            .head(top_k)
            .index
            .tolist()
        )
        print(f"Top {top_k} recipe IDs: {top_recipes}")
        return top_recipes
    except Exception as e:
        print(f"[collaborative_filtering] Error: {e}")
        return []



def filter_by_demographics(profiles_df: pd.DataFrame, target: dict) -> List[int]:
    """Filter profiles by same city, gender, and ±5 years age."""
    try:
        mask = (
            (profiles_df["city"] == target["city"]) &
            (profiles_df["gender"] == target["gender"]) &
            (profiles_df["age"].sub(target["age"]).abs() <= 5)
        )
        filtered = profiles_df[mask]["id"].tolist()
        print(f"[filter_by_demographics] {len(filtered)} profiles match demographics of {target['id']}.")
        return filtered
    except Exception as e:
        print(f"[filter_by_demographics] Error: {e}")
        return []


def hybrid_recommendation(profile_id: int) -> List[dict]:
    """Combine demographic filtering with collaborative filtering to recommend recipes."""
    try:
        ratings, profiles = fetch_data()
        target_profile = next((p for p in profiles if p["id"] == profile_id), None)
        print(f"Target profile: {target_profile}")
        if not target_profile:
            return []

        profiles_df = pd.DataFrame(profiles)
        demo_ids = filter_by_demographics(profiles_df, target_profile)
        print(f"Demographic IDs: {demo_ids}")

        relevant_ratings = [r for r in ratings if r["profile_id"] in demo_ids]
        print(f"Filtered ratings: {len(relevant_ratings)} relevant ratings found.")

        recommended_recipe_ids = collaborative_filtering(relevant_ratings, profile_id)
        print(f"Recommended recipe IDs: {recommended_recipe_ids}")
        if not recommended_recipe_ids:
            return []

        # Fetch recipes and preserve the order from recommended_recipe_ids
        recipes_result = (
            supabase.table("recipes")
            .select("*")
            .in_("id", recommended_recipe_ids)
            .execute()
            .data
        )
        
        # Create a mapping of recipe_id to recipe data for efficient lookup
        recipes_map = {recipe["id"]: recipe for recipe in recipes_result}
        
        # Return recipes in the same order as recommended_recipe_ids
        ordered_recipes = []
        for recipe_id in recommended_recipe_ids:
            if recipe_id in recipes_map:
                ordered_recipes.append(recipes_map[recipe_id])
        
        return ordered_recipes
    except Exception as e:
        print(f"[hybrid_recommendation] Error: {e}")
        return []

# ---------------------------------------------------------------------------
#  🎯 /recommend  — hybrid content‑based + collaborative recommendations
# ---------------------------------------------------------------------------

@app.post("/recommend")
def recommend_recipes() -> Any:
    payload = request.get_json(force=True)
    profile_id = payload.get("profile_id")
    if profile_id is None:
        return jsonify({"error": "profile_id is required"}), 400

    try:
        recipes = hybrid_recommendation(profile_id)
        return jsonify({"recipes": recipes})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

# ---------------------------------------------------------------------------
#  📊 /profile/ratings  — get all ratings by profile
# ---------------------------------------------------------------------------

@app.get("/profile/<int:profile_id>/ratings")
def get_profile_ratings(profile_id: int) -> Any:
    """Get all ratings given by a specific profile with recipe details."""
    try:
        # Join ratings with recipes to get recipe information
        ratings_result = supabase.table("ratings").select(
            "id, rating, description, created_at, recipe_id, recipes(id, name, image_url, category)"
        ).eq("profile_id", profile_id).order("created_at", desc=True).execute()
        
        if ratings_result.data:
            return jsonify({"ratings": ratings_result.data})
        else:
            return jsonify({"ratings": []})
    except Exception as exc:
        print(f"Error fetching profile ratings: {exc}")
        return jsonify({"error": str(exc)}), 500
    
#   --------------------------
#   Data from google analytics 
#   --------------------------

def check_google_credentials():
    creds, project = google.auth.default()
    print("Authenticated with project:", project)

def get_top_viewed_recipes(limit=10):
    check_google_credentials()

    client = Client(location="EU")

    # Get today and a year ago in YYYYMMDD format
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=365)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    query = f"""
    SELECT
      ep.value.string_value AS recipe_id,
      COUNT(*) AS views
    FROM `angularrecepies.analytics_540042299647.events_*`,
         UNNEST(event_params) AS ep
    WHERE
      _TABLE_SUFFIX BETWEEN '{start_str}' AND '{end_str}'
      AND event_name = 'view_recipe'
      AND ep.key = 'recipe_id'
    GROUP BY recipe_id
    ORDER BY views DESC
    LIMIT {limit}
    """

    query_job = client.query(query, location="EU")
    results = query_job.result()
    return [row.recipe_id for row in results]


@app.get("/recommend/popular")
def recommend_popular():
    try:
        popular_ids = get_top_viewed_recipes()
        if not popular_ids:
            return jsonify({"recipes": []})
        recipes = (
            supabase.table("recipes")
            .select("*")
            .in_("id", popular_ids)
            .execute()
            .data
        )
        return jsonify({"recipes": recipes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#
#
#

@app.get("/data/recipe-rating")
def data_demographics():
    ratings, profiles = fetch_data()
    recipes = supabase.table("recipes").select("id,name,servings").execute().data

    # Build df’s
    import pandas as pd
    ratings_df = pd.DataFrame(ratings)
    recipes_df = pd.DataFrame(recipes)

    # Aggregate avg rating and count
    agg = (
      ratings_df
        .groupby("recipe_id")["rating"]
        .agg(["mean","count"])
        .reset_index()
        .rename(columns={"mean":"avg_rating","count":"rating_count"})
    )

    merged = agg.merge(recipes_df, left_on="recipe_id", right_on="id")

    # Simply return a list of dicts:
    return merged[["name","avg_rating","rating_count","servings"]].to_dict(orient="records")

@app.get("/data/correlation")
def data_correlation():
    # 1. pull recipes table
    recipes = supabase.table("recipes")\
        .select("prep_time, cook_time, servings, calories_per_serving, protein_per_serving")\
        .execute().data

    df = pd.DataFrame(recipes)
    # drop any all-null rows
    df = df.dropna(how="all", subset=["prep_time","cook_time","servings","calories_per_serving","protein_per_serving"])

    # 2. calculate correlation matrix
    corr = df.corr()

    # 3. format for ngx-charts heatmap: array of { name: row, series: [ { name: col, value }... ] }
    data = []
    for row in corr.index:
        series = []
        for col in corr.columns:
            val = corr.at[row, col]
            series.append({
                "name": col,
                "value": 0 if pd.isna(val) else round(val, 2)
            })
        data.append({ "name": row, "series": series })

    return jsonify(data)
# ---------------------------------------------------------------------------
#  🏁 Main entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
