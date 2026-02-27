"""
app.py — The web server

Flask connects Python to the browser.
Routes:
  /               → serve the page
  /api/recommend  → run the ML, return JSON
  /api/notes      → available notes for the UI
  /api/brands     → brand list for the combobox
"""

from flask import Flask, render_template, request, jsonify
from recommender import recommend, get_notes_for_ui, get_stats, build_matrix
from data import get_brands

app = Flask(__name__)


@app.route("/")
def index():
    stats  = get_stats()
    brands = get_brands()
    return render_template("index.html", brands=brands, **stats)


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    body   = request.get_json()
    notes  = body.get("notes", [])
    gender = body.get("gender") or None
    brands = body.get("brands") or None
    results = recommend(selected_notes=notes, gender=gender, brands=brands, top_n=8)
    return jsonify({"results": results})


@app.route("/api/notes")
def api_notes():
    return jsonify(get_notes_for_ui())


if __name__ == "__main__":
    print("\n✦ Sillage — loading dataset…")
    build_matrix()
    print("✦ Ready at http://localhost:5000\n")
    app.run(debug=True, port=5000)
