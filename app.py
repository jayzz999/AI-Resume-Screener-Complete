from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import io
import re
import traceback
from typing import List, Dict, Any, Tuple

# Parsing libraries
try:
    import docx
except Exception:
    docx = None

try:
    import pdfminer
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:
    pdf_extract_text = None

# NLP / Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------- Utility Functions -----------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def read_txt(file_stream: io.BytesIO) -> str:
    try:
        return file_stream.read().decode("utf-8", errors="ignore")
    except Exception:
        file_stream.seek(0)
        return file_stream.read().decode("latin-1", errors="ignore")


def read_docx(path: str) -> str:
    if docx is None:
        raise RuntimeError("python-docx not installed")
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)


def read_pdf(path: str) -> str:
    if pdf_extract_text is None:
        raise RuntimeError("pdfminer.six not installed")
    return pdf_extract_text(path) or ""


def extract_text_from_file(path: str) -> str:
    ext = path.rsplit(".", 1)[1].lower()
    if ext == "txt":
        with open(path, "rb") as f:
            return read_txt(io.BytesIO(f.read()))
    if ext == "docx":
        return read_docx(path)
    if ext == "pdf":
        return read_pdf(path)
    raise ValueError(f"Unsupported file type: {ext}")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+.# ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skills(text: str, skill_catalog: List[str]) -> List[str]:
    text_norm = normalize_text(text)
    found = []
    for skill in skill_catalog:
        s = normalize_text(skill)
        # exact word/phrase containment
        if s and s in text_norm:
            found.append(skill)
    return sorted(list(set(found)))


def score_resume(resume_text: str, jd_text: str, skill_catalog: List[str]) -> Dict[str, Any]:
    resume_norm = normalize_text(resume_text)
    jd_norm = normalize_text(jd_text)

    # TF-IDF similarity
    tfidf = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = tfidf.fit_transform([jd_norm, resume_norm])
        sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
    except Exception:
        sim = 0.0

    # Skill matching
    jd_skills = extract_skills(jd_text, skill_catalog)
    resume_skills = extract_skills(resume_text, skill_catalog)
    common = set(map(str.lower, resume_skills)) & set(map(str.lower, jd_skills))
    skill_recall = len(common) / max(1, len(jd_skills))

    # Heuristic scoring
    # 70% TF-IDF sim, 30% skill recall
    total_score = 0.7 * sim + 0.3 * skill_recall

    return {
        "tfidf_similarity": sim,
        "jd_skills": jd_skills,
        "resume_skills": resume_skills,
        "matched_skills": sorted(list(common)),
        "skill_recall": skill_recall,
        "score": round(float(total_score), 4),
    }


def rank_candidates(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(items, key=lambda x: x.get("score", 0.0), reverse=True)


DEFAULT_SKILLS = [
    # Programming
    "python", "java", "c++", "javascript", "typescript", "node.js", "react", "angular",
    "flask", "django", "fastapi", "spring", "sql", "nosql", "mongodb", "postgresql",
    "mysql", "redis", "kafka", "docker", "kubernetes", "aws", "gcp", "azure",
    "terraform", "ansible", "git", "linux", "bash",
    # Data/ML
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "nlp", "computer vision",
    "llm", "mlops", "data engineering", "etl", "airflow", "dbt",
    # Soft / misc
    "communication", "leadership", "agile", "scrum", "jira", "confluence"
]

# ----------- API Endpoints -----------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/upload", methods=["POST"])
def upload_resume():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Unsupported file type"}), 400

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        text = extract_text_from_file(save_path)
        return jsonify({
            "filename": filename,
            "text": text,
            "size": os.path.getsize(save_path),
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/screen", methods=["POST"])
def screen_candidates():
    """
    Expected JSON:
    {
      "job_description": "...",
      "candidates": [
         {"id": "1", "text": "resume text"}  OR  {"id": "1", "file": (upload not supported here)}
      ],
      "skills": [optional list of skills to use]
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        job_description = data.get("job_description", "")
        skills = data.get("skills") or DEFAULT_SKILLS
        candidates = data.get("candidates") or []

        if not job_description:
            return jsonify({"error": "job_description is required"}), 400
        if not candidates:
            return jsonify({"error": "candidates is required and must be non-empty"}), 400

        results = []
        for c in candidates:
            cid = str(c.get("id", "")) or None
            text = c.get("text")
            if not text:
                return jsonify({"error": f"candidate {cid or '[unknown]'} missing text"}), 400
            s = score_resume(text, job_description, skills)
            item = {"id": cid, **s}
            results.append(item)

        ranked = rank_candidates(results)
        return jsonify({
            "job_description": job_description,
            "skills_used": skills,
            "results": ranked
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/screen-files", methods=["POST"])
def screen_candidates_files():
    """
    Multipart form-data:
      - job_description: text
      - skills: optional comma-separated skills
      - files: one or multiple resume files (pdf/docx/txt)
    """
    try:
        job_description = request.form.get("job_description", "")
        skills_raw = request.form.get("skills")
        skills = [s.strip() for s in skills_raw.split(",") if s.strip()] if skills_raw else DEFAULT_SKILLS

        if not job_description:
            return jsonify({"error": "job_description is required"}), 400

        if "files" not in request.files:
            return jsonify({"error": "No files part"}), 400

        files = request.files.getlist("files")
        candidates_payload = []
        for idx, file in enumerate(files, start=1):
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(path)
                text = extract_text_from_file(path)
                candidates_payload.append({"id": filename or str(idx), "text": text})

        if not candidates_payload:
            return jsonify({"error": "No valid files uploaded"}), 400

        results = []
        for c in candidates_payload:
            s = score_resume(c["text"], job_description, skills)
            results.append({"id": c["id"], **s})

        ranked = rank_candidates(results)
        return jsonify({
            "job_description": job_description,
            "skills_used": skills,
            "results": ranked
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
