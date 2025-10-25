import sqlite3
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseSetup:
    def __init__(self, db_path='database.db'):
        """Initialize database setup with path."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Create database connection."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database: {self.db_path}")
            return self.conn
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise
            
    def create_tables(self):
        """Create all required database tables."""
        try:
            # Resumes table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS resumes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_name TEXT NOT NULL,
                    email TEXT,
                    phone TEXT,
                    resume_text TEXT NOT NULL,
                    file_name TEXT,
                    file_type TEXT,
                    upload_date TEXT DEFAULT CURRENT_TIMESTAMP,
                    years_experience REAL,
                    education_level TEXT,
                    skills TEXT
                )
            ''')
            
            # Job descriptions table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS job_descriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_title TEXT NOT NULL,
                    jd_text TEXT NOT NULL,
                    required_skills TEXT,
                    min_experience REAL,
                    required_education TEXT,
                    created_date TEXT DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            # Matching results table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS matching_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_id INTEGER NOT NULL,
                    jd_id INTEGER NOT NULL,
                    match_score REAL,
                    matched_skills TEXT,
                    missing_skills TEXT,
                    match_date TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (resume_id) REFERENCES resumes(id),
                    FOREIGN KEY (jd_id) REFERENCES job_descriptions(id)
                )
            ''')
            
            # Training data table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_text TEXT NOT NULL,
                    jd_text TEXT NOT NULL,
                    match_label INTEGER,
                    created_date TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Model metadata table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_version TEXT,
                    accuracy REAL,
                    created_date TEXT DEFAULT CURRENT_TIMESTAMP,
                    file_path TEXT
                )
            ''')
            
            self.conn.commit()
            logger.info("All tables created successfully")
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")
            raise
            
    def insert_sample_data(self):
        """Insert sample job description and resume."""
        try:
            # Sample ML Engineer job description
            self.cursor.execute('''
                INSERT INTO job_descriptions 
                (job_title, jd_text, required_skills, min_experience, required_education, status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                'ML Engineer',
                'We are seeking an experienced ML Engineer to join our team. The ideal candidate will have strong experience in machine learning, deep learning, Python, TensorFlow, PyTorch, and model deployment.',
                'Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, AWS, Docker, Kubernetes',
                3.0,
                'Bachelor',
                'active'
            ))
            
            # Sample resume
            self.cursor.execute('''
                INSERT INTO resumes 
                (candidate_name, email, phone, resume_text, file_name, file_type, years_experience, education_level, skills)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                'John Doe',
                'john.doe@email.com',
                '+1-555-0123',
                'Experienced ML Engineer with 5 years of experience in developing and deploying machine learning models. Proficient in Python, TensorFlow, PyTorch, scikit-learn. Strong background in deep learning, computer vision, and NLP. Experience with AWS, Docker, and Kubernetes for model deployment.',
                'john_doe_resume.pdf',
                'pdf',
                5.0,
                'Master',
                'Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, AWS, Docker, Kubernetes, NLP, Computer Vision'
            ))
            
            self.conn.commit()
            logger.info("Sample data inserted successfully")
        except sqlite3.Error as e:
            logger.error(f"Error inserting sample data: {e}")
            raise
            
    def create_indexes(self):
        """Create indexes for better query performance."""
        try:
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_resume_email ON resumes(email)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_jd_status ON job_descriptions(status)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_match_resume ON matching_results(resume_id)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_match_jd ON matching_results(jd_id)')
            self.conn.commit()
            logger.info("Indexes created successfully")
        except sqlite3.Error as e:
            logger.error(f"Error creating indexes: {e}")
            raise
            
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
            
    def setup_complete_database(self):
        """Execute complete database setup process."""
        try:
            self.connect()
            self.create_tables()
            self.insert_sample_data()
            self.create_indexes()
            logger.info("Database setup completed successfully!")
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise
        finally:
            self.close()

def main():
    """Main function to run database setup."""
    db_setup = DatabaseSetup('database.db')
    db_setup.setup_complete_database()
    print("Database setup complete! Check database.db file.")

if __name__ == "__main__":
    main()
