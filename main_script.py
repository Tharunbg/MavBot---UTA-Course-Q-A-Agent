"""
UTA Course Q&A Agent - Production Grade with Enhanced Analytics
Author: AI/ML Engineering Team

✅ Complete course and professor analytics
✅ Grade-specific lookups with term filtering
✅ Course comparisons with side-by-side tables
✅ Topic-based course recommendations
✅ Enhanced intent detection
✅ Production-grade error handling
"""

# ========================
# CRITICAL: Set environment variables BEFORE any imports
# to prevent segmentation faults on macOS
# ========================
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ========================
# 1. IMPORTS
# ========================
import re
import logging
import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict, List, Optional, Tuple, Any
import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# ========================
# 2. CONFIGURATION MANAGEMENT
# ========================
@dataclass
class ModelConfig:
    """Configuration for all models used in the system."""
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    embed_model_id: str = "BAAI/bge-base-en-v1.5"
    torch_dtype: torch.dtype = torch.float32  # Changed from bfloat16 for macOS compatibility
    device_map: str = "cpu"  # Force CPU for broader compatibility
    max_new_tokens: int = 50  # Reduced for faster CPU inference
    temperature: float = 0.1
    top_k: int = 3

@dataclass
class DataConfig:
    """Configuration for data processing and indexing."""
    data_file: str = "/Users/priyankam/Documents/deploy_ml/project_data.csv"
    index_prefix: str = "uta_production"
    chunk_sizes: Dict[str, int] = field(default_factory=lambda: {
        'courses': 3,
        'professors': 3,
        'sections': 3
    })

@dataclass
class AppConfig:
    """Main application configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    log_level: str = "INFO"
    cache_size: int = 1000

# ========================
# 3. LOGGING SETUP
# ========================
class ProductionLogger:
    """Production-grade logging setup."""

    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def get_logger(self):
        return self.logger

# ========================
# 4. ERROR HANDLING
# ========================
class CourseQAError(Exception):
    """Base exception for Course Q&A system."""
    pass

class DataLoadingError(CourseQAError):
    """Raised when data loading fails."""
    pass

class ModelLoadingError(CourseQAError):
    """Raised when model loading fails."""
    pass

class IndexBuildingError(CourseQAError):
    """Raised when index building fails."""
    pass

# ========================
# 5. DATA PROCESSOR (ENHANCED WITH GRADE ANALYTICS)
# ========================
class DataProcessor:
    """Handles all data processing and aggregation operations with enhanced course information."""

    def __init__(self, config: DataConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.raw_df = None
        self.df_courses = None
        self.df_professors = None
        self.course_title_map = {}  # Cache for course titles

    def load_and_validate_data(self) -> None:
        """Load and validate input data with comprehensive error handling."""
        try:
            self.logger.info(f"Loading data from {self.config.data_file}")

            if not os.path.exists(self.config.data_file):
                raise DataLoadingError(f"Data file not found: {self.config.data_file}")

            self.raw_df = pd.read_csv(self.config.data_file)
            self._validate_dataframe()
            self._preprocess_data()
            self._create_aggregations()
            self._build_course_title_map()
            self._enhance_professor_analytics()

            self.logger.info("Data loading and validation completed successfully")

        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise DataLoadingError(f"Failed to load data: {str(e)}")

    def _validate_dataframe(self) -> None:
        """Validate the structure and content of the dataframe."""
        required_columns = ['Subject', 'Catalog Number', 'Course Name', 'Primary Instructor First Name',
                           'Primary Instructor Last Name', 'Description', 'Term', 'A', 'B', 'C', 'D', 'F', 'Total Grades']

        missing_columns = [col for col in required_columns if col not in self.raw_df.columns]
        if missing_columns:
            raise DataLoadingError(f"Missing required columns: {missing_columns}")

        # Check for optional columns and log warnings
        optional_columns = ['Course Career', 'GRAD']
        for col in optional_columns:
            if col not in self.raw_df.columns:
                self.logger.warning(f"Optional column '{col}' not found in dataset")

        if len(self.raw_df) == 0:
            raise DataLoadingError("Dataframe is empty")

    def _preprocess_data(self) -> None:
        """Preprocess the raw data with proper type handling."""
        # Numeric columns processing
        numeric_cols = ['A', 'B', 'C', 'D', 'F', 'W', 'P', 'I', 'Q', 'Z', 'R', 'Total Grades']
        for col in numeric_cols:
            if col in self.raw_df.columns:
                self.raw_df[col] = pd.to_numeric(self.raw_df[col], errors='coerce').fillna(0)

        # Text columns processing
        text_cols = ['Primary Instructor First Name', 'Primary Instructor Last Name',
                    'Description', 'Course Name', 'Term', 'Course Career', 'GRAD']
        for col in text_cols:
            if col in self.raw_df.columns:
                self.raw_df[col] = self.raw_df[col].fillna("").astype(str)

        # Create derived columns
        self.raw_df['course_code'] = (
            self.raw_df['Subject'].astype(str).str.strip() + ' ' +
            self.raw_df['Catalog Number'].astype(str).str.strip()
        )
        self.raw_df['instructor'] = (
            self.raw_df['Primary Instructor First Name'] + ' ' +
            self.raw_df['Primary Instructor Last Name']
        ).str.strip()

        # Calculate GPA and rates
        self.raw_df['graded_total'] = self.raw_df[['A', 'B', 'C', 'D', 'F']].sum(axis=1)
        gpa_numerator = (self.raw_df['A'] * 4 + self.raw_df['B'] * 3 +
                        self.raw_df['C'] * 2 + self.raw_df['D'] * 1)
        self.raw_df['gpa'] = (gpa_numerator / self.raw_df['graded_total'].replace(0, np.nan)).fillna(0)
        self.raw_df['a_rate'] = (self.raw_df['A'] / self.raw_df['graded_total'].replace(0, np.nan) * 100).fillna(0)
        self.raw_df['pass_rate'] = ((self.raw_df['A'] + self.raw_df['B'] + self.raw_df['C']) /
                                   self.raw_df['Total Grades'].replace(0, np.nan) * 100).fillna(0)
        self.raw_df['dfw_rate'] = ((self.raw_df['D'] + self.raw_df['F'] + self.raw_df['W']) /
                                  self.raw_df['Total Grades'].replace(0, np.nan) * 100).fillna(0)

    def _create_aggregations(self) -> None:
        """Create aggregated views of the data with enhanced course information."""
        # Course-level aggregation with additional fields
        aggregation_dict = {
            'title': ('Course Name', 'first'),
            'description': ('Description', 'first'),
            'avg_gpa': ('gpa', 'mean'),
            'avg_a_rate': ('a_rate', 'mean'),
            'avg_pass_rate': ('pass_rate', 'mean'),
            'avg_dfw_rate': ('dfw_rate', 'mean'),
            'total_students': ('Total Grades', 'sum'),
            'times_offered': ('Term', 'count')
        }

        # Add optional columns if they exist
        if 'Course Career' in self.raw_df.columns:
            aggregation_dict['course_career'] = ('Course Career', 'first')
        if 'GRAD' in self.raw_df.columns:
            aggregation_dict['grad'] = ('GRAD', 'first')

        self.df_courses = self.raw_df.groupby('course_code').agg(**aggregation_dict).reset_index()

        # Professor-level aggregation
        professor_df = self.raw_df[self.raw_df['instructor'] != ""].copy()
        self.df_professors = professor_df.groupby('instructor').agg(
            avg_gpa_given=('gpa', 'mean'),
            avg_a_rate=('a_rate', 'mean'),
            avg_pass_rate=('pass_rate', 'mean'),
            avg_dfw_rate=('dfw_rate', 'mean'),
            total_students=('Total Grades', 'sum'),
            courses_taught=('course_code', lambda x: sorted(x.unique().tolist())),
            terms_taught=('Term', 'count')
        ).reset_index()

    def _enhance_professor_analytics(self) -> None:
        """Add enhanced analytics for professors."""
        # Calculate professor teaching style based on grading patterns
        for idx, prof_row in self.df_professors.iterrows():
            prof_name = prof_row['instructor']
            prof_courses = self.raw_df[self.raw_df['instructor'] == prof_name]

            if len(prof_courses) > 0:
                # Calculate additional metrics
                total_graded = prof_courses['graded_total'].sum()
                a_plus_b_rate = ((prof_courses['A'].sum() + prof_courses['B'].sum()) /
                               total_graded * 100) if total_graded > 0 else 0

                # Determine teaching style
                if prof_row['avg_gpa_given'] >= 3.6 and prof_row['avg_a_rate'] >= 60:
                    teaching_style = "Generous Grader"
                elif prof_row['avg_gpa_given'] <= 2.8 or prof_row['avg_dfw_rate'] >= 20:
                    teaching_style = "Tough Grader"
                else:
                    teaching_style = "Balanced Grader"

                # Add to dataframe
                self.df_professors.at[idx, 'a_plus_b_rate'] = a_plus_b_rate
                self.df_professors.at[idx, 'teaching_style'] = teaching_style

    def _build_course_title_map(self) -> None:
        """Build a cache of course codes to titles for quick lookup."""
        self.course_title_map = dict(zip(self.df_courses['course_code'], self.df_courses['title']))
        self.logger.info(f"Built course title map with {len(self.course_title_map)} entries")

    def get_course_title(self, course_code: str) -> str:
        """Get course title with validation."""
        course_code = course_code.upper().strip()
        if course_code in self.course_title_map:
            return self.course_title_map[course_code]

        # Try to find close matches
        for code in self.course_title_map:
            if code.replace(" ", "") == course_code.replace(" ", ""):
                return self.course_title_map[code]

        self.logger.warning(f"Course title not found for: {course_code}")
        return "Title not available"

    def validate_course_code(self, course_code: str) -> bool:
        """Validate if course code exists in dataset."""
        course_code = course_code.upper().strip()
        return course_code in self.course_title_map

    def get_specific_grades(self, course_code: str, term: str = None, professor: str = None) -> Dict[str, Any]:
        """Get specific grade counts for a course/term/professor combination."""
        filters = [self.raw_df['course_code'] == course_code.upper()]

        if term:
            # Handle various term formats
            term_lower = term.lower()
            if 'spring' in term_lower:
                season = 'Spring'
            elif 'fall' in term_lower:
                season = 'Fall'
            elif 'summer' in term_lower:
                season = 'Summer'
            else:
                season = None

            year_match = re.search(r'(\d{4})', term)
            year = year_match.group(1) if year_match else None

            if season and year:
                term_filter = (self.raw_df['Term'].str.contains(season, case=False, na=False) &
                             self.raw_df['Term'].str.contains(year, na=False))
                filters.append(term_filter)

        if professor:
            prof_match = self.find_best_professor_match(professor)
            if prof_match:
                filters.append(self.raw_df['instructor'] == prof_match)

        # Apply filters
        mask = filters[0]
        for f in filters[1:]:
            mask = mask & f

        sections = self.raw_df[mask]

        if sections.empty:
            return {"error": "No matching sections found"}

        # Aggregate grades
        result = {
            'course_code': course_code,
            'term': term,
            'professor': professor,
            'total_sections': len(sections),
            'total_students': int(sections['Total Grades'].sum()),
            'grades': {
                'A': int(sections['A'].sum()),
                'B': int(sections['B'].sum()),
                'C': int(sections['C'].sum()),
                'D': int(sections['D'].sum()),
                'F': int(sections['F'].sum()),
                'W': int(sections['W'].sum())
            },
            'avg_gpa': float(sections['gpa'].mean()),
            'instructors': sections['instructor'].unique().tolist(),
            'terms': sections['Term'].unique().tolist()
        }

        # Calculate percentages
        total_graded = sum(result['grades'].values()) - result['grades']['W']
        if total_graded > 0:
            for grade in ['A', 'B', 'C', 'D', 'F']:
                result['grades'][f'{grade}_pct'] = (result['grades'][grade] / total_graded * 100)

        return result

    def find_best_professor_match(self, professor_query: str) -> Optional[str]:
        """Find best professor name match."""
        professors = self.df_professors['instructor'].tolist()
        query_parts = professor_query.lower().split()

        for professor in professors:
            professor_lower = professor.lower()
            if all(part in professor_lower for part in query_parts):
                return professor
        return None

    def get_courses_by_topic(self, topic: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Find courses related to a specific topic."""
        topic_lower = topic.lower()
        matching_courses = []

        for _, course in self.df_courses.iterrows():
            course_text = f"{course['title']} {course['description']}".lower()

            # Check for topic matches
            if (topic_lower in course_text or
                any(word in course_text for word in topic_lower.split())):

                matching_courses.append({
                    'course_code': course['course_code'],
                    'title': course['title'],
                    'avg_gpa': course['avg_gpa'],
                    'avg_pass_rate': course['avg_pass_rate'],
                    'total_students': course['total_students'],
                    'description': course['description'][:200] + "..." if len(course['description']) > 200 else course['description']
                })

        # Sort by GPA (highest first) for "easiest" queries
        if 'easy' in topic_lower or 'easiest' in topic_lower:
            matching_courses.sort(key=lambda x: x['avg_gpa'], reverse=True)
        elif 'hard' in topic_lower or 'hardest' in topic_lower:
            matching_courses.sort(key=lambda x: x['avg_gpa'])
        else:
            # Default sort by relevance (GPA as proxy for popularity/quality)
            matching_courses.sort(key=lambda x: x['avg_gpa'], reverse=True)

        return matching_courses[:max_results]

    def compare_courses(self, course_codes: List[str]) -> Dict[str, Any]:
        """Compare multiple courses side by side."""
        comparison_data = {}

        for course_code in course_codes:
            course_data = self.df_courses[self.df_courses['course_code'] == course_code]
            if not course_data.empty:
                course = course_data.iloc[0]
                comparison_data[course_code] = {
                    'title': course['title'],
                    'avg_gpa': course['avg_gpa'],
                    'avg_pass_rate': course['avg_pass_rate'],
                    'avg_dfw_rate': course['avg_dfw_rate'],
                    'total_students': course['total_students'],
                    'times_offered': course['times_offered'],
                    'description': course['description']
                }

        return comparison_data

    def create_search_chunks(self) -> Dict[str, pd.DataFrame]:
        """Create optimized search chunks for retrieval with enhanced information."""
        # Course chunks with career and grad information
        self.df_courses['search_chunk'] = self.df_courses.apply(
            lambda row: self._create_course_search_chunk(row), axis=1
        )

        self.df_courses['display_chunk'] = self.df_courses.apply(
            lambda row: self._create_course_display_chunk(row), axis=1
        )

        # Enhanced professor chunks with analytics
        self.df_professors['search_chunk'] = self.df_professors.apply(
            lambda row: self._create_professor_search_chunk(row), axis=1
        )

        self.df_professors['display_chunk'] = self.df_professors.apply(
            lambda row: self._create_professor_display_chunk(row), axis=1
        )

        # Section chunks with enhanced information
        self.raw_df['search_chunk'] = self.raw_df.apply(
            lambda row: self._create_section_search_chunk(row), axis=1
        )

        self.raw_df['display_chunk'] = self.raw_df.apply(
            lambda row: self._create_section_display_chunk(row), axis=1
        )

        return {
            'courses': self.df_courses,
            'professors': self.df_professors,
            'sections': self.raw_df
        }

    def _create_course_search_chunk(self, row: pd.Series) -> str:
        """Create search chunk for course information with career and grad data."""
        base_text = (
            f"Course: {row['course_code']} {row['title']}. "
            f"Description: {row['description']}. "
            f"Average GPA: {row['avg_gpa']:.2f}. "
            f"Pass Rate: {row['avg_pass_rate']:.1f}%. "
            f"Total Students: {row['total_students']}."
        )

        # Add career information if available
        if 'course_career' in row and row['course_career'] and row['course_career'] != "nan":
            base_text += f" Course Career: {row['course_career']}."

        # Add GRAD information if available
        if 'grad' in row and row['grad'] and row['grad'] != "nan":
            base_text += f" GRAD: {row['grad']}."

        return base_text

    def _create_course_display_chunk(self, row: pd.Series) -> str:
        """Create detailed display chunk for course information with all metadata."""
        # Base course information
        display_parts = [
            f"📚 Course: {row['course_code']} - {row['title']}",
            f"📖 Description: {row['description']}",
            f"📊 Overall Stats: Average GPA: {row['avg_gpa']:.2f}, "
            f"Pass Rate: {row['avg_pass_rate']:.1f}%, "
            f"Total Students: {row['total_students']}, "
            f"Times Offered: {row['times_offered']}"
        ]

        # Add career information if available
        if 'course_career' in row and row['course_career'] and row['course_career'] != "nan":
            display_parts.append(f"🎯 Course Career: {row['course_career']}")

        # Add GRAD information if available
        if 'grad' in row and row['grad'] and row['grad'] != "nan":
            display_parts.append(f"🎓 GRAD: {row['grad']}")

        # Add recent history
        history_df = self.raw_df[self.raw_df['course_code'] == row['course_code']].copy()
        if not history_df.empty:
            try:
                history_df['term_year'] = history_df['Term'].str.extract(r'(\d{4})').fillna('0').astype(int)
                recent_terms = history_df.sort_values('term_year', ascending=False).head(5)
            except:
                recent_terms = history_df.tail(5)

            history_lines = []
            for _, term_row in recent_terms.iterrows():
                history_lines.append(
                    f"  - {term_row['Term']}: {term_row['instructor']} (GPA: {term_row['gpa']:.2f}, Students: {term_row['Total Grades']})"
                )
            display_parts.append(f"\n📈 Recent Offerings:\n" + "\n".join(history_lines))

        return "\n".join(display_parts)

    def _create_professor_search_chunk(self, row: pd.Series) -> str:
        """Create search chunk for professor information with analytics."""
        return (
            f"Professor: {row['instructor']}. "
            f"Average GPA: {row['avg_gpa_given']:.2f}. "
            f"A Rate: {row['avg_a_rate']:.1f}%. "
            f"Pass Rate: {row['avg_pass_rate']:.1f}%. "
            f"Teaching Style: {row.get('teaching_style', 'Unknown')}. "
            f"Courses: {', '.join(row['courses_taught'][:5])}."
        )

    def _create_professor_display_chunk(self, row: pd.Series) -> str:
        """Create detailed display chunk for professor information."""
        display_parts = [
            f"👨‍🏫 Professor: {row['instructor']}",
            f"📊 Teaching Statistics:",
            f"  - Average GPA Given: {row['avg_gpa_given']:.2f}",
            f"  - A Rate: {row['avg_a_rate']:.1f}%",
            f"  - A+B Rate: {row.get('a_plus_b_rate', 0):.1f}%",
            f"  - Pass Rate: {row['avg_pass_rate']:.1f}%",
            f"  - DFW Rate: {row['avg_dfw_rate']:.1f}%",
            f"  - Total Students: {int(row['total_students'])}",
            f"  - Terms Taught: {row['terms_taught']}",
            f"  - Teaching Style: {row.get('teaching_style', 'Unknown')}"
        ]

        display_parts.append(f"\n📚 Courses Taught ({len(row['courses_taught'])}):")
        for i, course_code in enumerate(row['courses_taught'][:10], 1):
            title = self.get_course_title(course_code)
            display_parts.append(f"  {i}. {course_code}: {title}")

        if len(row['courses_taught']) > 10:
            display_parts.append(f"  ... and {len(row['courses_taught']) - 10} more courses")

        return "\n".join(display_parts)

    def _create_section_search_chunk(self, row: pd.Series) -> str:
        """Create search chunk for section information."""
        base_text = (
            f"Section: {row['course_code']} in {row['Term']} taught by {row['instructor']}. "
            f"Grades: A={row['A']}, B={row['B']}, C={row['C']}. "
            f"GPA: {row['gpa']:.2f}, Students: {row['Total Grades']}."
        )

        # Add career and grad information if available
        if 'Course Career' in row and row['Course Career'] and row['Course Career'] != "nan":
            base_text += f" Course Career: {row['Course Career']}."

        if 'GRAD' in row and row['GRAD'] and row['GRAD'] != "nan":
            base_text += f" GRAD: {row['GRAD']}."

        return base_text

    def _create_section_display_chunk(self, row: pd.Series) -> str:
        """Create display chunk for section information with all metadata."""
        display_parts = [
            f"📋 Section: {row['course_code']} in {row['Term']}",
            f"👨‍🏫 Instructor: {row['instructor']}",
            f"📊 Grades: A={row['A']}, B={row['B']}, C={row['C']}, D={row['D']}, F={row['F']}, W={row['W']}",
            f"🎯 Total Students: {row['Total Grades']}, GPA: {row['gpa']:.2f}"
        ]

        # Add career information if available
        if 'Course Career' in row and row['Course Career'] and row['Course Career'] != "nan":
            display_parts.append(f"🎯 Course Career: {row['Course Career']}")

        # Add GRAD information if available
        if 'GRAD' in row and row['GRAD'] and row['GRAD'] != "nan":
            display_parts.append(f"🎓 GRAD: {row['GRAD']}")

        return "\n".join(display_parts)

# ========================
# 6. VECTOR STORE MANAGER
# ========================
class VectorStoreManager:
    """Manages vector storage and retrieval with FAISS."""

    def __init__(self, config: DataConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.embed_model = None
        self.indices = {}

    def initialize_embedding_model(self, model_id: str) -> None:
        """Initialize the embedding model with error handling."""
        try:
            self.logger.info(f"Initializing embedding model: {model_id}")
            self.embed_model = SentenceTransformer(
                model_id,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            self.logger.info("Embedding model initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise ModelLoadingError(f"Embedding model loading failed: {str(e)}")

    def build_or_load_index(self, df: pd.DataFrame, text_column: str,
                          index_name: str, force_rebuild: bool = False) -> faiss.Index:
        """Build or load FAISS index with caching."""
        index_path = f"{self.config.index_prefix}_{index_name}.index"

        if not force_rebuild and os.path.exists(index_path):
            self.logger.info(f"Loading existing index: {index_path}")
            try:
                index = faiss.read_index(index_path)
                self.indices[index_name] = index
                return index
            except Exception as e:
                self.logger.warning(f"Failed to load index, rebuilding: {str(e)}")

        self.logger.info(f"Building new index: {index_name}")
        try:
            corpus = df[text_column].tolist()
            # Disable multiprocessing to avoid segmentation faults on macOS
            embeddings = self.embed_model.encode(
                corpus,
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=16,  # Reduced batch size for stability
                convert_to_numpy=True,
                device='cpu' if not torch.cuda.is_available() else 'cuda'
            )

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings.astype(np.float32))

            # Save index
            faiss.write_index(index, index_path)
            self.indices[index_name] = index

            self.logger.info(f"Index built and saved: {index_path}")
            return index

        except Exception as e:
            self.logger.error(f"Index building failed: {str(e)}")
            raise IndexBuildingError(f"Failed to build index {index_name}: {str(e)}")

    @lru_cache(maxsize=1000)
    def retrieve_similar(self, query: str, index_name: str, top_k: int = 3) -> List[int]:
        """Retrieve similar items with caching."""
        if index_name not in self.indices:
            raise IndexBuildingError(f"Index {index_name} not found")

        query_vec = self.embed_model.encode([query], normalize_embeddings=True)
        distances, indices = self.indices[index_name].search(query_vec.astype(np.float32), top_k)

        return indices[0].tolist()

# ========================
# 7. LLM MANAGER
# ========================
class LLMManager:
    """Manages LLM operations with optimizations."""

    def __init__(self, config: ModelConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.tokenizer = None
        self.model = None
        self.pipeline = None

    def initialize_models(self) -> None:
        """Initialize LLM models with optimizations."""
        try:
            self.logger.info(f"Loading tokenizer and model: {self.config.model_id}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=self.config.torch_dtype,
                device_map=self.config.device_map,
                trust_remote_code=True
            )

            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=self.config.torch_dtype,
                device_map=self.config.device_map
            )

            self.logger.info("LLM models initialized successfully")

        except Exception as e:
            self.logger.error(f"LLM initialization failed: {str(e)}")
            raise ModelLoadingError(f"LLM loading failed: {str(e)}")

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response with proper error handling."""
        try:
            start_time = time.time()

            generation_config = {
                'max_new_tokens': kwargs.get('max_new_tokens', self.config.max_new_tokens),
                'temperature': kwargs.get('temperature', self.config.temperature),
                'do_sample': True,
                'pad_token_id': self.tokenizer.eos_token_id
            }

            result = self.pipeline(prompt, **generation_config)
            generated_text = result[0]['generated_text']

            # Extract only the new generated content
            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text.strip()

            latency = time.time() - start_time
            self.logger.debug(f"Generation completed in {latency:.2f}s")

            return response

        except Exception as e:
            self.logger.error(f"Text generation failed: {str(e)}")
            return "I apologize, but I encountered an error while generating a response."

# ========================
# 8. QUERY PROCESSOR (ENHANCED WITH NEW FUNCTIONALITY)
# ========================
class QueryProcessor:
    """Processes and routes user queries with enhanced course information."""

    def __init__(self, data_processor: DataProcessor, vector_store: VectorStoreManager,
                 llm_manager: LLMManager, logger: logging.Logger):
        self.data_processor = data_processor
        self.vector_store = vector_store
        self.llm_manager = llm_manager
        self.logger = logger

        self.intent_patterns = {
            'direct_lookup': r'^[A-Z]{3,4}\s\d{4}$',
            'course_by_professor': r'([A-Z]{3,4}\s\d{4}).*(professor|instructor|by|teach|with)\s+([\w\s]+)',
            'professor_courses': r'(courses? by|professor|instructor)\s+([\w\s]+)',
            'course_comparison': r'compare|vs|versus|between.*and',
            'course_history': r'history|over time|semester.*history',
            'professor_lookup': r'^(professor|instructor)\s+([\w\s]+)$',
            'grade_lookup': r'(grade|a|b|c|d|f|w).*(spring|fall|summer)\s*\d{4}',
            'topic_search': r'(easy|hard|best|easiest|hardest|recommend).*(course|class)',
            'best_professor': r'(best|good|top|recommend|which).*(professor|instructor).*(for|teach|teaching)',
            'professor_by_subject': r'(professor|instructor).*(for|teaching|teach).*(\w+\s+\w+)'
        }

    def extract_course_codes(self, query: str) -> List[str]:
        """Extract course codes from query with better matching."""
        patterns = [
            r'\b([A-Z]{2,4})\s*(\d{4})\b',  # Standard: CSE 5334
            r'\b([A-Z]{2,4})(\d{4})\b',     # No space: CSE5334
        ]

        codes = []
        for pattern in patterns:
            matches = re.findall(pattern, query.upper())
            for match in matches:
                if len(match) == 2:
                    course_code = f"{match[0]} {match[1]}"
                    if self.data_processor.validate_course_code(course_code):
                        codes.append(course_code)

        return codes

    def detect_intent(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Detect user intent and extract parameters."""
        query_lower = query.lower()
        extracted_params = {}

        # NEW: Check for "best professor for X" queries FIRST (before topic_search)
        if re.search(r'(best|good|top|recommend|which).*(professor|instructor).*(for|teach|teaching)', query_lower):
            # Try to extract the subject
            subject_match = re.search(r'for\s+([\w\s]+)$', query_lower)
            subject = subject_match.group(1).strip() if subject_match else None
            
            # If no subject after "for", look for subject anywhere in query
            if not subject:
                # Remove common words and look for subject
                words = query_lower.split()
                common_words = {'best', 'good', 'top', 'recommend', 'which', 'professor', 'professors', 
                              'instructor', 'instructors', 'for', 'teach', 'teaching', 'course', 'courses'}
                potential_subjects = [word for word in words if word not in common_words]
                subject = ' '.join(potential_subjects) if potential_subjects else query_lower
            
            return 'best_professor', {'subject': subject, 'query': query}
        
        # NEW: Check for "professor for X" queries
        if re.search(r'(professor|instructor).*(for|teaching|teach)', query_lower):
            subject_match = re.search(r'for\s+([\w\s]+)$', query_lower)
            subject = subject_match.group(1).strip() if subject_match else None
            
            if subject:
                return 'professor_by_subject', {'subject': subject}

        # Check for complex multi-part queries first
        if any(word in query_lower for word in ['electrical', 'computer', 'programming', 'machine learning', 'data science', 'embedding']):
            if any(word in query_lower for word in ['high a rate', 'easy', 'best professor', 'related']):
                return 'topic_search', {'query': query}

        # Check for grade-specific queries first
        if re.search(r'(grade|a|b|c|d|f|w).*(spring|fall|summer)\s*\d{4}', query_lower):
            codes = self.extract_course_codes(query)
            if codes:
                # Extract term and professor
                term_match = re.search(r'(spring|fall|summer)\s*(\d{4})', query_lower)
                term = term_match.group(0) if term_match else None

                prof_match = re.search(r'(?:professor|instructor|by)\s+([\w\s]+)', query_lower)
                professor = prof_match.group(1).strip() if prof_match else None

                return 'grade_lookup', {
                    'course_code': codes[0],
                    'term': term,
                    'professor': professor
                }

        # Check for comparison queries
        if 'compare' in query_lower or ' vs ' in query_lower or ' versus ' in query_lower:
            codes = self.extract_course_codes(query)
            if len(codes) >= 2:
                return 'course_comparison', {'course_codes': codes}

        # Check for topic-based searches (but exclude professor queries)
        if (any(word in query_lower for word in ['easy', 'hard', 'easiest', 'hardest']) and 
            'professor' not in query_lower and 'instructor' not in query_lower):
            return 'topic_search', {'query': query}

        # Check for direct course lookup
        codes = self.extract_course_codes(query)
        if len(codes) == 1 and re.fullmatch(r'^[A-Z]{3,4}\s\d{4}$', query.upper().strip()):
            return 'direct_lookup', {'course_code': codes[0]}

        # Check each intent pattern
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, query_lower, re.IGNORECASE):
                if intent == 'course_by_professor':
                    match = re.search(pattern, query_lower, re.IGNORECASE)
                    if match and codes:
                        extracted_params = {
                            'course_code': codes[0],
                            'professor_query': match.group(3).strip()
                        }
                        return intent, extracted_params
                elif intent == 'professor_courses':
                    match = re.search(pattern, query_lower, re.IGNORECASE)
                    if match:
                        extracted_params = {'professor_query': match.group(2).strip()}
                        return intent, extracted_params
                elif intent == 'professor_lookup':
                    match = re.search(pattern, query_lower, re.IGNORECASE)
                    if match:
                        extracted_params = {'professor_query': match.group(2).strip()}
                        return intent, extracted_params

        # If we have course codes but no specific intent, use direct lookup
        if codes:
            return 'direct_lookup', {'course_code': codes[0]}

        # Default to semantic search
        return 'semantic_search', {'query': query}

    def find_professor_match(self, professor_query: str) -> Optional[str]:
        """Find best professor name match."""
        return self.data_processor.find_best_professor_match(professor_query)

    def retrieve_context(self, query: str, intent: str, params: Dict[str, Any]) -> str:
        """Retrieve relevant context based on intent with enhanced course information."""
        try:
            if intent == 'direct_lookup':
                course_code = params['course_code']
                course_data = self.data_processor.df_courses[
                    self.data_processor.df_courses['course_code'] == course_code
                ]
                if not course_data.empty:
                    return course_data.iloc[0]['display_chunk']
                return f"No data found for course: {course_code}"

            elif intent == 'course_by_professor':
                return self._handle_course_by_professor(params)

            elif intent == 'professor_courses':
                return self._handle_professor_courses(params)

            elif intent == 'professor_lookup':
                return self._handle_professor_lookup(params)

            elif intent == 'course_history':
                codes = self.extract_course_codes(query)
                if codes:
                    return self._handle_course_history(codes[0])

            elif intent == 'grade_lookup':
                return self._handle_grade_lookup(params)

            elif intent == 'course_comparison':
                return self._handle_course_comparison(params)

            elif intent == 'topic_search':
                return self._handle_topic_search(params)
            
            # NEW: Handle best professor queries
            elif intent == 'best_professor':
                return self._handle_best_professor(params)
            
            # NEW: Handle professor by subject queries
            elif intent == 'professor_by_subject':
                return self._handle_best_professor(params)  # Reuse same handler

            # Semantic search fallback
            indices_to_search = ['courses', 'professors', 'sections']
            all_contexts = []

            for index_name in indices_to_search:
                indices = self.vector_store.retrieve_similar(query, index_name)
                df = getattr(self.data_processor, f'df_{index_name}', None)
                if df is not None:
                    for idx in indices:
                        if 0 <= idx < len(df):
                            all_contexts.append(df.iloc[idx]['display_chunk'])

            return "\n\n".join(all_contexts[:5]) if all_contexts else "No relevant information found."

        except Exception as e:
            self.logger.error(f"Context retrieval failed: {str(e)}")
            return "Error retrieving context for your query."

    def _handle_course_by_professor(self, params: Dict[str, Any]) -> str:
        """Handle course by professor query with enhanced information."""
        course_code = params['course_code']
        professor_name = self.find_professor_match(params['professor_query'])

        if not professor_name:
            return f"Could not find professor matching: {params['professor_query']}"

        history_df = self.data_processor.raw_df[
            (self.data_processor.raw_df['course_code'] == course_code) &
            (self.data_processor.raw_df['instructor'] == professor_name)
        ]

        if not history_df.empty:
            history_lines = []
            for _, row in history_df.head(5).iterrows():
                line = f"- {row['Term']}: GPA {row['gpa']:.2f}, Students: {row['Total Grades']}"

                # Add career and grad information if available
                if 'Course Career' in row and row['Course Career'] and row['Course Career'] != "nan":
                    line += f", Career: {row['Course Career']}"
                if 'GRAD' in row and row['GRAD'] and row['GRAD'] != "nan":
                    line += f", GRAD: {row['GRAD']}"

                history_lines.append(line)

            history_text = "\n".join(history_lines)

            # Get course base information
            course_info = self.data_processor.df_courses[
                self.data_processor.df_courses['course_code'] == course_code
            ]
            base_info = ""
            if not course_info.empty:
                base_info = course_info.iloc[0]['display_chunk'].split('\n📈')[0] + "\n\n"

            return f"{base_info}📊 History for {course_code} taught by {professor_name}:\n{history_text}"
        else:
            return f"No history found for {course_code} taught by {professor_name}"

    def _handle_professor_courses(self, params: Dict[str, Any]) -> str:
        """Handle professor courses query with enhanced course information."""
        professor_name = self.find_professor_match(params['professor_query'])

        if not professor_name:
            return f"Could not find professor matching: {params['professor_query']}"

        prof_data = self.data_processor.df_professors[
            self.data_processor.df_professors['instructor'] == professor_name
        ]

        if not prof_data.empty:
            return prof_data.iloc[0]['display_chunk']

        return f"No courses found for professor: {professor_name}"

    def _handle_professor_lookup(self, params: Dict[str, Any]) -> str:
        """Handle professor lookup query."""
        professor_name = self.find_professor_match(params['professor_query'])

        if not professor_name:
            return f"Could not find professor matching: {params['professor_query']}"

        prof_data = self.data_processor.df_professors[
            self.data_processor.df_professors['instructor'] == professor_name
        ]

        if not prof_data.empty:
            return prof_data.iloc[0]['display_chunk']

        return f"No data found for professor: {professor_name}"

    def _handle_course_history(self, course_code: str) -> str:
        """Handle course history query with enhanced information."""
        history_df = self.data_processor.raw_df[
            self.data_processor.raw_df['course_code'] == course_code
        ]

        if not history_df.empty:
            # Get course base information first
            course_info = self.data_processor.df_courses[
                self.data_processor.df_courses['course_code'] == course_code
            ]
            base_info = ""
            if not course_info.empty:
                base_info = course_info.iloc[0]['display_chunk'].split('\n📈')[0] + "\n\n"

            history_lines = []
            for _, row in history_df.head(10).iterrows():
                line = f"- {row['Term']}: {row['instructor']} (GPA: {row['gpa']:.2f}, Students: {row['Total Grades']})"

                # Add career and grad information if available
                if 'Course Career' in row and row['Course Career'] and row['Course Career'] != "nan":
                    line += f", Career: {row['Course Career']}"
                if 'GRAD' in row and row['GRAD'] and row['GRAD'] != "nan":
                    line += f", GRAD: {row['GRAD']}"

                history_lines.append(line)

            history_text = "\n".join(history_lines)
            return f"{base_info}📈 Historical Data for {course_code}:\n{history_text}"

        return f"No history found for course: {course_code}"

    def _handle_grade_lookup(self, params: Dict[str, Any]) -> str:
        """Handle specific grade lookup queries."""
        course_code = params['course_code']
        term = params.get('term')
        professor = params.get('professor')

        grade_data = self.data_processor.get_specific_grades(course_code, term, professor)

        if 'error' in grade_data:
            return grade_data['error']

        # Format the response
        response = f"📊 **GRADE REPORT: {course_code}**\n\n"

        if term:
            response += f"**Term:** {term}\n"
        if professor:
            response += f"**Professor:** {professor}\n"

        response += f"**Total Sections:** {grade_data['total_sections']}\n"
        response += f"**Total Students:** {grade_data['total_students']}\n"
        response += f"**Average GPA:** {grade_data['avg_gpa']:.2f}\n\n"

        response += "**Grade Distribution:**\n"
        grades = grade_data['grades']
        for grade in ['A', 'B', 'C', 'D', 'F', 'W']:
            count = grades[grade]
            if f'{grade}_pct' in grades:
                pct = grades[f'{grade}_pct']
                response += f"- {grade}: {count} ({pct:.1f}%)\n"
            else:
                response += f"- {grade}: {count}\n"

        if len(grade_data['instructors']) > 0:
            response += f"\n**Instructors:** {', '.join(grade_data['instructors'])}"

        return response

    def _handle_course_comparison(self, params: Dict[str, Any]) -> str:
        """Handle course comparison queries."""
        course_codes = params['course_codes']
        comparison_data = self.data_processor.compare_courses(course_codes)

        if not comparison_data:
            return "No valid courses found for comparison."

        response = "📊 **COURSE COMPARISON**\n\n"

        # Create comparison table
        headers = ["Course", "Title", "GPA", "Pass Rate", "DFW Rate", "Students"]
        rows = []

        for course_code, data in comparison_data.items():
            rows.append([
                course_code,
                data['title'][:30] + "..." if len(data['title']) > 30 else data['title'],
                f"{data['avg_gpa']:.2f}",
                f"{data['avg_pass_rate']:.1f}%",
                f"{data['avg_dfw_rate']:.1f}%",
                f"{data['total_students']:,}"
            ])

        # Simple table formatting
        col_widths = [10, 35, 8, 12, 12, 12]
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        response += header_line + "\n"
        response += "-" * len(header_line) + "\n"

        for row in rows:
            response += " | ".join(str(item).ljust(w) for item, w in zip(row, col_widths)) + "\n"

        response += f"\n**Comparison Summary:**\n"

        # Find best in each category
        if len(comparison_data) > 1:
            highest_gpa = max(comparison_data.items(), key=lambda x: x[1]['avg_gpa'])
            highest_pass = max(comparison_data.items(), key=lambda x: x[1]['avg_pass_rate'])
            lowest_dfw = min(comparison_data.items(), key=lambda x: x[1]['avg_dfw_rate'])

            response += f"- Highest GPA: {highest_gpa[0]} ({highest_gpa[1]['avg_gpa']:.2f})\n"
            response += f"- Highest Pass Rate: {highest_pass[0]} ({highest_pass[1]['avg_pass_rate']:.1f}%)\n"
            response += f"- Lowest DFW Rate: {lowest_dfw[0]} ({lowest_dfw[1]['avg_dfw_rate']:.1f}%)\n"

        return response

    def _handle_topic_search(self, params: Dict[str, Any]) -> str:
        """Handle topic-based course searches."""
        query = params['query']
        matching_courses = self.data_processor.get_courses_by_topic(query)

        if not matching_courses:
            return f"No courses found matching: '{query}'"

        response = f"🔎 **FOUND {len(matching_courses)} COURSES:**\n\n"

        for i, course in enumerate(matching_courses, 1):
            response += f"**{course['course_code']}** - {course['title']}\n"
            response += f"   Difficulty: {'Easy' if course['avg_gpa'] >= 3.5 else 'Medium' if course['avg_gpa'] >= 3.0 else 'Hard'} | "
            response += f"GPA: {course['avg_gpa']:.2f} | Pass: {course['avg_pass_rate']:.1f}%\n"
            response += f"   {course['description']}\n\n"

        return response

    def _handle_best_professor(self, params: Dict[str, Any]) -> str:
        """Handle 'best professor for X' queries."""
        subject = params.get('subject', '').lower()
        
        self.logger.info(f"Analyzing best professors for subject: {subject}")
        
        # Find courses related to the subject
        related_courses = self.data_processor.get_courses_by_topic(subject, max_results=20)
        
        if not related_courses:
            return f"❌ No courses found related to '{subject}'. Try a different subject."
        
        # Get all professors who teach these courses
        professor_stats = {}
        
        for course in related_courses:
            course_code = course['course_code']
            
            # Find sections of this course
            sections = self.data_processor.raw_df[
                self.data_processor.raw_df['course_code'] == course_code
            ]
            
            for _, section in sections.iterrows():
                professor = section['instructor']
                if professor and professor != "":
                    if professor not in professor_stats:
                        professor_stats[professor] = {
                            'courses_taught': set(),
                            'total_students': 0,
                            'total_gpa': 0,
                            'section_count': 0,
                            'a_rate_total': 0,
                            'pass_rate_total': 0
                        }
                    
                    prof_data = professor_stats[professor]
                    prof_data['courses_taught'].add(course_code)
                    prof_data['total_students'] += section['Total Grades']
                    prof_data['total_gpa'] += section['gpa']
                    prof_data['section_count'] += 1
                    prof_data['a_rate_total'] += section.get('a_rate', 0)
                    prof_data['pass_rate_total'] += section.get('pass_rate', 0)
        
        if not professor_stats:
            return f"❌ No professors found teaching {subject} courses."
        
        # Calculate average metrics and rank professors
        ranked_professors = []
        
        for professor, stats in professor_stats.items():
            if stats['section_count'] > 0:
                avg_gpa = stats['total_gpa'] / stats['section_count']
                avg_a_rate = stats['a_rate_total'] / stats['section_count']
                avg_pass_rate = stats['pass_rate_total'] / stats['section_count']
                
                # Calculate a comprehensive score
                score = (
                    avg_gpa * 0.3 +           # GPA weight (30%)
                    avg_a_rate * 0.3 +        # A-rate weight (30%)
                    avg_pass_rate * 0.2 +     # Pass rate weight (20%)
                    (stats['total_students'] / 100) * 0.1 +  # Experience weight (10%)
                    len(stats['courses_taught']) * 0.1       # Versatility weight (10%)
                )
                
                # Get professor's overall data
                prof_overall = self.data_processor.df_professors[
                    self.data_processor.df_professors['instructor'] == professor
                ]
                
                teaching_style = "Unknown"
                if not prof_overall.empty:
                    teaching_style = prof_overall.iloc[0].get('teaching_style', 'Unknown')
                
                ranked_professors.append({
                    'name': professor,
                    'avg_gpa': avg_gpa,
                    'avg_a_rate': avg_a_rate,
                    'avg_pass_rate': avg_pass_rate,
                    'courses_count': len(stats['courses_taught']),
                    'courses': list(stats['courses_taught'])[:5],  # Top 5 courses
                    'total_students': stats['total_students'],
                    'section_count': stats['section_count'],
                    'teaching_style': teaching_style,
                    'score': score
                })
        
        # Sort by score (highest first)
        ranked_professors.sort(key=lambda x: x['score'], reverse=True)
        
        # Format response
        if not ranked_professors:
            return f"❌ No professor data available for '{subject}' courses."
        
        response = f"👨‍🏫 **TOP PROFESSORS FOR {subject.upper()} COURSES**\n\n"
        
        for i, prof in enumerate(ranked_professors[:5], 1):
            response += f"**{i}. {prof['name']}**\n"
            response += f"   📊 **Stats:** GPA: {prof['avg_gpa']:.2f} | "
            response += f"A Rate: {prof['avg_a_rate']:.1f}% | "
            response += f"Pass Rate: {prof['avg_pass_rate']:.1f}%\n"
            response += f"   🎯 **Style:** {prof['teaching_style']} | "
            response += f"Students: {prof['total_students']} | "
            response += f"Sections: {prof['section_count']}\n"
            response += f"   📚 **Courses:** {', '.join(prof['courses'])}"
            if prof['courses_count'] > 5:
                response += f" (+{prof['courses_count'] - 5} more)"
            response += "\n\n"
        
        # Add summary
        if ranked_professors:
            best = ranked_professors[0]
            response += f"🏆 **BEST OVERALL:** {best['name']} "
            response += f"(Score: {best['score']:.2f})\n"
            response += f"   • Highest combined GPA, A-rate, and experience for {subject} courses\n"
        
        return response

# ========================
# 9. MAIN APPLICATION
# ========================
class CourseQAAgent:
    """Main Course Q&A Agent application with enhanced course information."""

    def __init__(self, config: AppConfig = None):
        if config is None:
            config = AppConfig()
        self.config = config
        self.logger = ProductionLogger("CourseQAAgent", config.log_level).get_logger()

        # Initialize components
        self.data_processor = DataProcessor(config.data, self.logger)
        self.vector_store = VectorStoreManager(config.data, self.logger)
        self.llm_manager = LLMManager(config.model, self.logger)
        self.query_processor = None

        self.initialized = False

    def initialize(self) -> None:
        """Initialize the complete application."""
        try:
            self.logger.info("Initializing Course Q&A Agent...")

            # Step 1: Load and process data
            self.data_processor.load_and_validate_data()
            chunks = self.data_processor.create_search_chunks()

            # Step 2: Initialize embedding model
            self.vector_store.initialize_embedding_model(self.config.model.embed_model_id)

            # Step 3: Build indices
            for chunk_name, df in chunks.items():
                self.vector_store.build_or_load_index(
                    df, 'search_chunk', chunk_name
                )

            # Step 4: Initialize LLM
            self.llm_manager.initialize_models()

            # Step 5: Initialize query processor
            self.query_processor = QueryProcessor(
                self.data_processor, self.vector_store, self.llm_manager, self.logger
            )

            self.initialized = True
            self.logger.info("Course Q&A Agent initialized successfully!")

        except Exception as e:
            self.logger.error(f"Application initialization failed: {str(e)}")
            raise

    def process_query(self, query: str) -> str:
        """Process a user query and return response with enhanced course information."""
        if not self.initialized:
            return "System not initialized. Please call initialize() first."

        try:
            start_time = time.time()
            self.logger.info(f"Processing query: '{query}'")

            # Detect intent and retrieve context
            intent, params = self.query_processor.detect_intent(query)
            self.logger.info(f"Detected intent: {intent}, params: {params}")

            # Retrieve relevant context based on intent
            context = self.query_processor.retrieve_context(query, intent, params)

            # Check if context retrieval failed
            if "No relevant information" in context or "Error retrieving" in context:
                return "I couldn't find specific information about your query in the course database."

            # ========== HYBRID APPROACH: FACTUAL vs INTERPRETIVE ==========
            
            # FACTUAL QUERIES: Return data chunks directly (avoid hallucination)
            # These are structured lookups where we want 100% accurate facts from database
            factual_intents = [
                'direct_lookup',        # "CSE 5334" → raw course data
                'course_by_professor',  # "CSE 5334 by Dr. Smith" → historical data
                'professor_courses',    # "Courses by Dr. Smith" → professor's course list
                'professor_lookup',     # "Professor Smith" → professor info
                'grade_lookup',         # "Grades for CSE 5334 Spring 2023" → grade distribution
                'course_comparison',    # "Compare CSE 1234 vs CSE 5678" → comparison table
                'course_history',       # "CSE 5334 history" → semester-by-semester data
                'topic_search',         # "Easy courses" → structured course recommendations
                'best_professor',       # "Best professor for ML" → ranked professor list
                'professor_by_subject'  # "Professor for data science" → professor recommendations
            ]
            
            if intent in factual_intents:
                self.logger.info(f"Factual query detected - returning raw data chunks")
                latency = time.time() - start_time
                self.logger.info(f"Query processed in {latency:.2f}s with intent: {intent}")
                return context  # Return structured facts directly - NO hallucination risk
            
            # INTERPRETIVE QUERIES: Use LLM with strong grounding
            # These require reasoning, recommendations, or natural language explanation
            self.logger.info("Interpretive query detected - using LLM with grounding")
            
            # Strengthened prompt to reduce hallucination
            chat_prompt = [
                {
                    "role": "system",
                    "content": """You are a helpful UTA (University of Texas at Arlington) course assistant.

CRITICAL RULES TO PREVENT HALLUCINATION:
1. ONLY use information explicitly provided in the context below
2. If information is NOT in the context, say "I don't have that information in the database"
3. Do NOT make assumptions or infer facts not stated
4. Quote specific numbers (GPAs, percentages, counts) EXACTLY from the context
5. Do NOT add course details, professor names, or statistics not in the context

Your role: Answer conversationally but stay 100% grounded in the provided facts."""
                },
                {
                    "role": "user", 
                    "content": f"""Context from UTA course database:

{context}

Student Question: {query}

Provide a helpful answer using ONLY the information above:"""
                }
            ]
            
            # Apply chat template and generate response
            prompt = self.llm_manager.tokenizer.apply_chat_template(
                chat_prompt, tokenize=False, add_generation_prompt=True
            )
            
            response = self.llm_manager.generate_response(
                prompt, 
                max_new_tokens=200,   # Shorter to keep focused on context
                temperature=0.1       # Very low temperature for factual adherence
            )
            
            latency = time.time() - start_time
            self.logger.info(f"Query processed in {latency:.2f}s with intent: {intent}")
            
            return response

        except Exception as e:
            self.logger.error(f"Query processing failed: {str(e)}")
            return "I apologize, but I encountered an error while processing your query."
# ========================
# 10. COMMAND-LINE INTERFACE
# ========================
def main():
    """Main entry point for the application."""
    # Configuration
    config = AppConfig(
        log_level="INFO",
        cache_size=1000
    )

    # Create and initialize agent
    agent = CourseQAAgent(config)

    try:
        agent.initialize()

        print("\n" + "="*60)
        print("🚀 UTA Course Q&A Agent - Enhanced with Analytics")
        print("   Now with Grade Lookups, Professor Analytics & Comparisons")
        print("   Type 'exit' to quit")
        print("="*60 + "\n")

        # Chat loop
        while True:
            try:
                user_input = input("🧑‍🎓 You: ").strip()

                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\n👋 Thank you for using UTA Course Q&A!")
                    break

                if not user_input:
                    continue

                response = agent.process_query(user_input)
                print(f"\n💬 Agent: {response}\n")

            except KeyboardInterrupt:
                print("\n\n👋 Session ended by user.")
                break
            except Exception as e:
                print(f"\n⚠️  Error: {str(e)}")

    except Exception as e:
        print(f"❌ Failed to initialize application: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())