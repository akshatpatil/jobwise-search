import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(
    page_title="Job Resource Recommender",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    body, .stApp {
        background-color: #000000 !important; /* Main page color set to black */
        color: #D6CFE1 !important;
    }
    .main-header {
        font-size: 2.5rem;
        color: #D6CFE1;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #D6CFE1;
    }
    .resource-card {
        background-color: #232129; /* Very dark background for cards */
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 10px;
        border-left: 5px solid #D6CFE1;
    }
    .resource-title {
        font-weight: bold;
        font-size: 1.2rem;
        color: #D6CFE1;
    }
    .resource-desc {
        font-size: 1rem;
        color: #D6CFE1;
    }
    .resource-meta {
        font-size: 0.8rem;
        color: #D6CFE1;
    }
    .source-tag {
        background-color: #232129;
        border-radius: 15px;
        padding: 5px 10px;
        margin-right: 5px;
        font-size: 0.8rem;
        color: #D6CFE1;
        border: 1px solid #D6CFE1;
    }
    .stButton>button {
        background-color: #D6CFE1;
        color: #232129;
    }
    .stButton>button:hover {
        background-color: #232129;
        color: #D6CFE1;
        border: 1px solid #D6CFE1;
    }
    .stTextArea>div>div>textarea {
        background-color: #232129;
        color: #D6CFE1;
        border: 1px solid #D6CFE1;
    }
    .stSidebar, .css-1d391kg, .css-1lcbmhc {
        background-color: #8c7ba7 !important; /* Darker shade of lilac */
        color: #D6CFE1 !important;
    }
</style>
""", unsafe_allow_html=True)

# Application title
st.markdown("<h1 class='main-header'>Job Skill Resource Finder</h1>", unsafe_allow_html=True)
st.markdown("Find free open-source learning resources tailored to your job requirements")

# Create a database of free educational resources
@st.cache_data
def load_resources():
    resources = [
        {
            "title": "JavaScript Algorithms and Data Structures",
            "description": "Learn JavaScript fundamentals, ES6, regular expressions, debugging, data structures, OOP, functional programming, and algorithmic thinking",
            "skills": "javascript, data structures, algorithms, programming, es6, functional programming, oop",
            "duration": "300 hours",
            "source": "freeCodeCamp",
            "url": "https://www.freecodecamp.org/learn/javascript-algorithms-and-data-structures/"
        },
        {
            "title": "Responsive Web Design Certification",
            "description": "Learn HTML, CSS, visual design, accessibility, and responsive web design principles",
            "skills": "html, css, responsive design, web development, frontend, accessibility",
            "duration": "300 hours",
            "source": "freeCodeCamp",
            "url": "https://www.freecodecamp.org/learn/responsive-web-design/"
        },
        {
            "title": "Scientific Computing with Python",
            "description": "Python fundamentals, Python for data science, and completing scientific computing projects",
            "skills": "python, data science, scientific computing, programming",
            "duration": "300 hours",
            "source": "freeCodeCamp",
            "url": "https://www.freecodecamp.org/learn/scientific-computing-with-python/"
        },
        {
            "title": "Data Analysis with Python",
            "description": "Learn data analysis techniques using NumPy, Pandas, Matplotlib, and Seaborn",
            "skills": "python, data analysis, pandas, numpy, matplotlib, visualization",
            "duration": "300 hours",
            "source": "freeCodeCamp",
            "url": "https://www.freecodecamp.org/learn/data-analysis-with-python/"
        },
        {
            "title": "Information Security",
            "description": "Learn information security with HelmetJS, Python for penetration testing, and security concepts",
            "skills": "security, information security, python, penetration testing, cybersecurity",
            "duration": "300 hours",
            "source": "freeCodeCamp",
            "url": "https://www.freecodecamp.org/learn/information-security/"
        },
        {
            "title": "Machine Learning with Python",
            "description": "Learn TensorFlow and various machine learning algorithms and techniques",
            "skills": "machine learning, python, tensorflow, deep learning, neural networks, ai",
            "duration": "300 hours",
            "source": "freeCodeCamp",
            "url": "https://www.freecodecamp.org/learn/machine-learning-with-python/"
        },
        {
            "title": "Learn Python - Full Course for Beginners",
            "description": "A complete Python tutorial covering all the basics for beginners",
            "skills": "python, programming, beginners, fundamentals",
            "duration": "4 hours",
            "source": "freeCodeCamp YouTube",
            "url": "https://www.youtube.com/watch?v=rfscVS0vtbw"
        },
        {
            "title": "Learn SQL - Full Database Course for Beginners",
            "description": "A comprehensive introduction to SQL and database concepts",
            "skills": "sql, database, data engineering, postgresql, mysql",
            "duration": "4 hours",
            "source": "freeCodeCamp YouTube",
            "url": "https://www.youtube.com/watch?v=HXV3zeQKqGY"
        },
        {
            "title": "React Tutorial for Beginners",
            "description": "Learn the React JavaScript library from the ground up",
            "skills": "react, javascript, frontend, web development",
            "duration": "10 hours",
            "source": "freeCodeCamp YouTube",
            "url": "https://www.youtube.com/watch?v=bMknfKXIFA8"
        },
        {
            "title": "Git and GitHub for Beginners - Crash Course",
            "description": "Learn the basics of Git version control and GitHub",
            "skills": "git, github, version control, collaboration",
            "duration": "1 hour",
            "source": "freeCodeCamp YouTube",
            "url": "https://www.youtube.com/watch?v=RGOj5yH7evk"
        },
        {
            "title": "Docker Tutorial for Beginners",
            "description": "Full Docker course teaching containerization from scratch",
            "skills": "docker, devops, containerization, deployment",
            "duration": "3 hours",
            "source": "freeCodeCamp YouTube",
            "url": "https://www.youtube.com/watch?v=fqMOX6JJhGo"
        },
        {
            "title": "The Rust Programming Language Tutorial",
            "description": "Learn systems programming with Rust",
            "skills": "rust, systems programming, low-level, performance",
            "duration": "3 hours",
            "source": "freeCodeCamp YouTube",
            "url": "https://www.youtube.com/watch?v=MsocPEZBd-M"
        },
        {
            "title": "Object Oriented Programming in Python",
            "description": "Master OOP concepts using Python",
            "skills": "python, oop, object oriented programming, classes, inheritance",
            "duration": "1.5 hours",
            "source": "Programiz",
            "url": "https://www.programiz.com/python-programming/object-oriented-programming"
        },
        {
            "title": "Learn Node.js",
            "description": "Comprehensive guide to server-side JavaScript with Node.js",
            "skills": "nodejs, javascript, backend, server, express, api",
            "duration": "Various",
            "source": "MDN Web Docs",
            "url": "https://developer.mozilla.org/en-US/docs/Learn/Server-side/Node_server_without_framework"
        },
        {
            "title": "React Hooks Tutorial",
            "description": "Modern React development using functional components and hooks",
            "skills": "react, javascript, hooks, frontend, web development",
            "duration": "Various",
            "source": "React Docs",
            "url": "https://reactjs.org/docs/hooks-intro.html"
        },
        {
            "title": "Data Visualization with D3.js",
            "description": "Learn to create interactive data visualizations for the web",
            "skills": "d3js, data visualization, javascript, svg, web development",
            "duration": "Various",
            "source": "Observable",
            "url": "https://observablehq.com/@d3/learn-d3"
        },
        {
            "title": "Learn AWS Serverless",
            "description": "Build serverless applications on AWS",
            "skills": "aws, serverless, lambda, cloud computing, backend",
            "duration": "Various",
            "source": "AWS Workshops",
            "url": "https://aws.amazon.com/getting-started/hands-on/build-serverless-web-app-lambda-apigateway-s3-dynamodb-cognito/"
        },
        {
            "title": "Django Web Framework",
            "description": "Build web applications quickly with Django",
            "skills": "python, django, web development, backend, mvc, databases",
            "duration": "Various",
            "source": "Django Project",
            "url": "https://docs.djangoproject.com/en/stable/intro/tutorial01/"
        },
        {
            "title": "Flutter Mobile App Development",
            "description": "Build cross-platform mobile apps with Flutter",
            "skills": "flutter, dart, mobile development, ui design, cross-platform",
            "duration": "Various",
            "source": "Flutter Dev",
            "url": "https://flutter.dev/docs/get-started/codelab"
        },
        {
            "title": "CSS Grid and Flexbox",
            "description": "Master modern CSS layout techniques",
            "skills": "css, web development, frontend, responsive design, layout",
            "duration": "Various",
            "source": "CSS-Tricks",
            "url": "https://css-tricks.com/snippets/css/complete-guide-grid/"
        },
        {
            "title": "Kubernetes for Beginners",
            "description": "Learn container orchestration with Kubernetes",
            "skills": "kubernetes, docker, devops, orchestration, cloud computing",
            "duration": "4 hours",
            "source": "freeCodeCamp YouTube",
            "url": "https://www.youtube.com/watch?v=d6WC5n9G_sM"
        },
        {
            "title": "MongoDB Tutorial for Beginners",
            "description": "Learn NoSQL database concepts with MongoDB",
            "skills": "mongodb, nosql, database, json, backend",
            "duration": "2 hours",
            "source": "Programming with Mosh",
            "url": "https://www.youtube.com/watch?v=pWbMrx5rVBE"
        },
        {
            "title": "Linux Command Line Basics",
            "description": "Master the Linux terminal and command line interface",
            "skills": "linux, command line, bash, shell scripting, system administration",
            "duration": "3 hours",
            "source": "freeCodeCamp YouTube",
            "url": "https://www.youtube.com/watch?v=ZtqBQ68cfJc"
        },
        {
            "title": "Vue.js Complete Guide",
            "description": "Learn Vue.js framework for building user interfaces",
            "skills": "vuejs, javascript, frontend, web development, spa",
            "duration": "8 hours",
            "source": "Vue Mastery",
            "url": "https://www.vuemastery.com/courses/intro-to-vue-3/intro-to-vue3"
        },
        {
            "title": "Angular Tutorial for Beginners",
            "description": "Complete guide to Angular framework",
            "skills": "angular, typescript, javascript, frontend, web development",
            "duration": "6 hours",
            "source": "Programming with Mosh",
            "url": "https://www.youtube.com/watch?v=k5E2AVpwsko"
        }
    ]
    
    return pd.DataFrame(resources)

# Initialize TF-IDF vectorizer and compute similarity matrix
@st.cache_resource
def setup_similarity_search(_df):
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams for better matching
        max_features=5000    # Limit features for performance
    )
    
    # Create corpus by combining skills and descriptions
    corpus = _df['skills'] + " " + _df['description']
    
    # Fit and transform corpus to TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(corpus)
    
    return tfidf, tfidf_matrix

# Function to extract skills from job description
def extract_skills(job_description):
    common_skills = [
        "python", "javascript", "java", "c++", "ruby", "php", "html", "css",
        "react", "angular", "vue", "node", "express", "django", "flask",
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "sql", "mysql", "postgresql", "mongodb", "nosql", "database",
        "machine learning", "ai", "data science", "nlp", "computer vision",
        "git", "devops", "ci/cd", "jenkins", "github actions", "gitlab",
        "agile", "scrum", "kanban", "project management", "jira",
        "rest api", "graphql", "microservices", "serverless",
        "linux", "unix", "bash", "shell scripting",
        "frontend", "backend", "fullstack", "web development",
        "mobile development", "ios", "android", "flutter", "react native",
        "testing", "qa", "selenium", "cypress", "junit", "pytest",
        "security", "cybersecurity", "penetration testing",
        "data analysis", "data visualization", "tableau", "power bi",
        "blockchain", "ethereum", "smart contracts",
        "ux", "ui", "design", "figma", "adobe xd", "sketch",
        "typescript", "golang", "kotlin", "swift", "dart", "rust",
        "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn",
        "tensorflow", "pytorch", "keras", "opencv"
    ]
    
    job_description = job_description.lower()
    found_skills = []
    
    for skill in common_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', job_description):
            found_skills.append(skill)
    
    return found_skills

# Function to search for relevant resources using cosine similarity
def search_resources(job_description, tfidf, tfidf_matrix, df, k=10):
    # Extract key skills
    extracted_skills = extract_skills(job_description)
    skills_text = " ".join(extracted_skills)
    
    # Combine skills with job description for better matching
    search_text = job_description.lower() + " " + skills_text
    
    # Transform search text using the same TF-IDF vectorizer
    search_vector = tfidf.transform([search_text])
    
    # Calculate cosine similarity between search vector and all resources
    similarities = cosine_similarity(search_vector, tfidf_matrix).flatten()
    
    # Get indices of top k most similar resources
    top_indices = similarities.argsort()[-k:][::-1]
    
    # Filter out results with very low similarity (threshold = 0.1)
    filtered_indices = [idx for idx in top_indices if similarities[idx] > 0.1]
    
    if not filtered_indices:
        # If no resources meet the threshold, return top 3 anyway
        filtered_indices = top_indices[:3]
    
    # Get the matching resources
    results = df.iloc[filtered_indices].copy()
    results['score'] = [similarities[idx] * 100 for idx in filtered_indices]
    
    return results, extracted_skills

# Function to get personalized learning path
def get_learning_path(extracted_skills, df):
    """Generate a suggested learning path based on extracted skills"""
    
    # Define skill progression paths
    skill_paths = {
        'beginner_python': ['python', 'programming', 'fundamentals'],
        'data_science': ['python', 'data analysis', 'machine learning', 'data visualization'],
        'web_dev_frontend': ['html', 'css', 'javascript', 'react'],
        'web_dev_backend': ['node', 'python', 'django', 'database', 'api'],
        'devops': ['linux', 'docker', 'kubernetes', 'aws', 'ci/cd'],
        'mobile_dev': ['java', 'kotlin', 'swift', 'flutter', 'react native']
    }
    
    # Determine which path(s) match the extracted skills
    matching_paths = []
    for path_name, path_skills in skill_paths.items():
        overlap = set(extracted_skills) & set(path_skills)
        if overlap:
            matching_paths.append((path_name, len(overlap), path_skills))
    
    # Sort by overlap size
    matching_paths.sort(key=lambda x: x[1], reverse=True)
    
    return matching_paths[:2] if matching_paths else []

# Main app
def main():
    # Load resources
    df = load_resources()
    
    # Setup TF-IDF and similarity search
    tfidf, tfidf_matrix = setup_similarity_search(df)
    
    # Sidebar for input
    st.sidebar.markdown("## Input Job Details")
    
    # Job description input
    job_description = st.sidebar.text_area(
        "Paste Job Description or Skills Required",
        """
We are looking for a skilled Python Developer to join our data science team. 
The ideal candidate should have expertise in Python programming, data analysis, and machine learning. 
Experience with pandas, numpy, scikit-learn, and TensorFlow is required.
Knowledge of SQL and database concepts is a must.
        """,
        height=300
    )
    
    # Number of recommendations slider
    num_recommendations = st.sidebar.slider(
        "Number of Recommendations",
        min_value=3,
        max_value=15,
        value=8,
        step=1
    )
    
    # Additional filters
    st.sidebar.markdown("## Additional Filters")
    sources = list(df['source'].unique())
    selected_sources = st.sidebar.multiselect(
        "Filter by Source",
        sources,
        default=sources
    )
    
    # Duration filter
    duration_filter = st.sidebar.selectbox(
        "Preferred Duration",
        ["All Durations", "Short (< 5 hours)", "Medium (5-50 hours)", "Long (> 50 hours)"]
    )
    
    # Search button
    search_clicked = st.sidebar.button("Find Resources", type="primary")
    
    # About section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        """
        This app uses TF-IDF vectorization and cosine similarity to find relevant learning 
        resources based on job descriptions. It extracts key skills from your job 
        description and matches them with free and open-source educational content.
        
        **New Features:**
        - Skill-based learning paths
        - Duration filtering
        - Enhanced skill extraction
        """
    )
    
    # Main content
    if search_clicked and job_description:
        # Search for relevant resources
        with st.spinner("Analyzing job description and finding resources..."):
            results, extracted_skills = search_resources(
                job_description, tfidf, tfidf_matrix, df, k=num_recommendations
            )
        
        # Filter by selected sources
        if selected_sources:
            results = results[results['source'].isin(selected_sources)]
        
        # Apply duration filter
        if duration_filter != "All Durations":
            if duration_filter == "Short (< 5 hours)":
                results = results[~results['duration'].str.contains('300 hours')]
            elif duration_filter == "Medium (5-50 hours)":
                results = results[results['duration'].str.contains('hours') & 
                                ~results['duration'].str.contains('300 hours')]
            elif duration_filter == "Long (> 50 hours)":
                results = results[results['duration'].str.contains('300 hours')]
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display extracted skills
            st.markdown("### 🎯 Extracted Skills")
            if extracted_skills:
                skill_html = ' '.join([
                    f'<span class="source-tag" style="margin-bottom:8px; display:inline-block;">{skill}</span>'
                    for skill in extracted_skills
                ])
                st.markdown(f"<div style='margin-bottom: 16px;'>{skill_html}</div>", unsafe_allow_html=True)
            else:
                st.info("No specific skills detected. Showing general programming resources.")
            
            # Display results
            st.markdown("### 📚 Recommended Learning Resources")
            
            if len(results) > 0:
                for idx, (_, resource) in enumerate(results.iterrows(), 1):
                    with st.container():
                        # Add ranking number
                        ranking_color = "#4CAF50" if idx <= 3 else "#FF9800" if idx <= 6 else "#757575"
                        st.markdown(f"""
                        <div class="resource-card">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                <div style="flex-grow: 1;">
                                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                        <span style="background-color: {ranking_color}; color: white; border-radius: 50%; width: 25px; height: 25px; display: inline-flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 10px; font-size: 0.9rem;">
                                            {idx}
                                        </span>
                                        <div class="resource-title">{resource['title']}</div>
                                    </div>
                                    <div class="resource-desc">{resource['description']}</div>
                                    <div class="resource-meta" style="margin-top: 10px;">
                                        <span><b>Source:</b> {resource['source']}</span> | 
                                        <span><b>Duration:</b> {resource['duration']}</span> | 
                                        <span><b>Match Score:</b> {resource['score']:.1f}%</span>
                                    </div>
                                </div>
                            </div>
                            <div style="margin-top: 15px;">
                                <a href="{resource['url']}" target="_blank" style="background-color: #D6CFE1; color: #232129; padding: 8px 16px; border-radius: 5px; text-decoration: none; display: inline-block;">
                                    🚀 Start Learning
                                </a>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No matching resources found. Try adjusting your search filters or using different keywords.")
        
        with col2:
            # Learning path suggestions
            st.markdown("### 🛤️ Suggested Learning Paths")
            learning_paths = get_learning_path(extracted_skills, df)
            
            if learning_paths:
                for path_name, overlap_count, path_skills in learning_paths:
                    path_display_name = path_name.replace('_', ' ').title()
                    with st.expander(f"{path_display_name} ({overlap_count} matches)"):
                        st.write("**Recommended skill progression:**")
                        for skill in path_skills:
                            emoji = "✅" if skill in extracted_skills else "⏳"
                            st.write(f"{emoji} {skill.title()}")
            else:
                st.info("Enter more specific skills to get personalized learning path recommendations.")
            
            # Quick stats
            st.markdown("### 📊 Quick Stats")
            if len(results) > 0:
                avg_score = results['score'].mean()
                top_source = results['source'].value_counts().index[0]
                
                st.metric("Average Match Score", f"{avg_score:.1f}%")
                st.metric("Top Resource Source", top_source)
                st.metric("Resources Found", len(results))
        
        # Feature to suggest missing resources
        st.markdown("---")
        st.markdown("### 💡 Can't Find What You Need?")
        col1, col2 = st.columns([3, 1])
        with col1:
            missing_resource = st.text_input(
                "Suggest a resource to add:", 
                placeholder="e.g., Advanced React Patterns, Docker Swarm Tutorial..."
            )
        with col2:
            if st.button("Submit Suggestion"):
                if missing_resource:
                    st.success("Thank you for your suggestion! We'll consider adding it to our database.")
                    # Here you could save suggestions to a file or database
                else:
                    st.warning("Please enter a resource suggestion first.")
    
    else:
        # Display welcome message and instructions
        st.markdown("""
        <h3 style="text-align: center; color: #D6CFE1;">🚀 Welcome to the Job Skill Resource Finder</h3>
        
        <div style="background-color: #232129; padding: 20px; border-radius: 10px; margin: 20px 0;">
        This tool helps you find <strong>free, open-source learning resources</strong> that match the skills required in your job description.
        
        <h4>✨ How to use:</h4>
        1. 📝 Paste a job description or list of required skills in the sidebar
        2. 🔧 Optionally adjust filters (sources, duration, number of results)
        3. 🔍 Click "Find Resources" to get personalized recommendations
        4. 📈 Review your suggested learning path
        
        <h4>🌟 Features:</h4>
        • Automatically extracts key skills from job descriptions<br>
        • Uses advanced text similarity for accurate matching<br>
        • Focuses on free and open-source content<br>
        • Provides match scores and learning paths<br>
        • Supports 25+ programming skills and technologies
        </div>
        """, unsafe_allow_html=True)
        
        # Sample job roles for quick selection
        st.markdown("### 🎯 Try these sample job roles:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔬 Data Scientist", use_container_width=True):
                st.session_state.job_description = """
                Data Scientist position requiring expertise in Python, machine learning algorithms, 
                statistical analysis, deep learning, and data visualization. Experience with TensorFlow, 
                PyTorch, scikit-learn, and data manipulation using Pandas and NumPy is essential. 
                SQL skills and experience with cloud platforms (AWS/GCP) required.
                """
                
        with col2:
            if st.button("🌐 Full-Stack Developer", use_container_width=True):
                st.session_state.job_description = """
                Full-Stack Developer needed with strong JavaScript skills. Experience with React, 
                Node.js, Express, and MongoDB required. Knowledge of HTML/CSS, RESTful APIs, and 
                version control systems like Git is essential. TypeScript and cloud deployment 
                experience (AWS/Heroku) is a plus.
                """
                
        with col3:
            if st.button("⚙️ DevOps Engineer", use_container_width=True):
                st.session_state.job_description = """
                DevOps Engineer with expertise in CI/CD pipelines, Docker, Kubernetes, and cloud platforms 
                (AWS/Azure). Experience with infrastructure as code using Terraform or CloudFormation required. 
                Scripting skills in Python or Bash essential. Knowledge of monitoring and logging solutions,
                Jenkins, and GitLab CI/CD is highly valued.
                """

        # Display some featured resources
        st.markdown("### 🌟 Featured Free Learning Resources")
        featured_df = df.sample(n=4).reset_index(drop=True)
        
        cols = st.columns(2)
        for idx, (_, resource) in enumerate(featured_df.iterrows()):
            with cols[idx % 2]:
                st.markdown(f"""
                <div class="resource-card" style="height: 200px;">
                    <div class="resource-title">{resource['title']}</div>
                    <div class="resource-desc" style="font-size: 0.9rem; margin: 10px 0; overflow: hidden; text-overflow: ellipsis;">
                        {resource['description'][:100]}...
                    </div>
                    <div class="resource-meta">
                        <span><b>Source:</b> {resource['source']}</span><br>
                        <span><b>Duration:</b> {resource['duration']}</span>
                    </div>
                    <div style="margin-top: 10px;">
                        <a href="{resource['url']}" target="_blank" style="color: #D6CFE1;">
                            🔗 Explore Resource
                        </a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Add some statistics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Resources", len(df))
        with col2:
            st.metric("Resource Sources", df['source'].nunique())
        with col3:
            st.metric("Skill Categories", "25+")
        with col4:
            st.metric("Free Content", "100%")

# Run the app
if __name__ == "__main__":
    main()