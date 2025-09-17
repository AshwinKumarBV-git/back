from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import fitz  # PyMuPDF
import numpy as np
import faiss
from typing import Optional, List
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, auth, firestore, storage
from dotenv import load_dotenv
import tempfile
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import json
import time

# Load environment variables
load_dotenv()

# Initialize Firebase Admin
cred = credentials.Certificate("/etc/secrets/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET', 'eduprep-ai.appspot.com')
})

# Initialize Firestore
db = firestore.client()

# Initialize Firebase Storage
bucket = storage.bucket()

# Initialize Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')
rag_model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize embedding model for RAG
embedding_model = "models/text-embedding-004"

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Flutter app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LessonPlanRequest(BaseModel):
    syllabus: str
    grade: str
    subject: str
    token: str

class LessonPlanResponse(BaseModel):
    plan: str
    created_at: datetime

class SummaryResponse(BaseModel):
    title: str
    summary: str
    notes: str
    original_file_url: str
    content_type: str
    created_at: datetime

# Quiz generation models
class QuizOption(BaseModel):
    id: str  # A, B, C, D
    text: str

class QuizQuestion(BaseModel):
    question: str
    options: List[QuizOption]
    correctAnswer: str  # A, B, C, D
    bloomTag: str  # Knowledge, Understand, Apply, Analyze, Evaluate, Create

class QuizGenerationRequest(BaseModel):
    content: str
    numQuestions: int
    token: str
    title: str

class QuizGenerationResponse(BaseModel):
    title: str
    questions: List[QuizQuestion]
    created_at: datetime

# Speech-to-Plan Models
class SpeechToPlanRequest(BaseModel):
    transcript: str
    token: str

class SpeechToPlanResponse(BaseModel):
    plan: str
    generated_at: datetime

# Lesson Simulator Models
class LessonSimulationRequest(BaseModel):
    lesson_plan: str
    teacher_ideas: str
    student_age: int
    class_size: int
    subject_complexity: str # e.g., "beginner", "intermediate", "advanced"
    transcript: Optional[str] = None
    token: str

class SimulationFeedbackData(BaseModel):
    student_reactions: List[str]
    questions: List[str]
    suggestions: List[str]
    problem_areas: List[str]
    tone_feedback: Optional[str] = None
    improvement_tips: List[str]
    timestamp: datetime

# PDF related models
class QuizPdfRequest(BaseModel):
    file: Optional[UploadFile] = None
    youtube_url: Optional[str] = None
    numQuestions: int
    token: str
    title: Optional[str] = None

# RAG-related classes
class DocumentChunk:
    """Represents a chunk of document with metadata"""
    def __init__(self, text: str, chunk_id: int, source: str = "", start_pos: int = 0):
        self.text = text
        self.chunk_id = chunk_id
        self.source = source
        self.start_pos = start_pos
        self.embedding = None
        self.metadata = {}

class RAGProcessor:
    """Complete RAG implementation with proper embeddings"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_retries = 3
        self.retry_delay = 1.0
    
    def chunk_text_with_metadata(self, text: str, source: str = "") -> List[DocumentChunk]:
        """Split text into overlapping chunks with metadata"""
        chunks = []
        chunk_id = 0
        
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk_text = text[i:i + self.chunk_size].strip()
            
            if len(chunk_text) < 100:  # Skip very small chunks
                continue
                
            # Clean and preprocess chunk
            chunk_text = self.preprocess_chunk(chunk_text)
            
            chunk = DocumentChunk(
                text=chunk_text,
                chunk_id=chunk_id,
                source=source,
                start_pos=i
            )
            
            # Add metadata
            chunk.metadata = {
                'length': len(chunk_text),
                'word_count': len(chunk_text.split()),
                'has_numbers': any(char.isdigit() for char in chunk_text),
                'has_questions': '?' in chunk_text,
                'sentence_count': chunk_text.count('.') + chunk_text.count('!') + chunk_text.count('?')
            }
            
            chunks.append(chunk)
            chunk_id += 1
        
        return chunks
    
    def preprocess_chunk(self, text: str) -> str:
        """Clean and preprocess chunk text"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove page numbers and common PDF artifacts
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip lines that are likely page numbers or headers/footers
            if (len(line) < 3 or 
                line.isdigit() or 
                (line.count(' ') == 0 and len(line) < 20)):
                continue
            cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines)
    
    async def create_embeddings_batch(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Create embeddings for chunks using Gemini embedding API with batching"""
        embeddings_list = []
        
        # Process chunks in batches to handle rate limits
        batch_size = 10  # Gemini API batch limit
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = [chunk.text for chunk in batch_chunks]
            
            # Retry logic for API calls
            for attempt in range(self.max_retries):
                try:
                    # Create embeddings for batch
                    response = genai.embed_content(
                        model=embedding_model,
                        content=batch_texts,
                        task_type="retrieval_document"
                    )
                    
                    # Extract embeddings
                    batch_embeddings = []
                    if isinstance(response['embedding'], list) and isinstance(response['embedding'][0], list):
                        # Multiple embeddings returned
                        batch_embeddings = response['embedding']
                    else:
                        # Single embedding returned, wrap in list
                        batch_embeddings = [response['embedding']]
                    
                    # Store embeddings in chunks
                    for j, embedding in enumerate(batch_embeddings):
                        if i + j < len(chunks):
                            chunks[i + j].embedding = np.array(embedding, dtype=np.float32)
                            embeddings_list.append(embedding)
                    
                    break  # Success, break retry loop
                    
                except Exception as e:
                    print(f"Embedding API error (attempt {attempt + 1}): {str(e)}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        # Fallback: create random embedding for this batch
                        print(f"Using fallback embeddings for batch {i//batch_size + 1}")
                        for j in range(len(batch_chunks)):
                            fallback_embedding = np.random.rand(768).astype('float32')
                            chunks[i + j].embedding = fallback_embedding
                            embeddings_list.append(fallback_embedding.tolist())
        
        return np.array(embeddings_list, dtype=np.float32)
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index for similarity search"""
        dimension = embeddings.shape[1]
        
        # Use IndexIVFFlat for better performance on larger datasets
        if len(embeddings) > 100:
            # Create index with clustering
            quantizer = faiss.IndexFlatL2(dimension)
            nlist = min(100, max(1, len(embeddings) // 10))  # Number of clusters
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            # Train the index
            index.train(embeddings)
            index.add(embeddings)
        else:
            # Use simple flat index for small datasets
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
        
        return index
    
    async def create_query_embedding(self, query: str) -> np.ndarray:
        """Create embedding for query text"""
        for attempt in range(self.max_retries):
            try:
                response = genai.embed_content(
                    model=embedding_model,
                    content=query,
                    task_type="retrieval_query"
                )
                return np.array(response['embedding'], dtype=np.float32).reshape(1, -1)
                
            except Exception as e:
                print(f"Query embedding error (attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    # Fallback: return random embedding
                    return np.random.rand(1, 768).astype('float32')
    
    def retrieve_relevant_chunks(self, query_embedding: np.ndarray, index: faiss.Index, 
                               chunks: List[DocumentChunk], k: int = 5) -> List[DocumentChunk]:
        """Retrieve most relevant chunks using similarity search"""
        # Search for similar embeddings
        distances, indices = index.search(query_embedding, min(k, len(chunks)))
        
        # Return relevant chunks sorted by relevance
        relevant_chunks = []
        for idx in indices[0]:
            if idx < len(chunks):
                chunk = chunks[idx]
                relevant_chunks.append(chunk)
        
        return relevant_chunks
    
    def rerank_chunks(self, chunks: List[DocumentChunk], query: str) -> List[DocumentChunk]:
        """Rerank chunks based on additional heuristics"""
        def calculate_score(chunk: DocumentChunk) -> float:
            text = chunk.text.lower()
            query_lower = query.lower()
            
            # Basic keyword matching score
            query_words = set(query_lower.split())
            chunk_words = set(text.split())
            keyword_overlap = len(query_words.intersection(chunk_words))
            
            # Position score (earlier chunks might be more important)
            position_score = 1.0 / (chunk.chunk_id + 1)
            
            # Length score (prefer chunks with reasonable length)
            length_score = min(1.0, len(text) / 500)
            
            # Metadata-based scoring
            metadata_score = 0.0
            if chunk.metadata.get('has_questions', False) and '?' in query:
                metadata_score += 0.2
            if chunk.metadata.get('sentence_count', 0) > 3:
                metadata_score += 0.1
            
            return keyword_overlap + 0.1 * position_score + 0.1 * length_score + metadata_score
        
        # Sort by combined score
        chunks_with_scores = [(chunk, calculate_score(chunk)) for chunk in chunks]
        chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [chunk for chunk, score in chunks_with_scores]

# Global RAG processor instance
rag_processor = RAGProcessor()

async def verify_token(token: str):
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token['uid']
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

@app.post("/api/generate-lesson-plan", response_model=LessonPlanResponse)
async def generate_lesson_plan(request: LessonPlanRequest):
    # Verify Firebase token
    user_id = await verify_token(request.token)
    
    try:
        # Generate lesson plan using Gemini
        prompt = f"""Generate a detailed lesson plan with objectives, activities, and materials for: {request.subject} ({request.grade}) on the topic: {request.syllabus}

Please include:
1. Learning Objectives
2. Required Materials
3. Lesson Structure (with time allocations)
4. Teaching Methods
5. Student Activities
6. Assessment Methods
7. Differentiation Strategies
8. Homework/Extension Activities

Format the response in markdown."""

        response = model.generate_content(prompt)
        generated_plan = response.text
        created_at = datetime.now()

        # Save to Firestore
        doc_ref = db.collection(f'users/{user_id}/lessonPlans').document()
        doc_ref.set({
            'syllabus': request.syllabus,
            'grade': request.grade,
            'subject': request.subject,
            'plan': generated_plan,
            'createdAt': created_at.isoformat(),
            'userId': user_id
        })

        return LessonPlanResponse(
            plan=generated_plan,
            created_at=created_at
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#simple loadtest endpoint

@app.get("/")
async def health_check():
    return {"status": "ok"}


@app.post("/api/v1/speech-to-plan", response_model=SpeechToPlanResponse)
async def generate_speech_to_plan(request: SpeechToPlanRequest):
    uid = await verify_token(request.token) # Verify token from request body
    try:
        prompt = f"""
You are an expert instructional designer. A teacher has provided the following idea or topic for a lesson through a voice transcript:
'{request.transcript}'

Based on this, generate a comprehensive and structured lesson plan suitable for a classroom setting.
If the transcript mentions a specific grade or subject, use that. Otherwise, try to infer it or make a general assumption (e.g., middle school general topic).

The lesson plan should include, but not be limited to:
1.  **Lesson Title:** (A concise and descriptive title)
2.  **Learning Objectives:** (What students will know or be able to do - clear, measurable objectives)
3.  **Target Audience/Grade Level:** (Specify or infer)
4.  **Materials & Resources:** (List of necessary items, including any digital tools)
5.  **Lesson Activities & Procedure:** (A step-by-step breakdown, including approximate timings for each section: Introduction, Main Activities, Conclusion)
6.  **Differentiated Instruction:** (Suggestions for supporting diverse learners, e.g., for struggling students and for advanced learners)
7.  **Assessment:** (How learning will be checked, e.g., questions to ask, a short activity, exit ticket)
8.  **Estimated Total Duration:** (Approximate total time for the lesson)

Format the entire response in Markdown.
Ensure the plan is practical and engaging.
If the transcript is too short, unclear, or lacks sufficient detail to create a meaningful lesson plan, please state that you need more information and suggest what kind of details would be helpful. Do not attempt to generate a plan from insufficient input.
"""

        gemini_response = model.generate_content(prompt)
        generated_plan = gemini_response.text
        generated_at = datetime.now()

        # Optional: You could also save this generated plan to Firestore here if desired,
        # similar to the other lesson plan endpoint, associating it with the user (uid).
        # For now, just returning it as per the immediate requirement.

        return SpeechToPlanResponse(
            plan=generated_plan,
            generated_at=generated_at
        )
    except Exception as e:
        print(f"Error in generate_speech_to_plan: {e}") # Log the error server-side
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the lesson plan: {str(e)}")

@app.post("/api/v1/simulate-lesson", response_model=SimulationFeedbackData)
async def simulate_lesson_endpoint(request: LessonSimulationRequest):
    user_id = await verify_token(request.token)
    current_time = datetime.now()

    # Prepare input for Firestore (excluding the token)
    input_data_for_firestore = {
        "lesson_plan": request.lesson_plan,
        "teacher_ideas": request.teacher_ideas,
        "student_age": request.student_age,
        "class_size": request.class_size,
        "subject_complexity": request.subject_complexity,
        "transcript": request.transcript,
    }

    # Construct the prompt for Gemini
    # This prompt needs to be carefully engineered to request JSON output.
    prompt_parts = [
        f"You are an AI simulating a classroom to evaluate a lesson plan.",
        f"The lesson plan is: '{request.lesson_plan}'.",
        f"Additional teacher ideas/comments for delivery: '{request.teacher_ideas}'.",
        f"The target students are {request.student_age} years old.",
        f"The class size is {request.class_size} students.",
        f"The subject complexity is '{request.subject_complexity}'.",
    ]
    if request.transcript:
        prompt_parts.append(f"The teacher's speech transcript for part of the lesson is: '{request.transcript}'. Analyze this for pacing and clarity.")
    
    prompt_parts.extend([
        f"Based on this, simulate student interactions and provide feedback.",
        f"Please format your entire response as a single JSON object with the following keys:",
        f"  'student_reactions': A list of 3-5 diverse, typical student reactions or comments (e.g., 'This is engaging!', 'I'm a bit confused about X', 'Can we do an example?').",
        f"  'questions': A list of 2-3 pertinent questions students might ask based on the plan.",
        f"  'suggestions': A list of 2-3 actionable suggestions for improving the lesson's engagement, clarity, or structure.",
        f"  'problem_areas': A list of 1-2 potential problem areas or parts of the lesson where students might struggle.",
        f"  'tone_feedback': If a transcript was provided, a brief comment on the perceived tone, pacing, or clarity from the transcript (e.g., 'The explanation of the core concept seemed a bit fast.'). Otherwise, null.",
        f"  'improvement_tips': A list of 2-3 general teaching improvement tips relevant to the lesson plan.",
        f"Example of a student_reaction: 'The activity for XYZ seems fun!'",
        f"Example of a question: 'What's the difference between A and B again?'",
        f"Ensure all lists contain strings. Do not include any explanatory text outside of the JSON object itself."
    ])
    final_prompt = "\n".join(prompt_parts)

    try:
        gemini_response = model.generate_content(final_prompt)
        
        # Debug: Print raw Gemini response
        print(f"Raw Gemini Response for simulation: {gemini_response.text}")

        # Attempt to parse the response as JSON
        # Gemini might sometimes include markdown backticks around the JSON, try to remove them.
        cleaned_response_text = gemini_response.text.strip()
        if cleaned_response_text.startswith('```json'):
            cleaned_response_text = cleaned_response_text[7:]
        if cleaned_response_text.endswith('```'):
            cleaned_response_text = cleaned_response_text[:-3]
        
        feedback_json = json.loads(cleaned_response_text.strip())

        # Validate and structure the feedback using Pydantic model
        # The timestamp for the feedback itself is when it's processed by our server
        feedback_data = SimulationFeedbackData(
            student_reactions=feedback_json.get('student_reactions', []),
            questions=feedback_json.get('questions', []),
            suggestions=feedback_json.get('suggestions', []),
            problem_areas=feedback_json.get('problem_areas', []),
            tone_feedback=feedback_json.get('tone_feedback'),
            improvement_tips=feedback_json.get('improvement_tips', []),
            timestamp=current_time
        )

        # Save to Firestore
        simulation_doc_ref = db.collection('simulations').document(user_id).collection('user_simulations').document()
        simulation_doc_ref.set({
            'input_data': input_data_for_firestore,
            'feedback': feedback_data.model_dump(), # Use model_dump() for Pydantic v2
            'userId': user_id,
            'timestamp': current_time # Top-level timestamp for querying
        })

        return feedback_data

    except json.JSONDecodeError as e:
        print(f"JSONDecodeError parsing Gemini response: {e}")
        print(f"Problematic Gemini response text: {gemini_response.text}")
        raise HTTPException(status_code=500, detail="Error processing AI model's response: Invalid JSON format.")
    except Exception as e:
        print(f"Error in simulate_lesson_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during lesson simulation: {str(e)}")

# Helper functions for RAG (Updated with complete implementation)
def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def get_youtube_transcript(video_id):
    """Get transcript from YouTube video"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry["text"] for entry in transcript])
        return text
    except Exception as e:
        return None

def extract_youtube_id(url):
    """Extract YouTube video ID from URL"""
    try:
        if "youtu.be" in url:
            return url.split("/")[-1].split("?")[0]
        elif "youtube.com" in url:
            return url.split("v=")[1].split("&")[0]
    except:
        return None

async def enhanced_rag_generate_summary(text: str, title: str) -> str:
    """Generate summary using complete RAG implementation"""
    try:
        # Create chunks with metadata
        chunks = rag_processor.chunk_text_with_metadata(text, source=title)
        
        if not chunks:
            return f"""# Summary

Unable to process the document titled "{title}" due to insufficient content.

## Teaching Notes

- Please ensure the document contains readable text
- Check if the document is not corrupted or password-protected
- Try uploading a different format if available
"""
        
        # Create embeddings for all chunks
        embeddings = await rag_processor.create_embeddings_batch(chunks)
        
        # Build FAISS index
        index = rag_processor.build_faiss_index(embeddings)
        
        # Define queries for different types of content retrieval
        summary_queries = [
            "main concepts key ideas overview introduction",
            "important definitions terminology concepts",
            "conclusions summary findings results"
        ]
        
        teaching_queries = [
            "teaching methods pedagogy educational approaches",
            "student activities exercises practice problems",
            "assessment evaluation testing questions",
            "examples demonstrations illustrations case studies"
        ]
        
        # Retrieve relevant chunks for summary
        summary_chunks = []
        for query in summary_queries:
            query_embedding = await rag_processor.create_query_embedding(query)
            relevant_chunks = rag_processor.retrieve_relevant_chunks(
                query_embedding, index, chunks, k=3
            )
            summary_chunks.extend(relevant_chunks)
        
        # Retrieve relevant chunks for teaching notes
        teaching_chunks = []
        for query in teaching_queries:
            query_embedding = await rag_processor.create_query_embedding(query)
            relevant_chunks = rag_processor.retrieve_relevant_chunks(
                query_embedding, index, chunks, k=3
            )
            teaching_chunks.extend(relevant_chunks)
        
        # Remove duplicates while preserving order
        seen_chunk_ids = set()
        unique_summary_chunks = []
        for chunk in summary_chunks:
            if chunk.chunk_id not in seen_chunk_ids:
                unique_summary_chunks.append(chunk)
                seen_chunk_ids.add(chunk.chunk_id)
        
        seen_chunk_ids = set()
        unique_teaching_chunks = []
        for chunk in teaching_chunks:
            if chunk.chunk_id not in seen_chunk_ids:
                unique_teaching_chunks.append(chunk)
                seen_chunk_ids.add(chunk.chunk_id)
        
        # Rerank chunks for better relevance
        unique_summary_chunks = rag_processor.rerank_chunks(
            unique_summary_chunks, "summary main concepts overview"
        )
        unique_teaching_chunks = rag_processor.rerank_chunks(
            unique_teaching_chunks, "teaching methods student activities"
        )
        
        # Combine top chunks for context (limit to avoid token limits)
        summary_context = "\n\n".join([
            chunk.text for chunk in unique_summary_chunks[:4]
        ])
        
        teaching_context = "\n\n".join([
            chunk.text for chunk in unique_teaching_chunks[:4]
        ])
        
        # Generate enhanced summary with retrieved context
        prompt = f"""You are an educational content summarizer for teachers. Generate high-quality structured content for a document titled '{title}' using the retrieved relevant content below.

Your task is to create TWO CLEARLY SEPARATED SECTIONS:

# Summary

Provide a concise yet comprehensive summary of the main concepts and ideas based on the following content. Focus on the most important concepts, key findings, and overall message. This should be approximately 3-5 paragraphs.

## Teaching Notes

Create detailed teaching notes including:
- Key concepts and definitions extracted from the content
- Important facts, figures, or data points
- Suggested teaching approaches based on the material
- Student activities or discussion questions you can derive
- Assessment ideas based on the content
- Additional insights for classroom use

Format with proper markdown (headings, bullet points, etc.).

CONTENT FOR SUMMARY GENERATION:
{summary_context}

CONTENT FOR TEACHING NOTES:
{teaching_context}

ADDITIONAL CONTEXT (if different from above):
{text[:2000] if len(text) > len(summary_context + teaching_context) else "No additional context needed."}

FORMAT YOUR RESPONSE EXACTLY WITH THE TWO CLEARLY SEPARATED SECTIONS AS REQUESTED.
"""
        
        response = rag_model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"Enhanced RAG error: {str(e)}")
        # Fallback to simple processing
        return await simple_fallback_summary(text, title)

async def simple_fallback_summary(text: str, title: str) -> str:
    """Fallback summary generation without RAG"""
    # Simple text truncation for context
    context = text[:3000] if len(text) > 3000 else text
    
    prompt = f"""Generate a summary and teaching notes for "{title}".

# Summary

Provide a 3-4 paragraph summary of the main concepts.

## Teaching Notes

Include key points, definitions, and teaching suggestions.

Content: {context}
"""
    
    try:
        response = rag_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"""# Summary

Error generating summary for "{title}": {str(e)}

## Teaching Notes

- Manual review recommended
- Technical error occurred during processing
"""

@app.post("/api/summarize-resource", response_model=SummaryResponse)
async def summarize_resource(
    file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
    token: str = Form(...),
    title: Optional[str] = Form(None)
):
    # Verify Firebase token
    user_id = await verify_token(token)
    
    try:
        content_type = ""
        file_path = None
        file_url = ""
        extracted_text = ""
        
        # Process file upload or YouTube URL
        if file:
            # Save uploaded file to temporary location
            content_type = "pdf" if file.filename.lower().endswith('.pdf') else "video"
            
            # Create temp file with proper extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{content_type}") as temp_file:
                file_path = temp_file.name
            
            # Write uploaded content to temp file
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Generate unique filename for storage
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            storage_filename = f"{timestamp}_{file.filename}"
            
            # Upload to Firebase Storage
            try:
                blob = bucket.blob(f"users/{user_id}/{storage_filename}")
                blob.upload_from_filename(file_path)
                blob.make_public()
                file_url = blob.public_url
            except Exception as e:
                # If storage upload fails, still try to process the file
                print(f"Firebase storage upload error: {str(e)}")
                file_url = f"local://{file.filename}"
            
            # Extract text based on content type
            if content_type == "pdf":
                try:
                    extracted_text = extract_text_from_pdf(file_path)
                except Exception as e:
                    raise HTTPException(
                        status_code=422, 
                        detail=f"Could not extract text from PDF: {str(e)}"
                    )
            else:
                # For video, we'd need a different approach (not implemented)
                extracted_text = "Video text extraction not implemented yet."
                
            # Clean up temporary file
            try:
                os.unlink(file_path)
            except:
                pass
            
            # Use filename as title if not provided
            if not title:
                title = file.filename
            
        elif youtube_url:
            content_type = "video"
            video_id = extract_youtube_id(youtube_url)
            
            if not video_id:
                raise HTTPException(status_code=400, detail="Invalid YouTube URL")
            
            try:
                extracted_text = get_youtube_transcript(video_id)
                if not extracted_text:
                    raise HTTPException(status_code=404, detail="Could not retrieve transcript for the YouTube video. It might be disabled or the video unavailable.")
                # Set a default title if not provided
                if not title:
                    # Attempt to get video title (you might need a more robust way, e.g., using pytube)
                    try:
                        from pytube import YouTube
                        yt = YouTube(youtube_url)
                        title = yt.title
                    except Exception as e:
                        print(f"Could not fetch YouTube title: {e}")
                        title = f"Summary from {video_id}"
            except Exception as e:
                # Log the specific error for server-side diagnosis
                print(f"Error getting YouTube transcript for {video_id}: {e}")
                # Provide a more generic error to the client
                raise HTTPException(status_code=500, detail=f"Failed to process YouTube video: {e}")
            file_url = youtube_url  # Ensure this line is present
            
        else:
            raise HTTPException(status_code=400, detail="Either file or YouTube URL must be provided")
        
        # Process the extracted text using enhanced RAG
        if extracted_text:
            if len(extracted_text) < 100:
                # Not enough text to process meaningfully
                summary = "# Summary\n\nThe provided content is too short to generate a meaningful summary."
                notes = "## Teaching Notes\n\nNot enough content to generate teaching notes."
            else:
                # Use enhanced RAG processing
                try:
                    summary_content = await enhanced_rag_generate_summary(extracted_text, title)
                    
                    # Split the returned content into summary and notes
                    if "## Teaching Notes" in summary_content:
                        parts = summary_content.split("## Teaching Notes")
                        summary = parts[0].strip()
                        notes = "## Teaching Notes" + parts[1].strip()
                    else:
                        # If AI doesn't format as expected, provide reasonable fallback
                        summary = "# Summary\n\n" + summary_content
                        notes = "## Teaching Notes\n\nNo structured teaching notes were generated."
                except Exception as e:
                    # Fallback if enhanced RAG fails
                    print(f"Enhanced RAG error: {str(e)}")
                    summary = "# Summary\n\nError generating summary with enhanced RAG. Please try again later."
                    notes = "## Teaching Notes\n\nError generating teaching notes with enhanced RAG. Please try again later."
            
            # Ensure the summary starts with a markdown heading
            if not summary.strip().startswith("#"):
                summary = "# Summary\n\n" + summary
            
            # Ensure the notes section starts with a proper markdown heading
            if not notes.strip().startswith("#"):
                notes = "## Teaching Notes\n\n" + notes
            
            # Process markdown to ensure compatibility with our custom renderer
            # Explicitly format markdown for better display
            summary = summary.replace("**", "**").replace("*", "*")
            notes = notes.replace("**", "**").replace("*", "*")
            
            # Format numbered lists appropriately
            # This ensures our regex in the Flutter app can recognize them
            summary_lines = summary.split("\n")
            processed_summary_lines = []
            for line in summary_lines:
                if line.strip() and line.strip()[0].isdigit() and "." in line:
                    number_end = line.find(".")
                    if number_end > 0 and number_end < 5:  # Reasonable digit length
                        indent = line.find(line.strip())
                        spaces = " " * indent if indent > 0 else ""
                        number = line.strip()[:number_end+1]
                        rest = line.strip()[number_end+1:].strip()
                        line = f"{spaces}{number} {rest}"
                processed_summary_lines.append(line)
            summary = "\n".join(processed_summary_lines)
            
            # Do the same for notes
            notes_lines = notes.split("\n")
            processed_notes_lines = []
            for line in notes_lines:
                if line.strip() and line.strip()[0].isdigit() and "." in line:
                    number_end = line.find(".")
                    if number_end > 0 and number_end < 5:  # Reasonable digit length
                        indent = line.find(line.strip())
                        spaces = " " * indent if indent > 0 else ""
                        number = line.strip()[:number_end+1]
                        rest = line.strip()[number_end+1:].strip()
                        line = f"{spaces}{number} {rest}"
                processed_notes_lines.append(line)
            notes = "\n".join(processed_notes_lines)
            
            created_at = datetime.now()
            
            # Save to Firestore
            try:
                doc_ref = db.collection(f'users/{user_id}/summaries').document()
                doc_ref.set({
                    'title': title,
                    'summary': summary,
                    'notes': notes,
                    'originalFileUrl': file_url,
                    'contentType': content_type,
                    'createdAt': created_at.isoformat(),
                    'userId': user_id
                })
            except Exception as e:
                # If Firestore save fails, still return the generated content
                print(f"Firestore save error: {str(e)}")
            
            return SummaryResponse(
                title=title,
                summary=summary,
                notes=notes,
                original_file_url=file_url,
                content_type=content_type,
                created_at=created_at
            )
        else:
            raise HTTPException(status_code=400, detail="Could not extract text from the provided resource")
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the full error for debugging
        import traceback
        traceback.print_exc()
        # Return a user-friendly error
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

async def enhanced_rag_quiz_generation(text: str, num_questions: int, title: str, token: str) -> QuizGenerationResponse:
    """Generate quiz using enhanced RAG implementation"""
    try:
        # Create chunks with metadata
        chunks = rag_processor.chunk_text_with_metadata(text, source=title)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Insufficient content for quiz generation")
        
        # Create embeddings for all chunks
        embeddings = await rag_processor.create_embeddings_batch(chunks)
        
        # Build FAISS index
        index = rag_processor.build_faiss_index(embeddings)
        
        # Define queries for different Bloom's taxonomy levels
        bloom_queries = {
            "Knowledge": "facts definitions terms key concepts basic information",
            "Understand": "explain describe summarize main ideas concepts",
            "Apply": "examples applications use cases practical implementation",
            "Analyze": "compare contrast analyze relationships components",
            "Evaluate": "critique assess judge evaluate effectiveness",
            "Create": "design create synthesize combine generate new ideas"
        }
        
        # Retrieve relevant chunks for each Bloom's level
        bloom_chunks = {}
        for level, query in bloom_queries.items():
            query_embedding = await rag_processor.create_query_embedding(query)
            relevant_chunks = rag_processor.retrieve_relevant_chunks(
                query_embedding, index, chunks, k=max(2, num_questions // 3)
            )
            bloom_chunks[level] = rag_processor.rerank_chunks(relevant_chunks, query)
        
        # Distribute questions across Bloom's levels
        questions_per_level = {
            "Knowledge": max(1, num_questions // 4),
            "Understand": max(1, num_questions // 3),
            "Apply": max(1, num_questions // 4),
            "Analyze": max(1, num_questions // 6),
            "Evaluate": max(0, num_questions // 8),
            "Create": max(0, num_questions // 8)
        }
        
        # Adjust to match exact number requested
        total_planned = sum(questions_per_level.values())
        if total_planned < num_questions:
            questions_per_level["Understand"] += num_questions - total_planned
        elif total_planned > num_questions:
            # Reduce from higher levels first
            reduction_needed = total_planned - num_questions
            for level in ["Create", "Evaluate", "Analyze", "Apply"]:
                if reduction_needed <= 0:
                    break
                reduction = min(questions_per_level[level], reduction_needed)
                questions_per_level[level] -= reduction
                reduction_needed -= reduction
        
        # Generate context for each Bloom's level
        level_contexts = {}
        for level, level_chunks in bloom_chunks.items():
            if questions_per_level[level] > 0 and level_chunks:
                context = "\n\n".join([chunk.text for chunk in level_chunks[:3]])
                level_contexts[level] = context
        
        # Generate questions for each level
        all_questions = []
        for level, question_count in questions_per_level.items():
            if question_count > 0 and level in level_contexts:
                context = level_contexts[level]
                
                prompt = f"""Generate {question_count} multiple-choice questions at the "{level}" level of Bloom's Taxonomy based on the following content.

Bloom's {level} Level Guidelines:
- Knowledge: Recall facts, terms, basic concepts
- Understand: Explain ideas, summarize, describe
- Apply: Use information in new situations, solve problems
- Analyze: Break down information, find patterns, relationships
- Evaluate: Make judgments, critique, assess value
- Create: Combine ideas, design, formulate new approaches

Content for questions:
{context}

Format your response as a JSON array:
[
  {{
    "question": "Question text here",
    "options": [
      {{ "id": "A", "text": "Option A text" }},
      {{ "id": "B", "text": "Option B text" }},
      {{ "id": "C", "text": "Option C text" }},
      {{ "id": "D", "text": "Option D text" }}
    ],
    "correctAnswer": "A",
    "bloomTag": "{level}"
  }}
]

Make questions challenging but fair, with plausible distractors.
"""
                
                try:
                    response = model.generate_content(prompt)
                    response_text = response.text
                    
                    # Extract JSON from response
                    import re
                    json_match = re.search(r'\[[\s\S]*\]', response_text)
                    if json_match:
                        json_str = json_match.group(0)
                        level_questions = json.loads(json_str)
                        all_questions.extend(level_questions)
                
                except Exception as e:
                    print(f"Error generating {level} questions: {str(e)}")
                    # Continue with other levels
        
        # If no questions generated successfully, create fallback
        if not all_questions:
            all_questions = [{
                "question": "Error generating questions from content. Please try again with different content.",
                "options": [
                    {"id": "A", "text": "Try different content"},
                    {"id": "B", "text": "Check content length"},
                    {"id": "C", "text": "Contact support"},
                    {"id": "D", "text": "Report issue"}
                ],
                "correctAnswer": "A",
                "bloomTag": "Knowledge"
            }]
        
        # Convert to QuizQuestion objects
        quiz_questions = []
        for q in all_questions[:num_questions]:  # Limit to requested number
            question = QuizQuestion(
                question=q.get("question", ""),
                options=[
                    QuizOption(id=opt.get("id", ""), text=opt.get("text", ""))
                    for opt in q.get("options", [])
                ],
                correctAnswer=q.get("correctAnswer", "A"),
                bloomTag=q.get("bloomTag", "Knowledge")
            )
            quiz_questions.append(question)
        
        return QuizGenerationResponse(
            title=title,
            questions=quiz_questions,
            created_at=datetime.now()
        )
        
    except Exception as e:
        print(f"Enhanced RAG quiz generation error: {str(e)}")
        # Fallback to simple generation
        return await simple_quiz_fallback(text, num_questions, title)

async def simple_quiz_fallback(text: str, num_questions: int, title: str) -> QuizGenerationResponse:
    """Fallback quiz generation without RAG"""
    context = text[:2000] if len(text) > 2000 else text
    
    prompt = f"""Generate {num_questions} multiple-choice questions based on this content:

{context}

Format as JSON array with question, options (A-D), correctAnswer, and bloomTag fields."""
    
    try:
        response = model.generate_content(prompt)
        # Simple parsing and return basic structure
        questions = [QuizQuestion(
            question=f"Sample question {i+1} for {title}",
            options=[
                QuizOption(id="A", text="Option A"),
                QuizOption(id="B", text="Option B"),
                QuizOption(id="C", text="Option C"),
                QuizOption(id="D", text="Option D")
            ],
            correctAnswer="A",
            bloomTag="Knowledge"
        ) for i in range(num_questions)]
        
        return QuizGenerationResponse(
            title=title,
            questions=questions,
            created_at=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {str(e)}")

@app.post("/api/generate-quiz", response_model=QuizGenerationResponse)
async def generate_quiz(request: QuizGenerationRequest):
    # Verify Firebase token
    user_id = await verify_token(request.token)
    
    try:
        # Use enhanced RAG for quiz generation
        result = await enhanced_rag_quiz_generation(
            request.content, request.numQuestions, request.title, request.token
        )
        
        # Save to Firestore
        doc_ref = db.collection(f'users/{user_id}/quizzes').document()
        doc_ref.set({
            'title': request.title,
            'questions': [q.dict() for q in result.questions],
            'createdAt': result.created_at.isoformat(),
            'userId': user_id,
            'sourceType': 'manual',
            'pdfExported': False
        })
        
        return result
    
    except Exception as e:
        # Log the full error for debugging
        import traceback
        traceback.print_exc()
        # Return a user-friendly error
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/api/generate-quiz-from-file", response_model=QuizGenerationResponse)
async def generate_quiz_from_file(
    file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
    num_questions: int = Form(5),
    token: str = Form(...),
    title: Optional[str] = Form(None)
):
    # Verify Firebase token
    user_id = await verify_token(token)
    
    try:
        content_type = ""
        file_path = None
        file_url = ""
        extracted_text = ""
        
        # Process file upload or YouTube URL
        if file:
            # Save uploaded file to temporary location
            content_type = "pdf" if file.filename.lower().endswith('.pdf') else "video"
            
            # Create temp file with proper extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{content_type}") as temp_file:
                file_path = temp_file.name
            
            # Write uploaded content to temp file
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Generate unique filename for storage
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            storage_filename = f"{timestamp}_{file.filename}"
            
            # Upload to Firebase Storage
            try:
                blob = bucket.blob(f"users/{user_id}/{storage_filename}")
                blob.upload_from_filename(file_path)
                blob.make_public()
                file_url = blob.public_url
            except Exception as e:
                # If storage upload fails, still try to process the file
                print(f"Firebase storage upload error: {str(e)}")
                file_url = f"local://{file.filename}"
            
            # Extract text based on content type
            if content_type == "pdf":
                try:
                    extracted_text = extract_text_from_pdf(file_path)
                except Exception as e:
                    raise HTTPException(
                        status_code=422, 
                        detail=f"Could not extract text from PDF: {str(e)}"
                    )
            else:
                # For video, we'd need a different approach (not implemented)
                extracted_text = "Video text extraction not implemented yet."
                
            # Clean up temporary file
            try:
                os.unlink(file_path)
            except:
                pass
            
            # Use filename as title if not provided
            if not title:
                title = file.filename
            
        elif youtube_url:
            content_type = "video"
            video_id = extract_youtube_id(youtube_url)
            
            if not video_id:
                raise HTTPException(status_code=400, detail="Invalid YouTube URL")
            
            try:
                extracted_text = get_youtube_transcript(video_id)
                if not extracted_text:
                    raise HTTPException(status_code=404, detail="Could not retrieve transcript for the YouTube video. It might be disabled or the video unavailable.")
                # Set a default title if not provided
                if not title:
                    # Attempt to get video title (you might need a more robust way, e.g., using pytube)
                    try:
                        from pytube import YouTube
                        yt = YouTube(youtube_url)
                        title = yt.title
                    except Exception as e:
                        print(f"Could not fetch YouTube title: {e}")
                        title = f"Quiz from {video_id}"
            except Exception as e:
                # Log the specific error for server-side diagnosis
                print(f"Error getting YouTube transcript for {video_id}: {e}")
                # Provide a more generic error to the client
                raise HTTPException(status_code=500, detail=f"Failed to process YouTube video: {e}")

            file_url = youtube_url
        else:
            raise HTTPException(status_code=400, detail="Either file or YouTube URL must be provided")
        
        # Generate quiz using the extracted text with enhanced RAG
        if extracted_text:
            # Use enhanced RAG quiz generation
            result = await enhanced_rag_quiz_generation(extracted_text, num_questions, title, token)
            
            # Save to Firestore
            doc_ref = db.collection(f'users/{user_id}/quizzes').document()
            doc_ref.set({
                'title': title,
                'questions': [q.dict() for q in result.questions],
                'createdAt': result.created_at.isoformat(),
                'userId': user_id,
                'sourceType': 'file',
                'originalFileUrl': file_url,
                'contentType': content_type,
                'pdfExported': False
            })
            
            return result
        else:
            raise HTTPException(status_code=400, detail="Could not extract text from the provided resource")
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the full error for debugging
        import traceback
        traceback.print_exc()
        # Return a user-friendly error
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import asyncio
    uvicorn.run(app, host="0.0.0.0", port=8000)