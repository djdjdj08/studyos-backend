import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { createClient } from '@supabase/supabase-js';
import OpenAI from 'openai';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json({ limit: '25mb' }));

// Initialize OpenAI client
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Initialize Supabase client
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

// Chunk profiles configuration
const chunkProfiles = {
  short_form: { size: 400, overlap: 60 },
  default: { size: 900, overlap: 120 },
  long_book: { size: 1400, overlap: 180 }
};

/**
 * Helper: chunkText
 * Word-based text chunker with configurable profiles
 */
function chunkText(text, profile = 'default') {
  const { size, overlap } = chunkProfiles[profile] || chunkProfiles.default;
  const words = text.split(/\s+/);
  
  // If text is smaller than chunk size, return single chunk
  if (words.length <= size) {
    return [text];
  }
  
  const chunks = [];
  let start = 0;
  
  while (start < words.length) {
    const end = Math.min(start + size, words.length);
    const chunk = words.slice(start, end).join(' ');
    chunks.push(chunk);
    
    // Move start position, accounting for overlap
    start = end - overlap;
    
    // Prevent infinite loop if overlap >= size
    if (end === words.length) {
      break;
    }
  }
  
  return chunks;
}

/**
 * Helper: embedText
 * Generate embeddings using OpenAI
 */
async function embedText(input) {
  const response = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input
  });
  
  return response.data.map(item => item.embedding);
}

// POST /ingest_content - Ingest content into the system
app.post('/ingest_content', async (req, res) => {
  try {
    const {
      course,
      type,
      subtopic,
      assignment_type,
      chunk_profile,
      raw_text,
      source_name
    } = req.body;

    // Validate required fields
    if (!course || !type || !raw_text) {
      return res.status(400).json({ 
        error: 'Missing required fields: course, type, and raw_text are required' 
      });
    }

    // Validate type
    if (!['resource', 'instruction'].includes(type)) {
      return res.status(400).json({ 
        error: 'type must be "resource" or "instruction"' 
      });
    }

    // Chunk the text
    const chunks = chunkText(raw_text, chunk_profile);

    // Embed all chunks
    const embeddings = await embedText(chunks);

    // Insert each chunk into kb_chunks table
    const insertPromises = chunks.map((content, index) => {
      return supabase
        .from('kb_chunks')
        .insert({
          course,
          type,
          subtopic: subtopic || null,
          assignment_type: assignment_type || null,
          source_name: source_name || null,
          chunk_index: index,
          content,
          embedding: embeddings[index]
        })
        .select('id');
    });

    const results = await Promise.all(insertPromises);

    // Check for errors
    const errors = results.filter(r => r.error);
    if (errors.length > 0) {
      return res.status(500).json({ 
        error: 'Failed to insert some chunks', 
        details: errors.map(e => e.error.message) 
      });
    }

    const ids = results.map(r => r.data[0].id);

    res.status(201).json({ 
      success: true, 
      chunks_stored: chunks.length,
      ids 
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// POST /search_content - Search for content
app.post('/search_content', async (req, res) => {
  try {
    const {
      course,
      query,
      types,
      subtopic,
      assignment_type,
      top_k = 10,
      threshold = 0.7
    } = req.body;

    // Validate required fields
    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    // Embed the query
    const [queryEmbedding] = await embedText([query]);

    // Call Supabase RPC match_kb_chunks
    const { data, error } = await supabase.rpc('match_kb_chunks', {
      query_embedding: queryEmbedding,
      match_threshold: threshold,
      match_count: top_k,
      filter_course: course || null,
      filter_types: types || null,
      filter_subtopic: subtopic || null,
      filter_assignment_type: assignment_type || null
    });

    if (error) {
      return res.status(500).json({ error: error.message });
    }

    res.json({ success: true, results: data });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// POST /log_completion_result - Log completion results
app.post('/log_completion_result', async (req, res) => {
  try {
    const {
      course,
      assignment_type,
      subtopic,
      original_prompt,
      model_answer,
      outcome,
      score,
      teacher_feedback
    } = req.body;

    // Validate required fields
    if (!course || !model_answer || outcome === undefined) {
      return res.status(400).json({ 
        error: 'Missing required fields: course, model_answer, and outcome are required' 
      });
    }

    // Determine type based on outcome
    const type = outcome ? 'completion_good' : 'completion_bad';

    // Embed the model answer
    const [embedding] = await embedText([model_answer]);

    // Construct content with all relevant information
    const content = JSON.stringify({
      original_prompt,
      model_answer,
      score,
      teacher_feedback
    });

    // Insert as a single row into kb_chunks
    const { data, error } = await supabase
      .from('kb_chunks')
      .insert({
        course,
        type,
        subtopic: subtopic || null,
        assignment_type: assignment_type || null,
        source_name: null,
        chunk_index: 0,
        content,
        embedding
      })
      .select('id');

    if (error) {
      return res.status(500).json({ error: error.message });
    }

    res.status(201).json({ 
      success: true, 
      id: data[0].id 
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

// Start server
const server = app.listen(PORT, () => {
  console.log(`Backend running on port ${PORT}`);
});

export { app, server };
