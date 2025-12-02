import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { createClient } from '@supabase/supabase-js';
import OpenAI from 'openai';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Initialize Supabase client (only if environment variables are configured)
let supabase = null;
if (process.env.SUPABASE_URL && process.env.SUPABASE_SERVICE_ROLE_KEY) {
  supabase = createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY
  );
}

// Initialize OpenAI client (only if API key is configured)
let openai = null;
if (process.env.OPENAI_API_KEY) {
  openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
  });
}

// POST /ingest_content - Ingest content into the system
app.post('/ingest_content', async (req, res) => {
  try {
    if (!supabase || !openai) {
      return res.status(503).json({ error: 'Service not configured. Please check environment variables.' });
    }

    const { content, metadata } = req.body;

    if (!content) {
      return res.status(400).json({ error: 'Content is required' });
    }

    // Generate embedding for the content using OpenAI
    const embeddingResponse = await openai.embeddings.create({
      model: 'text-embedding-ada-002',
      input: content
    });

    const embedding = embeddingResponse.data[0].embedding;

    // Store content and embedding in Supabase
    const { data, error } = await supabase
      .from('content')
      .insert({
        content,
        metadata: metadata || {},
        embedding
      })
      .select();

    if (error) {
      return res.status(500).json({ error: error.message });
    }

    res.status(201).json({ success: true, data });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// POST /search_content - Search for content
app.post('/search_content', async (req, res) => {
  try {
    if (!supabase || !openai) {
      return res.status(503).json({ error: 'Service not configured. Please check environment variables.' });
    }

    const { query, limit = 10 } = req.body;

    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    // Generate embedding for the search query
    const embeddingResponse = await openai.embeddings.create({
      model: 'text-embedding-ada-002',
      input: query
    });

    const queryEmbedding = embeddingResponse.data[0].embedding;

    // Perform similarity search in Supabase
    const { data, error } = await supabase.rpc('match_content', {
      query_embedding: queryEmbedding,
      match_threshold: 0.7,
      match_count: limit
    });

    if (error) {
      return res.status(500).json({ error: error.message });
    }

    res.json({ success: true, data });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// POST /log_completion_result - Log completion results
app.post('/log_completion_result', async (req, res) => {
  try {
    if (!supabase) {
      return res.status(503).json({ error: 'Service not configured. Please check environment variables.' });
    }

    const { completion_id, result, metadata } = req.body;

    if (!completion_id || result === undefined) {
      return res.status(400).json({ error: 'completion_id and result are required' });
    }

    // Store completion result in Supabase
    const { data, error } = await supabase
      .from('completion_logs')
      .insert({
        completion_id,
        result,
        metadata: metadata || {},
        logged_at: new Date().toISOString()
      })
      .select();

    if (error) {
      return res.status(500).json({ error: error.message });
    }

    res.status(201).json({ success: true, data });
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
  console.log(`Server running on port ${PORT}`);
});

export { app, server };
