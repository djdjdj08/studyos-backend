// server.js
import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import OpenAI from 'openai';
import { createClient } from '@supabase/supabase-js';

const app = express();
app.use(cors());
app.use(express.json({ limit: '25mb' }));

// ---- CLIENTS ----
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

// ---- HELPERS ----

// Chunk size profiles
function getChunkConfig(profile = 'default') {
  switch (profile) {
    case 'short_form':
      return { sizeWords: 400, overlapWords: 60 };
    case 'long_book':
      return { sizeWords: 1400, overlapWords: 180 };
    default:
      return { sizeWords: 900, overlapWords: 120 };
  }
}

// Word-based chunker
function chunkText(text, profile = 'default') {
  const { sizeWords, overlapWords } = getChunkConfig(profile);
  const words = text.split(/\s+/).filter(Boolean);
  const chunks = [];

  if (words.length === 0) return [];

  // Tiny text → 1 chunk
  if (words.length <= sizeWords) {
    chunks.push(words.join(' '));
    return chunks;
  }

  let start = 0;
  while (start < words.length) {
    const end = Math.min(start + sizeWords, words.length);
    const chunkWords = words.slice(start, end);
    chunks.push(chunkWords.join(' '));

    if (end === words.length) break;
    start = end - overlapWords;
  }

  return chunks;
}

// Embedding helper
async function embedText(input) {
  const response = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input
  });
  return response.data.map(x => x.embedding);
}

// ---- ROUTES ----

// 1) INGEST CONTENT
app.post('/ingest_content', async (req, res) => {
  try {
    const {
      course,
      type,
      subtopic,
      assignment_type,
      chunk_profile = 'default',
      raw_text,
      source_name
    } = req.body;

    if (!raw_text || !course || !type) {
      return res.status(400).json({
        error: 'Missing required fields: course, type, raw_text'
      });
    }

    const chunks = chunkText(raw_text, chunk_profile);
    if (chunks.length === 0) {
      return res.status(400).json({ error: 'No content produced' });
    }

    const embeddings = await embedText(chunks);

    const rows = chunks.map((chunk, i) => ({
      course,
      type,
      subtopic,
      assignment_type,
      source_name,
      chunk_index: i,
      content: chunk,
      embedding: embeddings[i]
    }));

    const { data, error } = await supabase
      .from('kb_chunks')
      .insert(rows)
      .select('id');

    if (error) {
      console.error(error);
      return res.status(500).json({ error: 'Insert failed' });
    }

    return res.json({
      success: true,
      chunks_stored: rows.length,
      ids: data.map(x => x.id)
    });
  } catch (err) {
    console.error('INGEST ERROR:', err);
    res.status(500).json({ error: 'Internal error' });
  }
});

// 2) SEARCH CONTENT
app.post('/search_content', async (req, res) => {
  try {
    const {
      course,
      query,
      types,
      subtopic,
      assignment_type,
      top_k = 8,
      threshold = 0.7
    } = req.body;

    if (!query) {
      return res.status(400).json({ error: 'Missing query text' });
    }

    const [queryEmbedding] = await embedText(query);

    const { data, error } = await supabase.rpc('match_kb_chunks', {
      query_embedding: queryEmbedding,
      match_course: course || null,
      match_types: types?.length ? types : null,
      match_subtopic: subtopic || null,
      match_assignment_type: assignment_type || null,
      match_limit: top_k,
      match_threshold: threshold
    });

    if (error) {
      console.error(error);
      return res.status(500).json({ error: 'Search failed' });
    }

    return res.json({
      success: true,
      results: data || []
    });
  } catch (err) {
    console.error('SEARCH ERROR:', err);
    res.status(500).json({ error: 'Internal error' });
  }
});

// 3) LOG COMPLETION RESULTS
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

    if (!course || !assignment_type || !model_answer || !outcome) {
      return res.status(400).json({
        error: 'Missing required fields'
      });
    }

    const type =
      outcome === 'success' ? 'completion_good' : 'completion_bad';

    const [answerEmbedding] = await embedText(model_answer);

    const row = {
      course,
      type,
      subtopic,
      assignment_type,
      original_prompt,
      model_answer,
      outcome,
      score,
      teacher_feedback,
      chunk_index: 0,
      content: model_answer,
      embedding: answerEmbedding
    };

    const { data, error } = await supabase
      .from('kb_chunks')
      .insert(row)
      .select('id');

    if (error) {
      console.error(error);
      return res.status(500).json({ error: 'Insert failed' });
    }

    return res.json({
      success: true,
      id: data[0]?.id
    });
  } catch (err) {
    console.error('LOG RESULT ERROR:', err);
    res.status(500).json({ error: 'Internal error' });
  }
});

// ---- MCP MANIFEST ----
app.get('/.well-known/mcp.json', (req, res) => {
  res.json({
    version: "1.0.0",
    tools: [
      {
        name: "ingest_content",
        description: "Store user-provided resources or instructions into Supabase.",
        input_schema: {
          type: "object",
          properties: {
            course: { type: "string" },
            type: { type: "string" },
            subtopic: { type: "string" },
            assignment_type: { type: "string" },
            chunk_profile: { type: "string" },
            raw_text: { type: "string" },
            source_name: { type: "string" }
          },
          required: ["course", "type", "raw_text"]
        }
      },
      {
        name: "search_content",
        description: "Semantic search through stored resources, instructions, and past completions.",
        input_schema: {
          type: "object",
          properties: {
            course: { type: "string" },
            query: { type: "string" },
            types: {
              type: "array",
              items: { type: "string" }
            },
            subtopic: { type: "string" },
            assignment_type: { type: "string" },
            top_k: { type: "number" },
            threshold: { type: "number" }
          },
          required: ["query"]
        }
      },
      {
        name: "log_completion_result",
        description: "Store graded assignment results in Supabase to improve future outputs.",
        input_schema: {
          type: "object",
          properties: {
            course: { type: "string" },
            assignment_type: { type: "string" },
            subtopic: { type: "string" },
            original_prompt: { type: "string" },
            model_answer: { type: "string" },
            outcome: { type: "string" },
            score: { type: "number" },
            teacher_feedback: { type: "string" }
          },
          required: ["course", "assignment_type", "model_answer", "outcome"]
        }
      }
    ]
  });
});

// ---- SIMPLE HEALTH + ROOT CHECKS ----
app.get('/', (req, res) => {
  res.json({ status: 'ok', message: 'StudyOS backend root' });
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

// ---- SERVER ----
const port = process.env.PORT || 3001;
app.listen(port, () =>
  console.log(`StudyOS backend running on port ${port}`)
);
