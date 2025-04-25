export const runtime = 'nodejs';



import { OpenAI } from 'openai';
import pinecone from '@/utils/pinecone';

import fs from 'fs';
import path from 'path';
import Papa from 'papaparse';

// Load and parse the mapping CSV at runtime
interface MappingRow {
  [key: string]: string;
}

interface ScoreData {
  [key: string]: number;
}

interface InsightData {
  [key: string]: string;
}

interface ESGResult {
  [key: string]: number;
}

const mappingPath = path.join(process.cwd(), 'public', 'data', 'mapping-file-public.csv');
const mappingCsv = fs.readFileSync(mappingPath, 'utf8');
const { data: mappingData } = Papa.parse(mappingCsv, {
  header: true,
  skipEmptyLines: true
});

const mapping = Object.fromEntries(
  mappingData.map((row: unknown) => {
    const typedRow = row as MappingRow;
    return [typedRow['Original Column'], typedRow['Mapped Column']];
  })
);

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

interface RecordMetadata {
  text?: string;
  page_number?: number;
  [key: string]: any;
}

interface ScoredPineconeRecord<T> {
  id: string;
  score: number;
  metadata?: T;
  namespace?: string;
}

export async function POST(req: Request) {
  const {
    messages,
    docText,
    docName,
    consistencyResult,
    analysisResult,
    esgResult,
    tcfdAdvice,
    tcfdDrafts,
    csrdAdvice,
    csrdDrafts,
    griAdvice,
    griDrafts,
    sasbAdvice,
    sasbDrafts
  } = await req.json();
  
  
  // Declare these variables at the top so they're available everywhere:
  let sourceDocuments: string[] = [];
  let followUps: string[] = [];  
  const _scoreSummary = `


  
  ===== ANALYSIS SUMMARY =====

  📊 Reporting Quality:
  - Consistency Score: ${consistencyResult?.consistency_score ?? 'N/A'}
  - Readability Score: ${consistencyResult?.readability_score ?? 'N/A'}
  - Clarity Score: ${consistencyResult?.clarity_score ?? 'N/A'}



  🗣️ Cheap Talk Analysis:
  - Commitment Probability: ${analysisResult?.commitment_probability ?? 'N/A'}
  - Specificity Probability: ${analysisResult?.specificity_probability ?? 'N/A'}
  - Cheap Talk Score: ${analysisResult?.cheap_talk_probability ?? 'N/A'}
  - Safe Talk Score: ${analysisResult?.safe_talk_probability ?? 'N/A'}

  🌿 ESG Relevance:
  ${esgResult ? Object.entries(esgResult as ESGResult).map(([key, value]) => `- ${key}: ${((value as number) * 100).toFixed(1)}%`).join('\n') : 'No ESG scores available'}
  `;



  try {
    // Get the last user message
    const lastUserMessage = messages[messages.length - 1];
    console.log('Processing chat request with message:', lastUserMessage.content.substring(0, 100) + '...');
    console.log('Uploaded doc:', docName);

    if (!docText || !docName) {
      const response = await openai.chat.completions.create({
        model: 'gpt-4-turbo-preview',
        messages: [
          { role: 'system', content: 'You are a helpful assistant. No document was uploaded. Respond helpfully and clearly.' },
          ...messages,
        ],
        stream: true,
      });
    
      const stream = new ReadableStream({
        async start(controller) {
          const encoder = new TextEncoder();
          for await (const chunk of response) {
            const content = chunk.choices[0]?.delta?.content;
            if (content) {
              controller.enqueue(encoder.encode(content));
            }
          }
          controller.close();
        },
      });
    
      return new Response(stream, {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'X-Source-Documents': JSON.stringify(sourceDocuments),
          'X-Extra-Suggestions': JSON.stringify(followUps),
        }
      });
    }
    

    // Get embeddings for the query
    const embedding = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: lastUserMessage.content,
      encoding_format: "float"
    });

    console.log('Generated embedding with dimensions:', embedding.data[0].embedding.length);

    // Define namespace from document name
    const namespace = docName || 'default';
    const index = pinecone.index("embed-upload").namespace(namespace);

    const queryResponse = await index.query({
      vector: embedding.data[0].embedding,
      topK: 15,
      includeMetadata: true,
    }) as { matches: ScoredPineconeRecord<RecordMetadata>[] };

    
    
    const context = queryResponse.matches
      .filter(match => match.metadata?.text)
      .map(match => {
        const page = match.metadata?.page_number ? ` (p. ${match.metadata.page_number})` : '';
        return `[${match.namespace || namespace}${page}]: ${match.metadata?.text}`;
      })
      .join('\n\n');

    // Build a mapping from doc name to a Set of pages
    const sourceMap: { [doc: string]: Set<number> } = {};

    for (const match of queryResponse.matches) {
      const doc = match.namespace || namespace; // fallback
      const page = match.metadata?.page_number;

      if (page !== undefined) {
        if (!sourceMap[doc]) sourceMap[doc] = new Set();
        sourceMap[doc].add(page);
      }
    }

    // Format the sources nicely
    sourceDocuments = Object.entries(sourceMap).map(([doc, pages]) => {
      const sortedPages = Array.from(pages).sort((a, b) => a - b);
      const pageList = sortedPages.join(', ');
      return `${doc}, pp. ${pageList}`;
    });
    




    // --------------------------------------------
    // BUILD THE FINAL MESSAGES ARRAY, with context
    // --------------------------------------------

    const friendlyName = mapping[docName] || docName;

    const docNameMessage = {
      role: 'user',
      content: `(User uploaded: ${friendlyName})`
    };
    
    const buildAnalysisSummary = () => {
      if (!analysisResult && !consistencyResult && !esgResult) return '';
    
      const formatPercent = (val?: number) =>
        typeof val === 'number' ? `${(val * 100).toFixed(1)}%` : 'N/A';
    
      return `
    ===== ANALYSIS SUMMARY =====
    
    📊 Reporting Quality:
    - Consistency Score: ${formatPercent(consistencyResult?.consistency_score)}
    - Readability Score: ${formatPercent(consistencyResult?.readability_score)}
    - Clarity Score: ${formatPercent(consistencyResult?.clarity_score)}
    - Variability Score: ${formatPercent(consistencyResult?.consistency_variability)}

    
    🗣️ Cheap Talk Analysis:
    - Commitment Probability: ${formatPercent(analysisResult?.commitment_probability)}
    - Specificity Probability: ${formatPercent(analysisResult?.specificity_probability)}
    - Cheap Talk Score: ${formatPercent(analysisResult?.cheap_talk_probability)}
    - Safe Talk Score: ${formatPercent(analysisResult?.safe_talk_probability)}
    
    🌿 ESG Relevance:
    ${esgResult
      ? Object.entries(esgResult as ESGResult)
          .map(([key, value]) => `- ${key}: ${(value * 100).toFixed(1)}%`)
          .join('\n')
      : 'No ESG scores available'}
    `;
    };
    
    
    
    const scoreSummary = buildAnalysisSummary();

    let tcfdKnowledge = "";
    if (tcfdAdvice && Object.keys(tcfdAdvice).length > 0) {
      tcfdKnowledge += "\n📌 TCFD Advice:\n";
      for (const [_id, advice] of Object.entries(tcfdAdvice)) {
        tcfdKnowledge += `- ${advice}\n`;
      }
    }
    if (tcfdDrafts && Object.keys(tcfdDrafts).length > 0) {
      tcfdKnowledge += "\n📝 TCFD Draft Suggestions:\n";
      for (const [_id, draft] of Object.entries(tcfdDrafts)) {
        tcfdKnowledge += `- ${draft}\n`;
      }
    }

    let csrdKnowledge = "";
    if (csrdAdvice && Object.keys(csrdAdvice).length > 0) {
      csrdKnowledge += "\n📌 CSRD Advice:\n";
      for (const [_id, advice] of Object.entries(csrdAdvice)) {
        csrdKnowledge += `- ${advice}\n`;
      }
    }
    if (csrdDrafts && Object.keys(csrdDrafts).length > 0) {
      csrdKnowledge += "\n📝 CSRD Draft Suggestions:\n";
      for (const [_id, draft] of Object.entries(csrdDrafts)) {
        csrdKnowledge += `- ${draft}\n`;
      }
    }

    let griKnowledge = "";
    if (griAdvice && Object.keys(griAdvice).length > 0) {
      griKnowledge += "\n📌 GRI Advice:\n";
      for (const [_id, advice] of Object.entries(griAdvice)) {
        griKnowledge += `- ${advice}\n`;
      }
    }
    if (griDrafts && Object.keys(griDrafts).length > 0) {
      griKnowledge += "\n📝 GRI Draft Suggestions:\n";
      for (const [_id, draft] of Object.entries(griDrafts)) {
        griKnowledge += `- ${draft}\n`;
      }
    }

    let sasbKnowledge = "";
    if (sasbAdvice && Object.keys(sasbAdvice).length > 0) {
      sasbKnowledge += "\n📌 SASB Advice:\n";
      for (const [_id, advice] of Object.entries(sasbAdvice) as [string, string][]) {
        sasbKnowledge += `- ${advice}\n`;
      }
    }
    if (sasbDrafts && Object.keys(sasbDrafts).length > 0) {
      sasbKnowledge += "\n📝 SASB Draft Suggestions:\n";
      for (const [_id, draft] of Object.entries(sasbDrafts) as [string, string][]) {
        sasbKnowledge += `- ${draft}\n`;
      }
    }





    const contextualMessages = [
      {
        role: "system",
        content: `
      You are a helpful assistant that analyzes and explains company reports.
      
      🚫 NEVER say "excerpts", "segments", "chunks", or "provided content".
      ✅ ALWAYS refer to the uploaded document simply as **"the report"** — even if only parts of it are visible.
      
      Respond naturally and clearly, using only what is in the report. Cite page numbers (e.g. "p. 3") where possible.
      
      If something is not mentioned in the report, say:  
      👉 "The report does not mention..." or  
      👉 "There is no information in the report about..."
      
      ---
      
      Below is content from the report and some automated analysis scores.
      Use this information to answer the user's question clearly and helpfully.
            
    Only reference analysis scores (e.g. Cheap Talk, Safe talk, ESG, Consistency, Clarity, Readability) if the user explicitly asks about them. 
    Otherwise, rely solely on the content of the report.
    
    IMPORTANT: Always refer to the uploaded document as **"the report"**. 
    NEVER use terms like "segments", "excerpt", "excerpts", "extracted content", or "chunks" in your responses.

    You are an AI assistant analyzing uploaded company reports. Focus specifically on the uploaded document: ${friendlyName}.
    
    Use the following information to answer the user's question:\n\n${context}\n\n
    
    Refer to it simply as "the report" (not "extracted content" or "segments").

    In addition, the system provides three types of automated analysis:
    
    1. **Reporting Quality (Consistency) Score** – This measures how semantically consistent the document is across different segments. It is calculated by splitting the document into parts and comparing the similarity of language using cosine similarity of TF-IDF vectors. A higher score indicates a more consistent and coherent report.
    
    2. **Readability Score (Flesch Reading Ease)** – This measures how easy the document is to read, based on sentence length and word complexity. It is normalized between 0 and 1. Extremely low scores indicate overly complex, difficult-to-read text. Extremely high scores may suggest the text is too simplistic to convey technical or meaningful insights. The ideal is a balanced score that reflects accessible but informative language.
    
    3. **Clarity Score (Type-Token Ratio, or TTR)** – This measures the diversity of vocabulary by comparing the number of unique words (types) to the total number of words (tokens). Low TTR may indicate repetition or vague language, while extremely high TTR could reflect excessive complexity or lack of focus. The most effective reports often fall in the middle, combining variety with clarity and precision.
    
    4. **Variability Score** – This measures how much variation exists in language use across different parts of the report. A low variability score may indicate consistent or repetitive language, while a high score suggests diverse but potentially disjointed content. Ideal scores balance variety with coherence.



    4. **Cheap Talk Analysis** – This detects vague or non-committal language. It includes:
       - **Commitment Probability**: Likelihood that the text uses strong, forward-looking or obligation-based language.
       - **Specificity Probability**: Measures how specific and detailed the statements are.
       - **Cheap Talk Probability**: Defined as 1 - (commitment × specificity). High cheap talk implies vague or hollow language.
       - **Safe Talk**: Language that is specific but lacks strong commitments. Defined as (1 - commitment) × specificity.
    
    5. **ESG Category Scores** – Derived using FinBERT and other NLP techniques to estimate how much of the document relates to nine ESG categories (e.g., Climate Change, Human Capital). These are shown as percentages.
    
    ===== DOCUMENT ANALYSIS METRICS =====
    ${scoreSummary}
    
    ${tcfdKnowledge ? `\n\n📘 Additional TCFD Guidance:\n${tcfdKnowledge}` : ""}
    ${csrdKnowledge ? `\n\n📘 Additional CSRD Guidance:\n${csrdKnowledge}` : ""}
    ${griKnowledge ? `\n\n📘 Additional GRI Guidance:\n${griKnowledge}` : ""}
    ${sasbKnowledge ? `\n\n📘 Additional SASB Guidance:\n${sasbKnowledge}` : ""}

    Use these scores to interpret the overall quality and focus of the report. If asked, connect findings in the text (e.g., vagueness, strong commitments, ESG focus) to these metrics.
    
    IMPORTANT RULES:
    - Always cite page numbers where possible, e.g., (p. 3).
    - If unsure, acknowledge limitations but still attempt a helpful answer based on what's available in the report.
    - Use the scores to support your reasoning.
    - Do not hallucinate metrics not provided above.
    `
      },
      docNameMessage,
      ...messages
    ];
    
    
    
  // STREAM THE COMPLETION (GPT-4) BACK TO CLIENT
  const chatResponse = await openai.chat.completions.create({
    model: 'gpt-4-turbo-preview',
    messages: contextualMessages,
    stream: true,
  });

  let assistantResponse = '';
  const encoder = new TextEncoder();
  

  const stream = new ReadableStream({
    async start(controller) {
      // Stream the GPT-4 response
      for await (const chunk of chatResponse) {
        const content = chunk.choices[0]?.delta?.content;
        if (content) {
          assistantResponse += content;
          controller.enqueue(encoder.encode(content));
        }
      }
      controller.close();

      // After streaming is complete, generate follow-up suggestions
      try {
        const followUpRes = await openai.chat.completions.create({
          model: 'gpt-4-turbo-preview',
          messages: [
            {
              role: "system",
              content: `You are a helpful assistant. Based on the assistant's previous answer, suggest 2–3 concise follow-up questions the user might want to ask next.
  Be proactive, relevant, and helpful. Format your response as a plain numbered list without extra commentary.`
            },
            { role: "user", content: lastUserMessage.content },
            { role: "assistant", content: assistantResponse }
          ]
        });
        const followUpText = followUpRes.choices?.[0]?.message?.content || "";
        followUps = followUpText
          .split('\n')
          .filter(line => line.trim().match(/^\d+\./))
          .map(line => line.replace(/^\d+\.\s*/, '').trim());
      } catch (err) {
        console.warn("⚠️ Follow-up suggestion generation failed:", err);
      }

      // Enqueue a final newline so the stream ends properly
      controller.enqueue(encoder.encode('\n'));
    }
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'X-Source-Documents': JSON.stringify(sourceDocuments),
      'X-Extra-Suggestions': JSON.stringify(followUps)
    }
  });


  } catch (error) {
    console.error('Error in chat endpoint:', error);
    return new Response(
      JSON.stringify({ error: 'Failed to process request' }),
      { status: 500 }
    );
  }
} // 👈 ✅ THIS is what you need to add!
