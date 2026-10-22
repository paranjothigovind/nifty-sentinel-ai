import express, { Request, Response } from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn, ChildProcess } from 'child_process';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 3000;
const FASTAPI_PORT = 8050;
const FASTAPI_BASE = `http://127.0.0.1:${FASTAPI_PORT}`;

// Initialize Gemini Client server-side
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

let pythonProcess: ChildProcess | null = null;

async function checkFastAPIHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${FASTAPI_BASE}/health`, { signal: AbortSignal.timeout(1000) });
    return res.ok;
  } catch {
    return false;
  }
}

function startFastAPIService(): Promise<void> {
  return new Promise((resolve) => {
    checkFastAPIHealth().then((isHealthy) => {
      if (isHealthy) {
        console.log(`[Python FastAPI] Service already running on port ${FASTAPI_PORT}`);
        return resolve();
      }

      console.log(`[Python FastAPI] Spawning backend.main:app on 127.0.0.1:${FASTAPI_PORT}...`);
      pythonProcess = spawn('python3', ['-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', String(FASTAPI_PORT)], {
        cwd: __dirname,
        stdio: ['ignore', 'pipe', 'pipe'],
      });

      pythonProcess.stdout?.on('data', (d) => {
        const msg = d.toString();
        if (msg.includes('Application startup complete') || msg.includes('Uvicorn running on')) {
          console.log('[Python FastAPI] Ready.');
        }
      });

      pythonProcess.stderr?.on('data', (d) => {
        console.error(`[Python FastAPI] ${d.toString().trim()}`);
      });

      pythonProcess.on('exit', (code) => {
        console.warn(`[Python FastAPI] Process exited with code ${code}`);
        pythonProcess = null;
      });

      // Poll until ready or timeout after 5 seconds
      let attempts = 0;
      const interval = setInterval(async () => {
        attempts++;
        const ready = await checkFastAPIHealth();
        if (ready || attempts >= 10) {
          clearInterval(interval);
          resolve();
        }
      }, 500);
    });
  });
}

async function startServer() {
  const app = express();
  app.use(express.json());

  // Ensure FastAPI service is active
  await startFastAPIService();

  // 1. Proxy `/api/quant/*` to FastAPI on port 8050
  app.all('/api/quant/*', async (req: Request, res: Response) => {
    const targetPath = req.url.replace(/^\/api\/quant/, '');
    const targetUrl = `${FASTAPI_BASE}${targetPath}`;

    try {
      const headers: Record<string, string> = {
        'Content-Type': req.header('Content-Type') || 'application/json',
      };

      const fetchOptions: RequestInit = {
        method: req.method,
        headers,
      };

      if (['POST', 'PUT', 'PATCH'].includes(req.method) && req.body && Object.keys(req.body).length > 0) {
        fetchOptions.body = JSON.stringify(req.body);
      }

      const backendResponse = await fetch(targetUrl, fetchOptions);
      const data = await backendResponse.text();

      res.status(backendResponse.status);
      res.setHeader('Content-Type', backendResponse.headers.get('Content-Type') || 'application/json');
      res.send(data);
    } catch (err: any) {
      console.error(`Proxy error to ${targetUrl}:`, err.message);
      res.status(502).json({
        error: 'FastAPI quantitative backend unreachable',
        detail: err.message,
        targetUrl,
      });
    }
  });

  // 2. Educational Gemini explanation layer
  app.post('/api/explain', async (req: Request, res: Response) => {
    try {
      const { backtest, prediction, stats } = req.body;

      const prompt = `You are a quantitative finance researcher and educator specializing in market micro-structure and statistical learning.
Analyze the following empirical quantitative results for the NIFTY 50 index closing direction:

Empirical Dataset & Balance:
- Total sessions: ${stats?.usable_samples || 'N/A'}
- Empirical Target Balance: ${stats?.up_percentage || 'N/A'}% Up (${stats?.up_count} Up / ${stats?.down_count} Down)

Walk-Forward Out-of-Sample Performance (252-day seed):
- Model Directional Accuracy: ${backtest?.metrics?.model?.accuracy ? (backtest.metrics.model.accuracy * 100).toFixed(2) + '%' : 'N/A'}
- Baseline Majority Accuracy: ${backtest?.metrics?.baseline?.accuracy ? (backtest.metrics.baseline.accuracy * 100).toFixed(2) + '%' : 'N/A'}
- Model Brier Score: ${backtest?.metrics?.model?.brier_score || 'N/A'} (Baseline: ${backtest?.metrics?.baseline?.brier_score || 'N/A'})
- Model Log Loss: ${backtest?.metrics?.model?.log_loss || 'N/A'}

Latest Session Prediction:
- Date: ${prediction?.session_date} -> Target: ${prediction?.target_date}
- Predicted Direction: ${prediction?.predicted_direction}
- Probability P(Up): ${prediction?.p_up ? (prediction.p_up * 100).toFixed(2) + '%' : 'N/A'}
- Top Feature Drivers: ${JSON.stringify(prediction?.feature_attributions?.slice(0, 4) || [])}

Provide a concise, educational quantitative research briefing with 3 focused sections:
1. "Statistical Predictability Reality": Explain why next-day market direction is near 50/50, what the Brier score reveals about probability calibration, and why a ~51% directional hit rate is economically meaningful yet fragile.
2. "Regime & Feature Attribution": Explain what the current dominant signals (e.g. RSI, moving average distance, volatility) indicate about market tension.
3. "Risk & Uncertainty Disclaimer": Detail why educational models must not be used for live capital allocation without transaction costs, slippage, and regime changes.

Tone: Rigorous, institutional, mathematical, zero hype.`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: prompt,
      });

      res.json({
        explanation: response.text || 'Unable to generate analysis at this time.',
        generated_at: new Date().toISOString(),
      });
    } catch (err: any) {
      console.error('Gemini explanation error:', err.message);
      res.status(500).json({ error: 'Failed to generate quantitative explanation', details: err.message });
    }
  });

  // 3. Vite development middleware or production static serving
  const isProd = process.env.NODE_ENV === 'production';
  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  const server = app.listen(PORT, '0.0.0.0', () => {
    console.log(`[NIFTY Sentinel Gateway] Listening on http://0.0.0.0:${PORT}`);
  });

  // Clean shutdown
  const shutdown = () => {
    console.log('Shutting down services...');
    if (pythonProcess) {
      pythonProcess.kill();
    }
    server.close(() => {
      process.exit(0);
    });
  };

  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);
}

startServer().catch((err) => {
  console.error('Fatal server startup error:', err);
  process.exit(1);
});
