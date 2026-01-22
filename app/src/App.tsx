import React, { FormEvent, ReactNode, useEffect, useState } from 'react';

// ====== Tipos ======
type Sex = 'male' | 'female' | 'unknown';
type RiskLevel = 'BAIXO_RISCO' | 'INCERTO' | 'ALTO_RISCO';

interface AnalysisEcho {
  sex: Sex;
  localization: string;
  age: number | null;
}

interface AnalysisResult {
  prob_malignant: number;
  risk_level: RiskLevel;
  threshold: number;
  echo: AnalysisEcho;
  analyzed_at: string;
  disclaimer: string;
}

interface StoredAnalysis extends AnalysisResult {
  id: string;
}

type Step = 'home' | 'form' | 'loading' | 'result';

const LOCAL_STORAGE_KEY = 'pimple_history_v1';
const API_BASE_URL = 'http://localhost:8000';

// ====== helpers ======

function generateId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function callAnalyzeSkin(
  file: File,
  sex: Sex,
  localization: string,
  age: number | null,
): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('sex', sex);
  formData.append('localization', localization);
  if (age !== null) {
    formData.append('age', String(age));
  }

  const res = await fetch(`${API_BASE_URL}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(
      `Erro ao chamar API: ${res.status} ${res.statusText} - ${text}`,
    );
  }

  return res.json();
}

// ====== Layout básico ======

interface PageProps {
  title: string;
  subtitle: string;
  children: ReactNode;
}

const Page: React.FC<PageProps> = ({ title, subtitle, children }) => {
  return (
    <div className="app">
      <header className="app-header">
        <div className="container">
          <div className="app-brand">pimple</div>
          <div className="app-subtitle">{subtitle}</div>
        </div>
      </header>

      <main className="app-main">
        <div className="container">
          <h1 className="page-title">{title}</h1>
          {children}
        </div>
      </main>

      <footer className="app-footer">
        <div className="container">
          Este aplicativo é apenas para fins educacionais e não substitui
          avaliação médica profissional.
        </div>
      </footer>
    </div>
  );
};

const RiskBadge: React.FC<{ level: RiskLevel }> = ({ level }) => {
  let label = '';
  let cls = 'badge ';

  if (level === 'BAIXO_RISCO') {
    label = 'Baixo risco';
    cls += 'badge-low';
  } else if (level === 'ALTO_RISCO') {
    label = 'Alto risco';
    cls += 'badge-high';
  } else {
    label = 'Incerteza / risco intermediário';
    cls += 'badge-mid';
  }

  return <span className={cls}>{label}</span>;
};

// ====== Componente principal ======

const App: React.FC = () => {
  const [step, setStep] = useState<Step>('home');

  const [history, setHistory] = useState<StoredAnalysis[]>(() => {
    try {
      if (typeof window === 'undefined') return [];
      const raw = window.localStorage.getItem(LOCAL_STORAGE_KEY);
      if (!raw) return [];
      return JSON.parse(raw) as StoredAnalysis[];
    } catch {
      return [];
    }
  });

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [sex, setSex] = useState<Sex>('unknown');
  const [localization, setLocalization] = useState<string>('unknown');
  const [ageInput, setAgeInput] = useState<string>('');

  const [currentResult, setCurrentResult] = useState<StoredAnalysis | null>(
    null,
  );
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    try {
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(
          LOCAL_STORAGE_KEY,
          JSON.stringify(history),
        );
      }
    } catch {
      // ignore
    }
  }, [history]);

  function handleStartNew() {
    setSelectedFile(null);
    setSex('unknown');
    setLocalization('unknown');
    setAgeInput('');
    setCurrentResult(null);
    setErrorMessage(null);
    setStep('form');
  }

  function parseAge(): number | null {
    const trimmed = ageInput.trim();
    if (trimmed === '') return null;
    const n = Number(trimmed);
    return Number.isNaN(n) ? null : n;
  }

  async function handleSubmitForm(e: FormEvent) {
    e.preventDefault();
    setErrorMessage(null);

    if (!selectedFile) {
      setErrorMessage('Por favor, selecione uma imagem antes de continuar.');
      return;
    }

    const age = parseAge();
    if (ageInput.trim() !== '' && age === null) {
      setErrorMessage('Idade inválida. Use apenas números.');
      return;
    }

    try {
      setStep('loading');
      const result = await callAnalyzeSkin(
        selectedFile,
        sex,
        localization,
        age,
      );
      const stored: StoredAnalysis = {
        id: generateId(),
        ...result,
      };
      setHistory((prev) => [stored, ...prev]);
      setCurrentResult(stored);
      setStep('result');
    } catch (err: any) {
      console.error(err);
      setErrorMessage(
        err?.message ?? 'Erro inesperado ao analisar a imagem.',
      );
      setStep('form');
    }
  }

  function handleSelectFromHistory(item: StoredAnalysis) {
    setCurrentResult(item);
    setErrorMessage(null);
    setStep('result');
  }

  // ====== Render ======

  if (step === 'home') {
    return (
      <Page title="Histórico" subtitle="Resumo das análises">
        <div className="card card-header-row">
          <div className="card-title">Histórico</div>
          <button className="btn btn-primary" onClick={handleStartNew}>
            Nova análise
          </button>
        </div>

        {history.length === 0 ? (
          <div className="card">
            <p className="muted">
              Nenhuma análise anterior ainda. Quando você enviar uma foto, ela
              aparece aqui.
            </p>
          </div>
        ) : (
          <div className="card">
            <ul className="history-list">
              {history.map((item) => (
                <li
                  key={item.id}
                  className="history-item"
                  onClick={() => handleSelectFromHistory(item)}
                >
                  <div className="history-item-main">
                    <div className="history-date">
                      {new Date(item.analyzed_at).toLocaleString()}
                    </div>
                    <div className="history-summary">
                      {item.echo.sex} — {item.echo.localization}
                      {item.echo.age !== null &&
                        ` — ${item.echo.age} anos`}
                    </div>
                  </div>
                  <RiskBadge level={item.risk_level} />
                </li>
              ))}
            </ul>
          </div>
        )}
      </Page>
    );
  }

  if (step === 'loading') {
    return (
      <Page title="Analisando imagem..." subtitle="Processando seus dados">
        <div className="card">
          <p>
            Estamos processando sua foto e os dados informados. Isso pode levar
            alguns segundos.
          </p>
          <p className="muted">
            Esta etapa é apenas uma simulação. No futuro, aqui entra o modelo
            real de IA.
          </p>
        </div>
      </Page>
    );
  }

  if (step === 'result' && currentResult) {
    return (
      <Page title="Resultado da análise" subtitle="Veja o resumo da avaliação">
        <div className="card">
          <div className="result-header">
            <div>
              <div className="history-date">
                {new Date(currentResult.analyzed_at).toLocaleString()}
              </div>
              <div className="history-summary">
                {currentResult.echo.sex} — {currentResult.echo.localization}
                {currentResult.echo.age !== null &&
                  ` — ${currentResult.echo.age} anos`}
              </div>
            </div>
            <RiskBadge level={currentResult.risk_level} />
          </div>

          <div className="result-prob">
            Probabilidade estimada de malignidade:{' '}
            <strong>
              {(currentResult.prob_malignant * 100).toFixed(1)}%
            </strong>{' '}
            (threshold: {Math.round(currentResult.threshold * 100)}%)
          </div>

          <p className="muted result-disclaimer">
            {currentResult.disclaimer}
          </p>
        </div>

        <div className="button-row">
          <button className="btn btn-primary" onClick={handleStartNew}>
            Fazer nova análise
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => setStep('home')}
          >
            Voltar para o histórico
          </button>
        </div>
      </Page>
    );
  }

  // FORM
  return (
    <Page
      title="Nova análise"
      subtitle="Envie uma foto e responda ao questionário"
    >
      <form className="card form" onSubmit={handleSubmitForm}>
        <div className="form-group">
          <label className="form-label">Foto da lesão</label>
          <input
            type="file"
            accept="image/jpeg,image/png"
            onChange={(e) => {
              const file = e.target.files?.[0] ?? null;
              setSelectedFile(file);
            }}
          />
          <p className="muted">
            Use uma foto nítida, bem iluminada e próxima da área de interesse.
          </p>
        </div>

        <div className="form-grid">
          <div className="form-group">
            <label className="form-label">Sexo</label>
            <select
              value={sex}
              onChange={(e) => setSex(e.target.value as Sex)}
              className="form-control"
            >
              <option value="unknown">Prefiro não informar</option>
              <option value="male">Masculino</option>
              <option value="female">Feminino</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Localização</label>
            <select
              value={localization}
              onChange={(e) => setLocalization(e.target.value)}
              className="form-control"
            >
              <option value="unknown">Não sei / outro</option>
              <option value="face">Face</option>
              <option value="scalp">Couro cabeludo</option>
              <option value="back">Costas</option>
              <option value="chest">Tórax</option>
              <option value="upper extremity">Membro superior</option>
              <option value="lower extremity">Membro inferior</option>
              <option value="abdomen">Abdômen</option>
              <option value="ear">Orelha</option>
              <option value="hand">Mão</option>
              <option value="foot">Pé</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Idade (opcional)</label>
            <input
              type="number"
              min={0}
              max={120}
              value={ageInput}
              onChange={(e) => setAgeInput(e.target.value)}
              className="form-control"
              placeholder="Ex: 32"
            />
          </div>
        </div>

        {errorMessage && (
          <div className="form-error">{errorMessage}</div>
        )}

        <div className="button-row">
          <button type="submit" className="btn btn-primary">
            Analisar imagem
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => setStep('home')}
          >
            Cancelar
          </button>
        </div>
      </form>
    </Page>
  );
};

export default App;
