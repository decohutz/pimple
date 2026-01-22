// src/services/api.ts
import { AnalysisResult, Sex } from '../types/analysis';

const API_BASE_URL = 'http://localhost:8000';

interface AnalyzeParams {
  file: File;
  sex: Sex;
  localization: string;
  age?: number | null;
}

export async function analyzeSkin(params: AnalyzeParams): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append('image', params.file);
  formData.append('sex', params.sex);
  formData.append('localization', params.localization);

  if (params.age !== undefined && params.age !== null && !Number.isNaN(params.age)) {
    formData.append('age', String(params.age));
  }

  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(
      `Erro ao chamar API: ${response.status} ${response.statusText} - ${text}`,
    );
  }

  return response.json();
}
