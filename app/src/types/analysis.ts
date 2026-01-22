// src/types/analysis.ts

export type Sex = 'male' | 'female' | 'unknown';

export interface AnalysisEcho {
  sex: Sex;
  localization: string;
  age: number | null;
}

export type RiskLevel = 'BAIXO_RISCO' | 'INCERTO' | 'ALTO_RISCO';

export interface AnalysisResult {
  prob_malignant: number;
  risk_level: RiskLevel;
  threshold: number;
  echo: AnalysisEcho;
  analyzed_at: string;
  disclaimer: string;
}

export interface StoredAnalysis extends AnalysisResult {
  id: string;
}
