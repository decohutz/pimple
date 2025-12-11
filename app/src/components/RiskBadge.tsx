// src/components/RiskBadge.tsx
import React from 'react';
import { RiskLevel } from '../types/analysis';

interface Props {
  level: RiskLevel;
}

const RiskBadge: React.FC<Props> = ({ level }) => {
  let label = '';
  let className =
    'inline-flex px-3 py-1 rounded-full text-xs font-semibold border';

  if (level === 'BAIXO_RISCO') {
    label = 'Baixo risco';
    className += ' border-emerald-500 text-emerald-400';
  } else if (level === 'ALTO_RISCO') {
    label = 'Alto risco';
    className += ' border-red-500 text-red-400';
  } else {
    label = 'Incerteza / risco intermediário';
    className += ' border-amber-500 text-amber-400';
  }

  return <span className={className}>{label}</span>;
};

export default RiskBadge;
