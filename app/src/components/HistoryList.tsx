// src/components/HistoryList.tsx
import React from 'react';
import { StoredAnalysis } from '../types/analysis';
import RiskBadge from './RiskBadge';

interface Props {
  items: StoredAnalysis[];
  onSelect: (item: StoredAnalysis) => void;
}

const HistoryList: React.FC<Props> = ({ items, onSelect }) => {
  if (items.length === 0) {
    return (
      <p className="text-sm text-slate-400">
        Nenhuma análise anterior ainda. Quando você enviar uma foto, ela aparece aqui.
      </p>
    );
  }

  return (
    <ul className="space-y-3">
      {items.map((item) => (
        <li
          key={item.id}
          className="border border-slate-800 rounded-lg p-3 hover:border-slate-600 cursor-pointer transition-colors"
          onClick={() => onSelect(item)}
        >
          <div className="flex justify-between items-center">
            <div>
              <div className="text-xs text-slate-400">
                {new Date(item.analyzed_at).toLocaleString()}
              </div>
              <div className="text-sm mt-1">
                {item.echo.sex} — {item.echo.localization}
                {item.echo.age !== null && ` — ${item.echo.age} anos`}
              </div>
            </div>
            <RiskBadge level={item.risk_level} />
          </div>
        </li>
      ))}
    </ul>
  );
};

export default HistoryList;
