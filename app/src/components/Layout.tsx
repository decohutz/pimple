// src/components/Layout.tsx
import React from 'react';

interface LayoutProps {
  title: string;
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ title, children }) => {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 flex justify-center px-4 py-8">
      <div className="w-full max-w-3xl">
        <header className="mb-6">
          <h1 className="text-2xl font-semibold mb-1">pimple</h1>
          <p className="text-sm text-slate-400">{title}</p>
          <hr className="border-slate-800 mt-4" />
        </header>

        <main>{children}</main>

        <footer className="mt-8 text-xs text-slate-500">
          Este aplicativo é apenas para fins educacionais e não substitui avaliação
          médica profissional.
        </footer>
      </div>
    </div>
  );
};

export default Layout;
