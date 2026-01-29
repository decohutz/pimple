import React from "react";
import { NavLink, Outlet } from "react-router-dom";
import { cn } from "../lib/utils";
import {
  LayoutDashboard,
  Database,
  Microscope,
  History as HistoryIcon,
  Info
} from "lucide-react";

function NavItem({ to, icon, label }: { to: string; icon: React.ReactNode; label: string }) {
  return (
    <NavLink
      to={to}
      end={to === "/"}
      className={({ isActive }) =>
        cn(
          "flex items-center gap-2 rounded-xl px-3 py-2 text-sm transition",
          isActive ? "bg-blue-600 text-white" : "text-slate-600 hover:bg-slate-100"
        )
      }
    >
      <span className="inline-flex items-center justify-center">{icon}</span>
      <span className="font-medium">{label}</span>
    </NavLink>
  );
}

export default function AppShell() {
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 bg-white/80 backdrop-blur border-b border-slate-200">
        <div className="mx-auto max-w-6xl px-4 py-3 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-2xl bg-blue-600 flex items-center justify-center text-white font-bold">
              <Microscope size={18} />
            </div>
            <div className="leading-tight">
              <div className="font-semibold">Pimple</div>
              <div className="text-xs text-slate-500">Skin Lesion Analysis</div>
            </div>
          </div>

          <nav className="flex items-center gap-2">
            <NavItem to="/" icon={<LayoutDashboard size={16} />} label="Dashboard" />
            <NavItem to="/dataset" icon={<Database size={16} />} label="Dataset" />
            <NavItem to="/inference" icon={<Microscope size={16} />} label="Inferência" />
            <NavItem to="/history" icon={<HistoryIcon size={16} />} label="Histórico" />
            <NavItem to="/about" icon={<Info size={16} />} label="Sobre" />
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-8">
        <Outlet />
      </main>
    </div>
  );
}
