import { useState, useEffect } from 'react';
import { 
  ChevronLeft, ChevronRight, LayoutDashboard, 
  Database, Upload, Play, Activity, GitCommit, FileSearch, Shield
} from 'lucide-react';
import type { ViewMode, DataSource } from '../App';

interface SidebarProps {
  viewMode: ViewMode;
  setViewMode: (v: ViewMode) => void;
  dataSource: DataSource;
  setDataSource: (d: DataSource) => void;
}

export function Sidebar({ viewMode, setViewMode, dataSource, setDataSource }: SidebarProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [activeSection, setActiveSection] = useState('overview');

  const toggle = () => setIsExpanded(!isExpanded);

  useEffect(() => {
    if (viewMode !== 'dashboard') return;

    const handleScroll = () => {
      const sections = ['overview', 'forecast', 'trajectory', 'explainability', 'mitre'];
      for (let i = sections.length - 1; i >= 0; i--) {
        const el = document.getElementById(sections[i]);
        if (el) {
          const rect = el.getBoundingClientRect();
          if (rect.top <= 200) {
            setActiveSection(sections[i]);
            break;
          }
        }
      }
    };
    
    const mainEl = document.getElementById('main-scroll-container');
    if (mainEl) {
      mainEl.addEventListener('scroll', handleScroll);
      handleScroll(); // init
      return () => mainEl.removeEventListener('scroll', handleScroll);
    }
  }, [viewMode]);

  const scrollToSection = (id: string) => {
    setViewMode('dashboard');
    setTimeout(() => {
      const el = document.getElementById(id);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 100);
  };

  return (
    <>
      <div className={`md:hidden fixed inset-0 bg-black/50 z-40 transition-opacity ${isExpanded ? 'opacity-100' : 'opacity-0 pointer-events-none'}`} onClick={() => setIsExpanded(false)} />
      
      <div className={`absolute md:relative flex flex-col bg-surface border-r border-border transition-all duration-300 z-50 h-full 
        ${isExpanded ? 'w-[280px] translate-x-0' : 'w-[280px] -translate-x-full md:w-[72px] md:translate-x-0'}`}
      >
        
        {/* Toggle Button */}
        <button 
          onClick={toggle}
          className="absolute -right-4 top-6 bg-surface border border-border rounded-full p-1 text-muted hover:text-text hover:bg-border transition-colors z-50 shadow-lg hidden md:block"
        >
          {isExpanded ? <ChevronLeft size={18} /> : <ChevronRight size={18} />}
        </button>

        {/* Header */}
        <div className={`p-4 flex flex-col items-center justify-center ${isExpanded ? 'h-32' : 'h-20'}`}>
          <img 
            src="/assets/chronex-logo-clean.png" 
            alt="CHRONEX"
            className={`transition-all object-contain ${isExpanded ? 'w-[140px] mb-2' : 'w-[40px]'}`}
            onError={(e) => {
              e.currentTarget.style.display = 'none';
              e.currentTarget.parentElement!.innerHTML = `<div class="font-bold text-text ${isExpanded ? 'text-2xl' : 'text-xs'}">CHRONEX</div>`;
            }}
          />
          {isExpanded && (
            <div className="text-[11px] text-muted tracking-wide text-center">
              Network Threat Intelligence
            </div>
          )}
        </div>
        
        {isExpanded && (
          <div className="px-4 text-center pb-4">
            <div className="text-[11px] text-success font-semibold tracking-[1.5px] mb-1 flex items-center justify-center gap-2">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-success opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-success"></span>
              </span>
              ENGINE ONLINE
            </div>
            <div className="text-[10px] text-muted tracking-wider">LOCAL / OFFLINE</div>
          </div>
        )}
        
        <div className="flex-1 overflow-y-auto px-3 space-y-6 mt-2">
          {/* MODE */}
          <div>
            {isExpanded && <div className="text-[10px] font-bold tracking-[2px] text-muted mb-2 px-2 uppercase">DATA SOURCE</div>}
            <div className="space-y-1">
              <SidebarItem 
                icon={<Database size={18} />} 
                label="Dataset" 
                isActive={viewMode === 'dashboard' && dataSource === 'dataset'}
                isExpanded={isExpanded}
                onClick={() => { setViewMode('dashboard'); setDataSource('dataset'); }}
              />
              <SidebarItem 
                icon={<Upload size={18} />} 
                label="Upload CSV" 
                isActive={viewMode === 'dashboard' && dataSource === 'upload'}
                isExpanded={isExpanded}
                onClick={() => { setViewMode('dashboard'); setDataSource('upload'); }}
              />
            </div>
          </div>

          <hr className="border-border" />

          {/* ANALYSIS */}
          <div>
            {isExpanded && <div className="text-[10px] font-bold tracking-[2px] text-muted mb-2 px-2 uppercase">ANALYSIS</div>}
            <div className="space-y-1">
              <SidebarItem icon={<LayoutDashboard size={18} />} label="Overview" isExpanded={isExpanded} isActive={viewMode === 'dashboard' && activeSection === 'overview'} onClick={() => { scrollToSection('overview'); if(window.innerWidth < 768) setIsExpanded(false); }} />
              <SidebarItem icon={<Activity size={18} />} label="Forecast" isExpanded={isExpanded} isActive={viewMode === 'dashboard' && activeSection === 'forecast'} onClick={() => { scrollToSection('forecast'); if(window.innerWidth < 768) setIsExpanded(false); }} />
              <SidebarItem icon={<GitCommit size={18} />} label="Trajectory" isExpanded={isExpanded} isActive={viewMode === 'dashboard' && activeSection === 'trajectory'} onClick={() => { scrollToSection('trajectory'); if(window.innerWidth < 768) setIsExpanded(false); }} />
              <SidebarItem icon={<FileSearch size={18} />} label="Explainability" isExpanded={isExpanded} isActive={viewMode === 'dashboard' && activeSection === 'explainability'} onClick={() => { scrollToSection('explainability'); if(window.innerWidth < 768) setIsExpanded(false); }} />
              <SidebarItem icon={<Shield size={18} />} label="MITRE" isExpanded={isExpanded} isActive={viewMode === 'dashboard' && activeSection === 'mitre'} onClick={() => { scrollToSection('mitre'); if(window.innerWidth < 768) setIsExpanded(false); }} />
            </div>
          </div>

          <hr className="border-border" />

          {/* REPLAY */}
          <div>
            {isExpanded && <div className="text-[10px] font-bold tracking-[2px] text-muted mb-2 px-2 uppercase">PRESENTATION</div>}
            <div className="space-y-1">
              <SidebarItem 
                icon={<Play size={18} />} 
                label="SIH Demo" 
                isActive={viewMode === 'replay'}
                isExpanded={isExpanded}
                onClick={() => { setViewMode('replay'); if(window.innerWidth < 768) setIsExpanded(false); }}
                highlight
              />
            </div>
          </div>
        </div>
        
      </div>

      {/* Mobile persistent open button */}
      <button 
        onClick={() => setIsExpanded(true)}
        className={`md:hidden fixed left-0 top-6 bg-surface border border-l-0 border-border rounded-r-md p-2 text-muted hover:text-text z-40 shadow-lg transition-transform ${isExpanded ? '-translate-x-full' : 'translate-x-0'}`}
      >
        <ChevronRight size={18} />
      </button>
    </>
  );
}

function SidebarItem({ icon, label, isExpanded, isActive, onClick, highlight, disabled }: any) {
  return (
    <div className="relative group">
      <button
        onClick={onClick}
        disabled={disabled}
        className={`w-full flex items-center ${isExpanded ? 'px-3' : 'justify-center'} py-2 rounded-md transition-colors 
          ${isActive ? (highlight ? 'bg-primary/20 text-primary' : 'text-primary') : 'text-textMuted hover:bg-border/50 hover:text-text'}
          ${disabled ? 'opacity-50 cursor-not-allowed hover:bg-transparent' : ''}
        `}
      >
        <div className="relative flex items-center">
          {isActive && !highlight && (
            <div className={`absolute ${isExpanded ? '-left-3' : '-left-5'} w-1 h-5 bg-primary rounded-r-full shadow-[0_0_8px_rgba(14,165,233,0.5)]`} />
          )}
          <div className={isActive ? 'text-primary' : ''}>
            {icon}
          </div>
        </div>
        {isExpanded && (
          <span className={`ml-3 text-sm font-medium ${isActive ? 'text-text' : ''}`}>
            {label}
          </span>
        )}
      </button>

      {/* Tooltip for collapsed mode */}
      {!isExpanded && (
        <div className="absolute left-full ml-3 top-1/2 -translate-y-1/2 px-2 py-1 bg-surface border border-border text-text text-xs rounded-md shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all whitespace-nowrap z-[100]">
          {label}
        </div>
      )}
    </div>
  );
}
