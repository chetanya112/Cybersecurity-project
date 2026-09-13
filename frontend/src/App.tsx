import { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { Dashboard } from './pages/Dashboard';
import { Replay } from './pages/Replay';

export type ViewMode = 'dashboard' | 'replay';
export type DataSource = 'dataset' | 'upload';

function App() {
  const [viewMode, setViewMode] = useState<ViewMode>('dashboard');
  const [dataSource, setDataSource] = useState<DataSource>('dataset');

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar 
        viewMode={viewMode}
        setViewMode={setViewMode}
        dataSource={dataSource}
        setDataSource={setDataSource}
      />
      <main id="main-scroll-container" className="flex-1 overflow-y-auto relative">
        {viewMode === 'dashboard' ? (
          <Dashboard dataSource={dataSource} />
        ) : (
          <Replay />
        )}
      </main>
    </div>
  );
}

export default App;
