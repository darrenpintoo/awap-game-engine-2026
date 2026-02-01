import { useState, useEffect, useMemo, useRef } from 'react';
import {
  XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, AreaChart, Area, BarChart, Bar
} from 'recharts';
import {
  TrendingUp,
  Search, Zap, Users,
  Map as MapIcon, Home,
  Filter, BarChart3, Activity, List, Shield, Sword,
  Trophy, Flame, Clock, Award, DollarSign, Upload, Database
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';


// --- Types ---
interface Match {
  red_name: string;
  blue_name: string;
  map_name: string;
  map_type: 'official' | 'test';
  winner: string;
  duration: number;
  red_score: number;
  blue_score: number;
  timestamp: number;
}

interface MetaData {
  total_matches: number;
  completed_matches: number;
  last_updated: number;
}

interface Data {
  metadata: MetaData;
  matches: Match[];
}

interface BotStats {
  w: number;
  l: number;
  d: number;
  e: number;
  total: number;
  score: number;
  scoreDiff: number;
  scores: number[];
  opponents: Record<string, { w: number; l: number; total: number; scoreDiff: number }>;
  maps: Record<string, { w: number; l: number; d: number; total: number; score: number; scoreDiff: number }>;
  nemesis?: string;
  victim?: string;
}

// --- Constants ---
// Static heroes list for display - will also detect any Hero_ prefixed bots dynamically
const STATIC_HEROES = ["Hero_UltimateChampion", "Hero_BEST^3", "Hero_Sovereign", "Hero_BEST-Hydra", "Hero_BEST^2", "Hero_Relay"];
const HERO_COLORS: Record<string, string> = {
  "Hero_UltimateChampion": "#3b82f6",
  "Hero_BEST^3": "#8b5cf6",
  "Hero_Sovereign": "#10b981",
  "Hero_BEST-Hydra": "#f59e0b",
  "Hero_BEST^2": "#ec4899",
  "Hero_Relay": "#06b6d4",
};

// --- Utils ---
const cleanName = (n: string) => n.replace("Hero_", "").replace(".py", "").replace(/_/g, " ");
const isHero = (n: string) => n.startsWith("Hero_");
const safeDiv = (n: number, d: number) => d > 0 ? (n / d) * 100 : 0;
const getHeroColor = (name: string) => HERO_COLORS[name] || "#6366f1";


function App() {
  const [data, setData] = useState<Data | null>(null);
  const [activeTab, setActiveTab] = useState<'home' | 'maps' | 'bots' | 'matches' | 'stats'>('home');
  const [selectedMap, setSelectedMap] = useState<string | null>(null);
  const [selectedBot, setSelectedBot] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [matchFilter, setMatchFilter] = useState({ bot: "", map: "", outcome: "all", sort: "new" });
  const [dataSource, setDataSource] = useState<'live' | 'file'>('live');
  const [fileName, setFileName] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handle file upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const json = JSON.parse(event.target?.result as string);
        setData(json);
        setDataSource('file');
        setFileName(file.name);
      } catch (err) {
        console.error('Invalid JSON file:', err);
        alert('Invalid JSON file. Please select a valid tournament data file.');
      }
    };
    reader.readAsText(file);
  };

  // Reset to live data
  const resetToLive = () => {
    setDataSource('live');
    setFileName('');
  };

  useEffect(() => {
    // Only auto-fetch if in live mode
    if (dataSource !== 'live') return;

    const fetchData = async () => {
      try {
        const res = await fetch('/data.json?' + new Date().getTime());
        const json = await res.json();
        setData(json);
      } catch (e) { console.error(e); }
    };
    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, [dataSource]);


  const { playerStats, timeline, mapStats, globalStats, highScores } = useMemo(() => {
    if (!data) return {
      playerStats: {} as Record<string, BotStats>,
      timeline: [] as any[],
      mapStats: {} as Record<string, any>,
      globalStats: { scores: [] as number[], durations: [] as number[], totalMoney: 0 },
      highScores: [] as Match[]
    };

    const s: Record<string, BotStats> = {};
    const tl: any[] = [];
    const ms: Record<string, any> = {};
    const gs = { scores: [] as number[], durations: [] as number[], totalMoney: 0 };

    const chronological = [...data.matches].sort((a, b) => a.timestamp - b.timestamp);
    const sortedByScore = [...data.matches].sort((a, b) => Math.max(b.red_score, b.blue_score) - Math.max(a.red_score, a.blue_score));

    chronological.forEach((m, idx) => {
      // Timeline - track cumulative win rate for heroes
      if (idx % 25 === 0 && idx > 0) {
        const snap: any = { name: idx };
        STATIC_HEROES.forEach((h: string) => {
          const hst = s[h];
          // Calculate win rate excluding errors
          const validGames = hst ? (hst.total - hst.e) : 0;
          snap[cleanName(h)] = validGames > 0 ? ((hst.w / validGames) * 100) : 0;
        });
        tl.push(snap);
      }

      // Metadata
      gs.durations.push(m.duration);
      gs.scores.push(m.red_score);
      gs.scores.push(m.blue_score);
      gs.totalMoney += (m.red_score + m.blue_score);

      // Map Tracking
      if (!ms[m.map_name]) ms[m.map_name] = { score: 0, count: 0, matches: [], botPerformances: {} };
      ms[m.map_name].count++;
      ms[m.map_name].score += (m.red_score + m.blue_score);
      ms[m.map_name].matches.push(m);

      [m.red_name, m.blue_name].forEach(p => {
        const score = p === m.red_name ? m.red_score : m.blue_score;
        if (!ms[m.map_name].botPerformances[p]) ms[m.map_name].botPerformances[p] = { score: 0, count: 0, wins: 0 };
        ms[m.map_name].botPerformances[p].score += score;
        ms[m.map_name].botPerformances[p].count++;
        if (m.winner === p) ms[m.map_name].botPerformances[p].wins++;
      });

      // Player Stats
      [m.red_name, m.blue_name].forEach(p => {
        if (!s[p]) s[p] = { w: 0, l: 0, d: 0, e: 0, total: 0, score: 0, scoreDiff: 0, scores: [], opponents: {}, maps: {} };
        const isMe = p === m.red_name;
        const opp = isMe ? m.blue_name : m.red_name;
        const score = isMe ? m.red_score : m.blue_score;
        const oppScore = isMe ? m.blue_score : m.red_score;
        const result = m.winner === p ? 'W' : (m.winner === 'DRAW' ? 'D' : (['ERROR', 'TIMEOUT'].includes(m.winner) ? 'E' : 'L'));

        s[p].total++;
        s[p].score += score;
        s[p].scores.push(score);
        s[p].scoreDiff += (score - oppScore);

        if (!s[p].opponents[opp]) s[p].opponents[opp] = { w: 0, l: 0, total: 0, scoreDiff: 0 };
        s[p].opponents[opp].total++;
        s[p].opponents[opp].scoreDiff += (score - oppScore);

        if (!s[p].maps[m.map_name]) s[p].maps[m.map_name] = { w: 0, l: 0, d: 0, total: 0, score: 0, scoreDiff: 0 };
        s[p].maps[m.map_name].total++;
        s[p].maps[m.map_name].score += score;
        s[p].maps[m.map_name].scoreDiff += (score - oppScore);

        if (result === 'W') { s[p].w++; s[p].opponents[opp].w++; s[p].maps[m.map_name].w++; }
        else if (result === 'D') { s[p].d++; s[p].maps[m.map_name].d++; }
        else if (result === 'E') { s[p].e++; }
        else { s[p].l++; s[p].opponents[opp].l++; s[p].maps[m.map_name].l++; }
      });
    });

    Object.keys(s).forEach(p => {
      const oppArr = Object.entries(s[p].opponents);
      if (oppArr.length > 0) {
        s[p].nemesis = oppArr.sort((a, b) => a[1].scoreDiff - b[1].scoreDiff)[0][0];
        s[p].victim = oppArr.sort((a, b) => b[1].scoreDiff - a[1].scoreDiff)[0][0];
      }
    });

    return { playerStats: s, timeline: tl, mapStats: ms, globalStats: gs, highScores: sortedByScore.slice(0, 50) };
  }, [data]);

  const filteredMatches = useMemo(() => {
    if (!data) return [];
    let matches = data.matches.filter(m => {
      const botMatch = !matchFilter.bot || m.red_name.toLowerCase().includes(matchFilter.bot.toLowerCase()) || m.blue_name.toLowerCase().includes(matchFilter.bot.toLowerCase());
      const mapMatch = !matchFilter.map || m.map_name.toLowerCase().includes(matchFilter.map.toLowerCase());
      const outcomeMatch = matchFilter.outcome === 'all' ||
        (matchFilter.outcome === 'win' && m.winner !== 'DRAW' && m.winner !== 'ERROR') ||
        (matchFilter.outcome === 'draw' && m.winner === 'DRAW') ||
        (matchFilter.outcome === 'error' && (m.winner === 'ERROR' || m.winner === 'TIMEOUT'));
      return botMatch && mapMatch && outcomeMatch;
    });

    if (matchFilter.sort === 'score') {
      matches.sort((a, b) => Math.max(b.red_score, b.blue_score) - Math.max(a.red_score, a.blue_score));
    } else {
      matches.sort((a, b) => b.timestamp - a.timestamp);
    }

    return matches.slice(0, 500);
  }, [data, matchFilter]);

  if (!data) return <div className="flex h-screen items-center justify-center text-indigo-400 font-bold bg-[#020617]">Syncing Data Stream...</div>;

  const progress = (data.metadata.completed_matches / data.metadata.total_matches) * 100;
  const top10Bots = Object.entries(playerStats).sort((a, b) => safeDiv(b[1].w, b[1].total - b[1].e) - safeDiv(a[1].w, a[1].total - a[1].e)).slice(0, 10);

  return (
    <div className="flex h-screen bg-[#020617] text-slate-200 overflow-hidden font-sans">
      {/* Sidebar Navigation */}
      <nav className="w-20 border-r border-white/5 bg-[#0a0f1d] flex flex-col items-center py-8 gap-6 shrink-0 z-50">
        <div className="w-12 h-12 bg-indigo-600 rounded-2xl flex items-center justify-center text-white shadow-lg"><Award size={24} /></div>
        <div className="flex flex-col gap-3">
          <NavIcon active={activeTab === 'home'} onClick={() => setActiveTab('home')} icon={<Home size={20} />} label="Arena Home" />
          <NavIcon active={activeTab === 'bots'} onClick={() => setActiveTab('bots')} icon={<Users size={20} />} label="Bot Rankings" />
          <NavIcon active={activeTab === 'maps'} onClick={() => setActiveTab('maps')} icon={<MapIcon size={20} />} label="Map Analysis" />
          <NavIcon active={activeTab === 'matches'} onClick={() => setActiveTab('matches')} icon={<List size={20} />} label="Match Records" />
          <NavIcon active={activeTab === 'stats'} onClick={() => setActiveTab('stats')} icon={<BarChart3 size={20} />} label="Global Trends" />
        </div>
      </nav>

      {/* Main Viewport */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <header className="h-16 border-b border-white/5 px-8 flex items-center justify-between bg-[#0f172a]/20 backdrop-blur-xl shrink-0">
          <div className="flex items-center gap-6">
            <h1 className="text-xl font-black uppercase tracking-tighter">Arena <span className="text-indigo-500">Analytics</span></h1>
            <div className="h-4 w-[1px] bg-white/10" />
            <div className="flex gap-6 text-[10px] font-black uppercase text-slate-500 tracking-widest">
              <span className="flex items-center gap-2"><Activity size={12} className="text-indigo-500" /> Runtime: <span className="text-white font-mono">{data.metadata.completed_matches} Matches</span></span>
              <span>Total Yield: <span className="text-emerald-400 font-mono">${(globalStats.totalMoney / 1000).toFixed(1)}k</span></span>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {/* Data Source Controls */}
            <div className="flex items-center gap-2">
              {dataSource === 'file' ? (
                <div className="flex items-center gap-2 px-3 py-1.5 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                  <Database size={12} className="text-amber-400" />
                  <span className="text-[10px] font-bold text-amber-300">{fileName}</span>
                  <button
                    onClick={resetToLive}
                    className="text-[10px] font-black text-amber-400 hover:text-amber-200 underline ml-2"
                  >
                    ← Live
                  </button>
                </div>
              ) : (
                <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
                  <Activity size={12} className="text-emerald-400 animate-pulse" />
                  <span className="text-[10px] font-bold text-emerald-300">Live Data</span>
                </div>
              )}
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                accept=".json"
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center gap-2 px-3 py-1.5 bg-indigo-500/10 border border-indigo-500/30 rounded-lg hover:bg-indigo-500/20 transition-colors"
              >
                <Upload size={12} className="text-indigo-400" />
                <span className="text-[10px] font-bold text-indigo-300">Load JSON</span>
              </button>
            </div>
            <div className="w-48 h-1.5 bg-white/5 rounded-full overflow-hidden border border-white/5">
              <motion.div initial={{ width: 0 }} animate={{ width: `${progress}%` }} className="h-full bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.5)]" />
            </div>
            <span className="text-[10px] font-black text-slate-600 font-mono">{progress.toFixed(1)}%</span>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto p-8 custom-scrollbar">
          <AnimatePresence mode="wait">
            {activeTab === 'home' && (
              <motion.div key="home" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-8">
                {/* Hero Power Widgets */}
                <div className="grid grid-cols-4 gap-6">
                  {STATIC_HEROES.filter(h => playerStats[h]).map((h: string) => <HeroDetailCard key={h} name={h} stats={playerStats[h]} />)}
                  <div className="bg-indigo-600/10 border border-indigo-500/20 p-8 rounded-[40px] flex flex-col justify-center items-center text-center">
                    <Flame size={32} className="text-indigo-400 mb-4" />
                    <h3 className="text-lg font-black text-white uppercase leading-none">Gauntlet Pulse</h3>
                    <p className="text-[10px] text-indigo-300 font-bold uppercase tracking-widest mt-2">{((data.metadata.completed_matches / ((Date.now() - data.metadata.last_updated * 1000) / 1000) || 1)).toFixed(1)} msg/sec</p>
                  </div>
                </div>

                {/* Main Home Grid */}
                <div className="grid grid-cols-3 gap-8">
                  {/* Left Column: Top 10 Leaderboard */}
                  <div className="col-span-1 bg-white/[0.02] border border-white/5 rounded-[40px] overflow-hidden">
                    <div className="p-6 border-b border-white/5 flex justify-between items-center bg-white/[0.01]">
                      <h3 className="text-[10px] font-black uppercase text-slate-500 tracking-widest flex items-center gap-2"><Trophy size={14} className="text-yellow-500" /> Apex Leaderboard</h3>
                      <span className="text-[9px] font-black text-slate-600 uppercase">Top 10 bots</span>
                    </div>
                    <div className="p-4 space-y-1">
                      {top10Bots.map(([n, s], i) => (
                        <div key={n} className="flex items-center gap-3 p-3 rounded-2xl hover:bg-white/[0.03] transition-colors group">
                          <span className="text-[10px] font-black text-slate-600 w-4">#{i + 1}</span>
                          <div className="flex-1">
                            <div className={`text-[11px] font-bold ${isHero(n) ? 'text-indigo-400' : 'text-slate-300'}`}>{cleanName(n)}</div>
                            <div className="text-[9px] text-slate-600 font-bold uppercase">{s.total} Combat Runs</div>
                          </div>
                          <div className="text-right">
                            <div className="text-[11px] font-black text-emerald-500">{safeDiv(s.w, s.total - s.e).toFixed(1)}%</div>
                            <div className="text-[9px] font-black text-slate-700 uppercase">Win Rate</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Center/Right Column: Strategic Trends & High Scores */}
                  <div className="col-span-2 space-y-8">
                    {/* Dominance Chart */}
                    <div className="bg-white/[0.02] border border-white/5 rounded-[40px] p-8">
                      <div className="flex justify-between items-center mb-8">
                        <h3 className="text-[10px] font-black uppercase text-slate-500 tracking-widest flex items-center gap-2"><TrendingUp size={14} className="text-indigo-500" /> Hero Win Rate Over Time</h3>
                        <span className="text-[9px] text-slate-600 font-bold uppercase">Cumulative % as matches progress</span>
                      </div>
                      <div className="h-[280px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={timeline}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.02)" />
                            <XAxis dataKey="name" hide />
                            <YAxis domain={[0, 100]} tick={{ fontSize: 9, fill: '#475569' }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v}%`} width={40} />
                            <Tooltip contentStyle={{ background: '#020617', border: 'none', borderRadius: '12px', fontSize: '10px' }} formatter={(v) => [`${Number(v || 0).toFixed(1)}%`, '']} />
                            {STATIC_HEROES.filter(h => playerStats[h]).map((h: string) => (
                              <Area key={h} type="monotone" dataKey={cleanName(h)} stroke={HERO_COLORS[h]} fill={HERO_COLORS[h]} fillOpacity={0.03} strokeWidth={2.5} dot={false} />
                            ))}
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    {/* Hall of Fame: High Scoring Matches */}
                    <div className="bg-white/[0.02] border border-white/5 rounded-[40px] p-8">
                      <div className="flex justify-between items-center mb-6">
                        <h3 className="text-[10px] font-black uppercase text-slate-500 tracking-widest flex items-center gap-2"><Award size={14} className="text-emerald-500" /> Hall of Fame: Highest Scoring Games</h3>
                        <span className="text-[9px] font-black text-slate-600 uppercase">Top Performers</span>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        {highScores.slice(0, 10).map((m, i) => (
                          <div key={i} className="bg-white/[0.02] border border-white/5 p-4 rounded-3xl flex justify-between items-center group hover:border-emerald-500/30 transition-all">
                            <div className="flex items-center gap-3">
                              <span className="text-[10px] font-black text-slate-700">#{i + 1}</span>
                              <div className="flex flex-col">
                                <span className="text-[11px] font-black text-white">{cleanName(m.winner === 'DRAW' ? 'Tied' : m.winner)}</span>
                                <span className="text-[9px] text-slate-600 font-bold uppercase">{m.map_name.replace('maps/test-maps/', '').replace('.txt', '')}</span>
                              </div>
                            </div>
                            <div className="text-right">
                              <span className="text-lg font-black text-emerald-500 font-mono">${Math.max(m.red_score, m.blue_score)}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {activeTab === 'bots' && (
              <motion.div key="bots" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="flex gap-8 h-full overflow-hidden">
                <div className="w-80 flex flex-col gap-2 overflow-y-auto pr-2 custom-scrollbar shrink-0">
                  <div className="relative mb-4 px-2">
                    <Search className="absolute left-6 top-1/2 -translate-y-1/2 text-slate-600" size={14} />
                    <input type="text" placeholder="Global Rank Search..." className="w-full bg-white/5 border border-white/5 rounded-2xl px-12 py-3 text-xs focus:ring-1 ring-indigo-500/50 outline-none" value={searchTerm} onChange={e => setSearchTerm(e.target.value)} />
                  </div>
                  {Object.keys(playerStats)
                    .filter(n => n.toLowerCase().includes(searchTerm.toLowerCase()))
                    .sort((a, b) => safeDiv(playerStats[b].w, playerStats[b].total) - safeDiv(playerStats[a].w, playerStats[a].total))
                    .map((n, idx) => (
                      <button key={n} onClick={() => setSelectedBot(n)} className={`p-4 rounded-2xl border text-left transition-all ${selectedBot === n ? 'bg-indigo-600 border-indigo-500 shadow-xl' : 'bg-white/[0.02] border-white/5 hover:bg-white/5'}`}>
                        <div className="flex justify-between items-start">
                          <span className={`text-[10px] font-black uppercase ${selectedBot === n ? 'text-white' : 'text-slate-400'} truncate w-48`}>#{idx + 1} {cleanName(n)}</span>
                          {isHero(n) && <Zap size={10} className="text-yellow-400 shadow-[0_0_5px_yellow]" />}
                        </div>
                        <div className="flex justify-between mt-2 text-[9px] text-slate-500 font-bold uppercase tracking-widest">
                          <span>WR: {safeDiv(playerStats[n].w, playerStats[n].total).toFixed(1)}%</span>
                          <span className="font-mono">${(playerStats[n].score / playerStats[n].total).toFixed(0)} Avg</span>
                        </div>
                      </button>
                    ))}
                </div>
                <div className="flex-1 bg-white/[0.02] border border-white/5 rounded-[40px] p-8 overflow-y-auto custom-scrollbar">
                  {selectedBot ? <BotDeepDive name={selectedBot} stats={playerStats[selectedBot]} /> :
                    <div className="h-full flex flex-col items-center justify-center text-slate-700 gap-4 opacity-50">
                      <Users size={64} strokeWidth={1} />
                      <div className="text-[10px] font-black uppercase tracking-[0.3em]">Execute neural analysis on selected participant</div>
                    </div>
                  }
                </div>
              </motion.div>
            )}

            {activeTab === 'maps' && (
              <motion.div key="maps" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="flex gap-8 h-full overflow-hidden">
                <div className="w-80 flex flex-col gap-2 overflow-y-auto pr-2 custom-scrollbar shrink-0">
                  {Object.keys(mapStats).sort((a, b) => mapStats[b].score / mapStats[b].count - mapStats[a].score / mapStats[a].count).map((m, idx) => (
                    <button key={m} onClick={() => setSelectedMap(m)} className={`p-4 rounded-2xl border text-left transition-all ${selectedMap === m ? 'bg-emerald-600 border-emerald-500 shadow-xl' : 'bg-white/[0.02] border-white/5 hover:bg-white/5'}`}>
                      <div className="text-[10px] font-black uppercase text-white flex justify-between">
                        <span className="truncate w-44">#{idx + 1} {m.replace('maps/test-maps/', '').replace('.txt', '')}</span>
                        <span className="font-mono text-emerald-400">${(mapStats[m].score / mapStats[m].count / 2).toFixed(0)}</span>
                      </div>
                      <div className="text-[9px] text-slate-500 font-bold mt-1 uppercase tracking-widest">{mapStats[m].count} Total Combat Runs</div>
                    </button>
                  ))}
                </div>
                <div className="flex-1 bg-white/[0.01] border border-white/5 rounded-[40px] p-8 overflow-y-auto custom-scrollbar">
                  {selectedMap ? <MapDetailView name={selectedMap} data={mapStats[selectedMap]} /> :
                    <div className="h-full flex flex-col items-center justify-center text-slate-700 opacity-50"><MapIcon size={64} strokeWidth={1} /><div className="text-[10px] font-black mt-4 uppercase tracking-widest">Select Arena Blueprint</div></div>
                  }
                </div>
              </motion.div>
            )}

            {activeTab === 'matches' && (
              <motion.div key="matches" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-full flex flex-col">
                <div className="flex gap-4 mb-8">
                  <div className="flex-1 relative">
                    <Filter className="absolute left-5 top-1/2 -translate-y-1/2 text-slate-600" size={16} />
                    <input type="text" placeholder="Trace bot identifier in combat logs..." className="w-full bg-white/[0.02] border border-white/5 rounded-2xl px-14 py-4 text-xs focus:ring-1 ring-indigo-500/50 outline-none" onChange={e => setMatchFilter({ ...matchFilter, bot: e.target.value })} />
                  </div>
                  <select className="bg-[#0a0f1d] border border-white/10 rounded-2xl px-6 py-2 text-[10px] font-black uppercase text-slate-400 outline-none cursor-pointer hover:border-indigo-500 transition-colors" onChange={e => setMatchFilter({ ...matchFilter, outcome: e.target.value })}>
                    <option value="all">Analyze All Results</option>
                    <option value="win">Critical Victories</option>
                    <option value="draw">Tied Deadlocks</option>
                    <option value="error">Critical System Failures</option>
                  </select>
                  <button
                    onClick={() => setMatchFilter({ ...matchFilter, sort: matchFilter.sort === 'new' ? 'score' : 'new' })}
                    className={`px-6 py-2 rounded-2xl border text-[10px] font-black uppercase transition-all ${matchFilter.sort === 'score' ? 'bg-emerald-600 border-emerald-500 text-white' : 'bg-white/5 border-white/10 text-slate-400'}`}
                  >
                    {matchFilter.sort === 'score' ? 'Sort: Profit Potential' : 'Sort: Recent Signal'}
                  </button>
                </div>
                <div className="flex-1 bg-white/[0.01] border border-white/5 rounded-[40px] overflow-hidden overflow-y-auto custom-scrollbar shadow-2xl">
                  <table className="w-full text-left">
                    <thead className="bg-[#0f172a] text-[9px] font-black uppercase text-slate-500 tracking-[0.2em] sticky top-0 z-10 border-b border-white/10">
                      <tr>
                        <th className="p-6 pl-10">Arena Segment</th>
                        <th className="p-6">Origin Participant</th>
                        <th className="p-6">Target Opponent</th>
                        <th className="p-6">Yield Delta (R-B)</th>
                        <th className="p-6">Victor</th>
                        <th className="p-6 pr-10 text-right">Cycle Time</th>
                      </tr>
                    </thead>
                    <tbody className="text-[11px] divide-y divide-white/[0.03]">
                      {filteredMatches.map((m, i) => (
                        <tr key={i} className="hover:bg-white/[0.02] transition-colors group">
                          <td className="p-6 pl-10 text-slate-500 font-bold uppercase tracking-wider">{m.map_name.replace('maps/test-maps/', '').replace('.txt', '')}</td>
                          <td className="p-6 font-mono font-bold text-slate-300">{cleanName(m.red_name)}</td>
                          <td className="p-6 font-mono font-bold text-slate-300">{cleanName(m.blue_name)}</td>
                          <td className="p-6 font-bold">
                            <div className="flex items-center gap-2">
                              <span className="text-blue-400">${m.red_score}</span>
                              <span className="text-slate-600">vs</span>
                              <span className="text-indigo-400">${m.blue_score}</span>
                            </div>
                          </td>
                          <td className="p-6">
                            {m.winner === 'DRAW' ? <span className="text-slate-500 font-black">STALEMATE</span> :
                              (m.winner === 'ERROR' || m.winner === 'TIMEOUT') ? <span className="text-red-500 font-black border border-red-500/30 px-2 py-0.5 rounded text-[9px]">CRITICAL_FAIL</span> :
                                <span className="text-indigo-400 font-black flex items-center gap-1"><Award size={10} /> {cleanName(m.winner)}</span>}
                          </td>
                          <td className="p-6 pr-10 text-right font-mono text-slate-600 font-bold">{m.duration.toFixed(1)}s</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </motion.div>
            )}

            {activeTab === 'stats' && (
              <motion.div key="stats" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-8">
                <div className="grid grid-cols-2 gap-8 h-full">
                  <div className="bg-white/[0.02] border border-white/5 rounded-[40px] p-10 flex flex-col">
                    <h3 className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500 mb-10 flex items-center gap-2"><BarChart3 size={16} className="text-indigo-500" /> Universal Score bell curve</h3>
                    <div className="flex-1">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={getHistogram(globalStats.scores, 30)}>
                          <XAxis dataKey="bin" hide />
                          <YAxis hide />
                          <Tooltip contentStyle={{ background: '#020617', border: 'none', borderRadius: '12px' }} cursor={{ fill: 'rgba(99,102,241,0.05)' }} />
                          <Bar dataKey="count" fill="#6366f1" radius={[6, 6, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="mt-8 grid grid-cols-3 gap-4">
                      <StatMini label="Max Outlier" val={`$${Math.max(...globalStats.scores)}`} />
                      <StatMini label="Mean Yield" val={`$${(globalStats.scores.reduce((a, b) => a + b, 0) / globalStats.scores.length).toFixed(0)}`} />
                      <StatMini label="Stdev" val={`±${(Math.sqrt(globalStats.scores.map(x => Math.pow(x - (globalStats.scores.reduce((a, b) => a + b, 0) / globalStats.scores.length), 2)).reduce((a, b) => a + b) / globalStats.scores.length)).toFixed(0)}`} />
                    </div>
                  </div>
                  <div className="bg-white/[0.02] border border-white/5 rounded-[40px] p-10 flex flex-col">
                    <h3 className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500 mb-10 flex items-center gap-2"><Clock size={16} className="text-emerald-500" /> Duration probability density</h3>
                    <div className="flex-1">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={getHistogram(globalStats.durations, 20)}>
                          <XAxis dataKey="bin" hide />
                          <YAxis hide />
                          <Area type="monotone" dataKey="count" stroke="#10b981" fill="#10b981" fillOpacity={0.05} strokeWidth={3} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="mt-8 flex justify-between p-6 rounded-3xl bg-emerald-500/5 border border-emerald-500/10">
                      <div className="flex flex-col">
                        <span className="text-[10px] font-black text-slate-500 uppercase">Avg Run Time</span>
                        <span className="text-4xl font-black text-emerald-400">{(globalStats.durations.reduce((a, b) => a + b, 0) / globalStats.durations.length).toFixed(2)}s</span>
                      </div>
                      <Activity size={32} className="text-emerald-500/20" />
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}

// --- Data Visualization Sub-Components ---

function NavIcon({ active, onClick, icon, label }: { active: boolean, onClick: () => void, icon: any, label: string }) {
  return (
    <button onClick={onClick} className={`w-12 h-12 flex items-center justify-center rounded-2xl transition-all relative group ${active ? 'bg-indigo-600 text-white shadow-[0_0_15px_rgba(99,102,241,0.5)]' : 'text-slate-500 hover:bg-white/5 hover:text-slate-300'}`}>
      {icon}
      <div className="absolute left-16 px-3 py-1 bg-[#0f172a] border border-white/10 text-[9px] text-white rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-all transform translate-x-2 group-hover:translate-x-0 font-black uppercase tracking-widest z-[100] whitespace-nowrap shadow-2xl">
        {label}
      </div>
    </button>
  );
}

function StatMini({ label, val }: { label: string, val: string }) {
  return (
    <div className="p-4 rounded-2xl bg-white/[0.02] border border-white/5">
      <div className="text-[8px] font-black text-slate-600 uppercase mb-1">{label}</div>
      <div className="text-sm font-black text-slate-300 font-mono">{val}</div>
    </div>
  );
}

function KPI({ icon, label, val, color }: { icon: any, label: string, val: string, color: string }) {
  return (
    <div className="bg-white/[0.02] border border-white/5 p-5 rounded-3xl hover:border-white/10 transition-colors">
      <div className="flex items-center gap-2 text-slate-600 text-[9px] font-black uppercase tracking-widest mb-3">
        {icon} {label}
      </div>
      <div className={`text-xl font-black truncate tracking-tighter ${color}`}>{val}</div>
    </div>
  );
}

function BotDeepDive({ name, stats }: { name: string, stats: BotStats }) {
  const radarData = Object.entries(stats.maps).map(([m, s]) => ({
    subject: m.replace('maps/test-maps/', '').replace('.txt', ''),
    value: safeDiv(s.w, s.total)
  })).sort((a, b) => b.value - a.value);

  return (
    <div className="space-y-12">
      <header className="flex justify-between items-start">
        <div className="flex items-center gap-6">
          <div className="w-16 h-16 rounded-[28px] bg-indigo-600 flex items-center justify-center text-3xl font-black text-white shadow-2xl">{name[5] || 'B'}</div>
          <div>
            <h2 className="text-4xl font-black uppercase tracking-tighter leading-none">{cleanName(name)}</h2>
            <div className="flex gap-4 mt-3">
              <span className={`px-4 py-1.5 rounded-full border text-[9px] font-black uppercase ${isHero(name) ? 'bg-indigo-500/10 border-indigo-500/50 text-indigo-400' : 'bg-slate-500/10 border-slate-500/30 text-slate-500'}`}>
                {isHero(name) ? 'High Priority Candidate' : 'Community Participant'}
              </span>
            </div>
          </div>
        </div>
        <div className="text-right flex flex-col items-end">
          <div className="text-6xl font-black text-white tracking-tighter leading-none">{safeDiv(stats.w, stats.total).toFixed(1)}%</div>
          <div className="text-[10px] font-black text-indigo-500 uppercase tracking-widest mt-2 flex items-center gap-2">Success Velocity <Activity size={12} /></div>
        </div>
      </header>

      <div className="grid grid-cols-5 gap-6">
        <KPI icon={<Award size={16} />} label="Avg Game Profit" val={`$${(stats.score / stats.total).toFixed(0)}`} color="text-indigo-400" />
        <KPI icon={<Shield size={16} />} label="Resilience Index" val={`${safeDiv(stats.total - stats.e, stats.total).toFixed(1)}%`} color="text-emerald-400" />
        <KPI icon={<TrendingUp size={16} />} label="Net Score Margin" val={`${(stats.scoreDiff / stats.total).toFixed(0)}`} color="text-blue-400" />
        <KPI icon={<Sword size={16} />} label="Prime Nemesis" val={cleanName(stats.nemesis || "N/A")} color="text-red-400" />
        <KPI icon={<Users size={16} />} label="Principal Victim" val={cleanName(stats.victim || "N/A")} color="text-emerald-500" />
      </div>

      <div className="grid grid-cols-2 gap-10">
        <div className="space-y-6">
          <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500 flex items-center gap-2 px-2">Arena Dominance profile <MapIcon size={12} /></h4>
          <div className="h-[320px] bg-[#0f172a]/30 border border-white/5 rounded-[40px] p-8">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={radarData} layout="vertical">
                <XAxis type="number" hide />
                <YAxis dataKey="subject" type="category" width={110} tick={{ fontSize: 9, fill: '#64748b', fontWeight: 'bold' }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: '#020617', border: 'none', borderRadius: '12px' }} cursor={{ fill: 'rgba(255,255,255,0.02)' }} />
                <Bar dataKey="value" fill="#6366f1" radius={[0, 6, 6, 0]} barSize={10} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="space-y-6">
          <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500 flex items-center gap-2 px-2">Reward Distribution <DollarSign size={12} /></h4>
          <div className="h-[320px] bg-[#0f172a]/30 border border-white/5 rounded-[40px] p-8">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={getHistogram(stats.scores, 15)}>
                <XAxis dataKey="bin" hide />
                <YAxis hide />
                <Tooltip contentStyle={{ background: '#020617', border: 'none', borderRadius: '12px' }} cursor={{ fill: 'rgba(255,255,255,0.02)' }} />
                <Bar dataKey="count" fill="#10b981" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

function MapDetailView({ name, data }: { name: string, data: any }) {
  const bestBots = Object.entries(data.botPerformances)
    .sort((a: any, b: any) => (b[1].score / b[1].count) - (a[1].score / a[1].count))
    .slice(0, 15);

  return (
    <div className="space-y-12">
      <header className="flex justify-between items-end">
        <div>
          <h2 className="text-5xl font-black uppercase tracking-tighter leading-none">{name.replace('maps/test-maps/', '').replace('.txt', '')}</h2>
          <div className="text-[11px] font-black text-emerald-500 uppercase tracking-widest mt-4 flex items-center gap-2">Signal Strength: {data.count} Combat Cycles <Activity size={12} /></div>
        </div>
        <div className="text-right">
          <div className="font-mono font-black text-4xl text-white tracking-widest leading-none">${(data.score / data.count / 2).toFixed(0)}</div>
          <div className="text-[10px] text-slate-600 font-black uppercase tracking-widest mt-3">Mean Map Yield Matrix</div>
        </div>
      </header>

      <div className="bg-[#0f172a]/20 border border-white/5 rounded-[40px] overflow-hidden shadow-2xl">
        <table className="w-full text-left">
          <thead className="bg-[#0f172a] text-[9px] font-black uppercase text-slate-500 tracking-[0.2em] border-b border-white/10">
            <tr>
              <th className="p-6 pl-10">Global Arena Rank</th>
              <th className="p-6">Competitor ID</th>
              <th className="p-6">Mean Cyclic Yield</th>
              <th className="p-6">Success Frequency</th>
              <th className="p-6 pr-10 text-right">Run Count</th>
            </tr>
          </thead>
          <tbody className="text-[11px] divide-y divide-white/[0.02]">
            {bestBots.map(([bot, s]: any, idx) => (
              <tr key={bot} className="hover:bg-white/[0.02] transition-colors group">
                <td className="p-6 pl-10 font-mono text-slate-700 font-bold">NODE_{idx + 1}</td>
                <td className={`p-6 font-bold ${isHero(bot) ? 'text-indigo-400' : 'text-slate-300'} flex items-center gap-2`}>
                  {cleanName(bot)}
                  {idx === 0 && <Award size={14} className="text-yellow-500" />}
                </td>
                <td className="p-6 font-mono font-black text-white underline decoration-white/5 underline-offset-8">${(s.score / s.count).toFixed(0)}</td>
                <td className="p-6">
                  <div className="flex items-center gap-4">
                    <span className="font-black text-emerald-500 w-12">{safeDiv(s.wins, s.count).toFixed(1)}%</span>
                    <div className="w-24 h-1 bg-white/5 rounded-full overflow-hidden shrink-0">
                      <div className="h-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" style={{ width: `${safeDiv(s.wins, s.count)}%` }} />
                    </div>
                  </div>
                </td>
                <td className="p-6 pr-10 text-right text-slate-500 font-black tracking-widest font-mono">{s.count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function HeroDetailCard({ name, stats }: { name: string, stats: BotStats }) {
  const wr = safeDiv(stats.w, stats.total - stats.e).toFixed(1);
  const avgMargin = (stats.scoreDiff / stats.total).toFixed(0);

  return (
    <div className="bg-[#0f172a]/40 border border-white/5 p-8 rounded-[40px] relative overflow-hidden group hover:border-indigo-500/30 transition-all">
      <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/5 blur-3xl rounded-full -mr-16 -mt-16 group-hover:bg-indigo-500/10 transition-all duration-1000" />
      <div className="flex items-center gap-5 mb-10">
        <div className="w-14 h-14 rounded-[22px] flex items-center justify-center text-white text-2xl font-black shadow-2xl shrink-0 border border-white/5" style={{ backgroundColor: HERO_COLORS[name] }}>{name[5]}</div>
        <div>
          <h3 className="text-xl font-black text-white tracking-tighter leading-none">{cleanName(name)}</h3>
          <div className="text-[10px] text-slate-500 font-black uppercase tracking-widest flex items-center gap-1.5 mt-2"><Shield size={10} className="text-indigo-500/50" /> High Frequency Signal</div>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-10">
        <div className="flex flex-col">
          <div className="text-[9px] text-slate-600 font-black uppercase mb-1">Combat Success</div>
          <div className="text-4xl font-black tracking-tighter" style={{ color: HERO_COLORS[name] }}>{wr}%</div>
        </div>
        <div className="flex flex-col">
          <div className="text-[9px] text-slate-600 font-black uppercase mb-1">Win Margin</div>
          <div className={`text-4xl font-black tracking-tighter ${Number(avgMargin) > 0 ? 'text-emerald-500' : 'text-red-500'}`}>{Number(avgMargin) > 0 ? '+' : ''}{avgMargin}</div>
        </div>
      </div>
      <div className="mt-10 pt-6 border-t border-white/5 flex justify-between items-center text-[10px] font-black uppercase tracking-widest">
        <span className="text-slate-600">Mean: <span className="text-white font-mono">${(stats.score / stats.total).toFixed(0)}</span></span>
        <span className={`px-2 py-0.5 rounded ${Number(wr) > 50 ? 'bg-emerald-500/10 text-emerald-500' : 'bg-red-500/10 text-red-500'}`}>Target Stable</span>
      </div>
    </div>
  );
}

function getHistogram(data: number[], bins: number) {
  if (data.length === 0) return [];
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min;
  const step = range / bins || 1;
  const hist = Array.from({ length: bins }, (_, i) => ({ bin: (min + i * step).toFixed(0), count: 0 }));
  data.forEach(v => {
    const idx = Math.min(Math.floor((v - min) / step), bins - 1);
    hist[idx].count++;
  });
  return hist;
}

export default App;
