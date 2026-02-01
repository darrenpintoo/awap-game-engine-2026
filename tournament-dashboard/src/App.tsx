import { useState, useEffect, useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  LineChart, Line, CartesianGrid, AreaChart, Area, ComposedChart,
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';
import { Trophy, Activity, TrendingUp, DollarSign, Crosshair, AlertTriangle } from 'lucide-react';
import { motion } from 'framer-motion';

// --- Types ---
type Match = {
  red_name: string; blue_name: string;
  map_name: string; map_type: 'official' | 'test';
  winner: string; duration: number;
  red_score: number; blue_score: number;
  timestamp: number;
};
type Data = {
  metadata: { total_matches: number; completed_matches: number; last_updated: number };
  matches: Match[];
};

// --- Utils ---
const HEROES = ["Hero_BEST^3", "Hero_BEST^2", "Hero_Sovereign"];
const cleanName = (n: string) => n.replace("Hero_", "").replace(".py", "");
const isHero = (n: string) => n.startsWith("Hero_");
const safeDiv = (n: number, d: number) => d > 0 ? (n / d) * 100 : 0;

function App() {
  const [data, setData] = useState<Data | null>(null);

  useEffect(() => {
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
  }, []);

  // --- Compute Deep Stats ---
  const stats = useMemo(() => {
    if (!data) return null;
    const s: Record<string, any> = {};
    const timeline: any[] = [];
    let matchCount = 0;

    data.matches.sort((a, b) => a.timestamp - b.timestamp).forEach(m => {
      matchCount++;
      // Snapshot timeline every 50 matches to avoid chart clutter
      if (matchCount % 50 === 0) {
        const snap: any = { name: matchCount };
        HEROES.forEach(h => {
          const hStats = s[h];
          snap[cleanName(h)] = hStats && hStats.total > 0 ? (hStats.score / hStats.total) : 0;
        });
        timeline.push(snap);
      }

      [m.red_name, m.blue_name].forEach(p => {
        if (!s[p]) s[p] = {
          w: 0, l: 0, d: 0, e: 0, total: 0,
          score: 0,
          opponents: {}, maps: {}
        };

        const isMe = p === m.red_name;
        const opp = isMe ? m.blue_name : m.red_name;
        const score = isMe ? m.red_score : m.blue_score;
        const oppScore = isMe ? m.blue_score : m.red_score;
        const win = m.winner === p;

        s[p].total++;
        s[p].score += score;

        if (!s[p].opponents[opp]) s[p].opponents[opp] = { w: 0, l: 0, total: 0, scoreDiff: 0 };

        s[p].opponents[opp].total++;
        s[p].opponents[opp].scoreDiff += (score - oppScore);

        if (win) { s[p].w++; s[p].opponents[opp].w++; }
        else if (m.winner === 'DRAW') s[p].d++;
        else if (m.winner === 'ERROR' || m.winner === 'TIMEOUT') s[p].e++;
        else { s[p].l++; s[p].opponents[opp].l++; }
      });
    });
    return { s, timeline };
  }, [data]);

  if (!data || !stats) return <div className="flex h-screen items-center justify-center font-bold text-2xl text-slate-400">Loading Tournament Data...</div>;

  const { s: playerStats, timeline } = stats;
  const progress = (data.metadata.completed_matches / data.metadata.total_matches) * 100;

  // Chart Data
  const heroPerformance = HEROES.map(h => {
    const st = playerStats[h] || { w: 0, total: 0, score: 0, e: 0 };
    return {
      name: cleanName(h),
      wr: safeDiv(st.w, st.total - st.e),
      avgScore: st.total > 0 ? (st.score / st.total) : 0,
      playRate: st.total
    };
  });

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans pb-20">
      {/* Navbar */}
      <nav className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm/50 backdrop-blur-md bg-white/80">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 text-white p-2 rounded-lg"><Trophy size={20} /></div>
            <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-700 to-indigo-600">AWAP 2026 Finals</h1>
          </div>
          <div className="flex items-center gap-6 text-sm font-medium text-slate-500">
            <div className="flex flex-col items-end">
              <span className="text-xs uppercase tracking-wider text-slate-400">Match Progress</span>
              <span className="text-slate-900 font-mono">{data.metadata.completed_matches} / {data.metadata.total_matches}</span>
            </div>
            <div className="w-32 h-2 bg-slate-100 rounded-full overflow-hidden">
              <div className="h-full bg-blue-500 transition-all duration-500" style={{ width: `${progress}%` }} />
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 pt-10 space-y-8">

        {/* Hero KPI Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {HEROES.map((hero, i) => {
            const st = playerStats[hero] || { w: 0, total: 0, score: 0, opponents: {} };
            const wr = safeDiv(st.w, st.total - st.e).toFixed(1);

            // Calculate Nemesis
            let nemesis = "None"; let minDiff = Infinity;
            Object.entries(st.opponents).forEach(([op, d]: any) => {
              if (d.scoreDiff < minDiff && d.total > 0) { minDiff = d.scoreDiff; nemesis = cleanName(op); }
            });

            return (
              <motion.div
                initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1 }}
                key={hero} className="relative overflow-hidden bg-white p-6 rounded-2xl shadow-sm border border-slate-100 group hover:shadow-md transition-shadow"
              >
                <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                  <Trophy size={80} />
                </div>
                <h2 className="text-2xl font-bold text-slate-800 mb-1">{cleanName(hero)}</h2>
                <div className="flex items-baseline gap-2 mb-6">
                  <span className={`text-4xl font-black ${Number(wr) > 50 ? 'text-green-600' : 'text-orange-500'}`}>{wr}%</span>
                  <span className="text-slate-400 text-sm font-medium">Win Rate</span>
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg">
                    <div className="flex items-center gap-2 text-sm text-slate-600"><DollarSign size={16} /> Avg Earnings</div>
                    <div className="font-mono font-bold text-blue-600">${(st.total > 0 ? st.score / st.total : 0).toFixed(0)}</div>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-red-50 rounded-lg">
                    <div className="flex items-center gap-2 text-sm text-red-700"><AlertTriangle size={16} /> Nemesis</div>
                    <div className="font-bold text-red-700">{nemesis}</div>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-96">

          {/* WR Comparison */}
          <div className="lg:col-span-1 bg-white p-6 rounded-2xl shadow-sm border border-slate-100 flex flex-col">
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2"><Activity size={18} className="text-blue-500" /> Performance</h3>
            <div className="flex-1">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={heroPerformance}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                  <Tooltip cursor={{ fill: 'transparent' }} contentStyle={{ borderRadius: '8px' }} />
                  <Bar dataKey="wr" fill="#3b82f6" radius={[6, 6, 0, 0]} barSize={40} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Money Trend */}
          <div className="lg:col-span-2 bg-white p-6 rounded-2xl shadow-sm border border-slate-100 flex flex-col">
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2"><TrendingUp size={18} className="text-green-500" /> Average Score Trend</h3>
            <div className="flex-1">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timeline}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                  <XAxis dataKey="name" hide />
                  <YAxis />
                  <Tooltip contentStyle={{ borderRadius: '8px' }} />
                  <Legend />
                  {HEROES.map((h, i) => (
                    <Line
                      key={h}
                      type="monotone"
                      dataKey={cleanName(h)}
                      stroke={['#3b82f6', '#8b5cf6', '#10b981'][i]}
                      strokeWidth={3}
                      dot={false}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Leaderboard Table */}
        <div className="bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden">
          <div className="p-6 border-b border-slate-100 flex justify-between items-center">
            <h3 className="text-lg font-bold">Global Leaderboard</h3>
            <span className="text-xs font-medium bg-slate-100 px-3 py-1 rounded-full text-slate-500">
              Sorted by Win Rate (Excluding Errors)
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="bg-slate-50 text-slate-500 font-medium">
                <tr>
                  <th className="p-4 pl-6">Rank</th>
                  <th className="p-4">Bot Name</th>
                  <th className="p-4 text-center">Win Rate</th>
                  <th className="p-4 text-right">Avg Score</th>
                  <th className="p-4 text-right">Matches</th>
                  <th className="p-4 text-right pr-6">Record (W-L-D)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {Object.entries(playerStats)
                  .sort((a, b) => {
                    const wa = safeDiv(a[1].w, a[1].total - a[1].e);
                    const wb = safeDiv(b[1].w, b[1].total - b[1].e);
                    return wb - wa;
                  })
                  .slice(0, 50) // Top 50 only
                  .map(([name, s], idx) => {
                    const valid = s.total - s.e;
                    const wr = safeDiv(s.w, valid).toFixed(1);
                    const isTop = idx < 3;

                    return (
                      <tr key={name} className="hover:bg-slate-50/80 transition-colors group">
                        <td className="p-4 pl-6 font-mono text-slate-400 w-16">
                          {isTop ? <span className="text-yellow-500 font-bold text-lg">#{idx + 1}</span> : `#${idx + 1}`}
                        </td>
                        <td className={`p-4 font-medium ${isHero(name) ? 'text-blue-600' : 'text-slate-700'}`}>
                          {cleanName(name)}
                          {isHero(name) && <span className="ml-2 text-[10px] uppercase font-bold bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full tracking-wider">Hero</span>}
                        </td>
                        <td className="p-4 text-center w-32">
                          <div className={`font-bold ${Number(wr) > 50 ? 'text-green-600' : 'text-slate-500'}`}>{wr}%</div>
                        </td>
                        <td className="p-4 text-right font-mono text-slate-600">
                          ${(s.score / s.total).toFixed(0)}
                        </td>
                        <td className="p-4 text-right text-slate-400">{s.total}</td>
                        <td className="p-4 text-right pr-6 font-mono text-xs text-slate-400 group-hover:text-slate-600">
                          <span className="text-green-600 font-bold">{s.w}</span> - <span className="text-red-500 font-bold">{s.l}</span> - {s.d}
                        </td>
                      </tr>
                    );
                  })
                }
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
