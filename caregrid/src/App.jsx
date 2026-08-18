import { useState, useRef, useEffect, useCallback } from "react";
import {
  Activity,
  Bed,
  Clock,
  UserPlus,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  RotateCcw,
  PlayCircle,
  ScrollText,
  LayoutDashboard,
  Users,
  ListOrdered,
  Bot,
  Menu,
  Bell,
  Sun,
  Crown,
  Sparkles,
  Send,
  Loader2,
  LogOut,
  X,
  Search,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  AlertTriangle,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Design tokens — clinical monitor palette. Colors are functional (they map
// to the three scoring factors everywhere in the UI), not decorative.
// ---------------------------------------------------------------------------
const C = {
  bg: "#080B10",
  panel: "#0F161F",
  panelAlt: "#131B26",
  panelRaised: "#161F2B",
  border: "rgba(255,255,255,0.08)",
  borderStrong: "rgba(255,255,255,0.16)",
  text: "#E8EDF4",
  muted: "#7C8CA3",
  faint: "#4B5A70",
  severity: "#FF6B5B",
  survival: "#3FD9D0",
  wait: "#F4B942",
  admit: "#4ADE80",
  tie: "#C9A8FF",
  danger: "#FF6B5B",
};

const DISPLAY_FONT = "'Space Grotesk', system-ui, sans-serif";
const MONO_FONT = "'IBM Plex Mono', ui-monospace, monospace";

const MAX_WAIT_HOURS = 48;
const TIE_EPSILON = 3;
const BED_TOTAL = 6;

// ---------------------------------------------------------------------------
// Seed data
// ---------------------------------------------------------------------------
const SEED_PATIENTS = [
  { id: "P104", name: "Patient P104", severity: 92, survival: 88, arrivalTime: -30 },
  { id: "P107", name: "Patient P107", severity: 85, survival: 90, arrivalTime: -20 },
  { id: "P102", name: "Patient P102", severity: 78, survival: 95, arrivalTime: -10 },
];

let idCounter = 200;
const nextId = () => `P${idCounter++}`;

// ---------------------------------------------------------------------------
// Scoring engine
// ---------------------------------------------------------------------------
function scorePatients(patients, weights, simTime) {
  const sum = weights.severity + weights.survival + weights.wait || 1;
  const wS = weights.severity / sum;
  const wV = weights.survival / sum;
  const wW = weights.wait / sum;

  return patients.map((p) => {
    const waitHours = Math.max(simTime - p.arrivalTime, 0);
    const waitNorm = Math.min(waitHours / MAX_WAIT_HOURS, 1) * 100;
    const sevContrib = wS * p.severity;
    const survContrib = wV * p.survival;
    const waitContrib = wW * waitNorm;
    const score = sevContrib + survContrib + waitContrib;
    return { ...p, waitHours, waitNorm, sevContrib, survContrib, waitContrib, score, wS, wV, wW };
  });
}

function rankPatients(scored) {
  const sorted = [...scored].sort((a, b) => b.score - a.score);
  if (sorted.length === 0) return [];

  const clusters = [];
  let current = [sorted[0]];
  for (let i = 1; i < sorted.length; i++) {
    if (current.length && current[current.length - 1].score - sorted[i].score <= TIE_EPSILON) {
      current.push(sorted[i]);
    } else {
      clusters.push(current);
      current = [sorted[i]];
    }
  }
  if (current.length) clusters.push(current);

  const final = [];
  clusters.forEach((cluster) => {
    if (cluster.length === 1) {
      final.push({ ...cluster[0], tied: false, tieReason: null });
      return;
    }
    const ordered = [...cluster].sort((a, b) => {
      if (b.severity !== a.severity) return b.severity - a.severity;
      if (b.waitHours !== a.waitHours) return b.waitHours - a.waitHours;
      return a.arrivalTime - b.arrivalTime;
    });
    ordered.forEach((p, i) => {
      let reason;
      if (i === 0) {
        reason = `Leads a ${ordered.length}-way tie (\u0394\u2264${TIE_EPSILON} pts) highest severity in the group.`;
      } else {
        const leader = ordered[0];
        if (leader.severity !== p.severity) {
          reason = `Tie broken against ${leader.id}: lower severity (${p.severity} vs ${leader.severity}).`;
        } else if (leader.waitHours !== p.waitHours) {
          reason = `Severity equal to ${leader.id} tie broken by shorter wait (${p.waitHours.toFixed(0)}h vs ${leader.waitHours.toFixed(0)}h).`;
        } else {
          reason = `Severity and wait equal to ${leader.id} tie broken by later arrival (FCFS).`;
        }
      }
      final.push({ ...p, tied: true, tieReason: reason });
    });
  });

  return final.map((p, i) => ({ ...p, rank: i + 1 }));
}

// ---------------------------------------------------------------------------
// Anthropic API helper. Fails soft: callers fall back to a local summary.
// ---------------------------------------------------------------------------
async function askClaude(prompt) {
  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-6",
      max_tokens: 1000,
      messages: [{ role: "user", content: prompt }],
    }),
  });
  const data = await response.json();
  const text = (data.content || [])
    .map((b) => (b.type === "text" ? b.text : ""))
    .filter(Boolean)
    .join("\n")
    .trim();
  if (!text) throw new Error("empty response");
  return text;
}

function localExplanation(p, ranked) {
  const drivers = [
    { k: "severity", v: p.sevContrib, label: "severity" },
    { k: "survival", v: p.survContrib, label: "survival likelihood" },
    { k: "wait", v: p.waitContrib, label: "wait time" },
  ].sort((a, b) => b.v - a.v);
  const lead = drivers[0].label;
  const behind = ranked.find((x) => x.rank === p.rank + 1);
  let tail = "";
  if (behind) {
    tail = ` Leads #${behind.rank} (${behind.id}) by ${(p.score - behind.score).toFixed(1)} pts.`;
  }
  const tieTail = p.tied ? ` ${p.tieReason}` : "";
  return `${p.id} ranks #${p.rank} driven mainly by ${lead} (${drivers[0].v.toFixed(1)} pts).${tail}${tieTail}`;
}

// Builds a compact text snapshot of the whole system, used both for the
// AI Assistant chat and as the local fallback when the API is unavailable.
function buildSystemContext({ ranked, admitted, beds, simTime, weights, log }) {
  const queueLines = ranked
    .map(
      (p) =>
        `#${p.rank} ${p.id} score ${p.score.toFixed(1)} (severity ${p.severity}, survival ${p.survival}, waited ${p.waitHours.toFixed(0)}h)${p.tied ? " [tied: " + p.tieReason + "]" : ""}`
    )
    .join("\n");
  const admittedLines = admitted
    .map((p) => `${p.id} admitted at t=${p.admittedAt}h (severity ${p.severity}, survival ${p.survival})`)
    .join("\n");
  const logLines = log
    .slice(-6)
    .map((l) => `[t=${l.t}h] ${l.text}`)
    .join("\n");

  return `You are the AI Assistant embedded in CareGrid, an ICU patient-prioritization dashboard. Answer the user's question using ONLY the data below. Be concise (2-5 sentences unless a list is clearly needed), speak like a calm clinical operations assistant, and never invent patient data that isn't listed.

SIMULATION TIME: t=${simTime}h
BEDS: ${beds.available}/${beds.total} free
SCORING WEIGHTS: severity ${weights.severity}%, survival ${weights.survival}%, wait ${weights.wait}%

WAITING QUEUE (ranked, highest priority first):
${queueLines || "(empty no patients waiting)"}

CURRENTLY ADMITTED:
${admittedLines || "(no patients currently admitted)"}

RECENT AUDIT LOG:
${logLines || "(no log entries yet)"}

USER QUESTION: `;
}

// Simple pattern-matched fallback so the assistant still answers something
// useful if the API call fails (offline, rate limited, etc).
function localAssistantAnswer(question, { ranked, admitted, beds, simTime }) {
  const q = question.toLowerCase();
  if (!ranked.length && !admitted.length) {
    return "There are no patients in the system right now the queue and ICU are both empty.";
  }
  if (/(highest|top|first|next|who.*priority)/.test(q)) {
    const top = ranked[0];
    return top
      ? `${top.id} has the highest priority right now with a score of ${top.score.toFixed(1)} (severity ${top.severity}, survival ${top.survival}, waited ${top.waitHours.toFixed(0)}h).`
      : "The queue is currently empty, so there's no one waiting to prioritize.";
  }
  if (/(bed|capacity|room|space)/.test(q)) {
    return `There ${beds.available === 1 ? "is" : "are"} ${beds.available} of ${beds.total} ICU beds free. ${admitted.length} patient${admitted.length === 1 ? " is" : "s are"} currently admitted.`;
  }
  if (/(how many|queue|waiting)/.test(q)) {
    return `${ranked.length} patient${ranked.length === 1 ? " is" : "s are"} currently waiting in the queue.`;
  }
  if (/(tie|tied)/.test(q)) {
    const tied = ranked.filter((p) => p.tied);
    return tied.length
      ? `${tied.length} patient(s) are currently in a scoring tie: ${tied.map((p) => p.id).join(", ")}.`
      : "No patients are currently tied in score.";
  }
  const mentioned = ranked.find((p) => q.includes(p.id.toLowerCase())) || admitted.find((p) => q.includes(p.id.toLowerCase()));
  if (mentioned && mentioned.rank) {
    return localExplanation(mentioned, ranked);
  }
  if (mentioned) {
    return `${mentioned.id} is currently admitted to the ICU (severity ${mentioned.severity}, survival ${mentioned.survival}).`;
  }
  return `At t=${simTime}h: ${ranked.length} patient(s) waiting, ${admitted.length} admitted, ${beds.available}/${beds.total} beds free. Ask me about a specific patient (e.g. "why is P104 ranked first") or the ICU status.`;
}

// ---------------------------------------------------------------------------
// Small presentational pieces
// ---------------------------------------------------------------------------
function FactorBar({ label, color, contrib, weightPct }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5, marginBottom: 5 }}>
        <span style={{ color: C.text, display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: 2, background: color, display: "inline-block" }} />
          {label} <span style={{ color: C.faint }}>({weightPct}%)</span>
        </span>
        <span style={{ color: C.text, fontFamily: MONO_FONT }}>
          {contrib.toFixed(1)} <span style={{ color: C.faint }}>/ {weightPct}</span>
        </span>
      </div>
      <div style={{ height: 6, background: "rgba(255,255,255,0.06)", borderRadius: 3, overflow: "hidden" }}>
        <div
          style={{
            width: `${Math.min((contrib / weightPct) * 100, 100)}%`,
            height: "100%",
            background: color,
            borderRadius: 3,
            transition: "width 0.4s ease",
          }}
        />
      </div>
    </div>
  );
}

function WeightSlider({ label, color, value, onChange }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
        <span style={{ color: C.text, display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: 2, background: color, display: "inline-block" }} />
          {label}
        </span>
        <span style={{ color: C.muted, fontFamily: MONO_FONT }}>{value}</span>
      </div>
      <input
        type="range"
        min={0}
        max={100}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: "100%", accentColor: color, cursor: "pointer" }}
      />
    </div>
  );
}

function MetricCard({ icon, iconColor, value, label }) {
  return (
    <div
      style={{
        flex: "1 1 0",
        minWidth: 0,
        background: C.panel,
        border: `1px solid ${C.border}`,
        borderRadius: 10,
        padding: "10px 12px",
        display: "flex",
        alignItems: "center",
        gap: 10,
      }}
    >
      <div
        style={{
          width: 30,
          height: 30,
          borderRadius: 8,
          background: `${iconColor}1A`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
        }}
      >
        {icon}
      </div>
      <div style={{ minWidth: 0 }}>
        <div style={{ fontFamily: MONO_FONT, fontSize: 16, lineHeight: 1.1, whiteSpace: "nowrap" }}>{value}</div>
        <div
          style={{
            fontSize: 9.5,
            color: C.muted,
            textTransform: "uppercase",
            letterSpacing: 0.5,
            marginTop: 2,
            whiteSpace: "nowrap",
          }}
        >
          {label}
        </div>
      </div>
    </div>
  );
}

function NavItem({ icon, label, active, onClick, badge }) {
  return (
    <button
      onClick={onClick}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 11,
        width: "100%",
        background: active ? "rgba(63,217,208,0.10)" : "transparent",
        border: "none",
        borderLeft: active ? `2px solid ${C.survival}` : "2px solid transparent",
        color: active ? C.text : C.muted,
        padding: "9px 14px",
        fontSize: 13,
        fontWeight: active ? 600 : 500,
        cursor: "pointer",
        textAlign: "left",
        borderRadius: "0 8px 8px 0",
        fontFamily: DISPLAY_FONT,
      }}
    >
      {icon}
      <span style={{ flex: 1 }}>{label}</span>
      {badge !== undefined && badge !== null && (
        <span
          style={{
            fontFamily: MONO_FONT,
            fontSize: 10,
            color: active ? C.text : C.faint,
            background: "rgba(255,255,255,0.06)",
            borderRadius: 6,
            padding: "1px 6px",
          }}
        >
          {badge}
        </span>
      )}
    </button>
  );
}

function ScoreRing({ score }) {
  const r = 15;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(score, 100)) / 100;
  const color = score >= 80 ? C.severity : score >= 60 ? C.wait : C.survival;
  return (
    <svg width="38" height="38" viewBox="0 0 38 38">
      <circle cx="19" cy="19" r={r} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="4" />
      <circle
        cx="19"
        cy="19"
        r={r}
        fill="none"
        stroke={color}
        strokeWidth="4"
        strokeDasharray={c}
        strokeDashoffset={c * (1 - pct)}
        strokeLinecap="round"
        transform="rotate(-90 19 19)"
      />
    </svg>
  );
}

// Compact score-composition chart: one horizontal stacked bar per queued
// patient, segmented by how much each factor contributed. This replaces the
// old empty/underused line-graph area with something that directly explains
// *why* the queue is ordered the way it is, at a glance.
function ScoreCompositionChart({ ranked }) {
  return (
    <section
      style={{
        background: C.panel,
        border: `1px solid ${C.border}`,
        borderRadius: 12,
        padding: 14,
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
        flex: 1,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 2 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <Activity size={14} color={C.survival} />
          <h2
            style={{
              fontFamily: DISPLAY_FONT,
              fontSize: 12,
              textTransform: "uppercase",
              letterSpacing: 0.8,
              color: C.muted,
              margin: 0,
            }}
          >
            Score Composition
          </h2>
        </div>
        <span style={{ fontFamily: MONO_FONT, fontSize: 10, color: C.faint }}>
          {ranked.length} in queue
        </span>
      </div>
      <p style={{ fontSize: 10.5, color: C.faint, margin: "2px 0 10px" }}>
        What's driving each patient's priority score.
      </p>

      {ranked.length === 0 ? (
        <div style={{ color: C.faint, fontSize: 11.5, padding: "16px 0", textAlign: "center" }}>
          Queue is empty. Add a patient to see the breakdown.
        </div>
      ) : (
        <div style={{ overflowY: "auto", flex: 1, minHeight: 0, paddingRight: 2 }}>
          {ranked.map((p) => (
            <div key={p.id} style={{ marginBottom: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 3 }}>
                <span style={{ color: C.text, fontWeight: 600 }}>
                  #{p.rank} {p.id}
                </span>
                <span style={{ fontFamily: MONO_FONT, color: C.muted }}>{p.score.toFixed(1)}</span>
              </div>
              <div style={{ display: "flex", height: 10, borderRadius: 5, overflow: "hidden", background: "rgba(255,255,255,0.05)" }}>
                <div style={{ width: `${p.sevContrib}%`, background: C.severity }} title={`Severity: ${p.sevContrib.toFixed(1)}`} />
                <div style={{ width: `${p.survContrib}%`, background: C.survival }} title={`Survival: ${p.survContrib.toFixed(1)}`} />
                <div style={{ width: `${p.waitContrib}%`, background: C.wait }} title={`Wait: ${p.waitContrib.toFixed(1)}`} />
              </div>
            </div>
          ))}
        </div>
      )}

      <div style={{ display: "flex", gap: 14, marginTop: 8, paddingTop: 8, borderTop: `1px solid ${C.border}` }}>
        {[
          ["Severity", C.severity],
          ["Survival", C.survival],
          ["Wait", C.wait],
        ].map(([label, color]) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 10, color: C.muted }}>
            <span style={{ width: 7, height: 7, borderRadius: 2, background: color, display: "inline-block" }} />
            {label}
          </div>
        ))}
      </div>
    </section>
  );
}

// Ward capacity donut: the second useful visualization, showing bed
// occupancy at a glance instead of the removed empty ICU Beds tab.
function CapacityDonut({ beds, admitted }) {
  const r = 40;
  const c = 2 * Math.PI * r;
  const occupied = beds.total - beds.available;
  const pct = beds.total ? occupied / beds.total : 0;
  return (
    <section
      style={{
        background: C.panel,
        border: `1px solid ${C.border}`,
        borderRadius: 12,
        padding: 14,
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <Bed size={14} color={C.admit} />
        <h2
          style={{
            fontFamily: DISPLAY_FONT,
            fontSize: 12,
            textTransform: "uppercase",
            letterSpacing: 0.8,
            color: C.muted,
            margin: 0,
          }}
        >
          Ward Capacity
        </h2>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        <svg width="96" height="96" viewBox="0 0 96 96" style={{ flexShrink: 0 }}>
          <circle cx="48" cy="48" r={r} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="10" />
          <circle
            cx="48"
            cy="48"
            r={r}
            fill="none"
            stroke={pct >= 1 ? C.severity : pct >= 0.7 ? C.wait : C.admit}
            strokeWidth="10"
            strokeDasharray={c}
            strokeDashoffset={c * (1 - pct)}
            strokeLinecap="round"
            transform="rotate(-90 48 48)"
            style={{ transition: "stroke-dashoffset 0.5s ease" }}
          />
          <text x="48" y="44" textAnchor="middle" fill={C.text} fontSize="20" fontFamily={MONO_FONT} fontWeight="600">
            {occupied}/{beds.total}
          </text>
          <text x="48" y="60" textAnchor="middle" fill={C.faint} fontSize="9" fontFamily={MONO_FONT}>
            occupied
          </text>
        </svg>
        <div style={{ flex: 1, minWidth: 0 }}>
          {admitted.length === 0 ? (
            <p style={{ fontSize: 11, color: C.faint, margin: 0 }}>No patients currently admitted.</p>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
              {admitted.slice(0, 4).map((p) => (
                <div key={p.id} style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
                  <span style={{ color: C.text }}>{p.id}</span>
                  <span style={{ color: C.faint, fontFamily: MONO_FONT }}>t+{p.admittedAt}h</span>
                </div>
              ))}
              {admitted.length > 4 && (
                <span style={{ fontSize: 10.5, color: C.faint }}>+{admitted.length - 4} more</span>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function TrendBadge({ delta }) {
  if (!delta) {
    return (
      <span style={{ display: "inline-flex", alignItems: "center", color: C.faint }}>
        <Minus size={12} />
      </span>
    );
  }
  const up = delta > 0;
  return (
    <span style={{ display: "inline-flex", alignItems: "center", color: up ? C.admit : C.severity, fontSize: 10, fontFamily: MONO_FONT }}>
      {up ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
      {Math.abs(delta)}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Priority Queue row (used in both the Dashboard preview and the full tab)
// ---------------------------------------------------------------------------
function QueueRow({ p, expanded, onToggle, onSelectInsight, onAdmit, canAdmit, justMoved, detailed }) {
  return (
    <div
      style={{
        background: justMoved ? "rgba(63,217,208,0.08)" : C.panelAlt,
        border: `1px solid ${justMoved ? "rgba(63,217,208,0.35)" : C.border}`,
        borderRadius: 10,
        padding: "12px 14px",
        marginBottom: 8,
        transition: "background 0.6s ease, border-color 0.6s ease",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        {p.rank === 1 ? <Crown size={16} color={C.wait} style={{ flexShrink: 0 }} /> : (
          <span style={{ fontFamily: MONO_FONT, fontSize: 12, color: C.faint, width: 16, textAlign: "center", flexShrink: 0 }}>
            #{p.rank}
          </span>
        )}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontWeight: 600, fontSize: 13.5, color: C.text }}>{p.name || p.id}</span>
            <span style={{ fontSize: 10.5, color: C.faint }}>{p.waitHours.toFixed(0)}h waiting</span>
            {p.tied && (
              <span style={{ fontSize: 9.5, color: C.tie, background: "rgba(201,168,255,0.12)", borderRadius: 5, padding: "1px 6px" }}>
                TIE
              </span>
            )}
          </div>
          <div style={{ display: "flex", height: 5, borderRadius: 3, overflow: "hidden", marginTop: 6, background: "rgba(255,255,255,0.05)", maxWidth: 260 }}>
            <div style={{ width: `${p.sevContrib}%`, background: C.severity }} />
            <div style={{ width: `${p.survContrib}%`, background: C.survival }} />
            <div style={{ width: `${p.waitContrib}%`, background: C.wait }} />
          </div>
        </div>
        <div style={{ textAlign: "right", flexShrink: 0 }}>
          <div style={{ fontFamily: MONO_FONT, fontSize: 19, color: C.text, lineHeight: 1 }}>{p.score.toFixed(1)}</div>
          <div style={{ fontSize: 9, color: C.faint, marginTop: 2 }}>score</div>
        </div>
        {canAdmit && (
          <button
            onClick={() => onAdmit(p.id)}
            title="Admit to ICU"
            style={{
              background: "rgba(74,222,128,0.12)",
              border: `1px solid rgba(74,222,128,0.3)`,
              color: C.admit,
              borderRadius: 8,
              width: 30,
              height: 30,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: "pointer",
              flexShrink: 0,
            }}
          >
            <Bed size={14} />
          </button>
        )}
        <button
          onClick={() => onToggle(p.id)}
          style={{
            background: "transparent",
            border: `1px solid ${C.border}`,
            color: C.muted,
            borderRadius: 8,
            width: 30,
            height: 30,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
            flexShrink: 0,
          }}
        >
          {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
      </div>

      {(expanded || detailed) && (
        <div style={{ marginTop: 12, paddingTop: 12, borderTop: `1px solid ${C.border}` }}>
          <FactorBar label="Severity" color={C.severity} contrib={p.sevContrib} weightPct={Math.round(p.wS * 100)} />
          <FactorBar label="Survival likelihood" color={C.survival} contrib={p.survContrib} weightPct={Math.round(p.wV * 100)} />
          <FactorBar label="Wait time" color={C.wait} contrib={p.waitContrib} weightPct={Math.round(p.wW * 100)} />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10.5, color: C.faint, marginTop: 4 }}>
            <span>Raw: severity {p.severity} survival {p.survival} waited {p.waitHours.toFixed(0)}h</span>
            <button
              onClick={() => onSelectInsight(p.id)}
              style={{ background: "none", border: "none", color: C.survival, cursor: "pointer", fontSize: 10.5, fontFamily: DISPLAY_FONT }}
            >
              View full insight
            </button>
          </div>
          {p.tieReason && (
            <div style={{ marginTop: 8, fontSize: 10.5, color: C.tie, background: "rgba(201,168,255,0.08)", borderRadius: 6, padding: "6px 8px" }}>
              {p.tieReason}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// AI Assistant chat widget
// ---------------------------------------------------------------------------
function AssistantWidget({ open, onToggle, chat, chatInput, setChatInput, onSend, busy }) {
  const endRef = useRef(null);
  useEffect(() => {
    if (endRef.current) endRef.current.scrollIntoView({ behavior: "smooth" });
  }, [chat, open]);

  if (!open) {
    return (
      <button
        onClick={onToggle}
        style={{
          position: "fixed",
          left: 20,
          bottom: 20,
          background: "linear-gradient(135deg, #7C6CFF, #3FD9D0)",
          border: "none",
          borderRadius: 14,
          padding: "12px 18px",
          color: "#08110F",
          fontWeight: 700,
          fontSize: 13,
          fontFamily: DISPLAY_FONT,
          display: "flex",
          alignItems: "center",
          gap: 9,
          cursor: "pointer",
          boxShadow: "0 8px 24px rgba(63,217,208,0.25)",
          zIndex: 50,
        }}
      >
        <Bot size={17} />
        Ask AI Assistant
      </button>
    );
  }

  return (
    <div
      style={{
        position: "fixed",
        left: 20,
        bottom: 20,
        width: 360,
        maxWidth: "calc(100vw - 40px)",
        height: 460,
        maxHeight: "calc(100vh - 40px)",
        background: C.panelRaised,
        border: `1px solid ${C.borderStrong}`,
        borderRadius: 16,
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
        zIndex: 50,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "12px 14px",
          borderBottom: `1px solid ${C.border}`,
          background: C.panelAlt,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div
            style={{
              width: 26,
              height: 26,
              borderRadius: 8,
              background: "linear-gradient(135deg, #7C6CFF, #3FD9D0)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Bot size={14} color="#08110F" />
          </div>
          <div>
            <div style={{ fontSize: 12.5, fontWeight: 700, color: C.text, fontFamily: DISPLAY_FONT }}>AI Assistant</div>
            <div style={{ fontSize: 9.5, color: C.faint }}>Ask about patients, queue, or beds</div>
          </div>
        </div>
        <button
          onClick={onToggle}
          style={{ background: "none", border: "none", color: C.muted, cursor: "pointer", padding: 4 }}
        >
          <X size={16} />
        </button>
      </div>

      <div style={{ flex: 1, overflowY: "auto", padding: 14, display: "flex", flexDirection: "column", gap: 10 }}>
        {chat.length === 0 && (
          <div style={{ color: C.faint, fontSize: 11.5, lineHeight: 1.5 }}>
            Try asking:
            <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 8 }}>
              {["Who has the highest priority right now?", "How many ICU beds are free?", "Why is P104 ranked where it is?"].map((s) => (
                <button
                  key={s}
                  onClick={() => onSend(s)}
                  style={{
                    textAlign: "left",
                    background: C.panelAlt,
                    border: `1px solid ${C.border}`,
                    borderRadius: 8,
                    padding: "8px 10px",
                    color: C.muted,
                    fontSize: 11.5,
                    cursor: "pointer",
                  }}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {chat.map((m, i) => (
          <div
            key={i}
            style={{
              alignSelf: m.role === "user" ? "flex-end" : "flex-start",
              maxWidth: "88%",
              background: m.role === "user" ? "rgba(63,217,208,0.14)" : C.panelAlt,
              border: `1px solid ${m.role === "user" ? "rgba(63,217,208,0.3)" : C.border}`,
              borderRadius: 12,
              padding: "8px 11px",
              fontSize: 12.5,
              color: C.text,
              lineHeight: 1.45,
              whiteSpace: "pre-wrap",
            }}
          >
            {m.text}
          </div>
        ))}
        {busy && (
          <div style={{ alignSelf: "flex-start", display: "flex", alignItems: "center", gap: 6, color: C.faint, fontSize: 11.5 }}>
            <Loader2 size={13} className="spin" />
            thinking
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div style={{ display: "flex", gap: 8, padding: 12, borderTop: `1px solid ${C.border}` }}>
        <input
          value={chatInput}
          onChange={(e) => setChatInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && chatInput.trim() && !busy) {
              onSend(chatInput.trim());
            }
          }}
          placeholder="Ask a question"
          style={{
            flex: 1,
            background: C.panel,
            border: `1px solid ${C.border}`,
            borderRadius: 9,
            padding: "9px 11px",
            color: C.text,
            fontSize: 12.5,
            outline: "none",
          }}
        />
        <button
          onClick={() => chatInput.trim() && !busy && onSend(chatInput.trim())}
          disabled={!chatInput.trim() || busy}
          style={{
            background: chatInput.trim() && !busy ? C.survival : "rgba(255,255,255,0.06)",
            border: "none",
            borderRadius: 9,
            width: 36,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: chatInput.trim() && !busy ? "pointer" : "default",
            color: "#08110F",
          }}
        >
          <Send size={14} />
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main app
// ---------------------------------------------------------------------------
export default function App() {
  const [weights, setWeights] = useState({ severity: 50, survival: 30, wait: 20 });
  const [simTime, setSimTime] = useState(0);
  const [patients, setPatients] = useState(SEED_PATIENTS);
  const [admitted, setAdmitted] = useState([
  { id: "ICU1", name: "ICU Bed 1", severity: 0, survival: 0, admittedAt: 0 },
  { id: "ICU2", name: "ICU Bed 2", severity: 0, survival: 0, admittedAt: 0 },
  { id: "ICU3", name: "ICU Bed 3", severity: 0, survival: 0, admittedAt: 0 },
  { id: "ICU4", name: "ICU Bed 4", severity: 0, survival: 0, admittedAt: 0 },
  { id: "ICU5", name: "ICU Bed 5", severity: 0, survival: 0, admittedAt: 0 },
  { id: "ICU6", name: "ICU Bed 6", severity: 0, survival: 0, admittedAt: 0 },
]);
  const [log, setLog] = useState([
    { t: 0, text: "System initialized. ICU at full capacity (0/6 beds free). 3 patients queued.", tone: "info" },
  ]);
  const [expandedId, setExpandedId] = useState(null);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newPatient, setNewPatient] = useState({ name: "", severity: 60, survival: 60 });
  const [justMoved, setJustMoved] = useState({});
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeNav, setActiveNav] = useState("Dashboard");
  const [patientFilter, setPatientFilter] = useState("");

  const [insightId, setInsightId] = useState(null);
  const [explanation, setExplanation] = useState("");
  const [explaining, setExplaining] = useState(false);
  const [explainError, setExplainError] = useState(false);

  const [assistantOpen, setAssistantOpen] = useState(false);
  const [chat, setChat] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatBusy, setChatBusy] = useState(false);

  const prevRanksRef = useRef({});

  const scored = scorePatients(patients, weights, simTime);
  const ranked = rankPatients(scored);
  const beds = { total: BED_TOTAL, available: BED_TOTAL - admitted.length };
  const insightPatient = ranked.find((p) => p.id === insightId) || ranked[0] || null;
  const avgWait = ranked.length ? ranked.reduce((s, p) => s + p.waitHours, 0) / ranked.length : 0;

  const pushLog = useCallback(
    (text, tone = "info") => {
      setLog((l) => [...l, { t: simTime, text, tone }]);
    },
    [simTime]
  );

  // Detect rank movement vs. previous render, log it, and trigger the pulse.
  useEffect(() => {
    const prev = prevRanksRef.current;
    const moved = {};
    ranked.forEach((p) => {
      const before = prev[p.id];
      if (before !== undefined && before !== p.rank) {
        moved[p.id] = before - p.rank;
      }
    });
    if (Object.keys(moved).length) {
      setJustMoved(moved);
      const t = setTimeout(() => setJustMoved({}), 1800);
      const next = {};
      ranked.forEach((p) => (next[p.id] = p.rank));
      prevRanksRef.current = next;
      return () => clearTimeout(t);
    }
    const next = {};
    ranked.forEach((p) => (next[p.id] = p.rank));
    prevRanksRef.current = next;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ranked.map((p) => `${p.id}:${p.rank}`).join(",")]);

  // ---- actions -------------------------------------------------------
  const handleAddPatient = () => {
    const id = nextId();
    const sev = Math.max(1, Math.min(100, Number(newPatient.severity) || 60));
    const surv = Math.max(1, Math.min(100, Number(newPatient.survival) || 60));
    const p = {
      id,
      name: newPatient.name.trim() || `Patient ${id}`,
      severity: sev,
      survival: surv,
      arrivalTime: simTime,
    };
    setPatients((ps) => [...ps, p]);
    pushLog(`${p.name} (${id}) admitted to queue severity ${sev}, survival ${surv}.`, "info");
    setNewPatient({ name: "", severity: 60, survival: 60 });
    setShowAddForm(false);
  };

  const admitPatient = (id) => {
    if (beds.available <= 0) return;
    const p = ranked.find((x) => x.id === id);
    if (!p) return;
    setPatients((ps) => ps.filter((x) => x.id !== id));
    setAdmitted((a) => [...a, { id: p.id, name: p.name, severity: p.severity, survival: p.survival, admittedAt: simTime }]);
    pushLog(`${p.name || p.id} admitted to ICU from queue rank #${p.rank} (score ${p.score.toFixed(1)}).`, "admit");
  };

  const dischargePatient = (id) => {
    const p = admitted.find((x) => x.id === id);
    if (!p) return;
    setAdmitted((a) => a.filter((x) => x.id !== id));
    pushLog(`${p.name || p.id} discharged from ICU. Bed freed.`, "discharge");
  };

  const freeIcuBed = () => {
    if (admitted.length === 0) return;
    // Discharge the longest-admitted patient.
    const earliest = [...admitted].sort((a, b) => a.admittedAt - b.admittedAt)[0];
    dischargePatient(earliest.id);
  };

  const advanceTime = () => {
    setSimTime((t) => t + 6);
    pushLog(`Simulation time advanced to t=${simTime + 6}h.`, "info");
  };

  const runSimulation = () => {
    let freeBeds = beds.available;
    if (freeBeds <= 0 || ranked.length === 0) {
      pushLog(freeBeds <= 0 ? "Run simulation: no free beds available." : "Run simulation: queue is empty.", "info");
      return;
    }
    const toAdmit = ranked.slice(0, freeBeds);
    setPatients((ps) => ps.filter((p) => !toAdmit.find((t) => t.id === p.id)));
    setAdmitted((a) => [
      ...a,
      ...toAdmit.map((p) => ({ id: p.id, name: p.name, severity: p.severity, survival: p.survival, admittedAt: simTime })),
    ]);
    toAdmit.forEach((p) => pushLog(`Simulation: auto-admitted ${p.name || p.id} (rank #${p.rank}, score ${p.score.toFixed(1)}).`, "admit"));
  };

  const resetDemo = () => {
    idCounter = 200;
    setWeights({ severity: 50, survival: 30, wait: 20 });
    setSimTime(0);
    setPatients(SEED_PATIENTS);
    setAdmitted([]);
    setLog([{ t: 0, text: "System initialized. ICU at full capacity (0/6 beds free). 3 patients queued.", tone: "info" }]);
    setExpandedId(null);
    setInsightId(null);
    setExplanation("");
    setChat([]);
    prevRanksRef.current = {};
  };

  const generateExplanation = async () => {
    if (!insightPatient) return;
    setExplaining(true);
    setExplainError(false);
    const prompt = `${buildSystemContext({ ranked, admitted, beds, simTime, weights, log })}Explain in 2-3 sentences why ${insightPatient.id} is ranked #${insightPatient.rank} with score ${insightPatient.score.toFixed(1)}.`;
    try {
      const text = await askClaude(prompt);
      setExplanation(text);
    } catch (e) {
      setExplainError(true);
      setExplanation(localExplanation(insightPatient, ranked));
    } finally {
      setExplaining(false);
    }
  };

  const sendChat = async (text) => {
    setChat((c) => [...c, { role: "user", text }]);
    setChatInput("");
    setChatBusy(true);
    const prompt = buildSystemContext({ ranked, admitted, beds, simTime, weights, log }) + text;
    try {
      const reply = await askClaude(prompt);
      setChat((c) => [...c, { role: "assistant", text: reply }]);
    } catch (e) {
      setChat((c) => [...c, { role: "assistant", text: localAssistantAnswer(text, { ranked, admitted, beds, simTime }) }]);
    } finally {
      setChatBusy(false);
    }
  };

  const toggleExpand = (id) => setExpandedId((cur) => (cur === id ? null : id));
  const selectInsight = (id) => {
    setInsightId(id);
    setExplanation("");
    setExplainError(false);
  };

  const filteredPatients = ranked.filter(
    (p) => !patientFilter || p.id.toLowerCase().includes(patientFilter.toLowerCase()) || (p.name || "").toLowerCase().includes(patientFilter.toLowerCase())
  );

  const navItems = [
    { key: "Dashboard", icon: <LayoutDashboard size={16} /> },
    { key: "Patients", icon: <Users size={16} />, badge: ranked.length + admitted.length },
    { key: "Priority Queue", icon: <ListOrdered size={16} />, badge: ranked.length },
    { key: "Audit Log", icon: <ScrollText size={16} /> },
  ];

  return (
    <div
      style={{
        height: "100vh",
        width: "100%",
        background: C.bg,
        color: C.text,
        fontFamily: DISPLAY_FONT,
        display: "flex",
        overflow: "hidden",
      }}
    >
      <style>{`
        * { box-sizing: border-box; }
        input[type="range"] { -webkit-appearance: none; height: 4px; background: rgba(255,255,255,0.12); border-radius: 2px; }
        input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%; background: currentColor; cursor: pointer; }
        .spin { animation: spin 1s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        button:focus-visible, input:focus-visible { outline: 2px solid ${C.survival}; outline-offset: 1px; }
      `}</style>

      {/* Sidebar */}
      {sidebarOpen && (
        <aside
          style={{
            width: 200,
            flexShrink: 0,
            borderRight: `1px solid ${C.border}`,
            display: "flex",
            flexDirection: "column",
            padding: "16px 0",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 9, padding: "0 16px", marginBottom: 22 }}>
            <Activity size={19} color={C.survival} />
            <span style={{ fontWeight: 700, fontSize: 15.5 }}>CareGrid</span>
          </div>
          <nav style={{ display: "flex", flexDirection: "column", gap: 2, flex: 1 }}>
            {navItems.map((n) => (
              <NavItem
                key={n.key}
                icon={n.icon}
                label={n.key}
                badge={n.badge}
                active={activeNav === n.key}
                onClick={() => setActiveNav(n.key)}
              />
            ))}
          </nav>
          <div style={{ padding: "0 16px" }}>
            <div style={{ fontSize: 10, color: C.faint, lineHeight: 1.5 }}>
              "Transparent decisions.
              <br />
              Better outcomes."
            </div>
          </div>
        </aside>
      )}

      {/* Main column */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, overflow: "hidden" }}>
        {/* Top bar */}
        <header
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "11px 20px",
            borderBottom: `1px solid ${C.border}`,
            flexShrink: 0,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <button
              onClick={() => setSidebarOpen((s) => !s)}
              style={{ background: "none", border: "none", color: C.muted, cursor: "pointer", padding: 4 }}
            >
              <Menu size={17} />
            </button>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 7,
                background: C.panelAlt,
                border: `1px solid ${C.border}`,
                borderRadius: 999,
                padding: "5px 12px",
                fontSize: 11.5,
              }}
            >
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: C.admit, display: "inline-block" }} />
              SIMULATION MODE
              <span style={{ color: C.faint }}>|</span>
             <span style={{ color: C.muted }}>t={simTime}h</span>
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <Bell size={16} color={C.muted} />
            <Sun size={16} color={C.muted} />
            <div
              style={{
                width: 28,
                height: 28,
                borderRadius: "50%",
                background: "linear-gradient(135deg,#7C6CFF,#3FD9D0)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 12,
                fontWeight: 700,
                color: "#08110F",
              }}
            >
              N
            </div>
          </div>
        </header>

        {/* Metrics row — always visible, compact */}
        <div style={{ display: "flex", gap: 10, padding: "14px 20px 0", flexShrink: 0 }}>
          <MetricCard icon={<Bed size={15} color={C.admit} />} iconColor={C.admit} value={`${beds.available}/${beds.total}`} label="Beds free" />
          <MetricCard icon={<Users size={15} color={C.survival} />} iconColor={C.survival} value={ranked.length + admitted.length} label="In system" />
          <MetricCard icon={<Activity size={15} color={C.severity} />} iconColor={C.severity} value={ranked.length} label="In queue" />
          <MetricCard icon={<Clock size={15} color={C.wait} />} iconColor={C.wait} value={`${avgWait.toFixed(1)}h`} label="Avg wait" />
          <MetricCard icon={<PlayCircle size={15} color={C.tie} />} iconColor={C.tie} value={`${simTime}h`} label="Sim time" />
        </div>

        {/* Body — scrolls internally per-panel, page itself doesn't grow */}
        <div style={{ flex: 1, minHeight: 0, padding: 20, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          {activeNav === "Dashboard" && (
            <div style={{ flex: 1, minHeight: 0, display: "grid", gridTemplateColumns: "1.3fr 1fr", gap: 16 }}>
              {/* Left column */}
              <div style={{ display: "flex", flexDirection: "column", gap: 16, minHeight: 0 }}>
                <section
                  style={{
                    background: C.panel,
                    border: `1px solid ${C.border}`,
                    borderRadius: 12,
                    padding: 16,
                    display: "flex",
                    flexDirection: "column",
                    minHeight: 0,
                    flex: "1 1 55%",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 2 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <ListOrdered size={15} color={C.severity} />
                      <h2 style={{ fontFamily: DISPLAY_FONT, fontSize: 13, textTransform: "uppercase", letterSpacing: 0.8, color: C.muted, margin: 0 }}>
                        Priority Queue
                      </h2>
                    </div>
                    <button
                      onClick={() => setActiveNav("Priority Queue")}
                      style={{ background: "none", border: "none", color: C.survival, fontSize: 11, cursor: "pointer", fontFamily: DISPLAY_FONT }}
                    >
                      View full queue
                    </button>
                  </div>
                  <p style={{ fontSize: 11, color: C.faint, margin: "0 0 10px" }}>Real-time patient prioritization.</p>
                  <div style={{ overflowY: "auto", flex: 1, minHeight: 0, paddingRight: 2 }}>
                    {ranked.length === 0 ? (
                      <div style={{ color: C.faint, fontSize: 12, padding: "24px 0", textAlign: "center" }}>
                        No patients waiting. Add one from Quick Actions.
                      </div>
                    ) : (
                      ranked.slice(0, 5).map((p) => (
                        <QueueRow
                          key={p.id}
                          p={p}
                          expanded={expandedId === p.id}
                          onToggle={toggleExpand}
                          onSelectInsight={selectInsight}
                          onAdmit={admitPatient}
                          canAdmit={beds.available > 0}
                          justMoved={justMoved[p.id]}
                        />
                      ))
                    )}
                  </div>
                </section>

                <ScoreCompositionChart ranked={ranked} />
              </div>

              {/* Right column */}
              <div style={{ display: "flex", flexDirection: "column", gap: 16, minHeight: 0, overflowY: "auto", paddingRight: 2 }}>
                <section style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 12, padding: 16 }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
                    <h2 style={{ fontFamily: DISPLAY_FONT, fontSize: 13, textTransform: "uppercase", letterSpacing: 0.8, color: C.muted, margin: 0 }}>
                      Patient Insight
                    </h2>
                    {ranked.length > 0 && (
                      <select
                        value={insightPatient ? insightPatient.id : ""}
                        onChange={(e) => selectInsight(e.target.value)}
                        style={{
                          background: C.panelAlt,
                          border: `1px solid ${C.border}`,
                          color: C.text,
                          borderRadius: 7,
                          fontSize: 11.5,
                          padding: "4px 8px",
                        }}
                      >
                        {ranked.map((p) => (
                          <option key={p.id} value={p.id}>
                            {p.id}
                          </option>
                        ))}
                      </select>
                    )}
                  </div>

                  {!insightPatient ? (
                    <div style={{ color: C.faint, fontSize: 12 }}>No patients to inspect queue is empty.</div>
                  ) : (
                    <>
                      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
                        <ScoreRing score={insightPatient.score} />
                        <div style={{ flex: 1 }}>
                          <div style={{ fontWeight: 700, fontSize: 15 }}>{insightPatient.name || insightPatient.id}</div>
                          <div style={{ fontSize: 10.5, color: C.faint }}>{insightPatient.waitHours.toFixed(0)}h in queue</div>
                        </div>
                        <div
                          style={{
                            background: "rgba(255,255,255,0.05)",
                            borderRadius: 9,
                            padding: "6px 10px",
                            textAlign: "center",
                          }}
                        >
                          <div style={{ fontFamily: MONO_FONT, fontSize: 15, color: C.severity }}>#{insightPatient.rank}</div>
                          <div style={{ fontSize: 8.5, color: C.faint, textTransform: "uppercase" }}>rank</div>
                        </div>
                      </div>

                      <FactorBar label="Severity" color={C.severity} contrib={insightPatient.sevContrib} weightPct={Math.round(insightPatient.wS * 100)} />
                      <FactorBar label="Survival likelihood" color={C.survival} contrib={insightPatient.survContrib} weightPct={Math.round(insightPatient.wV * 100)} />
                      <FactorBar label="Wait time" color={C.wait} contrib={insightPatient.waitContrib} weightPct={Math.round(insightPatient.wW * 100)} />

                      <div style={{ background: C.panelAlt, border: `1px solid ${C.border}`, borderRadius: 10, padding: 12, marginTop: 6 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, fontWeight: 700, color: C.tie, marginBottom: 6 }}>
                          <Sparkles size={12} /> AI EXPLANATION
                        </div>
                        {explanation ? (
                          <p style={{ fontSize: 11.5, color: C.text, lineHeight: 1.5, margin: "0 0 8px" }}>
                            {explanation}
                            {explainError && <span style={{ color: C.faint }}> (offline fallback)</span>}
                          </p>
                        ) : (
                          <p style={{ fontSize: 11.5, color: C.faint, margin: "0 0 8px" }}>
                            Generate an explanation for why this patient is ranked where it is.
                          </p>
                        )}
                        <button
                          onClick={generateExplanation}
                          disabled={explaining}
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 6,
                            background: "rgba(201,168,255,0.12)",
                            border: `1px solid rgba(201,168,255,0.3)`,
                            color: C.tie,
                            borderRadius: 8,
                            padding: "6px 11px",
                            fontSize: 11.5,
                            cursor: explaining ? "default" : "pointer",
                          }}
                        >
                          {explaining ? <Loader2 size={12} className="spin" /> : <RotateCcw size={12} />}
                          {explaining ? "Generating" : explanation ? "Regenerate explanation" : "Generate explanation"}
                        </button>
                      </div>
                    </>
                  )}
                </section>

                <section style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 12, padding: 16 }}>
                  <h2 style={{ fontFamily: DISPLAY_FONT, fontSize: 13, textTransform: "uppercase", letterSpacing: 0.8, color: C.muted, margin: "0 0 12px" }}>
                    Scoring Weights
                  </h2>
                  <WeightSlider label="Severity" color={C.severity} value={weights.severity} onChange={(v) => setWeights((w) => ({ ...w, severity: v }))} />
                  <WeightSlider label="Survival" color={C.survival} value={weights.survival} onChange={(v) => setWeights((w) => ({ ...w, survival: v }))} />
                  <WeightSlider label="Wait time" color={C.wait} value={weights.wait} onChange={(v) => setWeights((w) => ({ ...w, wait: v }))} />
                  <div style={{ fontSize: 10, color: C.faint, marginTop: 2 }}>
                    Auto-normalized ({Math.round((weights.severity / (weights.severity + weights.survival + weights.wait || 1)) * 100)}% /{" "}
                    {Math.round((weights.survival / (weights.severity + weights.survival + weights.wait || 1)) * 100)}% /{" "}
                    {Math.round((weights.wait / (weights.severity + weights.survival + weights.wait || 1)) * 100)}%). Wait normalizes to 100% at {MAX_WAIT_HOURS}h.
                  </div>
                </section>

                <CapacityDonut beds={beds} admitted={admitted} />
              </div>
            </div>
          )}

          {activeNav === "Patients" && (
            <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column", gap: 14 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
                <div style={{ position: "relative", flex: 1, maxWidth: 320 }}>
                  <Search size={14} color={C.faint} style={{ position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)" }} />
                  <input
                    value={patientFilter}
                    onChange={(e) => setPatientFilter(e.target.value)}
                    placeholder="Search by ID or name"
                    style={{
                      width: "100%",
                      background: C.panel,
                      border: `1px solid ${C.border}`,
                      borderRadius: 9,
                      padding: "8px 10px 8px 32px",
                      color: C.text,
                      fontSize: 12.5,
                      outline: "none",
                    }}
                  />
                </div>
                <button
                  onClick={() => setShowAddForm(true)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 7,
                    background: C.survival,
                    border: "none",
                    borderRadius: 9,
                    padding: "8px 14px",
                    color: "#08110F",
                    fontSize: 12.5,
                    fontWeight: 700,
                    cursor: "pointer",
                  }}
                >
                  <UserPlus size={14} /> New patient
                </button>
              </div>

              <div style={{ flex: 1, minHeight: 0, overflowY: "auto", display: "flex", flexDirection: "column", gap: 18, paddingRight: 2 }}>
                <div>
                  <h3 style={{ fontFamily: DISPLAY_FONT, fontSize: 12, textTransform: "uppercase", letterSpacing: 0.8, color: C.muted, margin: "0 0 10px" }}>
                    Waiting queue {filteredPatients.length}
                  </h3>
                  {filteredPatients.length === 0 ? (
                    <div style={{ color: C.faint, fontSize: 12, padding: "10px 0" }}>No matching patients waiting.</div>
                  ) : (
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 12 }}>
                      {filteredPatients.map((p) => (
                        <div key={p.id} style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 12, padding: 14 }}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                            <div>
                              <div style={{ fontWeight: 700, fontSize: 13.5 }}>{p.name || p.id}</div>
                              <div style={{ fontSize: 10.5, color: C.faint, fontFamily: MONO_FONT }}>{p.id}</div>
                            </div>
                            <div
                              style={{
                                background: "rgba(255,255,255,0.05)",
                                borderRadius: 8,
                                padding: "3px 8px",
                                fontFamily: MONO_FONT,
                                fontSize: 11,
                                color: C.severity,
                              }}
                            >
                              #{p.rank}
                            </div>
                          </div>
                          <div style={{ display: "flex", gap: 10, margin: "10px 0", fontSize: 10.5, color: C.muted }}>
                            <span>Severity <b style={{ color: C.text }}>{p.severity}</b></span>
                            <span>Survival <b style={{ color: C.text }}>{p.survival}</b></span>
                            <span>Waited <b style={{ color: C.text }}>{p.waitHours.toFixed(0)}h</b></span>
                          </div>
                          <div style={{ display: "flex", height: 5, borderRadius: 3, overflow: "hidden", background: "rgba(255,255,255,0.05)", marginBottom: 12 }}>
                            <div style={{ width: `${p.sevContrib}%`, background: C.severity }} />
                            <div style={{ width: `${p.survContrib}%`, background: C.survival }} />
                            <div style={{ width: `${p.waitContrib}%`, background: C.wait }} />
                          </div>
                          <div style={{ display: "flex", gap: 8 }}>
                            <button
                              onClick={() => {
                                selectInsight(p.id);
                                setActiveNav("Dashboard");
                              }}
                              style={{
                                flex: 1,
                                background: "transparent",
                                border: `1px solid ${C.border}`,
                                color: C.muted,
                                borderRadius: 8,
                                padding: "7px 0",
                                fontSize: 11.5,
                                cursor: "pointer",
                              }}
                            >
                              View insight
                            </button>
                            <button
                              onClick={() => admitPatient(p.id)}
                              disabled={beds.available <= 0}
                              style={{
                                flex: 1,
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                gap: 5,
                                background: beds.available > 0 ? "rgba(74,222,128,0.12)" : "rgba(255,255,255,0.03)",
                                border: `1px solid ${beds.available > 0 ? "rgba(74,222,128,0.3)" : C.border}`,
                                color: beds.available > 0 ? C.admit : C.faint,
                                borderRadius: 8,
                                padding: "7px 0",
                                fontSize: 11.5,
                                cursor: beds.available > 0 ? "pointer" : "default",
                              }}
                            >
                              <Bed size={12} /> Admit
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div>
                  <h3 style={{ fontFamily: DISPLAY_FONT, fontSize: 12, textTransform: "uppercase", letterSpacing: 0.8, color: C.muted, margin: "0 0 10px" }}>
                    Currently admitted {admitted.length}
                  </h3>
                  {admitted.length === 0 ? (
                    <div style={{ color: C.faint, fontSize: 12, padding: "10px 0" }}>No patients currently admitted.</div>
                  ) : (
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 12 }}>
                      {admitted.map((p) => (
                        <div key={p.id} style={{ background: C.panel, border: `1px solid rgba(74,222,128,0.25)`, borderRadius: 12, padding: 14 }}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                            <div>
                              <div style={{ fontWeight: 700, fontSize: 13.5 }}>{p.name || p.id}</div>
                              <div style={{ fontSize: 10.5, color: C.faint, fontFamily: MONO_FONT }}>{p.id}</div>
                            </div>
                            <CheckCircle2 size={16} color={C.admit} />
                          </div>
                          <div style={{ display: "flex", gap: 10, margin: "10px 0", fontSize: 10.5, color: C.muted }}>
                            <span>Severity <b style={{ color: C.text }}>{p.severity}</b></span>
                            <span>Survival <b style={{ color: C.text }}>{p.survival}</b></span>
                            <span>Since <b style={{ color: C.text }}>t={p.admittedAt}h</b></span>
                          </div>
                          <button
                            onClick={() => dischargePatient(p.id)}
                            style={{
                              width: "100%",
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              gap: 6,
                              background: "rgba(255,107,91,0.1)",
                              border: `1px solid rgba(255,107,91,0.3)`,
                              color: C.severity,
                              borderRadius: 8,
                              padding: "7px 0",
                              fontSize: 11.5,
                              cursor: "pointer",
                            }}
                          >
                            <LogOut size={12} /> Discharge
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {activeNav === "Priority Queue" && (
            <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12, flexShrink: 0 }}>
                <div>
                  <h2 style={{ fontFamily: DISPLAY_FONT, fontSize: 15, margin: 0 }}>Full Priority Queue</h2>
                  <p style={{ fontSize: 11.5, color: C.faint, margin: "3px 0 0" }}>
                    Every waiting patient, ranked live, with the full factor breakdown behind each score.
                  </p>
                </div>
                {beds.available > 0 ? (
                  <span style={{ fontSize: 11, color: C.admit, background: "rgba(74,222,128,0.1)", borderRadius: 8, padding: "5px 10px" }}>
                    {beds.available} bed{beds.available !== 1 ? "s" : ""} free admit directly from here
                  </span>
                ) : (
                  <span style={{ fontSize: 11, color: C.severity, background: "rgba(255,107,91,0.1)", borderRadius: 8, padding: "5px 10px", display: "flex", alignItems: "center", gap: 5 }}>
                    <AlertTriangle size={12} /> ICU at full capacity
                  </span>
                )}
              </div>
              <div style={{ flex: 1, minHeight: 0, overflowY: "auto", paddingRight: 2 }}>
                {ranked.length === 0 ? (
                  <div style={{ color: C.faint, fontSize: 13, padding: "40px 0", textAlign: "center" }}>
                    The queue is empty. Add a patient from the Patients tab.
                  </div>
                ) : (
                  ranked.map((p) => (
                    <QueueRow
                      key={p.id}
                      p={p}
                      expanded
                      detailed
                      onToggle={toggleExpand}
                      onSelectInsight={(id) => {
                        selectInsight(id);
                        setActiveNav("Dashboard");
                      }}
                      onAdmit={admitPatient}
                      canAdmit={beds.available > 0}
                      justMoved={justMoved[p.id]}
                    />
                  ))
                )}
              </div>
            </div>
          )}

          {activeNav === "Audit Log" && (
            <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
              <h2 style={{ fontFamily: DISPLAY_FONT, fontSize: 15, margin: "0 0 12px", flexShrink: 0 }}>Audit Log</h2>
              <div
                style={{
                  flex: 1,
                  minHeight: 0,
                  overflowY: "auto",
                  background: C.panel,
                  border: `1px solid ${C.border}`,
                  borderRadius: 12,
                  padding: 16,
                  fontFamily: MONO_FONT,
                  fontSize: 12,
                }}
              >
                {[...log].reverse().map((l, i) => (
                  <div
                    key={i}
                    style={{
                      padding: "7px 0",
                      borderBottom: i < log.length - 1 ? `1px solid ${C.border}` : "none",
                      color: l.tone === "admit" ? C.admit : l.tone === "discharge" ? C.wait : C.muted,
                    }}
                  >
                    <span style={{ color: C.faint }}>[t={l.t}h]</span> {l.text}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Quick actions — always visible at the bottom, compact single row */}
        <div style={{ padding: "0 20px 16px", flexShrink: 0 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10 }}>
            <QuickAction icon={<UserPlus size={15} color={C.survival} />} label="New patient" sub="Add to queue" onClick={() => setShowAddForm(true)} />
            <QuickAction
              icon={<Bed size={15} color={C.wait} />}
              label="Free ICU bed"
              sub={admitted.length ? "Discharge earliest" : "Nothing to free"}
              onClick={freeIcuBed}
              disabled={admitted.length === 0}
            />
            <QuickAction icon={<Clock size={15} color={C.tie} />} label="Advance time" sub="+6 hours" onClick={advanceTime} />
            <QuickAction
              icon={<PlayCircle size={15} color={C.admit} />}
              label="Run simulation"
              sub="Auto-admit top ranks"
              onClick={runSimulation}
              disabled={beds.available === 0 || ranked.length === 0}
            />
            <QuickAction icon={<RotateCcw size={15} color={C.severity} />} label="Reset demo" sub="Start over" onClick={resetDemo} />
          </div>
        </div>
      </div>

      {/* Add patient modal */}
      {showAddForm && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.55)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 100,
          }}
          onClick={() => setShowAddForm(false)}
        >
          <div
            style={{ background: C.panelRaised, border: `1px solid ${C.borderStrong}`, borderRadius: 14, padding: 22, width: 340 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
              <h3 style={{ margin: 0, fontFamily: DISPLAY_FONT, fontSize: 15 }}>New patient</h3>
              <button onClick={() => setShowAddForm(false)} style={{ background: "none", border: "none", color: C.muted, cursor: "pointer" }}>
                <X size={16} />
              </button>
            </div>
            <label style={{ fontSize: 11, color: C.muted }}>Name (optional)</label>
            <input
              value={newPatient.name}
              onChange={(e) => setNewPatient((p) => ({ ...p, name: e.target.value }))}
              placeholder="e.g. Patient P210"
              style={{
                width: "100%",
                background: C.panel,
                border: `1px solid ${C.border}`,
                borderRadius: 8,
                padding: "8px 10px",
                color: C.text,
                fontSize: 12.5,
                margin: "5px 0 14px",
                outline: "none",
              }}
            />
            <WeightSlider label="Severity" color={C.severity} value={newPatient.severity} onChange={(v) => setNewPatient((p) => ({ ...p, severity: v }))} />
            <WeightSlider label="Survival likelihood" color={C.survival} value={newPatient.survival} onChange={(v) => setNewPatient((p) => ({ ...p, survival: v }))} />
            <button
              onClick={handleAddPatient}
              style={{
                width: "100%",
                background: C.survival,
                border: "none",
                borderRadius: 9,
                padding: "10px 0",
                color: "#08110F",
                fontWeight: 700,
                fontSize: 13,
                cursor: "pointer",
                marginTop: 8,
              }}
            >
              Add to queue
            </button>
          </div>
        </div>
      )}

      <AssistantWidget
        open={assistantOpen}
        onToggle={() => setAssistantOpen((o) => !o)}
        chat={chat}
        chatInput={chatInput}
        setChatInput={setChatInput}
        onSend={sendChat}
        busy={chatBusy}
      />
    </div>
  );
}

function QuickAction({ icon, label, sub, onClick, disabled }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        background: C.panel,
        border: `1px solid ${C.border}`,
        borderRadius: 11,
        padding: "11px 12px",
        cursor: disabled ? "default" : "pointer",
        opacity: disabled ? 0.45 : 1,
        textAlign: "left",
      }}
    >
      {icon}
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: C.text, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{label}</div>
        <div style={{ fontSize: 9.5, color: C.faint, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{sub}</div>
      </div>
    </button>
  );
}