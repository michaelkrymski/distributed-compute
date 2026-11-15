"use client";
import { useState } from "react";
import { phases } from "../lib/project";
export default function ProjectForecast() {
  const [days, setDays] = useState([3, 2, 4, 2, 3, 2]),
    [day, setDay] = useState(0);
  const total = days.reduce((a, b) => a + b, 0),
    current = Math.min(day, total);
  let elapsed = 0;
  return (
    <section className="card project-forecast">
      <div className="section-title">
        <div>
          <span className="eyebrow">WHAT COULD THE PROJECT LOOK LIKE?</span>
          <h2>Explore the whole timeline</h2>
        </div>
        <span className="badge orange">Hypothetical schedule</span>
      </div>
      <p className="muted">
        Set a duration for each phase, then scrub forward. This forecast does
        not update your team milestones.
      </p>
      <div className="forecast-phases">
        {phases.map((p, i) => {
          const start = elapsed;
          elapsed += days[i];
          const fraction = Math.max(
            0,
            Math.min(1, (current - start) / days[i]),
          );
          return (
            <label key={p.id}>
              <span>
                {String(i + 1).padStart(2, "0")} · {p.title}
              </span>
              <div className="forecast-phase-bar">
                <i style={{ width: `${fraction * 100}%` }} />
              </div>
              <span className="duration">
                <input
                  aria-label={`${p.title} planned days`}
                  type="number"
                  min="1"
                  max="60"
                  value={days[i]}
                  onChange={(e) =>
                    setDays((values) =>
                      values.map((v, j) =>
                        i === j
                          ? Math.max(
                              1,
                              Math.min(60, Number(e.target.value) || 1),
                            )
                          : v,
                      ),
                    )
                  }
                />
                <small>days</small>
                <b>
                  {fraction === 1
                    ? "Done"
                    : fraction > 0
                      ? "Active"
                      : "Upcoming"}
                </b>
              </span>
            </label>
          );
        })}
      </div>
      <div className="forecast-scrubber">
        <label htmlFor="project-day">
          Simulated day <b>{current.toFixed(1)}</b> / {total}
        </label>
        <input
          id="project-day"
          type="range"
          min="0"
          max={total}
          step="0.1"
          value={current}
          onChange={(e) => setDay(Number(e.target.value))}
        />
        <strong>{Math.round((current / total) * 100)}% of planned time</strong>
      </div>
      <p className="caption">
        Illustrative sequential phases and user-entered durations; no claim
        about feasibility or model quality. Phases may overlap in real work.
      </p>
    </section>
  );
}
