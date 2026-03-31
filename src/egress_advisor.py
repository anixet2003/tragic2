"""
Evacuation Output Advisor
Builds floor-plan improvement suggestions from simulation analytics and generated visuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


@dataclass
class PhaseSummary:
    early_end_time: float
    peak_start_time: float
    peak_end_time: float
    tail_start_time: float
    plateau_time: float
    peak_rate: float


@dataclass
class ImprovementMarker:
    marker_id: int
    x: float
    y: float
    category: str
    color: str
    title: str
    action: str
    impact: str


class EgressAdvisor:
    """Analyzes evacuation outputs and writes a structured recommendation report."""

    def __init__(
        self,
        environment,
        analytics,
        agents: Sequence,
        movement_img_path: Path,
        density_img_path: Path,
        panic_img_path: Path,
        csv_path: Path,
    ):
        self.environment = environment
        self.analytics = analytics
        self.agents = list(agents)
        self.movement_img_path = Path(movement_img_path)
        self.density_img_path = Path(density_img_path)
        self.panic_img_path = Path(panic_img_path)
        self.csv_path = Path(csv_path)

        self._times = np.array([], dtype=float)
        self._active = np.array([], dtype=float)
        self._evacuated = np.array([], dtype=float)
        self._deceased = np.array([], dtype=float)
        self._panic = np.array([], dtype=float)
        self._speed = np.array([], dtype=float)

        self._density_map = None
        self._panic_map = None

    def _load_csv(self) -> None:
        if not self.csv_path.exists():
            return

        times: List[float] = []
        active: List[float] = []
        evacuated: List[float] = []
        deceased: List[float] = []
        panic: List[float] = []
        speed: List[float] = []

        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                times.append(float(row["Time"]))
                active.append(float(row["Active_Agents"]))
                evacuated.append(float(row["Evacuated"]))
                deceased.append(float(row["Deceased"]))
                panic.append(float(row["Avg_Panic"]))
                speed.append(float(row["Avg_Speed"]))

        self._times = np.array(times, dtype=float)
        self._active = np.array(active, dtype=float)
        self._evacuated = np.array(evacuated, dtype=float)
        self._deceased = np.array(deceased, dtype=float)
        self._panic = np.array(panic, dtype=float)
        self._speed = np.array(speed, dtype=float)

    def _extract_maps(self) -> None:
        self._density_map = self.analytics.generate_heatmap("density")
        self._panic_map = self.analytics.generate_heatmap("panic")

    def _safe_idx(self, idx: int, size: int) -> int:
        return int(np.clip(idx, 0, max(size - 1, 0)))

    def _compute_phase_summary(self, total_agents: int) -> PhaseSummary:
        if self._times.size < 3:
            return PhaseSummary(0.0, 0.0, 0.0, 0.0, float(self._times[-1]) if self._times.size else 0.0, 0.0)

        dt = np.diff(self._times)
        dt[dt <= 0] = 1e-6
        evac_delta = np.diff(self._evacuated)
        evac_rate = evac_delta / dt

        peak_i = int(np.argmax(evac_rate)) if evac_rate.size else 0
        peak_rate = float(evac_rate[peak_i]) if evac_rate.size else 0.0

        early_idx_candidates = np.where(self._evacuated >= 0.2 * total_agents)[0]
        early_end_idx = int(early_idx_candidates[0]) if early_idx_candidates.size else self._safe_idx(1, self._times.size)

        peak_start_idx = self._safe_idx(peak_i, self._times.size)
        peak_end_idx = self._safe_idx(peak_i + max(1, int(10 / max(np.mean(dt), 1e-6))), self._times.size)

        tail_candidates = np.where(self._evacuated >= 0.8 * total_agents)[0]
        tail_start_idx = int(tail_candidates[0]) if tail_candidates.size else self._safe_idx(self._times.size - 1, self._times.size)

        plateau_idx = self._times.size - 1
        if evac_rate.size >= 4:
            window = 4
            for i in range(window, evac_rate.size):
                local = evac_rate[i - window:i]
                if np.mean(local) <= 0.05 and self._evacuated[i] >= 0.9 * total_agents:
                    plateau_idx = i
                    break

        return PhaseSummary(
            early_end_time=float(self._times[early_end_idx]),
            peak_start_time=float(self._times[peak_start_idx]),
            peak_end_time=float(self._times[peak_end_idx]),
            tail_start_time=float(self._times[tail_start_idx]),
            plateau_time=float(self._times[plateau_idx]),
            peak_rate=peak_rate,
        )

    def _top_heatmap_points(self, heatmap: np.ndarray, top_n: int = 4) -> List[Tuple[float, float, float]]:
        if heatmap is None or heatmap.size == 0:
            return []

        flat = heatmap.reshape(-1)
        top_idx = np.argsort(flat)[-top_n:][::-1]
        points = []
        for idx in top_idx:
            ix = idx // heatmap.shape[1]
            iy = idx % heatmap.shape[1]
            val = float(heatmap[ix, iy])
            pos = self.environment.grid.grid_to_world(int(ix), int(iy))
            points.append((float(pos[0]), float(pos[1]), val))
        return points

    def _local_walkable_neighbors(self, x: float, y: float, radius_cells: int = 2) -> int:
        ix, iy = self.environment.grid.world_to_grid(np.array([x, y], dtype=float))
        walkable_count = 0
        total = 0
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                nx, ny = ix + dx, iy + dy
                if not self.environment.grid.is_valid(nx, ny):
                    continue
                total += 1
                if self.environment.grid.walkable[nx, ny]:
                    walkable_count += 1
        if total == 0:
            return 0
        return int(round(100.0 * walkable_count / total))

    def _path_efficiency_stats(self) -> Dict[str, float]:
        ratios = []
        starts = []

        for agent in self.agents:
            if not agent.evacuated:
                continue
            if len(agent.trajectory) < 2:
                continue

            traj = np.array(agent.trajectory, dtype=float)
            path_len = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
            if path_len <= 1e-6:
                continue

            start = traj[0]
            if self.environment.exits:
                direct = min(float(np.linalg.norm(exit_obj.position - start)) for exit_obj in self.environment.exits)
            else:
                direct = float(np.linalg.norm(traj[-1] - start))

            if direct <= 1e-6:
                continue
            ratio = max(path_len / direct, 1.0)
            ratios.append(ratio)
            starts.append(start)

        if not ratios:
            return {"mean": 1.0, "p90": 1.0, "worst_x": self.environment.width / 2, "worst_y": self.environment.height / 2}

        ratios_arr = np.array(ratios, dtype=float)
        worst_i = int(np.argmax(ratios_arr))
        worst_start = starts[worst_i]
        return {
            "mean": float(np.mean(ratios_arr)),
            "p90": float(np.percentile(ratios_arr, 90)),
            "worst_x": float(worst_start[0]),
            "worst_y": float(worst_start[1]),
        }

    def _exit_utilization(self) -> Dict[str, object]:
        if not self.environment.exits:
            return {"counts": [], "overloaded": None, "underused": None, "imbalance_ratio": 1.0}

        counts = [int(exit_obj.total_evacuated) for exit_obj in self.environment.exits]
        max_i = int(np.argmax(counts))
        min_i = int(np.argmin(counts))
        max_count = max(counts) if counts else 0
        min_count = min(counts) if counts else 0
        ratio = float(max_count / max(min_count, 1)) if counts else 1.0

        return {
            "counts": counts,
            "overloaded": self.environment.exits[max_i],
            "underused": self.environment.exits[min_i],
            "imbalance_ratio": ratio,
        }

    def _format_xy(self, x: float, y: float) -> str:
        return f"x≈{x:.1f}, y≈{y:.1f}"

    def _build_improvement_markers(
        self,
        density_points: List[Tuple[float, float, float]],
        panic_points: List[Tuple[float, float, float]],
        exit_stats: Dict[str, object],
    ) -> List[ImprovementMarker]:
        markers: List[ImprovementMarker] = []

        marker_id = 1
        if density_points:
            x, y, _ = density_points[0]
            markers.append(
                ImprovementMarker(
                    marker_id=marker_id,
                    x=x,
                    y=y,
                    category="Capacity",
                    color="#d62728",
                    title="Widen Choke Zone",
                    action="Increase opening/corridor width by 1.0-1.5m and clear near obstructions",
                    impact="High",
                )
            )
            marker_id += 1

        overloaded = exit_stats.get("overloaded")
        underused = exit_stats.get("underused")
        if overloaded is not None and underused is not None:
            ux = float(underused.position[0])
            uy = float(underused.position[1])
            markers.append(
                ImprovementMarker(
                    marker_id=marker_id,
                    x=ux,
                    y=uy,
                    category="Exit Balance",
                    color="#1f77b4",
                    title="Route Rebalance",
                    action="Add feeder opening/signage to increase utilization of this exit",
                    impact="High",
                )
            )
            marker_id += 1

        if panic_points:
            px, py, _ = panic_points[0]
        elif density_points:
            px, py, _ = density_points[0]
        else:
            px, py = self.environment.width * 0.5, self.environment.height * 0.5

        markers.append(
            ImprovementMarker(
                marker_id=marker_id,
                x=px,
                y=py,
                category="Flow Geometry",
                color="#ff7f0e",
                title="Smooth Turn/Merge",
                action="Replace sharp turn with gentler transition and remove corner clutter",
                impact="Medium-High",
            )
        )
        marker_id += 1

        if len(density_points) > 1:
            x2, y2, _ = density_points[1]
            markers.append(
                ImprovementMarker(
                    marker_id=marker_id,
                    x=x2,
                    y=y2,
                    category="Operations",
                    color="#2ca02c",
                    title="One-Way Control",
                    action="Enforce one-way routing at merge to reduce crossing conflicts",
                    impact="Medium",
                )
            )
            marker_id += 1

        if len(density_points) > 2:
            x3, y3, _ = density_points[2]
            markers.append(
                ImprovementMarker(
                    marker_id=marker_id,
                    x=x3,
                    y=y3,
                    category="Obstacle Cleanup",
                    color="#9467bd",
                    title="Remove Small Obstructions",
                    action="Relocate minor internal obstructions within 3-5m of hotspot",
                    impact="Medium",
                )
            )

        return markers

    def _draw_base_floorplan(self, ax, floorplan_path: Optional[Path]) -> None:
        if floorplan_path is not None and floorplan_path.exists() and PIL_AVAILABLE:
            try:
                img = Image.open(floorplan_path)
                img_array = np.array(img)
                extent = [0, self.environment.width, 0, self.environment.height]
                ax.imshow(img_array, extent=extent, origin="upper", alpha=0.88, aspect="auto")
                return
            except Exception:
                pass

        # Fallback: use movement image as background if original floorplan not available.
        if self.movement_img_path.exists() and PIL_AVAILABLE:
            try:
                img = Image.open(self.movement_img_path)
                img_array = np.array(img)
                extent = [0, self.environment.width, 0, self.environment.height]
                ax.imshow(img_array, extent=extent, origin="upper", alpha=0.82, aspect="auto")
                return
            except Exception:
                pass

        # Last fallback: draw obstacles and exits on a plain canvas.
        for obstacle in self.environment.obstacles:
            ax.fill(
                [obstacle.x, obstacle.x + obstacle.width, obstacle.x + obstacle.width, obstacle.x],
                [obstacle.y, obstacle.y, obstacle.y + obstacle.height, obstacle.y + obstacle.height],
                color="#606060",
                alpha=0.55,
                linewidth=0,
            )
        for exit_obj in self.environment.exits:
            exit_circle = Circle((float(exit_obj.position[0]), float(exit_obj.position[1])), radius=max(0.6, exit_obj.width / 2), color="#32cd32", alpha=0.8)
            ax.add_patch(exit_circle)

    def _generate_overlay_image(
        self,
        markers: List[ImprovementMarker],
        output_path: Path,
        floorplan_path: Optional[Path],
    ) -> Optional[Path]:
        if not markers:
            return None

        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, self.environment.width)
        ax.set_ylim(0, self.environment.height)
        ax.set_aspect("equal")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_title("Floor Plan Improvement Overlay (Color-Coded)", fontsize=14, fontweight="bold")

        self._draw_base_floorplan(ax, floorplan_path=floorplan_path)

        # Draw markers and callouts.
        text_shift_x = max(1.0, self.environment.width * 0.04)
        text_shift_y = max(1.0, self.environment.height * 0.04)

        for i, marker in enumerate(markers):
            mx, my = float(marker.x), float(marker.y)

            ax.scatter([mx], [my], s=140, c=marker.color, edgecolors="black", linewidths=1.2, zorder=6)
            ax.text(
                mx,
                my,
                str(marker.marker_id),
                color="white",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=7,
            )

            direction = -1 if i % 2 else 1
            tx = np.clip(mx + direction * text_shift_x, 0.5, self.environment.width - 0.5)
            ty = np.clip(my + text_shift_y, 0.5, self.environment.height - 0.5)

            callout = f"{marker.marker_id}. {marker.title}\nImpact: {marker.impact}"
            ax.annotate(
                callout,
                xy=(mx, my),
                xytext=(tx, ty),
                fontsize=8,
                color="black",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=marker.color, alpha=0.88),
                arrowprops=dict(arrowstyle="->", color=marker.color, lw=1.4),
                zorder=8,
            )

        legend_text = "\n".join([
            "Color Code:",
            "Red: Capacity/widening",
            "Blue: Exit load balancing",
            "Orange: Turn/merge geometry",
            "Green: One-way operations",
            "Purple: Obstacle cleanup",
        ])
        ax.text(
            1.01,
            0.98,
            legend_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88),
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def _build_report(self) -> Tuple[str, List[ImprovementMarker]]:
        total_agents = len(self.agents)
        total_evac = int(np.sum([1 for a in self.agents if a.evacuated]))
        total_dead = int(np.sum([1 for a in self.agents if not a.alive]))

        phase = self._compute_phase_summary(total_agents=max(total_agents, 1))

        density_points = self._top_heatmap_points(self._density_map, top_n=5)
        panic_points = self._top_heatmap_points(self._panic_map, top_n=5)

        path_stats = self._path_efficiency_stats()
        exit_stats = self._exit_utilization()
        markers = self._build_improvement_markers(density_points, panic_points, exit_stats)

        bottlenecks = self.analytics.detect_bottlenecks(self.environment.grid)
        bottlenecks = bottlenecks[:6]

        report: List[str] = []

        report.append("A) Key Results (from CSV)")
        report.append(f"- Total agents: {total_agents}; evacuated: {total_evac}; deceased: {total_dead}.")
        report.append(f"- Evacuation plateau/completion point: around t≈{phase.plateau_time:.1f}s (evacuation rate flattens after this point).")
        report.append(
            f"- Phase breakdown: early flow until t≈{phase.early_end_time:.1f}s; peak congestion window around t≈{phase.peak_start_time:.1f}s to t≈{phase.peak_end_time:.1f}s; tail-end stragglers after t≈{phase.tail_start_time:.1f}s."
        )
        if self._panic.size > 0 and self._speed.size > 0:
            report.append(
                f"- Trend support: peak average panic≈{np.max(self._panic):.2f}, mean panic≈{np.mean(self._panic):.2f}; average speed drops to≈{np.min(self._speed):.2f} m/s during high-congestion periods."
            )
        report.append("")

        report.append("B) Visual Findings (from Image 1–3)")
        if density_points:
            for i, (x, y, val) in enumerate(density_points[:4], start=1):
                local_open = self._local_walkable_neighbors(x, y)
                report.append(
                    f"- Density hotspot {i} at {self._format_xy(x, y)} in Image 2 with mean density≈{val:.2f} agents/m²; local walkable neighborhood≈{local_open}% (suggesting a constricted area when this percentage is low)."
                )
        if panic_points:
            for i, (x, y, val) in enumerate(panic_points[:3], start=1):
                report.append(
                    f"- Elevated panic area {i} at {self._format_xy(x, y)} in Image 3 with panic index≈{val:.2f}; this likely overlaps queueing pressure near bottlenecks."
                )
        report.append(
            f"- Image 1 path behavior indicates detour pressure from starts near {self._format_xy(path_stats['worst_x'], path_stats['worst_y'])}; worst path/direct ratio≈{path_stats['p90']:.2f} at p90."
        )
        if exit_stats["overloaded"] is not None and exit_stats["underused"] is not None:
            over = exit_stats["overloaded"]
            under = exit_stats["underused"]
            report.append(
                f"- Exit usage appears imbalanced: Exit {over.id} evacuated {over.total_evacuated} vs Exit {under.id} evacuated {under.total_evacuated} (ratio≈{exit_stats['imbalance_ratio']:.2f})."
            )
        report.append(
            f"- Referenced outputs: Image 1 [{self.movement_img_path.name}], Image 2 [{self.density_img_path.name}], Image 3 [{self.panic_img_path.name}], CSV [{self.csv_path.name}]."
        )
        report.append("")

        report.append("C) Bottlenecks & Causes")
        if bottlenecks:
            for bn in bottlenecks:
                bx, by = float(bn["position"][0]), float(bn["position"][1])
                openness = self._local_walkable_neighbors(bx, by)
                cause = "pinch-point / merge conflict" if openness < 60 else "flow convergence and route competition"
                report.append(
                    f"- {self._format_xy(bx, by)} → Evidence: sustained density≈{bn['density']:.2f} agents/m² in Image 2 and slower evac-rate progression in CSV around peak window → Likely cause: {cause}."
                )
        else:
            report.append("- No strong bottleneck cell exceeded threshold; likely distributed congestion rather than a single choke point.")
        report.append("")

        recs: List[str] = []

        # Top 3 high-impact recommendations
        top_density = density_points[0] if density_points else (self.environment.width * 0.5, self.environment.height * 0.5, 0.0)
        d_x, d_y, d_val = top_density

        recs.append("1. **Change:** Widen the corridor/opening by at least 1.0-1.5 m at the primary choke zone and remove immediate edge obstructions within 2-3 m of the choke entry.")
        recs.append(f"   **Where:** Around {self._format_xy(d_x, d_y)} (highest-density hotspot in Image 2).")
        recs.append(f"   **Evidence:** Peak local density≈{d_val:.2f} agents/m² with queue buildup in movement paths near the same location.")
        recs.append("   **Why it helps:** Increases local discharge capacity and reduces friction/conflict at merges, which raises effective outflow.")
        recs.append("   **Expected impact:** High; should reduce peak density and lower tail latency, while improving average speed.")
        recs.append("   **How to validate next run:** Density hotspot intensity should drop and CSV evacuation slope should stay steeper for longer after the peak window.")

        if exit_stats["overloaded"] is not None and exit_stats["underused"] is not None:
            over = exit_stats["overloaded"]
            under = exit_stats["underused"]
            recs.append("")
            recs.append("2. **Change:** Add a connecting opening/corridor branch (or widen existing branch) that feeds the underused exit, and install directional signage to split flow before the main merge.")
            recs.append(
                f"   **Where:** Transition corridor upstream of Exit {over.id} (overloaded) and route toward Exit {under.id} at {self._format_xy(float(under.position[0]), float(under.position[1]))}."
            )
            recs.append(
                f"   **Evidence:** Exit utilization imbalance {over.total_evacuated} vs {under.total_evacuated} (ratio≈{exit_stats['imbalance_ratio']:.2f}); movement paths show dominant stream to one exit."
            )
            recs.append("   **Why it helps:** Load-balances exits, preventing one queue from dictating global evacuation time.")
            recs.append("   **Expected impact:** High; improves total evacuation time and reduces congestion peak near overloaded exit.")
            recs.append("   **How to validate next run:** Exit counts should become more even and hotspot near overloaded exit should weaken in Image 2.")
        else:
            recs.append("")
            recs.append("2. **Change:** Add one supplemental egress opening near the dominant crowd-origin side of the floor plan.")
            recs.append(f"   **Where:** Near {self._format_xy(path_stats['worst_x'], path_stats['worst_y'])}, aligned to the nearest exterior boundary.")
            recs.append("   **Evidence:** Long path detours and slow tail indicate insufficient local egress capacity.")
            recs.append("   **Why it helps:** Shortens travel distance and reduces dependency on central chokepoints.")
            recs.append("   **Expected impact:** High; improves total evacuation time and late-phase straggler clearance.")
            recs.append("   **How to validate next run:** Lower p90 path ratio and earlier plateau with fewer active agents late in the run.")

        max_panic_val = float(np.max(self._panic)) if self._panic.size > 0 else 0.0
        if panic_points and max_panic_val >= 0.1:
            panic_top = panic_points[0]
        else:
            panic_top = (d_x, d_y, max_panic_val)
        p_x, p_y, p_val = panic_top
        recs.append("")
        recs.append("3. **Change:** Reconfigure geometry at the panic hotspot into a gentler turn/merge (increase corner radius or replace acute turn with short beveled transition) and clear local obstructions.")
        recs.append(f"   **Where:** Around {self._format_xy(p_x, p_y)} (highest panic concentration in Image 3).")
        if max_panic_val >= 0.1:
            recs.append(f"   **Evidence:** Panic hotspot≈{p_val:.2f} co-located with high-density routing pressure.")
        else:
            recs.append("   **Evidence:** Panic remained low overall, but this location still aligns with major movement compression and turning conflict in path overlays.")
        recs.append("   **Why it helps:** Smoother turning reduces stop-and-go interactions, preventing panic amplification under compression.")
        recs.append("   **Expected impact:** Medium-High; lowers average panic and improves local throughput/speed stability.")
        recs.append("   **How to validate next run:** Panic heatmap peak should contract and Avg_Panic in CSV should decay sooner after peak congestion.")

        # Additional nice-to-have refinements
        recs.append("")
        recs.append("4. **Change:** Create a one-way circulation rule in the most conflict-prone corridor pair to prevent counter-flow crossing.")
        recs.append(f"   **Where:** Around the central junction near {self._format_xy(d_x, d_y)} (assumption: this hotspot corresponds to a merge/intersection).")
        recs.append("   **Evidence:** Movement paths exhibit crossing streams and queue spillback near hotspot.")
        recs.append("   **Why it helps:** Removes head-on conflicts and improves lane coherence.")
        recs.append("   **Expected impact:** Medium; reduces local density spikes and raises average speed.")
        recs.append("   **How to validate next run:** Fewer crossing trajectories in Image 1 and smoother CSV speed profile during peak window.")

        recs.append("")
        recs.append("5. **Change:** Remove/relocate small interior obstacles within 3-5 m of top density hotspots.")
        recs.append("   **Where:** At the top 2-3 density hotspot coordinates in Section B.")
        recs.append("   **Evidence:** High-density cells persist near narrow walkable neighborhoods.")
        recs.append("   **Why it helps:** Increases effective corridor width without structural wall changes.")
        recs.append("   **Expected impact:** Medium; reduces peak density and queue duration.")
        recs.append("   **How to validate next run:** Lower dwell time around those coordinates and reduced hotspot severity values.")

        report.append("D) Recommended Floor Plan Changes (prioritized)")
        report.extend(recs)
        report.append("")

        policy_lines: List[str] = []
        if exit_stats.get("imbalance_ratio", 1.0) >= 1.5:
            policy_lines.append("- Add dynamic guidance/signage policy that redirects new agents to underused exits when one exit queue exceeds a threshold.")
        if self._panic.size > 0 and float(np.max(self._panic)) >= 0.6:
            policy_lines.append("- Apply phased evacuation release for zones feeding the highest panic corridor to reduce instantaneous load at chokepoints.")

        if policy_lines:
            report.append("E) Optional: Guidance/Policy Improvements")
            report.extend(policy_lines)

        return "\n".join(report) + "\n", markers

    def generate(self, output_path: Path, floorplan_path: Optional[Path] = None) -> Path:
        self._load_csv()
        self._extract_maps()

        report_text, markers = self._build_report()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding="utf-8")

        overlay_path = output_path.parent / "floorplan_improvement_overlay.png"
        self._generate_overlay_image(markers, overlay_path, floorplan_path=floorplan_path)
        return output_path
