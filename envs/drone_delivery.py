from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - matplotlib optional
    plt = None

import gymnasium as gym
from gymnasium import spaces


# Actions: 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT, 4:STAY, 5:CHARGE
A_UP, A_DOWN, A_LEFT, A_RIGHT, A_STAY, A_CHARGE = range(6)


@dataclass(frozen=True)
class Cell:
    x: int
    y: int


class DroneDeliveryEnv(gym.Env):
    """Drone Delivery environment with wind slip and battery constraints.

    Observation: Discrete index encoding (x, y, battery, has_package)
    Action space: Discrete(6) [UP, DOWN, LEFT, RIGHT, STAY, CHARGE]

    Termination: delivery at dropoff, or battery depletion, or max_steps reached.

    Render modes:
    - "ansi": returns a string grid with agent, obstacles, pickup (P), dropoff (D)
    - "human": uses matplotlib if available; otherwise falls back to ansi print
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(
        self,
        *,
        width: int = 7,
        height: int = 7,
        max_battery: int = 5,
        charge_rate: int = 2,
        obstacles: Sequence[Tuple[int, int]] | None = None,
        pickup: Tuple[int, int] = (0, 0),
        dropoff: Tuple[int, int] = (6, 6),
        charging_stations: Sequence[Tuple[int, int]] | None = None,
        wind_slip: float = 0.1,
        max_steps: int = 200,
        render_mode: str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        assert width > 1 and height > 1
        assert max_battery >= 1
        assert 0.0 <= wind_slip <= 0.5, "slip must be in [0, 0.5]"
        self.width = width
        self.height = height
        self.max_battery = max_battery
        self.charge_rate = charge_rate
        self.pickup = Cell(*pickup)
        self.dropoff = Cell(*dropoff)
        self.wind_slip = wind_slip
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)

        obs = set(obstacles or [])
        assert all(0 <= x < width and 0 <= y < height for x, y in obs), "obstacle out of bounds"
        self.obstacles = {Cell(x, y) for x, y in obs}

        if charging_stations is None:
            charging_stations = [pickup]
        self.charging_stations = {Cell(*xy) for xy in charging_stations}

        # State encoding: (x, y, b, p) -> idx
        self.nS = width * height * (max_battery + 1) * 2
        self.nA = 6
        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

        # Episode state
        self._x = self.pickup.x
        self._y = self.pickup.y
        self._battery = max_battery
        self._has_package = 1
        self._steps = 0

        # Matplotlib state
        self._fig = None
        self._ax = None

    # --------------- Encoding helpers ---------------
    def _encode(self, x: int, y: int, b: int, p: int) -> int:
        return (((y * self.width) + x) * (self.max_battery + 1) + b) * 2 + p

    def _decode(self, idx: int) -> Tuple[int, int, int, int]:
        p = idx % 2
        idx //= 2
        b = idx % (self.max_battery + 1)
        idx //= (self.max_battery + 1)
        x = idx % self.width
        y = idx // self.width
        return x, y, b, p

    # --------------- Core API ---------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._x = self.pickup.x
        self._y = self.pickup.y
        self._battery = self.max_battery
        self._has_package = 1  # start carrying the package
        self._steps = 0
        obs = self._encode(self._x, self._y, self._battery, self._has_package)
        info = {}
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action)
        self._steps += 1

        terminated = False
        truncated = False
        reward = -1.0  # base, will be overridden by enumerated reward
        delivered = False
        collision = False

        # Sample transition from the enumerated model for consistency with DP
        s0 = self._encode(self._x, self._y, self._battery, self._has_package)
        candidates = self.enumerate_transitions(s0, action)
        # Fallback: if enumerate_transitions is unavailable, raise (should not happen)
        if not candidates:
            raise RuntimeError("No transition candidates produced by enumerate_transitions")
        r = self._rng.random()
        cum = 0.0
        chosen = candidates[-1]
        for prob, s2, rew, done in candidates:
            cum += prob
            if r < cum:
                chosen = (prob, s2, rew, done)
                break
        _, s2, rew, done = chosen
        x0, y0, b0, p0 = self._decode(s0)
        x2, y2, b2, p2 = self._decode(s2)

        # Update internal state according to sampled outcome
        reward = float(rew)
        terminated = bool(done)
        delivered = (p0 == 1 and p2 == 0)
        # collision if a movement was attempted with battery > 0 and position didn't change
        collision = (action in (A_UP, A_DOWN, A_LEFT, A_RIGHT)) and (b0 > 0) and (x2 == x0 and y2 == y0)

        self._x, self._y, self._battery, self._has_package = x2, y2, b2, p2

        if self._steps >= self.max_steps and not terminated:
            truncated = True

        # Clamp battery to [0, max_battery] for encoding safety
        self._battery = max(0, min(self._battery, self.max_battery))
        obs = self._encode(self._x, self._y, self._battery, self._has_package)
        info = {"delivered": delivered, "collision": collision, "battery": self._battery, "steps": self._steps}
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    # --------------- Model enumeration for DP ---------------
    def enumerate_transitions(self, s: int, a: int) -> List[Tuple[float, int, float, bool]]:
        """Return a list of (prob, s', reward, done) for taking action a in state s.
        This is computed analytically using the environment's rules (no sampling).
        """
        x, y, b, p = self._decode(s)
        results: List[Tuple[float, int, float, bool]] = []

        # Helper to finalize a candidate next state
        def finalize(nx: int, ny: int, nb: int, npack: int, base_reward: float) -> Tuple[float, int, float, bool]:
            terminated = False
            reward = base_reward
            # Delivery check
            if npack == 1 and nx == self.dropoff.x and ny == self.dropoff.y:
                reward += 50.0
                npack = 0
                terminated = True
            if nb == 0 and not terminated:
                reward += -10.0
                terminated = True
            s2 = self._encode(nx, ny, nb, npack)
            return (1.0, s2, reward, terminated)

        base_step_penalty = -1.0

        if a == A_CHARGE:
            nb = b
            if Cell(x, y) in self.charging_stations:
                nb = min(self.max_battery, b + self.charge_rate)
            prob, s2, r, done = finalize(x, y, nb, p, base_step_penalty)
            results.append((prob, s2, r, done))
            return results
        if a == A_STAY:
            prob, s2, r, done = finalize(x, y, b, p, base_step_penalty)
            results.append((prob, s2, r, done))
            return results

        # Movement actions: handle zero battery (cannot move)
        if b == 0:
            # Apply step penalty first; battery depletion (-10 and done=True)
            # is applied inside finalize for nb==0, mirroring step() ordering.
            prob, s2, r, done = finalize(x, y, 0, p, base_step_penalty)
            results.append((prob, s2, r, done))
            return results

        dirs = {
            A_UP: (0, -1),
            A_DOWN: (0, 1),
            A_LEFT: (-1, 0),
            A_RIGHT: (1, 0),
        }

        intended = a
        for prob, actual in self._slip_distribution(intended):
            if prob <= 0:
                continue
            dx, dy = dirs.get(actual, (0, 0))
            nx, ny = x + dx, y + dy
            nb = b - 1
            reward = base_step_penalty
            if not self._in_bounds(nx, ny) or Cell(nx, ny) in self.obstacles:
                # collision: stay in place
                nx, ny = x, y
                reward += -20.0
            _, s2, r, done = finalize(nx, ny, nb, p, reward)
            results.append((prob, s2, r, done))

        # Normalize numeric errors
        total_p = sum(p for p, *_ in results)
        if abs(total_p - 1.0) > 1e-8:
            results = [(p / total_p, s2, r, d) for (p, s2, r, d) in results]
        return results

    # --------------- Rendering ---------------
    def render(self):
        grid = [["."] * self.width for _ in range(self.height)]
        for c in self.obstacles:
            grid[c.y][c.x] = "#"
        grid[self.pickup.y][self.pickup.x] = "P"
        grid[self.dropoff.y][self.dropoff.x] = "D"
        grid[self._y][self._x] = "A" if self._has_package else "a"

        if self.render_mode == "ansi" or plt is None:
            buf = io.StringIO()
            for row in grid:
                buf.write(" ".join(row) + "\n")
            buf.write(f"battery={self._battery}, steps={self._steps}\n")
            s = buf.getvalue()
            print(s)
            return s
        else:
            # human with matplotlib - enhanced visual representation
            if self._fig is None or self._ax is None:
                self._fig, self._ax = plt.subplots(figsize=(8, 8))
                self._ax.set_xlim(-0.5, self.width - 0.5)
                self._ax.set_ylim(-0.5, self.height - 0.5)
                self._ax.set_aspect('equal')
                self._ax.invert_yaxis()
            self._ax.clear()
            
            # Draw grid cells with enhanced colors
            for x in range(self.width):
                for y in range(self.height):
                    color = "#F5F5F5"  # Light gray for free space
                    edgecolor = "#CCCCCC"
                    linewidth = 0.5
                    
                    if Cell(x, y) in self.obstacles:
                        color = "#2C3E50"  # Dark gray for obstacles
                        edgecolor = "#1A252F"
                        linewidth = 1.5
                    elif x == self.pickup.x and y == self.pickup.y:
                        color = "#AED6F1"  # Light blue for store/pickup
                        edgecolor = "#5DADE2"
                        linewidth = 2
                    elif x == self.dropoff.x and y == self.dropoff.y:
                        color = "#ABEBC6"  # Light green for house/dropoff
                        edgecolor = "#58D68D"
                        linewidth = 2
                    elif Cell(x, y) in self.charging_stations:
                        color = "#FAD7A0"  # Light orange for charging stations
                        edgecolor = "#F8C471"
                        linewidth = 1.5
                    
                    self._ax.add_patch(plt.Rectangle(
                        (x - 0.5, y - 0.5), 1, 1, 
                        facecolor=color, 
                        edgecolor=edgecolor,
                        linewidth=linewidth
                    ))
            
            # Draw icons/markers for special locations
            # Store/Pickup location - use 'S' marker with store icon
            self._ax.plot(self.pickup.x, self.pickup.y, marker='s', markersize=20, 
                         color='#3498DB', markeredgecolor='#2874A6', markeredgewidth=2, zorder=5)
            self._ax.text(self.pickup.x, self.pickup.y, 'S', fontsize=14, fontweight='bold',
                         ha='center', va='center', color='white', zorder=6)
            
            # House/Dropoff location - use house-like marker
            self._ax.plot(self.dropoff.x, self.dropoff.y, marker='^', markersize=22, 
                         color='#27AE60', markeredgecolor='#1E8449', markeredgewidth=2, zorder=5)
            self._ax.plot(self.dropoff.x, self.dropoff.y, marker='s', markersize=14, 
                         color='#27AE60', markeredgecolor='#1E8449', markeredgewidth=2, zorder=5)
            self._ax.text(self.dropoff.x, self.dropoff.y - 0.05, 'H', fontsize=12, fontweight='bold',
                         ha='center', va='center', color='white', zorder=6)
            
            # Obstacles - use X marker
            for obs in self.obstacles:
                self._ax.plot(obs.x, obs.y, marker='X', markersize=18, 
                             color='#34495E', markeredgecolor='#1C2833', markeredgewidth=2, zorder=5)
            
            # Charging stations - use lightning bolt symbol or star
            for station in self.charging_stations:
                if station.x != self.pickup.x or station.y != self.pickup.y:
                    self._ax.plot(station.x, station.y, marker='*', markersize=20, 
                                 color='#F39C12', markeredgecolor='#D68910', markeredgewidth=1.5, zorder=5)
            
            # Drone with enhanced representation
            # Draw drone body (main circle)
            drone_body = plt.Circle((self._x, self._y), 0.15, 
                                   facecolor='#2C3E50', edgecolor='#1A252F', linewidth=2, zorder=10)
            self._ax.add_patch(drone_body)
            
            # Draw drone propellers (4 small circles at corners)
            propeller_offset = 0.22
            propeller_positions = [
                (self._x - propeller_offset, self._y - propeller_offset),  # bottom-left
                (self._x + propeller_offset, self._y - propeller_offset),  # bottom-right
                (self._x - propeller_offset, self._y + propeller_offset),  # top-left
                (self._x + propeller_offset, self._y + propeller_offset),  # top-right
            ]
            for px, py in propeller_positions:
                propeller = plt.Circle((px, py), 0.08, 
                                      facecolor='#E74C3C', edgecolor='#922B21', linewidth=1.5, zorder=9)
                self._ax.add_patch(propeller)
            
            # Draw connecting arms (lines from body to propellers)
            for px, py in propeller_positions:
                self._ax.plot([self._x, px], [self._y, py], 
                             color='#34495E', linewidth=2, zorder=8)
            
            # Package indicator
            if self._has_package:
                # Draw package below drone
                package = plt.Rectangle((self._x - 0.12, self._y - 0.35), 0.24, 0.15,
                                       facecolor='#8B4513', edgecolor='#5D2906', 
                                       linewidth=2, zorder=7)
                self._ax.add_patch(package)
                # Package label
                self._ax.text(self._x, self._y - 0.275, 'PKG', fontsize=6, fontweight='bold',
                             ha='center', va='center', color='white', zorder=8)
                # Connection line (drone carrying package)
                self._ax.plot([self._x, self._x], [self._y - 0.15, self._y - 0.2], 
                             color='#7F8C8D', linewidth=1.5, linestyle='--', zorder=7)
            
            # Drone center indicator (small dot)
            self._ax.plot(self._x, self._y, marker='o', markersize=4, 
                         color='#ECF0F1', markeredgecolor='#BDC3C7', markeredgewidth=1, zorder=11)
            
            # Battery indicator with color coding
            battery_pct = self._battery / self.max_battery
            if battery_pct > 0.6:
                battery_color = '#27AE60'  # Green
                battery_status = 'HIGH'
            elif battery_pct > 0.3:
                battery_color = '#F39C12'  # Orange
                battery_status = 'MED'
            else:
                battery_color = '#E74C3C'  # Red
                battery_status = 'LOW'
            
            # Title with battery and steps
            title = f'Battery: {self._battery}/{self.max_battery} ({battery_status})  |  Steps: {self._steps}'
            if self._has_package:
                title += '  |  Package: CARRYING'
            else:
                title += '  |  Package: DELIVERED'
            
            self._ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#AED6F1', 
                          markersize=10, label='Store (Pickup)', markeredgecolor='#5DADE2'),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#ABEBC6', 
                          markersize=10, label='House (Dropoff)', markeredgecolor='#58D68D'),
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#2C3E50', 
                          markersize=10, label='Obstacle', markeredgecolor='#1A252F'),
                plt.Line2D([0], [0], marker='*', color='#F39C12', markersize=12, 
                          label='Charging Station', markeredgecolor='#D68910', markeredgewidth=1),
            ]
            self._ax.legend(handles=legend_elements, loc='upper left', 
                           bbox_to_anchor=(0, -0.05), ncol=4, frameon=False, fontsize=9)
            
            # Grid lines
            self._ax.set_xticks(range(self.width))
            self._ax.set_yticks(range(self.height))
            self._ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            self._fig.canvas.draw()
            plt.pause(0.001)

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig, self._ax = None, None

    # --------------- Helpers ---------------
    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _rotate_left(self, action: int) -> int:
        if action == A_UP:
            return A_LEFT
        if action == A_LEFT:
            return A_DOWN
        if action == A_DOWN:
            return A_RIGHT
        if action == A_RIGHT:
            return A_UP
        return action

    def _rotate_right(self, action: int) -> int:
        if action == A_UP:
            return A_RIGHT
        if action == A_RIGHT:
            return A_DOWN
        if action == A_DOWN:
            return A_LEFT
        if action == A_LEFT:
            return A_UP
        return action

    def _apply_penalties(self, reward: float, collision: bool, terminated: bool) -> Tuple[float, bool]:
        """Apply collision and battery depletion penalties consistently.

        This method centralizes the logic to avoid drift between branches.
        Behavior mirrors the previous inline logic:
        - Collision adds -20.0 but does not by itself terminate the episode.
        - If battery == 0 and not terminated yet (and delivery not achieved),
          add -10.0 and terminate.
        """
        if collision:
            reward += -20.0
        if self._battery == 0 and not terminated:
            reward += -10.0
            terminated = True
        return reward, terminated

    def _slip_distribution(self, intended: int) -> List[Tuple[float, int]]:
        """Return [(prob, action), ...] for wind slip around an intended direction.

        The distribution is: (1-2p) for intended, and p for each 90° rotation (left/right),
        where p = self.wind_slip and is clamped implicitly by construction. Probabilities sum to 1.
        """
        p = float(self.wind_slip)
        main_p = max(0.0, 1.0 - 2.0 * p)
        return [
            (main_p, intended),
            (p, self._rotate_left(intended)),
            (p, self._rotate_right(intended)),
        ]
