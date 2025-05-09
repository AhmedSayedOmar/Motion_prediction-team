#!/usr/bin/env python3
"""
Monolithic Non-ROS Simulation of the Research Pipeline:
  Collision Prediction (per collision_prediction.py) +
  SQP-based Overtaking (per sqp_avoidance_node.py) +
  Animated GIF Output

This script reproduces the research code’s collision prediction,
which integrates the longitudinal (s) coordinate with constant speeds.
It then calls an SQP solver that, given a collision region and a set of
“scaled waypoints” defining lateral boundaries, computes a full overtaking
trajectory in d (lateral coordinate) that starts on the raceline,
curves away for overtaking, and then merges back into the raceline.

After computing the overtaking plan the script “maps” it back in time:
the ego vehicle’s longitudinal state is integrated with its lateral deviation
(from the SQP plan) so that a time–trace is generated (similar to the initial
collision prediction trace). Two GIFs are then saved:
  - "collision_prediction.gif": shows the collision prediction trace.
  - "overtaking_plan.gif": shows a time-based simulation of both the ego and opponent
    vehicles moving along the track. In each frame, only the current positions are visible.
    
Dependencies:
  pip install numpy scipy matplotlib Pillow
"""

#!/usr/bin/env python3
import numpy as np
import math
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d

# -----------------------------------------------------------------------
# Data Structure: Scaled Waypoint (for lateral boundaries)
# -----------------------------------------------------------------------
class Wpnt:
    def __init__(self, s_m=0.0, d_left=0.05, d_right=-0.05):
        self.s_m = s_m
        self.d_left = d_left
        self.d_right = d_right

# -----------------------------------------------------------------------
# Collision Predictor (Research Code Style)
# -----------------------------------------------------------------------
class CollisionPredictor:
    """
    Integrates only the longitudinal coordinate (s) assuming constant speeds.
    When the gap (s_opp – s_ego) becomes less than the front threshold the collision 
    is assumed to start (recording cstart). When (s_ego – s_opp) exceeds the back 
    threshold the collision region ends (cend). In this simulation the opponent’s 
    lateral coordinate is fixed (d_opp = 0).
    """
    def __init__(self, time_steps=200, dt=0.02,
                 save_distance_front=0.6, save_distance_back=0.4):
        self.time_steps = time_steps
        self.dt = dt
        self.save_distance_front = save_distance_front  # m: collision onset threshold
        self.save_distance_back  = save_distance_back   # m: collision termination threshold

    def simulate(self, s_ego_init, s_opp_init, vs_ego, vs_opp, track_length):
        collision_started = False
        cstart = None
        cend = None
        s_ego = s_ego_init
        s_opp = s_opp_init
        d_opp = 0.0  # Opponent is exactly on raceline
        states = []  # save each timestep

        for i in range(self.time_steps):
            t = i * self.dt
            s_ego_next = s_ego + vs_ego * self.dt
            s_opp_next = s_opp + vs_opp * self.dt
            gap = (s_opp_next - s_ego_next) % track_length
            if gap > track_length/2:
                gap = track_length - gap
            states.append({'time': t, 's_ego': s_ego_next, 's_opp': s_opp_next, 'gap': gap, 'd_opp': d_opp})
            if (not collision_started) and (gap < self.save_distance_front*abs(vs_ego-vs_opp)):
                cstart = s_ego_next
                collision_started = True
            elif collision_started and ((s_ego_next - s_opp_next) > self.save_distance_back*abs(vs_ego-vs_opp)):
                cend = s_ego_next
                break
            s_ego, s_opp = s_ego_next, s_opp_next
        if collision_started and cend is None:
            cend = s_ego
        return cstart, cend, d_opp, states

# -----------------------------------------------------------------------
# SQP-based Overtaking Solver (Extended Domain)
# -----------------------------------------------------------------------
class SQPAvoidanceNode:
    """
    Computes an overtaking trajectory for the ego vehicle.
    
    The planning is extended to include an “entry” segment before the collision
    region and an “exit” segment after the collision region so that the overall
    trajectory spans a larger portion of the lap. In this extended domain:
      - For s <= s_start, the desired lateral value is 0 (raceline).
      - For s_start < s < s_end, a quadratic hump creates the lateral deviation.
      - For s >= s_end, the desired lateral value returns to 0.
    The SQP solver then computes a trajectory that tracks this desired profile.
    """
    def __init__(self, avoidance_resolution=20, spline_bound_mindist=0.002,
                 width_car=0.01, evasion_dist=0.005, max_kappa=5.0):
        self.avoidance_resolution = avoidance_resolution
        self.spline_bound_mindist = spline_bound_mindist
        self.width_car = width_car
        self.evasion_dist = evasion_dist
        self.max_kappa = max_kappa
        self.current_d = 0.0  # Ego’s current lateral coordinate (raceline)

    def plan_overtake(self, s_start, s_end, cur_d_ego,
                      scaled_wpnts, track_length, extension=2.0):
        from scipy.optimize import minimize
        #from scipy.interpolate import interp1d

        s_lower = max(0, s_start - extension)
        s_upper = min(track_length, s_end + extension)
        s_avoid = np.linspace(s_lower, s_upper, self.avoidance_resolution)

        d_min_arr = []
        d_max_arr = []
        for s in s_avoid:
            wp = min(scaled_wpnts, key=lambda wp: abs(wp.s_m - s))
            dmin = wp.d_right + self.spline_bound_mindist
            dmax = wp.d_left - self.spline_bound_mindist
            if dmin > dmax:
                mid = 0.5 * (dmin + dmax)
                dmin = mid
                dmax = mid
            d_min_arr.append(dmin)
            d_max_arr.append(dmax)

        d_min_arr = np.array(d_min_arr)
        d_max_arr = np.array(d_max_arr)

        # Initial guess: use current lateral coordinate.
        x0 = np.full(len(s_avoid), cur_d_ego)

        def curvature(d): #function from paper... allows us to cap max curavture, allows for overtaking in lower lateral accelerations (safety reasons as well)
            d = np.array(d)
            dd_ds = np.gradient(d, s_avoid) #lateral offset slope
            d2d_ds = np.gradient(dd_ds, s_avoid) #slopes rate of change
            kappa = np.abs(d2d_ds / (1 + dd_ds**2)**1.5) #frenet curvature formula
            #print("Max curvature:", np.max(kappa))
            return self.max_kappa - kappa

        # Define the desired lateral profile.
        def desired_d(s):
            if s <= s_start:
                return 0.0
            elif s >= s_end:
                return 0.0
            else:
                mid = (s_start + s_end) / 2.0
                half_width = (s_end - s_start) / 2.0
                return -0.015 * (1 - ((s - mid) / half_width) ** 2)

        # Objective: tracking plus smoothness
        def objective(d):
            d_arr = np.array(d)
            desired_profile = np.array([desired_d(s_val) for s_val in s_avoid])
            tracking_cost = np.sum((d_arr - desired_profile) ** 2)
            smooth_cost = np.sum(np.diff(np.diff(d_arr)) ** 2)
            first_derivative_cost = (np.diff(d_arr)[0] ** 2)
            return 1000 * tracking_cost + 100 * smooth_cost + 1000 * first_derivative_cost

        bounds = [(d_min_arr[i], d_max_arr[i]) for i in range(len(s_avoid))]
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=[{'type':'ineq','fun':curvature}], options={'maxiter': 50})
        #res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'maxiter': 50})
        if not res.success:
            return s_avoid, x0, False
        return s_avoid, res.x, True

# -----------------------------------------------------------------------
# Scaled Waypoint Generator (with small lateral dimensions)
# -----------------------------------------------------------------------
def generate_scaled_wpnts(track_length, num_points=200):
    s_lin = np.linspace(0, track_length, num_points)
    wpnts = []
    # For this simulation, assume the track’s lateral boundaries are ±0.05 m.
    for s in s_lin:
        wpnts.append(Wpnt(s_m=s, d_left=0.05, d_right=-0.05))
    return wpnts

# -----------------------------------------------------------------------
# Animation Functions
# -----------------------------------------------------------------------
def animate_collision(trace, track_length, filename="collision_prediction.gif"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, track_length)
    ax.set_ylim(-0.02, 0.02)
    ax.set_xlabel("Longitudinal Position s (m)")
    ax.set_ylabel("Lateral Position d (m)")
    ax.set_title("Collision Prediction Trace")
    ego_point, = ax.plot([], [], 'bo', markersize=8, label="Ego")
    opp_point, = ax.plot([], [], 'ro', markersize=8, label="Opponent")
    time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)
    ax.legend()

    def init():
        ego_point.set_data([], [])
        opp_point.set_data([], [])
        time_text.set_text('')
        return ego_point, opp_point, time_text

    def update(frame):
        state = trace[frame]
        # wrap in lists so they're sequences:
        ego_point.set_data([state['s_ego']], [0.0])
        opp_point.set_data([state['s_opp']], [state['d_opp']])
        time_text.set_text(f"t = {state['time']:.3f} s")
        return ego_point, opp_point, time_text

    anim = animation.FuncAnimation(fig, update, frames=len(trace),
                                   init_func=init, blit=True, interval=50)
    writer = animation.PillowWriter(fps=10)
    anim.save(filename, writer=writer)
    plt.close(fig)
    print(f"Saved collision prediction animation to {filename}")

def animate_extended_overtaking(trace, filename="overtaking_plan.gif"):
    """
    Animate the full time-based simulation trace of both vehicles.
    In each frame, only the current positions of the ego and opponent vehicles are drawn.
    This way, as the simulation proceeds, previous positions are not retained.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    s_vals = [state['s_ego'] for state in trace]
    ax.set_xlim(min(s_vals)-1, max(s_vals)+1)
    d_vals = [state.get('d_ego', 0.0) for state in trace]
    ax.set_ylim(min(d_vals)-0.005, max(d_vals)+0.005)
    ax.set_xlabel("Longitudinal Position s (m)")
    ax.set_ylabel("Lateral Position d (m)")
    ax.set_title("Extended Overtaking Trace")
    # We will use scatter plots that update to show only the current state.
    ego_scatter = ax.scatter([], [], c='b', s=80, label="Ego")
    opp_scatter = ax.scatter([], [], c='r', s=80, label="Opponent")
    time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)
    ax.legend()

    def init():
        ego_scatter.set_offsets([[0,0]])
        opp_scatter.set_offsets([[0,0]])
        time_text.set_text('')
        return ego_scatter, opp_scatter, time_text

    def update(frame):
        state = trace[frame]
        ego_coord = np.array([[state['s_ego'], state.get('d_ego', 0.0)]])
        opp_coord = np.array([[state['s_opp'], state['d_opp']]])
        ego_scatter.set_offsets(ego_coord)
        opp_scatter.set_offsets(opp_coord)
        time_text.set_text(f"t = {state['time']:.3f} s")
        return ego_scatter, opp_scatter, time_text

    anim = animation.FuncAnimation(fig, update, frames=len(trace),
                                   init_func=init, blit=True, interval=100)
    writer = animation.PillowWriter(fps=10)
    anim.save(filename, writer=writer)
    plt.close(fig)
    print(f"Saved extended overtaking animation to {filename}")

# -----------------------------------------------------------------------
# Simulation: Mapping SQP Output Back to a Time-based Trace
# -----------------------------------------------------------------------
def simulate_extended_overtaking_trace(sqpnode, collision_region, sqp_plan, s_ego_init, vs_ego, s_opp_init, vs_opp, track_length, dt=0.02, t_final=1.68):
    """
    Given the SQP plan (s_ot, d_ot) computed over an extended domain, 
    generate a time-based trace:
      - s_ego(t) = s_ego_init + vs_ego * t.
      - d_ego(t) is given by linearly interpolating the SQP plan when s_ego is 
        within [s_ot[0], s_ot[-1]], and 0 otherwise.
      - s_opp(t) = s_opp_init + vs_opp * t, with d_opp(t)=0.
    Returns the list of states for t in [0, t_final].
    """
    s_ot, d_ot, _ = sqp_plan
    f_overtake = interp1d(s_ot, d_ot, kind='linear', fill_value=0.0, bounds_error=False)
    
    trace = []
    t = 0.0
    while t <= t_final:
        s_ego = s_ego_init + vs_ego * t
        s_opp = s_opp_init + vs_opp * t
        if s_ot[0] <= s_ego <= s_ot[-1]:
            d_ego = float(f_overtake(s_ego))
        else:
            d_ego = 0.0
        gap = (s_opp - s_ego) % track_length
        if gap > track_length/2:
            gap = track_length - gap
        state = {'time': t, 's_ego': s_ego, 's_opp': s_opp, 'gap': gap, 'd_ego': d_ego, 'd_opp': 0.0}
        trace.append(state)
        t += dt
    return trace

# -----------------------------------------------------------------------
# Full Pipeline Simulation with Extended Overtaking and GIF Output
# -----------------------------------------------------------------------
def simulate_full_pipeline_trace():
    track_length = 100.0   # m
    s_ego_init = 0.0       # initial s position of ego
    vs_ego = 10.0           # ego speed (m/s)
    s_opp_init = 3.0       # initial s position of opponent
    vs_opp = 3.0           # opponent speed (m/s)
    dt = 0.02
    
    print("----- Collision Prediction Trace -----")
    coll_pred = CollisionPredictor(time_steps=200, dt=dt,
                                   save_distance_front=0.6, save_distance_back=0.4)
    cstart, cend, d_collision, trace_collision = coll_pred.simulate(s_ego_init, s_opp_init, vs_ego, vs_opp, track_length)
    for state in trace_collision:
        print(f"t={state['time']:.3f} s, s_ego={state['s_ego']:.3f} m, s_opp={state['s_opp']:.3f} m, gap={state['gap']:.3f} m, d_opp={state['d_opp']:.3f} m")
    print("\nFinal Collision Region:")
    if cstart is None:
        print("No collision predicted => no overtake needed.")
        return None, None, None, track_length
    else:
        print(f"s_start = {cstart:.3f} m, s_end = {cend:.3f} m, opponent d = {d_collision:.3f} m")
        
    scaled_wpnts = generate_scaled_wpnts(track_length, num_points=200)
    
    extension = 5.0  # Extended planning horizon before and after the collision region
    sqp_node = SQPAvoidanceNode(avoidance_resolution=20, spline_bound_mindist=0.002,
                                  width_car=0.01, evasion_dist=0.005)
    s_ot, d_ot, success = sqp_node.plan_overtake(cstart, cend, cur_d_ego=0.0,
                                                 scaled_wpnts=scaled_wpnts, 
                                                 track_length=track_length, 
                                                 extension=extension)

    print("\n----- Extended SQP-based Overtaking Plan -----")
    if not success:
        print("SQP solver did not converge. Using initial guess:")
        for s_val, d_val in zip(s_ot, d_ot):
            print(f"  s={s_val:.3f} m, d={d_val:.3f} m")
    else:
        print(f"SQP plan computed with {len(s_ot)} points:")
        for s_val, d_val in zip(s_ot, d_ot):
            print(f"  s={s_val:.3f} m, d={d_val:.3f} m")
    
    t_final = 3.0
    extended_trace = simulate_extended_overtaking_trace(sqp_node, (cstart, cend, d_collision),
                                                        (s_ot, d_ot, success),
                                                        s_ego_init, vs_ego, s_opp_init, vs_opp,
                                                        track_length, dt=dt, t_final=t_final)

    return trace_collision, extended_trace, (s_ot, d_ot, success), track_length

# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
if __name__ == "__main__":
    trace_collision, extended_trace, sqp_plan, track_length = simulate_full_pipeline_trace()
    if trace_collision is None:
        exit()
    (s_ot, d_ot, success) = sqp_plan
    
    print("\nFinal Collision Region:")
    print("\nExtended SQP-based Overtaking Plan:")
    for s_val, d_val in zip(s_ot, d_ot):
        print(f"  s={s_val:.3f} m, d={d_val:.3f} m")
    
    animate_collision(trace_collision, track_length, filename="collision_prediction.gif")
    animate_extended_overtaking(extended_trace, filename="overtaking_plan_withcurvature.gif")