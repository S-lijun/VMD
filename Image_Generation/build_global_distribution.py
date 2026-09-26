# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print("[ROOT]", ROOT)


def print_stats(name, values):

    print(f"\n{name} statistics")
    print("samples :", len(values))
    print("min     :", np.min(values))
    print("max     :", np.max(values))
    print("median  :", np.median(values))
    print("mean    :", np.mean(values))
    print("std     :", np.std(values))


def clean_balabit(df):

    df = df.rename(columns={
        "client timestamp":"time",
        "x":"x",
        "y":"y",
        "state":"state"
    })

    df = df[df["state"] == "Move"]
    df = df[(df["x"] < 65535) & (df["y"] < 65535)]
    df = df.drop_duplicates()

    for c in ["x","y","time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x","y","time"])


def clean_chaoshen(df):

    df = df.rename(columns={
        "X":"x",
        "Y":"y",
        "Timestamp":"time",
        "EventName":"event"
    })

    df = df[df["event"]=="Move"]

    for c in ["x","y","time"]:
        df[c] = pd.to_numeric(df[c],errors="coerce")

    return df.dropna(subset=["x","y","time"])


def clean_dfl(df):

    df.columns = [c.strip().lower() for c in df.columns]

    if "client timestamp" in df.columns:
        df = df.rename(columns={"client timestamp":"time"})
    elif "timestamp" in df.columns:
        df = df.rename(columns={"timestamp":"time"})

    if "state" in df.columns:
        df = df[df["state"].str.lower()=="move"]

    for c in ["x","y","time"]:
        df[c] = pd.to_numeric(df[c],errors="coerce")

    return df.dropna(subset=["x","y","time"])


def clean_twos(df):

    df = df.rename(columns={
        "timestamp": "time",
        "x": "x",
        "y": "y",
        "event": "event",
    })

    df = df[df["event"] == "Mouse Moved"]
    df = df[(df["x"] < 65535) & (df["y"] < 65535)]
    df = df.drop_duplicates()

    for c in ["x", "y", "time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x", "y", "time"])


# ------------------------------------------------
# velocity
# ------------------------------------------------

def compute_velocity(xs,ys,ts):

    dx = xs[1:] - xs[:-1]
    dy = ys[1:] - ys[:-1]
    dt = ts[1:] - ts[:-1]

    dt = np.maximum(dt,1e-5)

    v = np.sqrt(dx**2 + dy**2)/dt

    out = np.zeros(len(xs))
    out[1:] = v

    return out

# ------------------------------------------------
# velocity differences
# ------------------------------------------------

def compute_velocity_pair(xs, ys, ts, chunk_size=150):

    if len(xs) < chunk_size:
        return np.array([])

    v = compute_velocity(xs, ys, ts)

    results = []
    n_chunks = len(xs) // chunk_size

    for i in range(n_chunks):

        v_chunk = v[i*chunk_size:(i+1)*chunk_size]

        # pairwise difference
        diff = np.abs(v_chunk[:, None] - v_chunk[None, :])

        # upper triangle
        upper = diff[np.triu_indices(chunk_size, k=1)]

        results.append(upper)

    if results:
        return np.concatenate(results)

    return np.array([])


# ------------------------------------------------
# vx vy
# ------------------------------------------------

def compute_vxvy(xs,ys,ts):

    dx = xs[1:] - xs[:-1]
    dy = ys[1:] - ys[:-1]
    dt = ts[1:] - ts[:-1]

    dt = np.maximum(dt,1e-5)

    vx = np.zeros(len(xs))
    vy = np.zeros(len(xs))

    vx[1:] = dx/dt
    vy[1:] = dy/dt

    return vx,vy


# ------------------------------------------------
# acceleration (FIXED)
# ------------------------------------------------

def compute_acceleration(xs, ys, ts):

    T = len(xs)

    if T < 3:
        return np.array([])

    # -------- velocity --------
    dx = xs[1:] - xs[:-1]
    dy = ys[1:] - ys[:-1]
    dt = ts[1:] - ts[:-1]

    dt = np.maximum(dt, 1e-5)

    v_segment = np.sqrt(dx**2 + dy**2) / dt

    v = np.zeros(T)
    v[1:] = v_segment


    # -------- acceleration --------
    a = np.zeros(T)

    a[1:] = (v[1:] - v[:-1]) / dt


    # --------  --------
    a = a[np.isfinite(a)]

    return a

# ------------------------------------------------
# ax ay (NEW)
# ------------------------------------------------

def compute_axay(xs, ys, ts):

    T = len(xs)

    if T < 3:
        return np.array([]), np.array([])

    # velocity
    dx = xs[1:] - xs[:-1]
    dy = ys[1:] - ys[:-1]
    dt = ts[1:] - ts[:-1]

    dt = np.maximum(dt, 1e-5)

    vx = dx / dt
    vy = dy / dt

    vx = np.concatenate([[vx[0]], vx])
    vy = np.concatenate([[vy[0]], vy])

    # acceleration (strict physical)
    ax = np.zeros(T)
    ay = np.zeros(T)

    ax[1:] = (vx[1:] - vx[:-1]) / dt
    ay[1:] = (vy[1:] - vy[:-1]) / dt

    # clean
    ax = ax[np.isfinite(ax)]
    ay = ay[np.isfinite(ay)]

    return ax, ay


# ------------------------------------------------
# curvature (geometric, vx/vy + ax/ay from prev/next neighbors)
# ------------------------------------------------

def compute_curvature(xs, ys, ts):
    """
    Geometric curvature at each point using neighbors i-1 and i+1.

    For interior i: segment velocities v_prev (i-1->i), v_next (i->i+1);
    central velocity/acceleration; kappa = |vx*ay - vy*ax| / (vx^2 + vy^2)^(3/2).
    Endpoints padded from nearest interior value (same length as input).
    """
    T = len(xs)

    if T < 3:
        return np.array([])

    kappa = np.zeros(T)
    eps = 1e-8

    for i in range(1, T - 1):
        dt_prev = max(ts[i] - ts[i - 1], eps)
        dt_next = max(ts[i + 1] - ts[i], eps)
        dt_c = max(ts[i + 1] - ts[i - 1], eps)

        vx_prev = (xs[i] - xs[i - 1]) / dt_prev
        vy_prev = (ys[i] - ys[i - 1]) / dt_prev
        vx_next = (xs[i + 1] - xs[i]) / dt_next
        vy_next = (ys[i + 1] - ys[i]) / dt_next

        vx = 0.5 * (vx_prev + vx_next)
        vy = 0.5 * (vy_prev + vy_next)
        ax = (vx_next - vx_prev) / dt_c
        ay = (vy_next - vy_prev) / dt_c

        speed_sq = vx * vx + vy * vy
        if speed_sq < eps:
            kappa[i] = 0.0
        else:
            kappa[i] = abs(vx * ay - vy * ax) / (speed_sq ** 1.5 + eps)

    kappa[0] = kappa[1]
    kappa[-1] = kappa[-2]

    kappa = kappa[np.isfinite(kappa)]

    return kappa


# ------------------------------------------------
# angular velocity (|d theta / dt|, theta = atan2(vy, vx))
# ------------------------------------------------

def compute_angular_velocity(xs, ys, ts):
    """
    Angular speed |omega| from heading theta = atan2(vy, vx).

    vx/vy from compute_vxvy; theta unwrapped; omega[i] = |theta[i+1]-theta[i]|/dt.
    Low-speed points (|v|^2 < eps) set to 0. Endpoints padded like curvature.
    """
    T = len(xs)

    if T < 2:
        return np.array([])

    vx, vy = compute_vxvy(xs, ys, ts)
    eps = 1e-8
    speed_sq = vx * vx + vy * vy

    theta = np.arctan2(vy, vx)
    if speed_sq[1] >= eps:
        theta[0] = theta[1]
    else:
        theta[0] = 0.0

    theta = np.unwrap(theta)

    dt = np.maximum(np.diff(ts), eps)
    omega = np.zeros(T)
    omega[:-1] = np.abs(np.diff(theta)) / dt
    omega[-1] = omega[-2]

    omega[speed_sq < eps] = 0.0

    omega = omega[np.isfinite(omega)]

    return omega


# ------------------------------------------------
# timediff node
# ------------------------------------------------

def compute_timediff_node(ts):

    if len(ts)<2:
        return np.array([])

    dt = ts[1:] - ts[:-1]
    dt = dt[dt>0]

    return dt


# ------------------------------------------------
# timediff pair
# ------------------------------------------------

def compute_timediff_pair(ts,chunk_size=150):

    if len(ts)<chunk_size:
        return np.array([])

    results=[]
    n_chunks=len(ts)//chunk_size

    for i in range(n_chunks):

        chunk=ts[i*chunk_size:(i+1)*chunk_size]

        diff=np.abs(chunk[:,None]-chunk[None,:])

        upper=diff[np.triu_indices(chunk_size,k=1)]

        results.append(upper)

    if results:
        return np.concatenate(results)

    return np.array([])


# ============================================================
# distribution builder
# ============================================================

def build_distribution(dataset,feature,training_root):

    print("\n[Step] Building distribution:",feature)

    all_values=[]
    all_vx=[]
    all_vy=[]

    users=sorted(os.listdir(training_root))

    for user in users:

        user_dir=os.path.join(training_root,user)

        if not os.path.isdir(user_dir):
            continue

        print("\nUser:",user)

        session_files=sorted(os.listdir(user_dir))

        for file in session_files:

            path=os.path.join(user_dir,file)

            if not os.path.isfile(path):
                continue

            print("   session:",file)

            df=pd.read_csv(path)

            if dataset=="balabit":
                df=clean_balabit(df)
            elif dataset=="chaoshen":
                df=clean_chaoshen(df)
            elif dataset=="dfl":
                df=clean_dfl(df)
            elif dataset=="twos":
                df=clean_twos(df)
            else:
                raise ValueError(f"Unsupported dataset: {dataset}")

            xs=df["x"].values
            ys=df["y"].values
            ts=df["time"].values

            if len(xs)<2:
                continue


            if feature=="velocity":

                v=compute_velocity(xs,ys,ts)
                v=v[np.isfinite(v)]
                all_values.append(v)


            elif feature=="vxvy":

                vx,vy=compute_vxvy(xs,ys,ts)
                vx=vx[np.isfinite(vx)]
                vy=vy[np.isfinite(vy)]

                all_vx.append(vx)
                all_vy.append(vy)
            
            elif feature=="velocity_pair":

                dv = compute_velocity_pair(xs, ys, ts, 150)

                if len(dv) > 0:
                    all_values.append(dv)


            elif feature=="acceleration":

                a=compute_acceleration(xs,ys,ts)

                if len(a) > 0:
                    a = np.abs(a)
                    a = a[np.abs(a) > 1e-8]
                    all_values.append(a)
                
            elif feature=="axay":

                ax, ay = compute_axay(xs, ys, ts)

                if len(ax) > 0:
                    all_vx.append(ax)
                if len(ay) > 0:
                    all_vy.append(ay)

            elif feature=="curvature":

                kappa = compute_curvature(xs, ys, ts)

                if len(kappa) > 0:
                    all_values.append(kappa)

            elif feature=="angular_velocity":

                omega = compute_angular_velocity(xs, ys, ts)

                if len(omega) > 0:
                    all_values.append(omega)

            elif feature=="timediff_node":

                dt=compute_timediff_node(ts)

                if len(dt)>0:
                    all_values.append(dt)


            elif feature=="timediff_pair":

                dt=compute_timediff_pair(ts,150)

                if len(dt)>0:
                    all_values.append(dt)


    if feature in ["vxvy", "axay"]:

        vx_values=np.concatenate(all_vx)
        vy_values=np.concatenate(all_vy)

        print("\nSummary")
        print_stats("VX",vx_values)
        print_stats("VY",vy_values)

        return vx_values,vy_values

    else:

        values=np.concatenate(all_values)

        print("\nSummary")
        print_stats(feature.upper(),values)

        return values


# ============================================================
# CLI
# ============================================================

def main():

    parser=argparse.ArgumentParser()

    parser.add_argument("--dataset",
        required=True,
        choices=["balabit","chaoshen","dfl","twos"])

    parser.add_argument("--feature",
        required=True,
        choices=[
            "velocity",
            "vxvy",
            "velocity_pair",
            "acceleration",
            "axay",
            "curvature",
            "angular_velocity",
            "timediff_node",
            "timediff_pair"
        ])

    parser.add_argument("--training_root",required=True)
    parser.add_argument("--out_dir",required=True)

    args=parser.parse_args()

    training_root=os.path.join(ROOT,args.training_root)
    out_path=os.path.join(ROOT,args.out_dir)


    if args.feature in ["vxvy", "axay"]:

        vx,vy=build_distribution(args.dataset,args.feature,training_root)

        np.savez_compressed(out_path,vx=vx,vy=vy)

    else:

        values=build_distribution(args.dataset,args.feature,training_root)

        np.savez_compressed(out_path,values=values)

    print("\nSaved to:",out_path)


if __name__=="__main__":
    main()