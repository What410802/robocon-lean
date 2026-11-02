#!/usr/bin/env python3
"""Evaluate the N-W-wolf `black_locked_wheels` JIT policy with a locked-wheel adapter.

The N-W-wolf policy is a 12-action, 6-frame-history HIMLoco policy for blackW:
16-DoF robot contract, policy actions only for the 12 leg joints; wheel targets
stay at default q=0 with kp=80/kd=3 (see their config.yaml).  This script is a
standalone diagnostic, not the delivery policy runner.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import mujoco
import torch

ROOT = Path(__file__).resolve().parents[2]
LEGS = ("FL", "FR", "RL", "RR")
Q_DEFAULT = np.array([0.0,0.8014,-1.527,0.0, 0.0,-0.8014,1.527,0.0,
                      -0.0,0.8014,-1.527,0.0, 0.0,-0.8014,1.527,0.0])
POLICY_INDICES = [0,1,2,4,5,6,8,9,10,12,13,14]
ACTION_SCALE = 0.25
KP = np.array([50,50,50,80, 50,50,50,80, 50,50,50,80, 50,50,50,80], float)
KD = np.array([1.2,1.2,1.2,3.0, 1.2,1.2,1.2,3.0, 1.2,1.2,1.2,3.0, 1.2,1.2,1.2,3.0])
TORQUE = np.array([23.7,23.7,59.25,20]*4)
CMD_SCALE = np.array([2.0,2.0,0.25])

def quat_inv_rotate(q, v):
    w,x,y,z = q; qv=np.array([x,y,z]); qc=np.array([-x,-y,-z])
    # q^-1 * v * q (vector part), q assumed unit
    t=2*np.cross(qc, v); return v + w*t + np.cross(qc,t)

def yaw(q):
    w,x,y,z=q; return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))

def build_obs(q, dq, base_quat, command, last_action):
    grav = quat_inv_rotate(base_quat, np.array([0.,0.,-1.]))
    ang = dq[3:6].copy()  # free-joint angular velocity (world-ish); use body via quat
    ang = quat_inv_rotate(base_quat, ang)
    # q/dq are 16 joint values in policy order (already selected via qadr/vadr).
    qrel = q[POLICY_INDICES] - Q_DEFAULT[POLICY_INDICES]
    dqj = dq[POLICY_INDICES]
    return np.concatenate([command*CMD_SCALE, ang*0.25, grav, qrel*1.0, dqj*0.05, last_action])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--scene', default=str(ROOT/'demo/robots/blackW/smoke_flat.xml'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--steps', type=int, default=300)
    args=ap.parse_args()
    m=mujoco.MjModel.from_xml_path(args.scene); d=mujoco.MjData(m)
    net=torch.jit.load(args.policy,map_location='cpu'); net.eval()
    dt=m.opt.timestep
    # joint qpos addresses by name in policy order
    names=['FL_hip_joint','FL_thigh_joint','FL_calf_joint','FL_wheel_joint','FR_hip_joint','FR_thigh_joint','FR_calf_joint','FR_wheel_joint','RL_hip_joint','RL_thigh_joint','RL_calf_joint','RL_wheel_joint','RR_hip_joint','RR_thigh_joint','RR_calf_joint','RR_wheel_joint']
    qadr=np.array([m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in names])
    vadr=np.array([m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in names])
    # actuator index for each policy-order joint (XML actuator order differs)
    act_ids=np.array([
        [a for a in range(m.nu) if int(m.actuator_trnid[a,0]) == mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)][0]
        for n in names
    ])
    # root free joint qpos/qvel addresses
    d.qpos[:]=0; d.qvel[:]=0; d.qpos[0:3]=[0,0,0.69]; d.qpos[3:7]=[1,0,0,0]; d.qpos[qadr]=Q_DEFAULT
    mujoco.mj_forward(m,d)
    # settle with default wheel lock
    def apply(tau):
        d.ctrl[act_ids] = np.clip(tau, -TORQUE, TORQUE)
    def pd_targets(qpos, qvel, target):
        tau=KP*(target-qpos)-KD*qvel
        return np.clip(tau,-TORQUE,TORQUE)
    for _ in range(500):
        q=d.qpos[qadr]; dq=d.qvel[vadr]
        tau=pd_targets(q,dq,Q_DEFAULT)
        apply(tau); mujoco.mj_step(m,d)
    hist=[]; last=np.zeros(12)
    cmd=np.array([0.0,0.0,0.0])
    logs=[]
    for step in range(args.steps):
        # command profile: 0-50 stand, 50-200 forward 0.2, 200-300 yaw +0.3
        if step<50: cmd=np.array([0.,0.,0.])
        elif step<200: cmd=np.array([0.5,0.,0.])
        else: cmd=np.array([0.,0.,0.8])
        q=d.qpos[qadr]; dq=d.qvel[vadr]
        ob=build_obs(q,dq,np.array(d.qpos[3:7]),cmd,last)
        hist.append(ob)
        if len(hist)>6: hist.pop(0)
        while len(hist)<6: hist.insert(0,ob.copy())
        # N-W-wolf ObservationBuffer returns newest-first history.
        inp=torch.tensor(np.concatenate(list(reversed(hist))),dtype=torch.float32).view(1,-1)
        with torch.no_grad(): action=net(inp).numpy()[0]
        if step < 5 or step % 50 == 0:
            print('step', step, 'obs', np.round(inp.numpy()[0,:12],3).tolist(), 'act', np.round(action,3).tolist(), 'z', float(d.qpos[2]), flush=True)
        last=action
        target=Q_DEFAULT.copy(); target[POLICY_INDICES]+=ACTION_SCALE*action
        # 10 physics steps @2ms = 20ms control
        for _ in range(10):
            tqq=d.qpos[qadr]; tdq=d.qvel[vadr]
            apply(pd_targets(tqq,tdq,target)); mujoco.mj_step(m,d)
        logs.append((step*10*dt, cmd.copy(), d.qpos[:3].copy(), yaw(d.qpos[3:7]), d.qvel[0:3].copy()))
    arr=np.array([[l[0], *l[1], *l[2], l[3], *l[4]] for l in logs])
    # columns: t, cmd_vx, cmd_vy, cmd_wz, x, y, z, yaw, vx, vy, vz
    print('t cmd_vx cmd_vy cmd_wz x y z yaw vx vy')
    for i in range(0,len(arr),50):
        r=arr[i]; print(f'{r[0]:5.2f} {r[1]:+.2f} {r[2]:+.2f} {r[3]:+.2f}  {r[4]:+.2f} {r[5]:+.2f} {r[6]:.2f} {math.degrees(r[7]):+6.1f} {r[8]:+.2f} {r[9]:+.2f}')
    if args.out:
        np.save(args.out,arr); print('saved',args.out)

if __name__=='__main__': main()
