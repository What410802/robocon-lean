#!/usr/bin/env python3
"""Publish adaptive forward commands on /joy from /eval_state (deployment trial)."""
import argparse, json, math, sys, time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Joy
from std_msgs.msg import String

def wrap(a):
    return (a + math.pi) % (2*math.pi) - math.pi

class A(Node):
    def __init__(self, a):
        super().__init__('adaptive_joy')
        self.a=a; self.pose=None; self.prev=None
        self.pub=self.create_publisher(Joy,'joy',QoSProfile(depth=1))
        self.create_subscription(String,'/eval_state',self.cb,qos_profile_sensor_data)
    def cb(self,msg):
        try:
            d=json.loads(msg.data); self.pose=(float(d['base'][0]),float(d['base'][1]),float(d['base'][2]),
                math.atan2(2*(d['quat'][0]*d['quat'][3]+d['quat'][1]*d['quat'][2]),
                           1-2*(d['quat'][2]*d['quat'][2]+d['quat'][3]*d['quat'][3])), float(d.get('t',0)))
        except Exception: pass
    def run(self):
        lane_yaw=math.radians(self.a.lane_yaw_deg); cy=self.a.center_y
        t0=time.monotonic(); last_t=None
        while time.monotonic()-t0 < self.a.seconds:
            rclpy.spin_once(self, timeout_sec=0.0)
            if self.pose is None: time.sleep(0.02); continue
            x,y,z,yaw,t=self.pose
            spsi,cpsi=math.sin(lane_yaw),math.cos(lane_yaw)
            e=-(x-0)*spsi+(y-cy)*cpsi
            vy=float(max(-0.2,min(0.2,-0.5*e)))
            h=wrap(yaw-lane_yaw)
            wzb=0.0
            if self.prev is not None and t>self.prev[3]:
                dt=t-self.prev[3]
                vxw=(x-self.prev[0])/dt; vyw=(y-self.prev[1])/dt
                wzb=-spsi*vxw+cpsi*vyw
            wz=float(max(-0.7,min(0.7,-0.8*e-2.0*h-0.6*wzb)))
            m=Joy(); m.buttons=[0]*11
            m.axes=[vy/1.5, self.a.vx/1.5, 0.0, wz/1.5, 0.0,0.0,0.0,0.0]
            self.pub.publish(m); self.prev=(x,y,z,t)
            time.sleep(0.02)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--vx',type=float,default=0.5)
    ap.add_argument('--center-y',type=float,default=0.0); ap.add_argument('--lane-yaw-deg',type=float,default=0.0)
    ap.add_argument('--seconds',type=float,default=18.0)
    a=ap.parse_args(); rclpy.init(); n=A(a)
    import threading; th=None
    try:
        n.run()
    finally:
        n.destroy_node(); rclpy.shutdown()
    return 0
if __name__=='__main__': sys.exit(main())
