import carla
import math

class CarlaInterface:
    def __init__(self, host='127.0.0.1', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.ego_vehicle = None
        
        # Try to find an existing ego vehicle, or you'll need to spawn one in main
        for actor in self.world.get_actors().filter('vehicle.*'):
            if actor.attributes.get('role_name') == 'ego_vehicle':
                self.ego_vehicle = actor
                break

    def get_ego_state(self):
        if not self.ego_vehicle:
            return 0.0, 0.0, 0.0, 0.0
            
        transform = self.ego_vehicle.get_transform()
        velocity = self.ego_vehicle.get_velocity()
        
        x = transform.location.x
        y = transform.location.y
        # Convert CARLA degrees to radians for our controller
        yaw = -math.radians(transform.rotation.yaw)

        print(f"[M6 EGO] CARLA yaw={transform.rotation.yaw:.2f}°, M6 yaw={yaw:.3f} rad")
        
        # Calculate speed in m/s
        speed = math.hypot(velocity.x, velocity.y)
        
        return x, y, yaw, speed

    def apply_control(self, steer, throttle, brake, hand_brake=False):
        if not self.ego_vehicle:
            return
            
        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = brake
        control.hand_brake = hand_brake
        control.manual_gear_shift = False
        
        self.ego_vehicle.apply_control(control)