import carla
import random
import time
import math

def main():
    # Connect to CARLA
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(5.0)
    world = client.get_world()

    # Get the blueprint library and filter for a standard car
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle_bp.set_attribute('role_name', 'ego_vehicle')

    # Find a valid spawn point
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()

    # Spawn the vehicle
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(f"Spawned ego_vehicle at {spawn_point.location}")
    print("Spectator camera attached. Leave this script running...")
    
    # Get the simulator's spectator camera
    spectator = world.get_spectator()
    
    try:
        while True:
            # Constantly update the camera to follow the car
            transform = vehicle.get_transform()
            yaw = math.radians(transform.rotation.yaw)
            
            # Position the camera 8 meters behind and 3 meters above the car
            camera_x = transform.location.x - 8.0 * math.cos(yaw)
            camera_y = transform.location.y - 8.0 * math.sin(yaw)
            camera_z = transform.location.z + 3.0
            
            spectator_transform = carla.Transform(
                carla.Location(x=camera_x, y=camera_y, z=camera_z),
                carla.Rotation(pitch=-15.0, yaw=transform.rotation.yaw)
            )
            spectator.set_transform(spectator_transform)
            
            time.sleep(0.05) # Prevent the loop from maxing out CPU

    except KeyboardInterrupt:
        print("\nDestroying vehicle...")
        vehicle.destroy()

if __name__ == '__main__':
    main()