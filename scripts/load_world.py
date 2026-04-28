#!/usr/bin/env python
import argparse
import sys
import time
import carla


def load_world_with_layers(minimal_layers=True):
    try:
        print("Connecting to CARLA server...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        
        print("Loading Town05_Opt with minimal layers...")
        world = client.load_world('Town05_Opt', map_layers=carla.MapLayer.NONE)
        new_settings = world.get_settings()
        new_settings.synchronous_mode = True
        new_settings.fixed_delta_seconds = 0.1
        world.apply_settings(new_settings)
        time.sleep(1)
        minimal_layers = True

        if minimal_layers:
            print("Configuring map layers (minimal mode - unloading specific layers)...")
            try:
                world.load_map_layer(carla.MapLayer.All)
                world.unload_map_layer(carla.MapLayer.Buildings)
                world.unload_map_layer(carla.MapLayer.ParkedVehicles)
                world.unload_map_layer(carla.MapLayer.Walls)
                world.unload_map_layer(carla.MapLayer.Foliage)
                world.unload_map_layer(carla.MapLayer.Decals)
                world.unload_map_layer(carla.MapLayer.StreetLights)
                world.unload_map_layer(carla.MapLayer.Props)
                print("Minimal layers configuration applied.")
            except RuntimeError as e:
                print(f"Error configuring minimal map layers: {e}")
                return 1
        else:
            print("Loading all map layers (full mode)...")
            try:
                world.load_map_layer(carla.MapLayer.All)
                print("All layers loaded successfully.")
            except RuntimeError as e:
                print(f"Error loading all map layers: {e}")
                return 1

        current_world = client.get_world()
        print(f"World after loading and configuration: '{current_world.get_map().name}'")
        world.tick()
        
        layer_mode = "minimal" if minimal_layers else "full"
        print(f"World loaded and configured successfully with {layer_mode} layers!")
        return 0
    except Exception as e:
        print(f"Error loading world: {e}")
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load CARLA world with configurable layer options')
    parser.add_argument(
        '--minimal-layers',
        action='store_true',
        help='Use minimal layers by unloading buildings, vehicles, walls, etc.'
    )
    
    args = parser.parse_args()
    sys.exit(load_world_with_layers(minimal_layers=args.minimal_layers)) 