So, how do we actually use tactile sensors in Isaac Lab via TacEx?

In general you need to
- use the robot asset with the sensor models
- initialize the sensors

>[!NOTE]
> Depending on the tactile simulation you have different limitations.
> For more details see [Reference Table].
> The GPU based taxim approach with phyx rigid bodies works with Isaac Lab and can be used for direct and manager based workflow.
> The UIPC simulation currently only works with a direct workflow and requires the use of `UipcRLEnv` (instead of the default `DirectRLEnv`).

In the following an example which shows you how to we added GelSight sensors to the existing factory environment of Isaac Lab.

The full env can be found in `tacex/source/tacex_tasks/tacex_tasks`.

# Direct Workflow

We used TacEx primarily for envs in the direct workflow, since this allows for more flexibility in the implementation details.
The UIPC simulation is currently only implemented for a direct workflow.

- copy the factory task folder from `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/factory` into `tacex/source/tacex_tasks/tacex_tasks`
- adjust the `factory_env_cfg.py`:
  - from tacex_assets import TACEX_ASSETS_DATA_DIR
  - change the spawn config of the robot to use our Gripper asset `usd_path=f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Gripper/physx_rigid_gelpads.usd",`
  - add configs for the tactile sensors
```python
    gsmini_left = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
        sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix = "/Camera",
            update_period= 0,
            resolution = (32,32),
        ),
        device = "cuda",
        debug_vis=True, # for rendering sensor output in the gui
        # update Taxim cfg
        marker_motion_sim_cfg=None,
        data_types=["tactile_rgb"], #marker_motion
    )
    # settings for optical sim
    gsmini_left.optical_sim_cfg = gsmini_left.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(32, 32),
    )
    gsmini_right = gsmini_left.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
    )
```

- register the env's by adjusting `__init__.py`:
  - id=`TacEx-Factory-PegInsert-Direct-v0`, ?
  - entry_point=`f"{__name__}.factory_env:FactoryEnv"`


- spawn the sensors by adding this to `_setup_scene()` method of `factory_env.py`
```python
        # sensors
        self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
        self.scene.sensors["gsmini_left"] = self.gsmini_left

        self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
        self.scene.sensors["gsmini_right"] = self.gsmini_right

```

You can train the envs via:
> Call from source dir of the TacEx repo

- `isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-PegInsert-Direct-v0 --num_envs 206 --enable_cameras`
- `isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-GearMesh-Direct-v0 --num_envs 206 --enable_cameras`
- `isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-NutThread-Direct-v0 --num_envs 206 --enable_cameras`
