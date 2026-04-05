import gymnasium as gym

from . import agents, flat_env_cfg


gym.register(
    id="Hopping-Flat-Go2-v0",
    entry_point="whole_body_tracking.tasks.hopping.go2_hopping_env:Go2HoppingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.go2_hopping_flat_env_cfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2HoppingFlatPPORunnerCfg",
    },
)

gym.register(
    id="Hopping-Flat-Go2-TrackingCfg-v0",
    entry_point="whole_body_tracking.tasks.hopping.go2_hopping_env:Go2HoppingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.go2_hopping_flat_tracking_cfg_env_cfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2HoppingFlatPPORunnerCfg",
    },
)

gym.register(
    id="Hopping-Trampoline-Go2-TrackingCfg-v0",
    entry_point="whole_body_tracking.tasks.hopping.go2_hopping_env:Go2HoppingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.go2_hopping_trampoline_tracking_cfg_env_cfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2HoppingTrampolineTrackingCfgPPORunnerCfg",
    },
)

